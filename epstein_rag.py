#!/usr/bin/env python3
"""
Download + RAG over DOJ Epstein Library datasets.

Official pages:
- https://www.justice.gov/epstein
- https://www.justice.gov/epstein/doj-disclosures

This script:
- Scrapes dataset pages for file links
- Downloads with retry + resume
- Extracts PDF text (PyMuPDF)
- Builds embeddings index (SentenceTransformers + FAISS)
- Answers questions with citations to source files (local Ollama or OpenAI)

NOTE on age gate:
DOJ uses an age-verification gate. In practice, setting a cookie often works:
  Cookie: justiceGovAgeVerified=true
If that ever changes, open a dataset page in a browser, click "Yes", then export
cookies and adapt `AGE_COOKIE_NAME/VAL` below.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# PDF text extraction
import fitz  # PyMuPDF

# Embeddings + vector index
import faiss
from sentence_transformers import SentenceTransformer


BASE = "https://www.justice.gov"
DOJ_DISCLOSURES = "https://www.justice.gov/epstein/doj-disclosures"

# Best-effort age verification cookie (commonly used on justice.gov age gates)
AGE_COOKIE_NAME = "justiceGovAgeVerified"
AGE_COOKIE_VAL = "true"

DEFAULT_OUTDIR = Path("epstein_downloads")
DEFAULT_INDEXDIR = Path("epstein_index")

# --- Basic redaction helpers (to reduce accidental victim/PII leakage in Q&A outputs) ---
PII_PATTERNS = [
    # emails
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[REDACTED_EMAIL]"),
    # phone-like
    (re.compile(r"\b(\+?\d[\d\s().-]{7,}\d)\b"), "[REDACTED_PHONE]"),
    # SSN-like
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
]

def redact_pii(text: str) -> str:
    out = text
    for pat, repl in PII_PATTERNS:
        out = pat.sub(repl, out)
    return out


@dataclass
class Chunk:
    doc_path: str
    page: int
    chunk_id: str
    text: str


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "epstein-rag/1.0 (research; respectful; contact: none)",
        "Accept": "text/html,application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    })
    s.cookies.set(AGE_COOKIE_NAME, AGE_COOKIE_VAL, domain="www.justice.gov")
    return s


def fetch_html(session: requests.Session, url: str, timeout: int = 60) -> str:
    r = session.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.text


def discover_dataset_pages(session: requests.Session) -> Dict[str, str]:
    """
    Returns mapping: dataset_name -> dataset_page_url
    """
    html = fetch_html(session, DOJ_DISCLOSURES)
    soup = BeautifulSoup(html, "html.parser")

    links = {}
    for a in soup.select("a"):
        label = (a.get_text() or "").strip()
        href = a.get("href") or ""
        if "data-set" in href and "files" in href and "View files" in label:
            # The preceding text usually indicates "Data Set N"
            # We'll read nearby content for a dataset number if possible.
            parent_text = a.find_parent().get_text(" ", strip=True) if a.find_parent() else label
            m = re.search(r"Data Set\s+(\d+)", parent_text, re.I)
            ds = f"DataSet {m.group(1)}" if m else href
            links[ds] = urljoin(BASE, href)

    # Fallback: if the "View files" anchors are not easy to parse,
    # scan all dataset links by pattern.
    if not links:
        for a in soup.select("a[href]"):
            href = a["href"]
            if re.search(r"/epstein/doj-disclosures/data-set-\d+-files", href):
                ds_num = re.search(r"data-set-(\d+)-files", href).group(1)
                links[f"DataSet {ds_num}"] = urljoin(BASE, href)

    if not links:
        raise RuntimeError("Could not discover dataset pages. DOJ page structure may have changed.")

    return dict(sorted(links.items(), key=lambda kv: int(re.search(r"\d+", kv[0]).group(0))))


def discover_file_links_in_dataset(session: requests.Session, dataset_page_url: str) -> List[str]:
    """
    Walks pagination within a dataset page, returns absolute URLs to files.
    """
    seen_pages = set()
    file_urls: List[str] = []

    def extract_links(page_url: str) -> Tuple[List[str], List[str]]:
        html = fetch_html(session, page_url)
        soup = BeautifulSoup(html, "html.parser")

        # File links: often like /epstein/files/DataSet%2012/EFTA....pdf
        files = []
        for a in soup.select("a[href]"):
            href = a["href"]
            if re.search(r"/epstein/files/", href) and re.search(r"\.(pdf|zip|mp4|mp3|wav|jpg|jpeg|png|tif|tiff)$", href, re.I):
                files.append(urljoin(BASE, href))

        # Pagination links (often "Next", page numbers)
        next_pages = []
        for a in soup.select("a[href]"):
            txt = (a.get_text() or "").strip().lower()
            href = a["href"]
            if txt in {"next", "last"} or re.fullmatch(r"\d+", txt):
                if "data-set" in href and "files" in href:
                    next_pages.append(urljoin(BASE, href))

        return files, next_pages

    queue = [dataset_page_url]
    while queue:
        page = queue.pop(0)
        if page in seen_pages:
            continue
        seen_pages.add(page)

        files, next_pages = extract_links(page)
        file_urls.extend(files)

        for npg in next_pages:
            if npg not in seen_pages:
                queue.append(npg)

    # de-dup
    return sorted(set(file_urls))


def safe_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = os.path.basename(path)
    # in case of weirdness
    return name or hashlib.sha256(url.encode()).hexdigest()


def download_one(session: requests.Session, url: str, outpath: Path, retries: int = 5) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = outpath.with_suffix(outpath.suffix + ".part")

    # Resume if partial exists
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=120, headers=headers, allow_redirects=True) as r:
                # If server doesn't support Range, it may return 200 with full content
                if r.status_code not in (200, 206):
                    r.raise_for_status()

                mode = "ab" if r.status_code == 206 and resume_from > 0 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            tmp.rename(outpath)
            return
        except Exception as e:
            if attempt == retries:
                raise
            backoff = min(60, 2 ** attempt)
            time.sleep(backoff)


def download_all(file_urls: List[str], outdir: Path, workers: int = 6) -> None:
    session = make_session()
    # Group by dataset folder when possible
    tasks = []
    for u in file_urls:
        # try to preserve dataset directory structure
        m = re.search(r"/epstein/files/(DataSet%20?\d+|DataSet\s?\d+|DataSet%201\d)/(.+)$", u, re.I)
        if m:
            ds = m.group(1).replace("%20", " ")
            rel = m.group(2)
            target = outdir / ds / safe_filename_from_url(u)
        else:
            target = outdir / "misc" / safe_filename_from_url(u)
        tasks.append((u, target))

    failures = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, session, u, p): (u, p) for (u, p) in tasks}
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Downloading"):
            u, p = futures[fut]
            try:
                fut.result()
            except Exception as e:
                failures.append({"url": u, "path": str(p), "error": str(e)})

    if failures:
        fail_path = outdir / "download_failures.json"
        fail_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"\nSome downloads failed. See: {fail_path}")


def iter_pdfs(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.pdf"):
        if p.is_file():
            yield p


def chunk_text(text: str, max_chars: int = 2400, overlap: int = 300) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks


def extract_pdf_chunks(pdf_path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return chunks

    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            txt = page.get_text("text") or ""
        except Exception:
            txt = ""
        txt = redact_pii(txt)
        for k, c in enumerate(chunk_text(txt)):
            cid = hashlib.sha1(f"{pdf_path}:{page_num}:{k}".encode()).hexdigest()
            chunks.append(Chunk(
                doc_path=str(pdf_path),
                page=page_num + 1,
                chunk_id=cid,
                text=c
            ))
    return chunks


def build_index(download_dir: Path, index_dir: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "chunks.sqlite3"
    faiss_path = index_dir / "index.faiss"
    meta_path = index_dir / "meta.json"

    # SQLite for chunk storage
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_path TEXT,
            page INTEGER,
            text TEXT
        )
    """)
    conn.commit()

    # Load embedding model
    embedder = SentenceTransformer(model_name)

    # Collect chunks
    all_chunks: List[Chunk] = []
    pdfs = list(iter_pdfs(download_dir))
    for pdf in tqdm(pdfs, desc="Extracting PDFs"):
        all_chunks.extend(extract_pdf_chunks(pdf))

    # Dedup already-indexed
    existing = set(r[0] for r in cur.execute("SELECT chunk_id FROM chunks").fetchall())
    new_chunks = [c for c in all_chunks if c.chunk_id not in existing]
    if not new_chunks:
        print("No new chunks to index.")
        return

    # Insert into sqlite
    cur.executemany(
        "INSERT OR IGNORE INTO chunks(chunk_id, doc_path, page, text) VALUES (?, ?, ?, ?)",
        [(c.chunk_id, c.doc_path, c.page, c.text) for c in new_chunks]
    )
    conn.commit()

    # Embed
    texts = [c.text for c in new_chunks]
    embs = embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)

    # Load or create FAISS
    dim = embs.shape[1]
    if faiss_path.exists():
        index = faiss.read_index(str(faiss_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        index = faiss.IndexFlatIP(dim)
        meta = {"model": model_name, "ids": []}

    index.add(embs)
    meta["ids"].extend([c.chunk_id for c in new_chunks])

    faiss.write_index(index, str(faiss_path))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Indexed {len(new_chunks)} new chunks.")


def search_index(index_dir: Path, query: str, k: int = 8) -> List[Dict]:
    db_path = index_dir / "chunks.sqlite3"
    faiss_path = index_dir / "index.faiss"
    meta_path = index_dir / "meta.json"
    if not (db_path.exists() and faiss_path.exists() and meta_path.exists()):
        raise RuntimeError("Index not found. Run: python epstein_rag.py index")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(faiss_path))
    embedder = SentenceTransformer(meta["model"])

    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)

    scores, idxs = index.search(q_emb, k)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    ids = meta["ids"]
    picked = []
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        if i < 0 or i >= len(ids):
            continue
        picked.append({"chunk_id": ids[i], "score": float(s), "rank": rank})

    # hydrate from sqlite
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    results = []
    for item in picked:
        row = cur.execute(
            "SELECT doc_path, page, text FROM chunks WHERE chunk_id=?",
            (item["chunk_id"],)
        ).fetchone()
        if not row:
            continue
        doc_path, page, text = row
        results.append({
            "rank": item["rank"],
            "score": item["score"],
            "doc_path": doc_path,
            "page": page,
            "text": text
        })
    return results


def answer_with_ollama(model: str, question: str, contexts: List[Dict]) -> str:
    import ollama

    ctx = "\n\n".join(
        [f"[{i+1}] File: {c['doc_path']} (page {c['page']})\n{c['text']}"
         for i, c in enumerate(contexts)]
    )

    system = (
        "You are a careful analyst answering questions about a set of documents. "
        "Use ONLY the provided excerpts as evidence. "
        "If the answer is not supported by the excerpts, say so. "
        "Avoid repeating personal identifying information about victims or private individuals."
    )

    prompt = (
        f"Question:\n{question}\n\n"
        f"Excerpts:\n{ctx}\n\n"
        "Answer in a concise way, and include citations like [1], [2] referring to the excerpts."
    )

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp["message"]["content"]


def answer_with_openai(model: str, question: str, contexts: List[Dict]) -> str:
    from openai import OpenAI
    client = OpenAI()

    ctx = "\n\n".join(
        [f"[{i+1}] File: {c['doc_path']} (page {c['page']})\n{c['text']}"
         for i, c in enumerate(contexts)]
    )

    system = (
        "You are a careful analyst answering questions about a set of documents. "
        "Use ONLY the provided excerpts as evidence. "
        "If the answer is not supported by the excerpts, say so. "
        "Avoid repeating personal identifying information about victims or private individuals."
    )

    user = (
        f"Question:\n{question}\n\n"
        f"Excerpts:\n{ctx}\n\n"
        "Answer concisely and include citations like [1], [2] that refer to excerpt numbers."
    )

    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return r.choices[0].message.content


def cmd_download(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    session = make_session()
    datasets = discover_dataset_pages(session)

    if args.dataset.lower() != "all":
        # accept "12" or "DataSet 12"
        wanted = args.dataset.strip().lower()
        filtered = {}
        for name, url in datasets.items():
            if wanted == name.lower() or wanted == re.search(r"\d+", name).group(0):
                filtered[name] = url
        if not filtered:
            raise SystemExit(f"Dataset not found. Available: {', '.join(datasets.keys())}")
        datasets = filtered

    all_files = []
    for name, page_url in datasets.items():
        print(f"\nDiscovering files in {name}: {page_url}")
        files = discover_file_links_in_dataset(session, page_url)
        print(f"  found {len(files)} files")
        all_files.extend(files)

    all_files = sorted(set(all_files))
    (outdir / "discovered_files.json").write_text(json.dumps(all_files, indent=2), encoding="utf-8")
    print(f"\nTotal unique files discovered: {len(all_files)}")
    print(f"Saved list to: {outdir / 'discovered_files.json'}")

    if args.list_only:
        return

    download_all(all_files, outdir, workers=args.workers)


def cmd_index(args: argparse.Namespace) -> None:
    build_index(Path(args.downloaddir), Path(args.indexdir), model_name=args.embed_model)


def cmd_ask(args: argparse.Namespace) -> None:
    contexts = search_index(Path(args.indexdir), args.question, k=args.k)

    print("\nTop matches:")
    for c in contexts:
        print(f"  [{c['rank']}] score={c['score']:.3f} {c['doc_path']} (p.{c['page']})")

    if args.no_llm:
        print("\n--- Context excerpts ---")
        for c in contexts:
            print(f"\n[{c['rank']}] {c['doc_path']} (page {c['page']})\n{c['text']}")
        return

    if args.provider == "ollama":
        ans = answer_with_ollama(args.model, args.question, contexts)
    elif args.provider == "openai":
        ans = answer_with_openai(args.model, args.question, contexts)
    else:
        raise SystemExit("provider must be 'ollama' or 'openai'")

    print("\n--- Answer ---")
    print(ans)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="Discover and download dataset files from DOJ Epstein Library")
    dl.add_argument("--dataset", default="all", help="e.g. all | 12 | 'DataSet 12'")
    dl.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    dl.add_argument("--workers", type=int, default=6)
    dl.add_argument("--list-only", action="store_true", help="Only discover and write discovered_files.json")
    dl.set_defaults(func=cmd_download)

    ix = sub.add_parser("index", help="Build local RAG index from downloaded PDFs")
    ix.add_argument("--downloaddir", default=str(DEFAULT_OUTDIR))
    ix.add_argument("--indexdir", default=str(DEFAULT_INDEXDIR))
    ix.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ix.set_defaults(func=cmd_index)

    ask = sub.add_parser("ask", help="Ask questions using RAG (Ollama or OpenAI)")
    ask.add_argument("question")
    ask.add_argument("--indexdir", default=str(DEFAULT_INDEXDIR))
    ask.add_argument("-k", type=int, default=8)
    ask.add_argument("--no-llm", action="store_true", help="Only print retrieved excerpts")
    ask.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    ask.add_argument("--model", default="llama3.1", help="Ollama model name OR OpenAI model name")
    ask.set_defaults(func=cmd_ask)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
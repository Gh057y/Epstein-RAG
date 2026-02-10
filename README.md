# Epstein-RAG
Download and RAG to query Epstein files

5min work with ChatGPT - untested
## install dependencies


> python -m venv .venv
> source .venv/bin/activate
>
> pip install -U pip
> pip install requests beautifulsoup4   tqdm pymupdf sentence-transformers faiss-cpu numpy
> 
> optional (recommended) for local LLM chat:
>> pip install ollama
>
> optional for OpenAI:
>> pip install openai

### requirements.txt

> requests
> beautifulsoup4
> tqdm
> pymupdf
> sentence-transformers
> faiss-cpu
> numpy
> ollama
> openai

### make executable

> chmod +x epstein_rag.py
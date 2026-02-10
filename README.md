# Epstein-RAG
Download and RAG to query Epstein files

5min work with ChatGPT - untested

## Setup

### install dependencies


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

## usage:

### Discover file links (no downloads yet)

> python epstein_rag.py download --dataset all --list-only

### Download (example: just dataset 12)

> python epstein_rag.py download --dataset 12 --workers 8

### Build the index from downloaded PDFs

> python epstein_rag.py index --downloaddir epstein_downloads --indexdir epstein_index

## Ask Questions (ollama):

### pull ollama model

> ollama pull llama3.1

### Ask question

> python epstein_rag.py ask "What does the evidence list say about items seized?" --provider ollama --model llama3.1

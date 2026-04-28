# WikiRAG — Local Wikipedia RAG System

A fully local, ChatGPT-style question-answering system that answers questions about famous people and places using Wikipedia data, a local embedding model, ChromaDB, and a locally-hosted LLM via Ollama.

> **No external API calls. Runs entirely on your laptop.**

---

## Demo Video

🎥 [Insert your Loom/YouTube link here]

---

## System Overview

```
Wikipedia → SQLite (raw text) → Chunker → ChromaDB (vectors)
                                                  ↓
User Query → Classifier → Retriever → Prompt Builder → Ollama LLM → Answer
```

- **Ingestion:** Fetches Wikipedia articles and stores raw text in SQLite
- **Chunking:** Splits documents into 500-character chunks with 50-character overlap
- **Embedding:** Uses `nomic-embed-text` (via Ollama) to embed chunks
- **Storage:** Two ChromaDB collections — `people` and `places`
- **Retrieval:** Classifies query, fetches top-5 relevant chunks
- **Generation:** `llama3.2:3b` generates a grounded answer from retrieved context
- **UI:** Streamlit web app + CLI fallback

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- ~8 GB free disk space (models + vector DB)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/wikirag.git
cd wikirag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull required Ollama models

```bash
# Pull the language model
ollama pull llama3.2:3b

# Pull the embedding model
ollama pull nomic-embed-text
```

Verify Ollama is running:
```bash
ollama list
```

---

## Ingest Data

This fetches Wikipedia articles for all 40+ entities and stores them in SQLite, then chunks and embeds them into ChromaDB.

```bash
python ingest.py
```

You will see progress output as each article is fetched, chunked, and embedded.

> **Note:** Ingestion is idempotent — safe to re-run without duplicating data.

Expected output:
```
[✓] Fetched: Albert Einstein (person)
[✓] Fetched: Marie Curie (person)
...
[✓] Embedded 847 chunks into 'people' collection
[✓] Embedded 634 chunks into 'places' collection
Ingestion complete.
```

---

## Start the Application

### Option A — Streamlit Web UI (recommended)

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### Option B — Command Line Interface

```bash
python cli.py
```

---

## Usage

### Streamlit UI

1. Type your question in the chat input box
2. Press Enter to submit
3. Toggle **"Show source chunks"** in the sidebar to see retrieved context
4. Click **"Clear conversation"** in the sidebar to reset

### CLI

```
WikiRAG > Who was Albert Einstein?
WikiRAG > What is the Eiffel Tower?
WikiRAG > Compare Lionel Messi and Cristiano Ronaldo
WikiRAG > quit
```

---

## Example Queries

### People
```
Who was Albert Einstein and what is he known for?
What did Marie Curie discover?
Why is Nikola Tesla famous?
Compare Lionel Messi and Cristiano Ronaldo
What is Frida Kahlo known for?
```

### Places
```
Where is the Eiffel Tower located?
Why is the Great Wall of China important?
What was the Colosseum used for?
Where is Mount Everest?
What is Machu Picchu?
```

### Mixed
```
Which famous place is located in Turkey?
Which person is associated with electricity?
Compare the Eiffel Tower and the Statue of Liberty
Compare Albert Einstein and Nikola Tesla
```

### Expected Failures (I don't know)
```
Who is the president of Mars?
Tell me about John Doe
What is the capital of Atlantis?
```

---

## Project Structure

```
wikirag/
├── app.py                  # Streamlit web UI
├── cli.py                  # Command-line interface
├── ingest.py               # Main ingestion entry point
│
├── wikirag/
│   ├── __init__.py
│   ├── config.py           # Constants and configuration
│   ├── fetcher.py          # Wikipedia API fetcher
│   ├── chunker.py          # Text chunking logic
│   ├── embedder.py         # Embedding via Ollama / sentence-transformers
│   ├── store.py            # ChromaDB operations
│   ├── database.py         # SQLite operations
│   ├── classifier.py       # Query classification (person/place/both)
│   ├── retriever.py        # Vector similarity retrieval
│   └── generator.py        # LLM answer generation via Ollama
│
├── data/
│   ├── entities.py         # List of all people and places to ingest
│   └── wikirag.db          # SQLite database (created at runtime)
│
├── chroma_db/              # ChromaDB persistent storage (created at runtime)
│
├── requirements.txt
├── README.md
├── product_prd.md
└── recommendation.md
```

---

## Configuration

Edit `wikirag/config.py` to adjust:

```python
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks
TOP_K = 5                 # Number of chunks to retrieve
RELEVANCE_THRESHOLD = 1.2 # Cosine distance above which to return "I don't know"
OLLAMA_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
```

---

## Requirements

```
chromadb>=0.4.0
requests>=2.31.0
streamlit>=1.32.0
ollama>=0.1.7
```

See `requirements.txt` for pinned versions.

---

## Troubleshooting

**Ollama not found / connection refused**
```bash
ollama serve   # Start Ollama if it's not running
```

**ChromaDB version conflicts**
```bash
pip install chromadb==0.4.24
```

**Slow ingestion**
- Embedding 40+ articles with nomic-embed-text takes 5–15 minutes on CPU
- You can reduce the entity list in `data/entities.py` for faster testing

**Model not responding**
```bash
ollama pull llama3.2:3b   # Re-pull if model seems missing
```

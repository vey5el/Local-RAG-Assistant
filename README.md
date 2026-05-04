# WikiRAG — Local Wikipedia RAG System
### BLG483E Project 3

A fully local, ChatGPT-style question-answering system that answers questions about famous people and places using Wikipedia data, local embedding models, ChromaDB, and locally-hosted LLMs via Ollama.

> **No external API calls. No internet required at query time. Runs entirely on your laptop.**

---

## Demo Video

🎥 [[Loom link here](https://www.loom.com/share/cddd83540be346779f81c9a1c9f2866e)]

---
## GitHub Repo

[[GitHub Repo link is here](https://github.com/vey5el/Local-RAG-Assistant)]

---

## System Overview

```
Wikipedia API
     ↓
SQLite (raw text cache)
     ↓
Chunker — 5 configurable profiles (tiny / small / medium / large / xl)
     ↓
Embedder — minilm (sentence-transformers) or nomic (Ollama)
     ↓
ChromaDB — 10 collection pairs (5 profiles × 2 embedders)
     ↓
User Query → Classifier → Entity-Aware Retriever → Prompt Builder
                                                          ↓
                                              Ollama LLM (llama3 / phi3 / mistral)
                                                          ↓
                                                       Answer
```

---

## What Makes This System Special

### Beyond the spec
The assignment required a basic RAG system. This implementation goes further:

- **5 chunk profiles** — compare tiny vs xl chunks side by side in the same UI
- **2 embedding models** — minilm (fast, CUDA) and nomic (higher quality, Ollama)
- **3 LLMs** — switch between llama3, phi3, and mistral instantly without re-ingestion
- **Entity-aware retrieval** — when a query mentions "Mahatma Gandhi", the retriever filters ChromaDB to Gandhi's chunks *before* similarity ranking, preventing semantic drift
- **Comprehensive benchmark** — 1,950 LLM calls across all 30 combinations, full timing and quality analysis

---

## Selectable Options

| Dimension | Options | Needs re-ingestion? |
|-----------|---------|-------------------|
| **Chunk profile** | tiny / small / medium / large / xl | ✅ Yes |
| **Embedding model** | minilm / nomic | ✅ Yes |
| **LLM** | llama3 / phi3 / mistral | ❌ No — switch instantly |

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- ~8 GB free disk space (models + vectors)
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
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull Ollama models

```bash
# Start Ollama
ollama serve

# Pull LLMs (pull at least llama3.2:3b)
ollama pull llama3.2:3b
ollama pull phi3
ollama pull mistral

# Pull embedding model (only needed for nomic embedder)
ollama pull nomic-embed-text
```

Verify:
```bash
ollama list
```

---

## Ingest Data

Ingestion fetches Wikipedia articles → chunks them → embeds them → stores in ChromaDB.
Raw article text is cached in SQLite — re-ingesting different profiles/embedders skips the network.
- You can also download preprocessed data from my [drive](https://drive.google.com/file/d/1mTtLt9B8xY3JUTDtiAHjrvVVe-NO7nTS/view?usp=sharing) which can save you couple of minutes to hours. Just extract it the project folder.



### Quickstart — default profile (medium) + default embedder (minilm)
```bash
python ingest.py
```

### Ingest all 10 combinations (recommended)
```bash
python ingest.py --all
```

### Ingest specific combinations
```bash
python ingest.py --all --embedder minilm       # all 5 profiles, minilm only (fast)
python ingest.py --all --embedder nomic        # all 5 profiles, nomic (slower)
python ingest.py --profile large --embedder minilm
```

### Parallel ingestion (faster on multi-core machines)
```bash
python parallel_ingest.py                      # all 10 combos in parallel
python parallel_ingest.py --embedder minilm    # 5 threads, minilm only
```

### Check ingestion status
```bash
python ingest.py --list
```

### Reset and re-ingest
```bash
python ingest.py --profile medium --embedder minilm --reset
```

---

## Start the Application

### Streamlit Web UI (recommended)

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

### Command Line Interface

```bash
python cli.py
```

---

## Streamlit UI Guide

The sidebar has three independent selectors:

**🗂 Chunk Profile** — how documents were split
| Profile | Chunk size | Overlap | Best for |
|---------|-----------|---------|---------|
| tiny | 150 chars | 15 | Very precise fact lookup |
| small | 300 chars | 30 | Precise retrieval |
| medium | 500 chars | 50 | Balanced — default |
| large | 1000 chars | 100 | Rich context answers |
| xl | 2000 chars | 200 | Comparison questions |

**🧠 Embedding Model** — how text was vectorized
- `minilm` — paraphrase-MiniLM-L3-v2, 384-dim, fast, CUDA-compatible
- `nomic` — nomic-embed-text, 768-dim, higher quality via Ollama

**🤖 Language Model** — which LLM generates the answer
- `llama3` — llama3.2:3b, fastest, excellent quality
- `phi3` — Microsoft Phi-3, compact and efficient
- `mistral` — Mistral 7B, most detailed answers

> Switching profile or embedder clears the chat — they use different ChromaDB collections.
> Switching LLM works instantly mid-conversation.

Toggle **"Show source chunks"** to see which Wikipedia passages were used.

---

## CLI Guide

```
WikiRAG [medium/minilm/llama3] > Who was Albert Einstein?
WikiRAG [medium/minilm/llama3] > profile large
WikiRAG [large/minilm/llama3]  > embedder nomic
WikiRAG [large/nomic/llama3]   > llm mistral
WikiRAG [large/nomic/mistral]  > Compare the Eiffel Tower and the Statue of Liberty
WikiRAG [large/nomic/mistral]  > sources on
WikiRAG [large/nomic/mistral]  > status
WikiRAG [large/nomic/mistral]  > quit
```

Available commands: `profile <name>`, `embedder <name>`, `llm <name>`, `status`, `sources on/off`, `clear`, `help`, `quit`

---

## Example Queries

### People
```
Who was Albert Einstein and what is he known for?
What did Marie Curie discover?
Why is Nikola Tesla famous?
Compare Lionel Messi and Cristiano Ronaldo
What is Frida Kahlo known for?
Who was Mustafa Kemal Atatürk?
What did Stephen Hawking contribute to science?
```

### Places
```
Where is the Eiffel Tower located?
Why is the Great Wall of China important?
What was the Colosseum used for?
Where is Mount Everest?
What is Machu Picchu?
What is the Hagia Sophia?
What is Göbekli Tepe?
```

### Mixed / Comparison
```
Which famous place is located in Turkey?
Which person is associated with electricity?
Compare the Eiffel Tower and the Statue of Liberty
Compare Albert Einstein and Nikola Tesla
Who founded a country?
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
├── app.py                   # Streamlit web UI (3 live selectors)
├── cli.py                   # Command-line interface
├── ingest.py                # Sequential ingestion pipeline
├── parallel_ingest.py       # Threaded parallel ingestion
├── test.py                  # Benchmark suite (65 queries × 30 combos)
├── merge_reports.py         # Merges individual reports into master
├── requirements.txt
├── README.md
├── product_prd.md
├── recommendation.md
│
├── lib/
│   ├── config.py            # Chunk profiles, embedders, LLMs
│   ├── fetcher.py           # Wikipedia REST API fetcher with retry
│   ├── chunker.py           # Fixed-size chunking with overlap
│   ├── embedder.py          # sentence-transformers + Ollama backends
│   ├── store.py             # ChromaDB read/write operations
│   ├── database.py          # SQLite article cache
│   ├── classifier.py        # Rule-based query classification
│   ├── retriever.py         # Entity-aware vector retrieval
│   └── generator.py         # Ollama LLM prompt + generation
│
├── data/
│   ├── entities.py          # 25 people + 25 places
│   └── wikirag.db           # SQLite database (created at runtime)
│
├── chroma_db/               # ChromaDB storage (created at runtime)
└── .streamlit/
    └── config.toml          # Suppresses watcher warnings
```

---

## Configuration

Edit `lib/config.py` to tune:

```python
TOP_K               = 5      # chunks retrieved per query
RELEVANCE_THRESHOLD = 1.5    # cosine distance cutoff
OLLAMA_MODEL        = "llama3.2:3b"
```

---

## GPU Acceleration

### NVIDIA GPU (CUDA)
```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
sentence-transformers detects CUDA automatically — 10–20x faster ingestion.



---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ollama: command not found` | Install from https://ollama.com |
| Connection refused on port 11434 | Run `ollama serve` |
| ChromaDB version error | `pip install chromadb==0.4.24` |
| Ingestion frozen on nomic | It's working — nomic embeds one chunk at a time. A per-chunk bar shows progress |
| Low CPU during minilm | Normal on 2-core machines — use NVIDIA GPU for real speedup |
| Wrong entity returned (src mismatch) | Entity-aware retriever filters by name — ensure entity is in `data/entities.py` |

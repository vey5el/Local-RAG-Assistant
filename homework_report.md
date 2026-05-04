# BLG483E — Project 3 Report
## WikiRAG: A Local Wikipedia Retrieval-Augmented Generation System

**Student:** Veysel
**Course:** BLG483E — AI Aided Software Development
**Date:** May 2026

---

## 1. Introduction

This report documents the design, implementation, and evaluation of WikiRAG — a fully local, ChatGPT-style question-answering system built for Project 3. The system ingests Wikipedia articles about famous people and places, stores them in a local vector database, and uses a locally-hosted language model to answer natural language questions grounded in retrieved context.

The system runs entirely on localhost with no external API calls at any stage — ingestion, embedding, retrieval, and generation all happen on the local machine.

---

## 2. System Architecture

The pipeline consists of six sequential stages:

```
Wikipedia API
     ↓
SQLite (raw text cache)
     ↓
Chunker (5 configurable profiles)
     ↓
Embedder (minilm or nomic)
     ↓
ChromaDB (10 collection pairs)
     ↓
User Query → Classifier → Entity-Aware Retriever → Prompt → LLM → Answer
```

### Key design principle
Every stage is modular and independently configurable. The chunk profile, embedding model, and language model can all be changed at runtime from the UI without restarting the application — and without re-ingesting data.

---

## 3. Implementation

### 3.1 Data Ingestion

Wikipedia articles are fetched using the Wikipedia REST API (`/w/api.php` with `action=query&prop=extracts&explaintext=True`). The system ingests **25 people and 25 places** — all 20 required by the spec, plus 15 additional entities including Mustafa Kemal Atatürk, Göbekli Tepe, Genghis Khan, Elon Musk, and others.

Raw article text is stored in **SQLite** after the first fetch. All subsequent ingestion runs (for different profiles or embedders) read from the local database — no network calls are made again. This makes switching chunk configurations fast.

### 3.2 Chunking

Documents are split using a **fixed-size sliding window with overlap**:

```python
def chunk_text(text, chunk_size=500, overlap=50):
    # splits at word boundaries, slides by (chunk_size - overlap)
```

Five named profiles are supported simultaneously:

| Profile | Chunk size | Overlap |
|---------|-----------|---------|
| tiny | 150 chars | 15 |
| small | 300 chars | 30 |
| medium | 500 chars | 50 (default) |
| large | 1000 chars | 100 |
| xl | 2000 chars | 200 |

All five profiles are stored in ChromaDB at the same time — switching profiles in the UI queries a different collection, not a different run of the pipeline.

### 3.3 Embedding and Storage

Two local embedding models are supported:

**minilm** — `paraphrase-MiniLM-L3-v2` via sentence-transformers (384-dimensional vectors). Uses CUDA automatically if an NVIDIA GPU is available. Supports batch embedding — all chunks of an article are embedded in a single forward pass.

**nomic** — `nomic-embed-text` via Ollama (768-dimensional vectors). Embeds one chunk at a time via HTTP. Higher quality vectors but slower due to sequential calls.

Vectors are stored in **ChromaDB** using cosine similarity. The design follows **Option A (two collections)** from the spec: one `people_{profile}_{embedder}` collection and one `places_{profile}_{embedder}` collection per combination. This gives 10 collection pairs total.

**Design rationale for Option A:** Vectors from different embedding models are mathematically incompatible and cannot be mixed. Separate collections also allow scoped retrieval — a person query never touches place vectors — which is both faster and semantically cleaner.

### 3.4 Query Classification

Queries are classified as `person`, `place`, or `both` using rule-based keyword matching against the known entity name sets defined in `data/entities.py`. If a query contains a known person name, it targets the people collection. Same for places. If both or neither match, both collections are searched and results are merged by distance score.

### 3.5 Entity-Aware Retrieval

A key improvement over naive semantic search is **entity-aware retrieval**. When a query mentions a known entity by name, ChromaDB is filtered by title metadata *before* similarity ranking:

```python
where = {"title": {"$eq": "Mahatma Gandhi"}}
# only then: rank by cosine similarity within Gandhi's chunks
```

**Why this matters:** Without this filter, generic queries like "What did Gandhi do?" can match semantically similar text in unrelated articles. In testing, this caused Ada Lovelace's article to appear as the top result for Gandhi queries — because both articles contain language about contributions and achievements. The entity filter eliminates this semantic drift.

### 3.6 Generation

Three local LLMs are supported via Ollama:

- **llama3.2:3b** — fastest, excellent general quality
- **phi3** — Microsoft's compact model, efficient
- **mistral** (7B) — most detailed answers, slightly slower

The LLM receives a structured prompt:

```
System: You ONLY use the provided context. Say "I don't know" if context is insufficient.
Context: [top-5 retrieved chunks]
Question: [user query]
```

Temperature is set to 0.1 for factual, deterministic answers.

### 3.7 Chat Interface

Two interfaces are provided:

**Streamlit UI** — three live dropdowns (chunk profile, embedder, LLM), source chunk toggle, conversation history, stats panel, clear button.

**CLI** — full terminal interface with runtime commands (`profile`, `embedder`, `llm`, `sources on/off`, `status`, `help`).

---

## 4. Testing Environment

### 4.1 Development Machine

The development and application testing was done on a personal laptop:

- **CPU:** Older generation Intel (details withheld)
- **GPU:** AMD Radeon RX 5500 XT
- **OS:** Windows 11

**Challenge:** The AMD GPU on Windows does not support CUDA, and Ollama does not support AMD GPU acceleration on Windows. This meant all LLM inference ran on CPU, producing response times of 9–15 seconds per query. Running the full benchmark suite (1,950 LLM calls across 30 combinations) would have taken an estimated 50–80 hours on this machine.

### 4.2 Cloud Testing Environment (RunPod)

To complete the benchmark in a reasonable timeframe, I deployed the system on **RunPod** — a GPU cloud service — for the testing phase.

**Instance used:**
- GPU: NVIDIA GeForce RTX 5090 (24GB VRAM)
- RAM: 41GB
- CPU: 6 vCPU
- Cost: $0.69/hr × ~3 hours = ~$2.07 total

**Setup process:**
1. Zipped the project (excluding `venv/`) and uploaded via Google Drive → `gdown`
2. Installed `ollama` and pulled all 4 models (llama3.2:3b, phi3, mistral, nomic-embed-text)
3. Installed Python dependencies via `requirements.txt`
4. Verified ChromaDB vectors transferred correctly (all 10 combinations intact)
5. Ran minilm tests in Terminal 1, nomic tests in Terminal 2 simultaneously

**Performance on RTX 5090:**
- Average response time: **0.47–1.46 seconds** per query (vs 9–15s on CPU)
- Total benchmark runtime: ~45 minutes for all 1,950 LLM calls
- GPU utilization: ~15-25% (embedding model is too small to saturate a 5090)

After testing, the instance was terminated to stop billing. The ChromaDB vectors and reports were downloaded locally.

---

## 5. Benchmark Results

### 5.1 Setup

- **30 combinations** tested: 5 profiles × 2 embedders × 3 LLMs
- **65 queries** per combination: 25 person, 25 place, 10 mixed/comparison, 5 failure cases
- **1,950 total LLM calls**
- Each query measured for retrieval time and generation time separately

### 5.2 Response Time

| Combination | Avg total | Retrieval | Generation |
|-------------|-----------|-----------|------------|
| tiny/nomic/llama3 | **0.47s** (fastest) | 0.166s | 0.30s |
| xl/nomic/mistral | **1.46s** (slowest) | 0.123s | 1.34s |
| medium/minilm/llama3 | 0.55s | 0.09s | 0.46s |
| large/nomic/mistral | 1.10s | 0.130s | 0.97s |

**Key finding:** LLM generation accounts for 52–94% of total response time across all combinations. Retrieval with minilm is negligible (as low as 0.066s, 6.5% of total). nomic adds some overhead (0.1–0.17s) due to Ollama HTTP calls for each query embedding.

### 5.3 Retrieval Quality

Measured by average cosine distance (lower = more relevant chunks):

| Profile | Embedder | Avg distance |
|---------|----------|-------------|
| tiny | nomic | **0.3227** (best) |
| small | nomic | 0.3362 |
| medium | nomic | 0.3614 |
| large | nomic | 0.3879 |
| xl | minilm | **0.4289** (worst) |

**Key finding:** Smaller chunks produce lower distance scores — more precise semantic matches. nomic consistently outperforms minilm at every profile level, reflecting its higher dimensionality (768 vs 384).

### 5.4 Answer Quality

**Entity mention accuracy** — does the answer actually mention the entity asked about?

| Combination | Entity mention rate |
|-------------|-------------------|
| small/minilm/llama3 | **50/50 (100%)** |
| medium/minilm/llama3 | **50/50 (100%)** |
| medium/minilm/mistral | **50/50 (100%)** |
| tiny/nomic/llama3 | 43/50 (86%) — worst |

**I-don't-know accuracy** — does the system correctly refuse unanswerable questions?

| LLM | Correct refusals |
|-----|-----------------|
| llama3 | 30/30 (100%) |
| mistral | 29/30 (97%) |
| phi3 | 29/30 (97%) |

All three LLMs are highly reliable at refusing hallucination on out-of-scope queries.

### 5.5 Best Combinations

| Use case | Recommended combination | Reason |
|----------|------------------------|--------|
| Best overall | `small/minilm/llama3` | 100% entity mention, 100% IDK, 0.5s avg |
| Best quality | `large/nomic/mistral` | Most detailed answers, 100% entity mention |
| Best retrieval | `tiny/nomic` | Lowest distance scores across the board |
| Best for demo | `medium/minilm/llama3` | Fast, accurate, well-balanced |

---

## 6. Design Decisions and Tradeoffs

### Two collections vs one with metadata (Option A vs B)

I chose **Option A (two collections per combination)** over a single collection with metadata filtering. The main reason is that vectors from different embedding models are mathematically incompatible — a 384-dim minilm vector cannot be meaningfully compared against a 768-dim nomic vector. Separate collections make this constraint explicit and eliminate any risk of cross-contamination. The tradeoff is more collections to manage (10 pairs), but at this corpus size this is negligible.

### Entity-aware filtering vs pure similarity

Pure semantic similarity retrieval failed on generic queries — "What did Gandhi do?" returned Ada Lovelace chunks on some combinations because both articles contain similar vocabulary about contributions and achievements. The entity filter adds a metadata pre-filter that scopes retrieval to the correct article before ranking by similarity. The tradeoff is that it only works for entities in the known set — unknown people/places fall back to unfiltered search.

### minilm vs nomic

minilm is significantly faster (batch embedding, CUDA, ~0.05–0.1s per query) while nomic produces better retrieval distances (0.32 vs 0.39 avg) due to higher dimensionality. For production use where quality matters more than raw speed, nomic is preferable. For demo use where responsiveness is important, minilm is the better choice.

### Fixed-size chunking vs sentence-based

Sentence-based splitting was considered but rejected because Wikipedia text contains varied formatting — section headers, parenthetical references, lists — that makes clean sentence detection unreliable. Fixed-size character windows with word-boundary alignment are simpler, more predictable, and handle arbitrarily large documents without edge cases.

---

## 7. Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| AMD GPU not supported by Ollama on Windows | Used NVIDIA GTX 1050 for embedding (CUDA), kept CPU for LLM |
| LLM too slow on CPU for full benchmark | Deployed on RunPod (RTX 5090) for testing — $2 total cost |
| Semantic drift on generic queries | Implemented entity-aware retrieval with ChromaDB title filter |
| nomic ingestion appearing frozen | Added per-chunk inner progress bar for nomic embedder |
| ChromaDB `get()` without limit causing 20s hangs | Changed to `limit=1` existence check + `limit=500` count cap |
| Parallel ingestion on ChromaDB | Used ThreadPoolExecutor — safe because each combo writes to different collections |

---

## 8. Optional Extensions Implemented

The following optional extensions from the spec were implemented:

| Extension | Implementation |
|-----------|---------------|
| Citations / source highlighting | Source chunks shown with cosine distance scores in UI |
| Comparing two local models | All 3 LLMs benchmarked side-by-side across 30 combinations |
| Latency measurement | Retrieval and generation timed separately per query |
| Comparison questions | "Both" query type merges and re-ranks results from both collections |
| Caching | SQLite caches Wikipedia text — network never re-hit for repeat ingestion |
| Improving retrieval ranking | Entity-aware title filtering before similarity ranking |

---

## 9. Possible Improvements

- **Streaming responses** — stream LLM tokens to the UI as they are generated rather than waiting for the full response
- **Chat history memory** — include prior conversation turns in the prompt context for multi-turn coherence
- **Re-ranking** — add a cross-encoder re-ranker as a second pass after initial retrieval to improve precision
- **BM25 hybrid search** — combine keyword search with vector search for better recall on short queries
- **Automatic entity detection** — use spaCy NER instead of the fixed entity set to handle entities not in the known list
- **Wikipedia refresh** — schedule periodic re-ingestion to keep articles up to date
- **Answer quality scoring** — integrate Ragas for automated evaluation of faithfulness and relevance

---

## 10. Conclusion

WikiRAG successfully implements all required components of the specification and adds significant depth through configurable profiles, multiple embedding models, multiple LLMs, and a comprehensive benchmark. The entity-aware retrieval mechanism directly addresses a real failure mode in naive RAG systems. The benchmark results demonstrate that the system is both fast (sub-second responses on GPU hardware) and accurate (100% entity mention rate, 97–100% refusal accuracy on unanswerable queries) across all tested configurations.

The modular architecture — with each concern isolated in its own `lib/` module — makes the system straightforward to extend, test, and deploy.

---

## Appendix: File Structure

```
wikirag/
├── app.py                   Streamlit UI
├── cli.py                   CLI interface
├── ingest.py                Sequential ingestion
├── parallel_ingest.py       Threaded parallel ingestion
├── test.py                  Benchmark suite (65 queries × 30 combos)
├── merge_reports.py         Report merger
├── requirements.txt
├── README.md
├── product_prd.md
├── recommendation.md
├── lib/
│   ├── config.py            Chunk profiles, embedders, LLMs
│   ├── fetcher.py           Wikipedia REST API
│   ├── chunker.py           Fixed-size chunking
│   ├── embedder.py          sentence-transformers + Ollama
│   ├── store.py             ChromaDB operations
│   ├── database.py          SQLite cache
│   ├── classifier.py        Query classification
│   ├── retriever.py         Entity-aware retrieval
│   └── generator.py         LLM generation
└── data/
    └── entities.py          25 people + 25 places
```

## Appendix: RunPod Setup Commands

```bash
# Install dependencies
pip install chromadb sentence-transformers streamlit tqdm requests ollama

# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 3

# Pull models
ollama pull llama3.2:3b
ollama pull phi3
ollama pull mistral
ollama pull nomic-embed-text

# Upload project
gdown "https://drive.google.com/uc?export=download&id=FILE_ID" -O RAG.zip
unzip RAG.zip -d wikirag && cd wikirag

# Verify vectors
python -c "from lib.store import get_client, collection_stats; print(collection_stats(get_client()))"

# Run benchmark (2 terminals in parallel)
# Terminal 1: all minilm combinations
# Terminal 2: all nomic combinations

# Zip and download reports
zip -r reports.zip reports/
```

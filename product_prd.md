# Product Requirements Document
## WikiRAG — Local Wikipedia RAG System
### BLG483E Project 3

---

## 1. Overview

### Product Name
WikiRAG — A Local Retrieval-Augmented Generation System

### Purpose
WikiRAG is a fully local, ChatGPT-style question-answering system that uses Wikipedia content about famous people and places as its knowledge base. The system combines vector-based retrieval with locally-run language models to answer natural language questions, grounded entirely in retrieved factual context — with no external API calls.

### Problem Statement
General-purpose LLMs hallucinate facts and lack grounding in specific knowledge bases. Sending queries to cloud APIs raises privacy and cost concerns. WikiRAG solves both problems by running entirely on localhost: all ingestion, embedding, retrieval, and generation happens on the user's machine.

---

## 2. Goals

### Goals
- Ingest Wikipedia articles for ≥20 people and ≥20 places (implemented: 25 + 25)
- Support 5 chunk profiles and 2 embedding models as independently selectable dimensions
- Support 3 local LLMs selectable at query time with no re-ingestion required
- Store each chunk profile + embedder combination in its own ChromaDB collection pair
- Classify queries as person, place, or both and retrieve from the correct collection(s)
- Use entity-aware filtering to prevent semantic drift in retrieval
- Generate grounded answers that cite only retrieved context
- Return "I don't know" when no relevant context is found
- Provide a Streamlit UI with live selectors for all three dimensions
- Provide a CLI interface with runtime switching of all dimensions
- Include a comprehensive benchmark suite measuring retrieval and generation quality

### Non-Goals
- No external LLM or embedding API usage
- No cloud database or storage
- No real-time Wikipedia updates
- No user authentication or multi-user support

---

## 3. Users

**Primary:** Students and instructors evaluating the system as an academic deliverable.

**Secondary:** Developers learning local RAG architecture and LLM evaluation methodology.

---

## 4. Functional Requirements

### 4.1 Ingestion
| ID | Requirement |
|----|-------------|
| F1 | Fetch Wikipedia article text via the Wikipedia REST API (no scraping, no paid API) |
| F2 | Ingest ≥20 people and ≥20 places — implemented as 25 + 25 |
| F3 | Cache raw article text in SQLite with title, entity type, URL, and fetch timestamp |
| F4 | Re-ingestion reads from SQLite cache — no network call when profile or embedder changes |
| F5 | Ingestion is idempotent — re-running skips already-indexed articles |
| F6 | Show per-article progress bar (tqdm) for outer loop |
| F7 | For nomic embedder: show per-chunk inner progress bar to confirm activity |
| F8 | Support parallel ingestion via threaded workers (one thread per combination) |

### 4.2 Chunking
| ID | Requirement |
|----|-------------|
| F9  | Support 5 named chunk profiles: tiny, small, medium, large, xl |
| F10 | Each profile defines chunk_size (characters) and overlap (characters) |
| F11 | Chunks split at word boundaries — no mid-word cuts |
| F12 | Chunk metadata includes: title, entity_type, chunk_index, url, profile, embedder |

### 4.3 Embedding
| ID | Requirement |
|----|-------------|
| F13 | Support minilm: paraphrase-MiniLM-L3-v2 via sentence-transformers (384-dim) |
| F14 | Support nomic: nomic-embed-text via Ollama (768-dim) |
| F15 | minilm uses GPU (CUDA) automatically when available |
| F16 | minilm uses batch embedding for speed |
| F17 | nomic embeds one chunk at a time via Ollama HTTP API |

### 4.4 Vector Storage
| ID | Requirement |
|----|-------------|
| F18 | Use ChromaDB with cosine similarity metric |
| F19 | Each profile + embedder combination gets its own collection pair |
| F20 | Collection naming: `people_{profile}_{embedder}` / `places_{profile}_{embedder}` |
| F21 | Maximum 10 collection pairs (5 profiles × 2 embedders) |
| F22 | LLM selection does NOT affect collections — switching LLM needs no re-ingestion |

### 4.5 Query Classification
| ID | Requirement |
|----|-------------|
| F23 | Classify query as person, place, or both using keyword/rule-based matching |
| F24 | Match against known entity name sets from `data/entities.py` |
| F25 | Unrecognized queries default to searching both collections |

### 4.6 Retrieval
| ID | Requirement |
|----|-------------|
| F26 | When query contains a known entity name, filter ChromaDB by title metadata before similarity ranking (entity-aware retrieval) |
| F27 | Entity filter prevents semantic drift — "What did Gandhi do?" returns Gandhi chunks, not Ada Lovelace chunks |
| F28 | For "both" queries: retrieve from both collections and merge by distance score |
| F29 | Return top-k chunks (default: 5) passing the relevance distance threshold |
| F30 | If no chunks pass threshold, signal "not found" — do not generate |

### 4.7 Generation
| ID | Requirement |
|----|-------------|
| F31 | Support 3 LLMs: llama3.2:3b, phi3, mistral — all via Ollama |
| F32 | System prompt instructs model to answer only from provided context |
| F33 | System prompt instructs model to respond "I don't know" if context is insufficient |
| F34 | Temperature set to 0.1 for factual, deterministic answers |
| F35 | Graceful error message if Ollama is unreachable |

### 4.8 Streamlit UI
| ID | Requirement |
|----|-------------|
| F36 | Sidebar dropdown: chunk profile (shows only ingested profiles) |
| F37 | Sidebar dropdown: embedding model (shows only embedders ingested for selected profile) |
| F38 | Sidebar dropdown: LLM (all 3 always available) |
| F39 | Switching profile or embedder clears conversation (different collection) |
| F40 | Switching LLM works instantly mid-conversation |
| F41 | Stats panel shows chunk counts per ingested combination |
| F42 | Toggle to show/hide source chunks with distance scores |
| F43 | Clear conversation button |

### 4.9 CLI
| ID | Requirement |
|----|-------------|
| F44 | Runtime commands: profile, embedder, llm, status, sources on/off, clear, help, quit |
| F45 | Prompt shows active configuration: [profile/embedder/llm] |

### 4.10 Benchmark Suite
| ID | Requirement |
|----|-------------|
| F46 | Test all ingested combinations with configurable query set |
| F47 | Measure retrieval time and generation time separately |
| F48 | Record avg similarity distance, top source, entity mention accuracy |
| F49 | Track I-don't-know accuracy on failure-case queries |
| F50 | Save report after every query (crash-safe incremental saves) |
| F51 | Generate Markdown report with side-by-side LLM comparison |

---

## 5. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NF1 | All components run on localhost — zero external API calls at query time |
| NF2 | System is fully reproducible by following README instructions only |
| NF3 | Codebase is modular — each concern in its own file under `lib/` |
| NF4 | Ingestion is idempotent and cache-aware |

---

## 6. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                            │
│       Streamlit (3 dropdowns)  /  CLI (runtime commands)    │
└──────────────────────┬──────────────────────────────────────┘
                       │ query + (profile, embedder, llm)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Query Classifier                               │
│       keyword/rule-based → person / place / both           │
│       + entity name extraction from known sets             │
└──────────┬──────────────────────────────┬──────────────────┘
           │ person                       │ place
           ▼                              ▼
┌─────────────────────┐        ┌─────────────────────┐
│ ChromaDB            │        │ ChromaDB            │
│ people_{p}_{e}      │        │ places_{p}_{e}      │
│ filter by title     │        │ filter by title     │
│ then rank by cosine │        │ then rank by cosine │
└──────────┬──────────┘        └──────────┬──────────┘
           └──────────────┬───────────────┘
                          │ top-k relevant chunks
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Prompt Builder                             │
│      system prompt + context chunks + user query           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Local LLM via Ollama                             │
│      llama3.2:3b  /  phi3  /  mistral                      │
│      (selected at query time — no re-ingestion needed)     │
└──────────────────────┬──────────────────────────────────────┘
                       │ answer
                       ▼
                  User Interface
```

---

## 7. Design Decisions

### 7.1 Two Collections per Combination (Option A)

**Decision:** Two ChromaDB collections per profile+embedder pair (`people_*` / `places_*`)

**Rationale:**
- Vectors from different embedding models are mathematically incompatible — they must be stored separately
- Scoped retrieval is faster — person queries never touch place vectors
- Collection naming makes storage immediately transparent
- For "both" queries, results from both same-config collections are merged and re-ranked by distance

**Tradeoff:** More collections to manage (10 pairs max), but cleaner semantics and no risk of cross-contamination.

### 7.2 LLM Not Part of Collection Name

**Decision:** LLM selection is runtime-only, not encoded in collection names

**Rationale:**
- LLM selection affects generation only, not retrieval or stored vectors
- Storing per-LLM collections would triple storage with zero retrieval benefit
- Switching LLMs mid-conversation is a useful demo feature

### 7.3 Entity-Aware Retrieval

**Decision:** When a query mentions a known entity by name, filter ChromaDB by title metadata before similarity ranking

**Rationale:**
- Without filtering, generic queries like "What did Gandhi do?" can match semantically similar text in unrelated articles (e.g. Ada Lovelace's article discussing contributions)
- Title filtering guarantees the retriever searches Gandhi's own chunks first
- Falls back to unfiltered search if filtered results are empty or title is not in the known entity set

**Tradeoff:** Requires the entity to be in `data/entities.py` — unknown entities get unfiltered search.

### 7.4 Query Classification: Rule-Based

**Decision:** Keyword matching against known entity name sets

**Rationale:**
- Simple, transparent, deterministic — no extra LLM call required
- Sufficient for a closed-world corpus of 50 known entities
- Unrecognized queries default to searching both collections conservatively

### 7.5 Five Chunk Profiles

**Decision:** Named profiles with preset chunk_size + overlap pairs

**Rationale:**
- Named profiles are easier to communicate in a demo than raw numbers
- Overlap is proportional to chunk size (~10%) for consistency
- 5 profiles cover the full range from very precise to maximum context
- All 5 coexist in ChromaDB simultaneously — no re-ingestion when switching

---

## 8. Chunk Profiles

| Profile | Chunk size | Overlap | Avg chunks/article | Best for |
|---------|-----------|---------|-------------------|----------|
| tiny | 150 chars | 15 | 400–800 | Precise fact lookup |
| small | 300 chars | 30 | 200–400 | Balanced precision |
| medium | 500 chars | 50 | 120–250 | General QA (default) |
| large | 1000 chars | 100 | 60–130 | Rich context, comparisons |
| xl | 2000 chars | 200 | 30–70 | Maximum context per chunk |

---

## 9. Benchmark Results Summary

Tested across 30 combinations (5 profiles × 2 embedders × 3 LLMs), 65 queries each = 1,950 total LLM calls.

| Metric | Best | Worst |
|--------|------|-------|
| Avg response time | 0.47s (tiny/nomic/llama3) | 1.46s (xl/nomic/mistral) |
| Retrieval distance | 0.3227 (tiny/nomic) | 0.4289 (xl/minilm) |
| Entity mention accuracy | 100% (multiple combos) | 94% (tiny/nomic/llama3) |
| I-don't-know accuracy | 100% (llama3) | 98% (phi3, mistral) |
| Retrieval % of total time | 6.5% (xl/minilm/mistral) | 47% (tiny/minilm/llama3) |

**Recommended combination for demo:** `medium/minilm/llama3` — 0.55s avg, 100% entity mention, 100% IDK accuracy, minimal overhead.

---

## 10. Optional Extensions Implemented

| Extension | Implementation |
|-----------|---------------|
| Citations / source highlighting | Source chunks shown with cosine distance scores |
| Comparing two local models | 3 LLMs selectable; full benchmark comparison generated |
| Latency measurement | Retrieval and generation timed separately per query |
| Comparison questions | "Both" query type merges results from both collections |
| Caching | SQLite caches Wikipedia text — network never re-hit |
| Improving retrieval ranking | Entity-aware title filtering before similarity ranking |

---

## 11. Out of Scope

- Streaming LLM responses
- Chat history memory across sessions
- Automatic Wikipedia refresh / scheduled re-ingestion
- Re-ranking with a cross-encoder model
- Multi-user support

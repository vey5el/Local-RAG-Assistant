# Product Requirements Document
## Local Wikipedia RAG System — BLG483E Project 3

---

## 1. Overview

### Product Name
WikiRAG — A Local Retrieval-Augmented Generation System

### Purpose
WikiRAG is a fully local, ChatGPT-style question-answering system that uses Wikipedia content about famous people and places as its knowledge base. The system combines vector-based retrieval with a locally-run language model to answer natural language questions, grounded in retrieved factual context.

### Problem Statement
General-purpose LLMs hallucinate and lack grounding in specific, curated knowledge bases. Sending queries to external APIs raises privacy and cost concerns. WikiRAG addresses both by running entirely on localhost with no external API calls — providing factual, citation-backed answers from a controlled corpus.

---

## 2. Goals and Non-Goals

### Goals
- Ingest Wikipedia articles for ≥20 people and ≥20 places
- Chunk, embed, and store documents in a local vector database
- Classify user queries as about a person, place, or both
- Retrieve relevant chunks and generate grounded answers via a local LLM
- Provide a simple, interactive chat interface (CLI or Streamlit)
- Return "I don't know" when the answer cannot be found in the data

### Non-Goals
- No external LLM API usage (OpenAI, Anthropic, etc.)
- No cloud database or storage
- No real-time Wikipedia updates (static ingestion only)
- No user authentication or multi-user support

---

## 3. Users

**Primary User:** Students and instructors evaluating the system as an academic project demo.

**Secondary User:** Any developer exploring local RAG architectures for privacy-preserving QA.

---

## 4. Functional Requirements

### 4.1 Data Ingestion
| ID | Requirement |
|----|-------------|
| F1 | Fetch Wikipedia article text via the Wikipedia API (no scraping) |
| F2 | Support ≥20 people and ≥20 places as defined in the spec |
| F3 | Store raw article text in SQLite with metadata (title, type, url, fetched_at) |
| F4 | Ingestion must be idempotent — re-running does not duplicate records |

### 4.2 Chunking
| ID | Requirement |
|----|-------------|
| F5 | Split documents into fixed-size chunks with configurable overlap |
| F6 | Default: 500 tokens per chunk, 50-token overlap |
| F7 | Each chunk retains metadata: source title, entity type (person/place), chunk index |
| F8 | Design must handle large documents gracefully |

### 4.3 Embedding and Storage
| ID | Requirement |
|----|-------------|
| F9 | Generate embeddings locally using `nomic-embed-text` via Ollama or `sentence-transformers` |
| F10 | Store embeddings in ChromaDB with two collections: `people` and `places` |
| F11 | Each vector document includes: chunk text, source title, entity type, chunk index |

### 4.4 Query Classification
| ID | Requirement |
|----|-------------|
| F12 | Classify incoming query as: `person`, `place`, or `both` |
| F13 | Classification uses keyword/rule-based logic (no extra model call required) |
| F14 | "Both" classification queries both collections and merges results |

### 4.5 Retrieval
| ID | Requirement |
|----|-------------|
| F15 | Embed the user query using the same local embedding model |
| F16 | Retrieve top-k (default: 5) most similar chunks from the relevant collection(s) |
| F17 | Return ranked chunks with similarity scores |

### 4.6 Generation
| ID | Requirement |
|----|-------------|
| F18 | Construct a prompt combining the user query and retrieved context chunks |
| F19 | Send prompt to local Ollama model (llama3.2, phi3, or mistral) |
| F20 | Generated answer must be grounded in context; model is instructed not to hallucinate |
| F21 | If no relevant chunks found (score below threshold), return "I don't know" |
| F22 | Optionally display the source chunks used |

### 4.7 Chat Interface
| ID | Requirement |
|----|-------------|
| F23 | Provide a Streamlit web UI and/or a CLI interface |
| F24 | User can ask questions and receive formatted answers |
| F25 | User can toggle display of retrieved source chunks |
| F26 | User can clear/reset the conversation history |
| F27 | System displays which collection(s) were queried per question |

---

## 5. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NF1 | All components run on localhost — zero external API calls |
| NF2 | Ingestion of the full 40+ entity dataset completes in under 10 minutes |
| NF3 | Query-to-response latency under 30 seconds on consumer hardware (M1/M2 Mac or equivalent) |
| NF4 | System is reproducible: following README instructions must produce a working system |
| NF5 | Codebase is modular — ingestion, embedding, retrieval, and generation are separate modules |

---

## 6. Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        User Interface                     │
│              Streamlit App  /  CLI                        │
└────────────────────────┬─────────────────────────────────┘
                         │ query
                         ▼
┌──────────────────────────────────────────────────────────┐
│                    Query Classifier                       │
│         (keyword/rule-based: person / place / both)       │
└──────┬───────────────────────────────────────┬───────────┘
       │ person                                │ place
       ▼                                       ▼
┌─────────────┐                        ┌──────────────┐
│  ChromaDB   │                        │  ChromaDB    │
│  "people"   │                        │  "places"    │
│  collection │                        │  collection  │
└──────┬──────┘                        └──────┬───────┘
       └────────────────┬──────────────────────┘
                        │ top-k chunks
                        ▼
┌──────────────────────────────────────────────────────────┐
│                    Prompt Builder                         │
│         [System prompt + context chunks + query]          │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                  Local LLM (Ollama)                       │
│            llama3.2:3b / phi3 / mistral                   │
└────────────────────────┬─────────────────────────────────┘
                         │ answer
                         ▼
                    User Interface
```

### Storage
- **SQLite** — raw article text, metadata, ingestion log
- **ChromaDB** — two persistent collections (`people`, `places`) with embeddings

### Embedding Model
- Primary: `nomic-embed-text` via Ollama
- Fallback: `all-MiniLM-L6-v2` via `sentence-transformers`

### LLM
- Primary: `llama3.2:3b` via Ollama
- Alternatives: `phi3`, `mistral`

---

## 7. Design Decisions

### Two Collections vs. One with Metadata
**Decision: Two collections (`people`, `places`)**

**Rationale:**
- Enables faster, scoped retrieval — person queries never touch place vectors and vice versa
- Cleaner separation of concern for query routing
- Slightly higher storage cost is acceptable at this scale
- For "both" queries, we merge ranked results from both collections with a simple interleave strategy

**Tradeoff:** Slightly more complex management code, but the retrieval precision benefit outweighs this.

### Query Classification
**Decision: Rule-based keyword matching**

Known person names and place names are used as lookup sets. If the query contains a known person name → `person`. Known place name → `place`. Both → `both`. No match → search both.

**Rationale:** Simple, transparent, no extra LLM call, deterministic. Sufficient for the scope of this project.

---

## 8. Example Queries

### People
- "Who was Albert Einstein and what is he known for?"
- "What did Marie Curie discover?"
- "Compare Lionel Messi and Cristiano Ronaldo"

### Places
- "Where is the Eiffel Tower located?"
- "What was the Colosseum used for?"
- "Where is Mount Everest?"

### Mixed
- "Which famous place is located in Turkey?"
- "Which person is associated with electricity?"
- "Compare the Eiffel Tower and the Statue of Liberty"

### Failure Cases
- "Who is the president of Mars?" → Expected: "I don't know"
- "Tell me about John Doe" → Expected: "I don't know"

---

## 9. Milestones

| Milestone | Description |
|-----------|-------------|
| M1 | Ingestion pipeline: Wikipedia → SQLite |
| M2 | Chunking and embedding pipeline: SQLite → ChromaDB |
| M3 | Query classification + retrieval working end-to-end |
| M4 | LLM generation with grounded prompts |
| M5 | Chat UI (Streamlit) functional |
| M6 | README, PRD, recommendation docs complete |
| M7 | Demo video recorded and linked |

---

## 10. Out of Scope (Future Work)
- Streaming LLM responses
- Chat history memory across sessions
- Multi-model comparison UI
- Latency benchmarking dashboard
- Automatic Wikipedia refresh/updates

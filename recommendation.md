# Production Deployment Recommendation
## WikiRAG — From Localhost to Production

---

## Executive Summary

The current WikiRAG system is architected as a fully local application designed for offline, single-user use on a developer laptop. This document outlines the key architectural changes, technology choices, and tradeoffs required to deploy WikiRAG as a scalable, reliable production service.

---

## Current Architecture (Localhost)

| Component | Local Implementation |
|-----------|----------------------|
| LLM | Ollama (llama3.2:3b, phi3, mistral) |
| Embeddings | nomic-embed-text via Ollama |
| Vector DB | ChromaDB (file-based, persistent) |
| Metadata DB | SQLite |
| UI | Streamlit / CLI |
| Ingestion | One-time Python script |

**Limitations in production context:**
- Cannot serve concurrent users (Ollama is single-threaded by default)
- ChromaDB file-based storage does not scale horizontally
- SQLite is not safe for concurrent writes
- No authentication, logging, or observability
- No automatic knowledge refresh

---

## Recommended Production Stack

### 1. Language Model — Self-Hosted LLM Inference

**Replace:** Ollama (single-process)
**With:** [vLLM](https://github.com/vllm-project/vllm) or [TGI (Text Generation Inference)](https://github.com/huggingface/text-generation-inference)

**Why:**
- vLLM supports batched inference, PagedAttention, and continuous batching — enabling hundreds of concurrent requests on a single GPU
- TGI adds streaming, quantization support, and a production-ready HTTP API
- Both expose an OpenAI-compatible API endpoint, requiring minimal code changes

**Model recommendation for production:**
- `Mistral-7B-Instruct-v0.3` (strong quality, open weights, efficient on A10G)
- `Llama-3-8B-Instruct` (excellent instruction following, widely supported)
- For higher quality: `Mixtral-8x7B-Instruct` if budget allows

**Infrastructure:** AWS `g5.xlarge` (1x A10G, 24GB VRAM) or GCP `n1-standard-4` + T4

---

### 2. Embeddings — Dedicated Embedding Service

**Replace:** Ollama nomic-embed-text (CPU, serial)
**With:** [TEI (Text Embeddings Inference)](https://github.com/huggingface/text-embeddings-inference) by Hugging Face

**Why:**
- GPU-accelerated, handles 1000+ embeddings/second
- REST API with batching support
- Drop-in replacement — same `nomic-embed-text` or `bge-large-en-v1.5` model

**Alternative:** If cost is a priority, `sentence-transformers` served via FastAPI with async batching is acceptable for moderate traffic.

---

### 3. Vector Database — Managed or Self-Hosted at Scale

**Replace:** ChromaDB (file-based)
**With:** [Qdrant](https://qdrant.tech) or [Weaviate](https://weaviate.io)

**Why Qdrant:**
- Handles billions of vectors with horizontal scaling
- Supports payload filtering (equivalent to our person/place metadata filtering)
- Self-hostable on Kubernetes or available as managed cloud service
- HNSW index with excellent recall/speed tradeoff

**Why Weaviate:**
- Strong native multi-tenancy support
- Built-in hybrid search (BM25 + vector) for better retrieval on short queries
- GraphQL and REST APIs

**For moderate scale (< 10M vectors):** ChromaDB itself can be run as a separate HTTP server (`chroma run`) and is sufficient.

---

### 4. Relational Database — Replace SQLite

**Replace:** SQLite
**With:** PostgreSQL (managed: AWS RDS, Supabase, or Railway)

**Why:**
- Concurrent read/write safety
- Full-text search via `pg_trgm` for optional BM25 hybrid retrieval
- Mature tooling, backups, replication

---

### 5. API Layer

**Add:** FastAPI application wrapping retrieval + generation

```
Client → FastAPI → [Classifier → Retriever → vLLM] → Response
```

- Async endpoints for non-blocking inference
- `/query` POST endpoint with JSON request/response
- `/ingest` POST endpoint for triggering re-ingestion
- `/health` GET endpoint for load balancer checks
- Rate limiting via `slowapi` or API gateway

---

### 6. Frontend — Replace Streamlit

**Replace:** Streamlit
**With:** Next.js (React) or SvelteKit frontend consuming the FastAPI backend

**Why:** Streamlit is not designed for concurrent users and does not support proper session management. A JavaScript frontend is more performant, customizable, and scalable.

**Alternatively:** Keep Streamlit for internal/demo use and expose the raw API for third-party integrations.

---

### 7. Ingestion Pipeline — Scheduled Refresh

**Replace:** One-time `ingest.py` script
**With:** Scheduled pipeline using **Apache Airflow** or **Prefect**

- Weekly Wikipedia refresh for all entities
- Incremental ingestion: only re-embed articles that have changed
- Dead letter queue for failed fetches

---

### 8. Observability and Monitoring

Add the following production essentials:

| Concern | Tool |
|---------|------|
| Logging | Structured JSON logs → CloudWatch / Loki |
| Metrics | Prometheus + Grafana (latency, retrieval scores, LLM throughput) |
| Tracing | OpenTelemetry → Jaeger / Tempo |
| Error tracking | Sentry |
| LLM eval | Ragas (for retrieval and answer quality metrics) |

---

## Production Architecture Diagram

```
                        ┌──────────────────────┐
                        │    CDN / Load Balancer│
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │   FastAPI Backend      │
                        │  (Auto-scaled pods)    │
                        └──┬──────────────┬─────┘
                           │              │
              ┌────────────▼───┐    ┌─────▼──────────────┐
              │  Qdrant Vector │    │  PostgreSQL (RDS)   │
              │  DB (Managed)  │    │  (metadata, logs)   │
              └────────────────┘    └────────────────────┘
                           │
              ┌────────────▼──────────────────────┐
              │           vLLM / TGI               │
              │  (GPU inference server, A10G)       │
              └───────────────────────────────────┘
                           │
              ┌────────────▼──────────────────────┐
              │           TEI                      │
              │  (Embedding server, GPU or CPU)    │
              └───────────────────────────────────┘
```

---

## Cost Estimate (AWS, moderate traffic ~1000 queries/day)

| Component | Service | Estimated Monthly Cost |
|-----------|---------|----------------------|
| LLM inference | 1x g5.xlarge (on-demand) | ~$300 |
| Embedding server | 1x t3.medium (CPU, TEI) | ~$30 |
| Vector DB | Qdrant Cloud (1M vectors) | ~$25 |
| PostgreSQL | RDS t3.micro | ~$15 |
| API + Frontend | 2x t3.small | ~$30 |
| **Total** | | **~$400/month** |

> Savings tip: Use Spot Instances for LLM inference (up to 70% cost reduction) with automatic fallback.

---

## Key Tradeoffs

| Decision | Local | Production | Tradeoff |
|----------|-------|------------|---------|
| LLM runtime | Ollama | vLLM | Higher ops complexity; much higher throughput |
| Vector DB | ChromaDB file | Qdrant managed | Cost; gains HA and horizontal scale |
| Embedding | CPU via Ollama | GPU via TEI | Cost; gains 50-100x throughput |
| DB | SQLite | PostgreSQL | Migration effort; gains concurrency safety |
| Query routing | Keyword rules | Could add ML classifier | Complexity; marginal precision gain at this corpus size |
| Knowledge freshness | Static | Scheduled refresh | Airflow complexity; gains up-to-date answers |

---

## Migration Path (Recommended Order)

1. **Week 1:** Extract business logic into FastAPI; keep Ollama + ChromaDB for now
2. **Week 2:** Replace SQLite with PostgreSQL; add basic auth and rate limiting
3. **Week 3:** Deploy ChromaDB as HTTP server; add observability (Prometheus, Sentry)
4. **Week 4:** Migrate to vLLM on GPU; benchmark latency and throughput
5. **Month 2:** Migrate to Qdrant; implement scheduled ingestion refresh
6. **Month 3:** Add Next.js frontend; add Ragas-based answer quality monitoring

---

## Summary

The localhost WikiRAG system is a solid foundation. The primary production concerns are **concurrency** (replace Ollama with vLLM), **scalability** (replace file-based ChromaDB with Qdrant), and **reliability** (replace SQLite with PostgreSQL, add observability). The core retrieval and generation logic requires minimal changes — the modular architecture makes this migration tractable.

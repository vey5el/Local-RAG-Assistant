# Production Deployment Recommendation
## WikiRAG — From Localhost to Production
### BLG483E Project 3

---

## Executive Summary

WikiRAG is currently architected as a single-user localhost application. This document outlines the changes required to deploy it as a scalable, reliable production service — covering infrastructure, technology replacements, cost estimates, and a phased migration path.

---

## Current Localhost Architecture

| Component | Implementation | Limitation |
|-----------|---------------|------------|
| LLM | Ollama (single-process) | Single user, no concurrency |
| Embeddings | sentence-transformers / Ollama | CPU or single GPU, no batching across requests |
| Vector DB | ChromaDB (file-based) | No horizontal scaling, single writer |
| Metadata DB | SQLite | Not safe for concurrent writes |
| UI | Streamlit | Not designed for multiple concurrent users |
| Ingestion | One-time script | No scheduled refresh, no failure recovery |

---

## Recommended Production Stack

### 1. Language Model — Replace Ollama with vLLM

**Replace:** Ollama (single-process, CPU-friendly)
**With:** [vLLM](https://github.com/vllm-project/vllm)

**Why:**
- vLLM supports continuous batching — serves hundreds of concurrent requests on a single GPU
- PagedAttention memory management — efficient VRAM usage
- OpenAI-compatible API — minimal code changes required
- Exposes `/v1/chat/completions` — drop-in replacement for the current Ollama HTTP call

**Recommended models for production:**
- `Mistral-7B-Instruct-v0.3` — strong quality, open weights, efficient on A10G
- `Llama-3-8B-Instruct` — excellent instruction following, widely supported

**Infrastructure:** AWS `g5.xlarge` (1× A10G, 24GB VRAM) or GCP `n1-standard-4` + T4

---

### 2. Embeddings — Replace with Dedicated Inference Server

**Replace:** sentence-transformers on CPU / Ollama nomic (sequential)
**With:** [TEI (Text Embeddings Inference)](https://github.com/huggingface/text-embeddings-inference) by Hugging Face

**Why:**
- GPU-accelerated, handles 1,000+ embeddings/second
- Native batching via REST API
- Supports the same models (nomic-embed-text, MiniLM) — no retraining needed
- Docker image available, trivial to deploy

**Alternative for moderate load:** sentence-transformers served via FastAPI with async batching is acceptable up to ~100 requests/minute.

---

### 3. Vector Database — Replace ChromaDB File Storage

**Replace:** ChromaDB (file-based PersistentClient)
**With:** [Qdrant](https://qdrant.tech) or [Weaviate](https://weaviate.io)

**Why Qdrant:**
- Handles billions of vectors with horizontal scaling
- Native payload filtering — equivalent to the current title/entity_type metadata filtering
- HNSW index with tunable recall/speed tradeoff
- Available as self-hosted (Kubernetes) or managed cloud

**Why Weaviate:**
- Built-in hybrid search (BM25 + vector) improves recall on short or ambiguous queries
- Strong multi-tenancy for isolating user data
- GraphQL and REST APIs

**For moderate scale (<10M vectors):** ChromaDB can be upgraded to HTTP server mode (`chroma run`) which adds concurrency with minimal migration effort.

---

### 4. Metadata Database — Replace SQLite with PostgreSQL

**Replace:** SQLite
**With:** PostgreSQL (managed: AWS RDS, Supabase, or Railway)

**Why:**
- Concurrent read/write safety
- Full-text search via `pg_trgm` for optional BM25 hybrid retrieval
- JSONB support for flexible metadata schemas
- Mature backup, replication, and monitoring tooling

---

### 5. API Layer — Add FastAPI Backend

**Add:** FastAPI application wrapping retrieval + generation

```
Client → FastAPI → [Classifier → Retriever → vLLM] → Response
```

- Async endpoints for non-blocking inference
- `POST /query` — accepts query, profile, embedder, llm; returns answer + sources
- `POST /ingest` — triggers re-ingestion pipeline
- `GET /health` — for load balancer health checks
- Rate limiting via `slowapi` or an API gateway
- OpenAPI docs auto-generated

---

### 6. Frontend — Replace Streamlit

**Replace:** Streamlit (single-threaded, session-based)
**With:** Next.js (React) or SvelteKit consuming the FastAPI backend

**Why:**
- Streamlit is not designed for concurrent users and does not support proper session management
- A JavaScript frontend enables real-time streaming responses via SSE
- Better mobile experience and customization

**Short-term alternative:** Keep Streamlit for internal/demo use, expose the raw FastAPI for third-party integration.

---

### 7. Ingestion Pipeline — Scheduled Refresh

**Replace:** One-time `ingest.py` script
**With:** Scheduled pipeline using Apache Airflow or Prefect

- Weekly Wikipedia refresh for all entities
- Incremental ingestion: detect changed articles using Wikipedia's revision API
- Dead letter queue for failed fetches
- Alerting on ingestion failures

---

### 8. Observability

| Concern | Tool |
|---------|------|
| Structured logging | JSON logs → CloudWatch / Grafana Loki |
| Metrics | Prometheus + Grafana (latency, retrieval scores, GPU utilization) |
| Distributed tracing | OpenTelemetry → Jaeger |
| Error tracking | Sentry |
| RAG quality monitoring | Ragas (automated retrieval + answer quality scoring) |

---

## Production Architecture Diagram

```
                    ┌────────────────────────┐
                    │   CDN / Load Balancer  │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │    FastAPI Backend      │
                    │  (auto-scaled pods)     │
                    └──────┬─────────┬────────┘
                           │         │
           ┌───────────────▼──┐  ┌───▼────────────────┐
           │  Qdrant Vector   │  │  PostgreSQL (RDS)   │
           │  DB (Managed)    │  │  metadata + logs    │
           └──────────────────┘  └────────────────────┘
                           │
           ┌───────────────▼──────────────────┐
           │        vLLM Inference Server      │
           │     (GPU: A10G, batched)          │
           └───────────────────────────────────┘
                           │
           ┌───────────────▼──────────────────┐
           │        TEI Embedding Server       │
           │     (GPU or CPU, batched)         │
           └───────────────────────────────────┘
```

---

## Cost Estimate (AWS, ~1,000 queries/day)

| Component | Service | Estimated Monthly Cost |
|-----------|---------|----------------------|
| LLM inference | g5.xlarge (A10G) on-demand | ~$300 |
| Embedding server | t3.medium (CPU, TEI) | ~$30 |
| Vector DB | Qdrant Cloud (starter) | ~$25 |
| PostgreSQL | RDS t3.micro | ~$15 |
| API + load balancer | 2× t3.small | ~$30 |
| **Total** | | **~$400/month** |

> **Cost optimization:** Use Spot Instances for the LLM server (up to 70% savings) with automatic fallback. Scale down to zero during off-hours with auto-scaling groups.

---

## Key Tradeoffs

| Decision | Localhost | Production | Tradeoff |
|----------|-----------|------------|---------|
| LLM runtime | Ollama | vLLM | Higher ops complexity → much higher throughput |
| Vector DB | ChromaDB file | Qdrant managed | Cost → horizontal scale and HA |
| Embedding | CPU / Ollama | GPU via TEI | Cost → 50–100× throughput |
| Database | SQLite | PostgreSQL | Migration effort → concurrency safety |
| Query routing | Keyword rules | Could add ML classifier | Complexity → marginal precision gain at this scale |
| Knowledge freshness | Static | Scheduled refresh | Airflow complexity → up-to-date answers |
| UI | Streamlit | Next.js | Rewrite effort → concurrent users, streaming |

---

## Migration Path (Recommended Order)

| Phase | Week | Actions |
|-------|------|---------|
| 1 | 1–2 | Extract business logic into FastAPI; keep Ollama + ChromaDB initially |
| 2 | 3–4 | Replace SQLite with PostgreSQL; add auth and rate limiting |
| 3 | 5–6 | Deploy ChromaDB as HTTP server; add Prometheus + Sentry |
| 4 | 7–8 | Migrate LLM to vLLM on GPU; benchmark latency and throughput |
| 5 | Month 2 | Migrate to Qdrant; implement scheduled ingestion refresh |
| 6 | Month 3 | Replace Streamlit with Next.js frontend; add Ragas quality monitoring |

---

## Summary

The localhost WikiRAG system is a solid RAG foundation. The primary production concerns are:

1. **Concurrency** — replace Ollama with vLLM for batched GPU inference
2. **Scalability** — replace file-based ChromaDB with Qdrant
3. **Reliability** — replace SQLite with PostgreSQL, add observability

The core retrieval and generation logic — chunking, entity-aware filtering, prompt construction — requires minimal changes. The modular `lib/` architecture was designed with this migration in mind.

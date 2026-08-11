# Operations Guide

Tuning, troubleshooting and production notes. Start with the [README](README.md) for what the project is and how to run it.

---

## Configuration presets

Every value lives in `.env`. Restart the backend to apply — note that `docker compose restart` reuses the existing container's environment, so changes to `.env` need a recreate:

```bash
docker compose up -d --force-recreate backend
```

Hot reload during development lives in an override rather than the default file, since
`--reload` restarts the backend on every edit and each start reloads ~2GB of models:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### By document type

| | Long reports, books | Articles (default) | Emails, short notes | Contracts, legal |
|---|---|---|---|---|
| `CHUNK_SIZE` | 1500 | 1000 | 500 | 800 |
| `CHUNK_OVERLAP` | 300 | 200 | 100 | 240 |
| `MIN_CHUNK_SIZE` | 200 | 100 | 50 | 100 |
| `RETRIEVAL_TOP_K` | 7 | 5 | 4 | 5 |
| `RERANK_THRESHOLD` | 0.30 | 0.35 | 0.35 | 0.50 |

Larger chunks preserve argument and context; smaller ones sharpen retrieval precision. Overlap carries context across a split so a sentence cut in half is still findable from either side.

### By priority

**Accuracy** — more query phrasings, a wider candidate pool, strict reranking:

```bash
USE_MULTI_QUERY=true
NUM_QUERY_VARIANTS=5
RETRIEVAL_CANDIDATES=30
USE_RERANKING=true
RERANK_THRESHOLD=0.45
```

**Speed** — one search, no cross-encoder. Roughly halves latency and drops the reranker's memory entirely:

```bash
USE_MULTI_QUERY=false
RETRIEVAL_CANDIDATES=10
USE_RERANKING=false
```

**Cost** — query expansion spends one extra LLM call per question. Disabling it is the single biggest saving:

```bash
USE_MULTI_QUERY=false
DEFAULT_MODEL=gemini:gemini-2.5-flash
MAX_CHAT_HISTORY=10
```

### By language

`BAAI/bge-m3` (default) covers 100+ languages at 1024 dimensions, and lets a question in one language retrieve passages written in another. English-only alternatives are smaller and faster:

```bash
# English only, ~440MB instead of ~2.2GB
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIM=768
RERANKER_MODEL=BAAI/bge-reranker-base
```

`EMBEDDING_DIM` must match the model. If it doesn't, the mismatch is detected at startup and the real dimension wins — but a persisted index built at the old dimension is discarded, so the corpus needs re-ingesting.

### Hybrid search balance

`BM25_WEIGHT` and `SEMANTIC_WEIGHT` set how lexical and semantic hits are weighted during fusion.

| Corpus | BM25 | Semantic |
|---|---|---|
| Codes, part numbers, names | 0.7 | 0.3 |
| General documents (default) | 0.3 | 0.7 |
| Conceptual, cross-language | 0.2 | 0.8 |

Raise BM25 when users search for exact strings; raise semantic when they describe what they mean.

---

## Measured performance

On an RTX 3050 Laptop (4GB VRAM), `gemini-2.5-flash`, 89-chunk corpus:

| Operation | Time |
|---|---|
| Embed 64 texts (GPU) | 0.50s |
| Analysis + one chart | 8–9s |
| Three-chart dashboard | 11–13s |
| RAG query, cross-language | 25–30s |
| Cold start (models into VRAM) | ~45s |
| 1 retrieval, 3 query variants | 0.83s |
| 5 concurrent retrievals | 2.80s for all five |

RAG queries dominate because they add query expansion, hybrid search over every variant, and cross-encoder reranking before the answer call. Turn off `USE_MULTI_QUERY` and `USE_RERANKING` to see the floor.

These are one machine's numbers, not a benchmark. Measure your own.

---

## Hardware

GPU is auto-detected; without one, everything runs on CPU. `DEVICE` in the logs at startup tells you which path is live.

| | Embedder | Reranker | Total |
|---|---|---|---|
| `bge-m3` + `bge-reranker-v2-m3` | ~2.1GB | ~2.1GB | ~4.2GB VRAM |
| English alternatives | ~0.5GB | ~0.5GB | ~1.0GB VRAM |

Those are fp32 figures. With `USE_FP16_ON_GPU=true` (the default) the pair is ~2.7GB resident, which is what makes a 4GB card workable — and why `INFERENCE_CONCURRENCY` defaults to 1. Raising it on a card this size makes concurrent calls evict each other, turning a 0.05s reranker batch into 50-100s.

If loading still fails, set `USE_RERANKING=false` to drop the second model, or force CPU:

```bash
CUDA_VISIBLE_DEVICES= docker compose up -d --force-recreate backend
```

The FAISS index itself always runs on CPU — this project ships `faiss-cpu`. Only embedding and reranking use CUDA.

---

## Operations

```bash
docker compose ps                     # container health
docker compose logs -f backend        # follow the agent loop
curl localhost:8000/status            # documents, vectors, active model
curl localhost:8000/health            # liveness

docker compose exec backend python -m unittest discover -s tests -t .   # 163 offline tests
docker compose exec backend python tests/e2e.py                         # end-to-end, spends API calls
```

The backend logs each turn of the agent loop: the tools it chose, what they returned, and how long the whole request took. That log is the fastest way to understand a disappointing answer.

---

## Troubleshooting

**Answers say the service is unavailable.** Both providers failed. Check the log for the underlying error — a 429 with `insufficient_quota` means the account is out of credit (never retried, by design); a 429 with `RESOURCE_EXHAUSTED` is a rate limit and does get retried with backoff.

**Retrieval misses obvious content.** Lower `RERANK_THRESHOLD` first — it is the most common cause, since passages are dropped after reranking. Then raise `RETRIEVAL_CANDIDATES`, then `NUM_QUERY_VARIANTS`.

**Retrieval returns irrelevant passages.** The opposite: raise `RERANK_THRESHOLD` toward 0.5.

**A knowledge base file did not appear.** The folder is only read at startup, and files are content-hashed — an unchanged file is skipped deliberately. Restart after adding one. Check the log for `⏭ unchanged` versus `✓ … chunks`.

**Out of memory on upload.** Lower `EMBEDDING_BATCH_SIZE` to 8 or 16, and `CHUNK_SIZE` to 500.

**Redis connection refused.** `docker compose ps` should show `ai_agent_redis` healthy. Sessions, chat history and dataframes all live there; the vector index does not, so a Redis restart loses conversations but not documents.

**The index looks wrong after changing the embedding model.** Dimensions no longer match, so the persisted index is discarded on load. Re-ingest, or delete the volume: `docker compose down && docker volume rm fileuploader_chatbot_index_data`.

---

## Production notes

This runs as a demo out of the box. Before exposing it:

- **Add authentication.** Session ids are client-supplied. They isolate data but do not authenticate it — anyone who guesses an id inherits that session until it expires.
- **Set `CORS_ORIGINS`** to your real frontend origin. It ships restricted to localhost.
- **Set `DEBUG_MODE=false`** and `LOG_LEVEL=WARNING`. Debug logging includes document content and tool arguments.
- **Put the API behind a rate limiter.** Every request can trigger several LLM calls, so an unmetered endpoint is a billing risk more than a load one.
- **Mount `/app/data` on durable storage.** The index is written there; a named volume is fine for one host, but not for a cluster.

### Scaling

Two pieces of state prevent naive horizontal scaling:

- **The vector index is per-process and in-memory.** Two backend replicas hold two different indexes, so a request will hit one or the other. A shared vector database (Qdrant, pgvector, Milvus) is the real fix.
- **`llm_service.model` is process-global.** Switching model via `/models/select` affects every user on that replica.

Redis already handles sessions, chat history and dataframes correctly across replicas.

Vertical scaling is the simpler path here: one backend with a GPU handles a lot, because the expensive parts (embedding, reranking) are batched and the LLM calls are I/O-bound and fully async.

### Backup

```bash
docker run --rm -v fileuploader_chatbot_index_data:/d -v "$PWD:/b" alpine tar czf /b/index.tar.gz -C /d .
docker compose exec redis redis-cli SAVE
```

The knowledge base folder is the source of truth for shared documents — back that up and the index can always be rebuilt by restarting.

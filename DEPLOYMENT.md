# Operations Guide

Tuning, troubleshooting and production notes. Start with the [README](README.md) for what the project is and how to run it.

---

## Configuration presets

Every value lives in `.env`. Restart the backend to apply — note that `docker compose restart` reuses the existing container's environment, so changes to `.env` need a recreate:

```bash
docker compose up -d --force-recreate backend
```

### By document type

| Setting | Long reports, books | Articles (default) | Emails, short notes | Contracts, legal |
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
DEFAULT_MODEL=gemini:gemini-3.5-flash-lite
MAX_CHAT_HISTORY=10
```

**Repeatability** — `LLM_TEMPERATURE` defaults to `0.0`, and should stay there for
analysis. At `0.1`, the same twelve questions asked twice diverged on five: a subset filter
applied once and skipped once, two files' totals summed once and correctly refused once,
"typical order size" read as a median value once and a median unit count once. Every figure
was exact both times — what varied was interpretation. An agent obeying a dozen rules per
turn cannot afford sampling noise. Raise it only if you want variety over repeatability.

Query expansion is the exception and keeps its own higher temperature in
[`llm.py`](app/services/llm.py): phrasing one question several ways is the entire point of
that call.

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

### Measured retrieval quality

```bash
docker compose exec backend python -m tests.retrieval_eval
```

A 21-passage corpus with a known correct answer for each of 23 questions, built to separate
the halves of hybrid search rather than flatter them: questions sharing rare tokens with
their passage, questions phrased with none of its words, an English question whose answer is
in Arabic, two whose answer spans a pair of passages, two turning on a number rather than a
topic, a negation pair, and near-duplicate passages that punish matching on subject alone.

| Configuration | recall@1 | recall@5 | MRR | s/query |
|---|--:|--:|--:|--:|
| BM25 only | 61% | 70% | 0.63 | 0.13 |
| Semantic only | 96% | 100% | 0.98 | 0.13 |
| Hybrid, no rerank | 78% | 100% | 0.87 | 0.14 |
| **Hybrid + rerank (shipped)** | **96%** | **96%** | **0.96** | **2.67** |
| Lexical-heavy hybrid (0.7/0.3) | 96% | 96% | 0.96 | 2.59 |

Recall@1 by question type, shipped configuration: rare token 100%, paraphrase 100%,
cross-language 100%, near-duplicate distractor 75%. Multi-hop, numeric and negation
questions all rank first.

Three things worth knowing before tuning:

- **BM25 alone cannot serve a multilingual corpus** — 0% on the cross-language question,
  50% on paraphrases, because an English question and an Arabic passage share no tokens. It
  is still the half that guarantees an exact identifier like `CMP-4410-A` is found.
- **Reranking buys ranking, and the threshold costs recall.** Reranking lifts recall@1 from
  78% to 96%. But `RERANK_THRESHOLD` then discards everything below it — an average of 3 of
  5 passages — taking recall@5 from 100% down to 96%. **Any value between 0.25 and 0.60
  behaves identically here**, because the cross-encoder's scores are bimodal: passages land
  well above 0.6 or well below 0.25, and the threshold merely cuts the gap. Lowering it from
  0.35 to 0.25 therefore changes nothing; only going below ~0.2 admits more. It also gets
  **stricter as the corpus grows** — on 15 passages it kept 2.0 on average, on 21 it keeps
  1.1, so a question whose answer spans two passages is the first casualty.
- **On this corpus semantic-only matches the shipped configuration and is 20× faster.** That
  is a result about 21 passages and one embedding model, not a recommendation — `bge-m3` is
  unusually strong multilingually, and a corpus of codes and part numbers would likely
  invert it. It is stated because measuring it was the point.

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

The application auto-detects CUDA and falls back to CPU; `Hardware Accelerator Detected` in the startup log tells you which path is live.

Docker will not expose a GPU unless one is requested, and requesting a device that does not exist fails the container — so the compose file asks for `${GPU_COUNT:-0}` and runs anywhere by default. On an NVIDIA host with the container toolkit installed, set `GPU_COUNT=1` in `.env` and recreate. That is the only switch; nothing else changes between the two modes.

| Models | Embedder | Reranker | Total |
|---|---|---|---|
| `bge-m3` + `bge-reranker-v2-m3` | ~2.1GB | ~2.1GB | ~4.2GB VRAM |
| English alternatives | ~0.5GB | ~0.5GB | ~1.0GB VRAM |

Those are fp32 figures. With `USE_FP16_ON_GPU=true` (the default) the pair is ~2.7GB resident, which is what makes a 4GB card workable — and why `INFERENCE_CONCURRENCY` defaults to 1. Raising it on a card this size makes concurrent calls evict each other, turning a 0.05s reranker batch into 50-100s.

If loading still fails, set `USE_RERANKING=false` to drop the second model, or force CPU by taking the device away in `.env` — a shell variable will not do it, because the container's environment comes from `.env` and the compose `environment:` block, not from the shell that ran `docker compose`:

```bash
GPU_COUNT=0    # in .env, then:
docker compose up -d --force-recreate backend
```

The FAISS index itself always runs on CPU — this project ships `faiss-cpu`. Only embedding and reranking use CUDA.

---

## Operations

```bash
docker compose ps                     # container health
docker compose logs -f backend        # follow the agent loop
curl localhost:8000/status            # documents, vectors, active model
curl localhost:8000/health            # liveness

docker compose exec backend python -m unittest discover -s tests -t .   # 558 offline tests
docker compose exec backend python tests/e2e.py             # end-to-end, spends API calls
docker compose exec backend python -m tests.acceptance      # regression list, spends API calls
```

The backend logs each turn of the agent loop: the tools it chose, what they returned, and how long the whole request took. That log is the fastest way to understand a disappointing answer — and the same lines are streamed to the UI per request, so you rarely need to reach for `docker compose logs` while a question is running.

```bash
curl -N -X POST localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","query":"revenue by region"}'
```

---

## Troubleshooting

**Answers arrive but the model shown is not the one selected.** That is failover working: the caption under each answer reports the model that actually produced it, while the footer shows what is selected. A `429` in the log naming the selected provider explains the switch.

**Answers say the service is unavailable.** Both providers failed. Check the log for the underlying error. Every 429 is retried with backoff and then failed over, whether it is a rate limit (`RESOURCE_EXHAUSTED`) or an exhausted account (`insufficient_quota`) — the two are indistinguishable by status, and retrying the second costs latency rather than money, since a refused request generates no completion. If the log shows `insufficient_quota` on both providers, add credit; no amount of retrying will clear it.

**An answer looks plausible but the totals feel wrong.** Check the table profile the agent was given — `docker compose logs backend | grep "⚠"` shows the data-quality warnings it saw. Inconsistent label spellings, a currency column duplicated in two units, and exactly repeated rows are each measured per table and stated in the prompt, because none is visible from a column list.

**Retrieval misses obvious content.** Passages are dropped after reranking, so `RERANK_THRESHOLD` is the first suspect — but lower it *properly*: measurement shows every value from 0.25 to 0.60 keeps the same passages, because the cross-encoder's scores are bimodal. Going from 0.35 to 0.25 will appear to do nothing. Try `0.0` to see what is being discarded, then raise `RETRIEVAL_CANDIDATES` and `NUM_QUERY_VARIANTS`. Run `tests/retrieval_eval.py` against your own corpus rather than guessing.

**Retrieval returns irrelevant passages.** The opposite: raise `RERANK_THRESHOLD` toward 0.5.

**A knowledge base file did not appear.** The folder is only read at startup, and files are content-hashed — an unchanged file is skipped deliberately. Restart after adding one. Check the log for `⏭ unchanged` versus `✓ … chunks`. The skip verifies the document still has vectors as well as a registry entry: when only the registry survived, every file looked unchanged and was skipped, so the corpus came up empty and no restart could repair it. That case now logs `↻ registered but not indexed, re-ingesting`.

**Out of memory on upload.** Lower `EMBEDDING_BATCH_SIZE` to 8 or 16, and `CHUNK_SIZE` to 500.

**The backend will not start on Windows: "an attempt was made to access a socket in a way forbidden by its access permissions".** Nothing is listening on the port — Windows has *reserved* it. Hyper-V and WSL2 claim blocks of dynamic ports at boot, and 8000 often lands inside one:

```bash
netsh int ipv4 show excludedportrange protocol=tcp
```

If a range covers 8000, a reboot usually clears it — the blocks are claimed afresh at boot and may land elsewhere. To stop them landing on 8000 at all, reserve it explicitly from an Administrator prompt:

```bash
net stop winnat
```

```bash
netsh int ipv4 add excludedportrange protocol=tcp startport=8000 numberofports=1
```

```bash
net start winnat
```

An explicit exclusion is removed from the dynamic pool but stays bindable on purpose, which is exactly what Docker needs. Failing that, stop publishing the port: the frontend reaches the backend over the compose network at `http://backend:8000`, so publishing it is only needed for `curl` from the host. Drop it with an untracked `docker-compose.override.yml`:

```yaml
services:
  backend:
    ports: !override []
```

**`docker compose restart` did not pick up my change.** It reuses the existing container, so edits to `.env`, `ports` or `environment` are ignored — the container keeps whatever it was created with, which can leave a port unpublished long after the compose file gained it. Use `docker compose up -d --force-recreate backend`.

**Redis connection refused.** `docker compose ps` should show `ai_agent_redis` healthy. Sessions, chat history and dataframes all live there; the vector index does not, so a Redis restart loses conversations but not documents.

**The index looks wrong after changing the embedding model.** Dimensions no longer match, so the persisted index is discarded on load — the log says `Persisted index does not match documents … discarding`. Knowledge base files re-ingest themselves on the next start. Uploaded documents do not, because their bytes are gone: re-upload them, or delete the volume and start clean with `docker compose down && docker volume rm fileuploader_chatbot_index_data`.

**Answers arrive in the wrong language.** The log line `🗣 Answered as:` records the language the model declared for the question, which is the only place it states what it thought it read. If that says English and the prose is not, the drift is the model's rather than a detection failure — it happens on flash-class models, most often on questions with no clear ask. A reasoning-tier model is the fix; see the model table in the [README](README.md).

---

## Production notes

This runs as a single-host application and is not hardened. Before exposing it to
anyone but yourself:

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
docker run --rm \
  -v fileuploader_chatbot_index_data:/d -v "$PWD:/b" \
  alpine tar czf /b/index.tar.gz -C /d .
docker compose exec redis redis-cli SAVE
```

The knowledge base folder is the source of truth for shared documents — back that up and the index can always be rebuilt by restarting.

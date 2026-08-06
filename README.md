# AI Data Agent — Multilingual RAG for Document Q&A

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![FAISS](https://img.shields.io/badge/vector%20search-FAISS-orange)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

**Chat with your PDFs, Word documents, Excel files and CSVs — in any language.** Upload documents, ask questions in natural language, and get answers with charts, statistics and source citations.

A Retrieval-Augmented Generation (RAG) agent combining **hybrid search** — BM25 keyword matching plus FAISS dense vector retrieval — with Reciprocal Rank Fusion and cross-encoder reranking, and LLM **function calling** for statistical analysis and data visualisation. It parses documents into both a searchable text index and queryable dataframes, then lets the model decide whether a question needs a document lookup, a calculation, a filter, or a chart — and answer with all of them at once.

- **Backend** FastAPI · **Frontend** Streamlit · **Sessions** Redis · **Vector store** FAISS
- **Models** OpenAI and Google Gemini, switchable at runtime, with automatic failover
- **Multilingual** `BAAI/bge-m3` embeddings and reranking — ask in English, retrieve from Arabic, French or Chinese; answers come back in the language you asked in

### What you can use it for

- **Chat with PDFs** — ask questions across a stack of reports, contracts or research papers
- **Spreadsheet analysis** — "average revenue by region", "top 10 by margin", correlations
- **Document comparison** — query several files at once and cite which one each fact came from
- **Automatic charts** — bar, line, pie, scatter and histogram generated from your data on request
- **Cross-language search** — a knowledge base in one language, questions in another

---

## Quick start

```bash
cp .env.example .env      # add OPENAI_API_KEY and/or GEMINI_API_KEY
docker compose up --build
```

Frontend on <http://localhost:8501>, API docs on <http://localhost:8000/docs>.

The first build downloads ~2GB of model weights and bakes them into the image, so startup afterwards is fast. Subsequent builds reuse the cache.

`./start.sh local` runs the stack without Docker, against a local Redis.

---

## How retrieval works

A single question fans out into several searches and is narrowed back down:

```
        query
          │
          ▼
  query expansion ──────────►  n phrasings (LLM, 1 call)
          │
          ├──────────────┬──────────────┐        per phrasing, in parallel
          ▼              ▼              ▼
        BM25          FAISS           BM25 …
     (lexical)      (semantic)
          └──────────────┴──────────────┘
                         │
                         ▼
        Reciprocal Rank Fusion  (BM25_WEIGHT / SEMANTIC_WEIGHT)
                         │
                         ▼
              cross-encoder rerank
                         │
                         ▼
           drop below RERANK_THRESHOLD  →  top-k passages
```

**Why both searches.** BM25 matches literal tokens — invoice numbers, product codes, names. Embeddings match meaning — "how much did we make" against a passage about revenue. Each fails where the other works, so results are fused by rank rather than score (RRF), which avoids having to calibrate two incomparable scoring scales.

**Why expand the query.** One phrasing retrieves one neighbourhood of the embedding space. Asking the model for alternative phrasings and searching all of them widens recall before the reranker narrows for precision. Expansion is best-effort: if the call fails or returns junk, retrieval proceeds with the original query alone.

**Why rerank.** Bi-encoders embed the query and passage separately, so they can only measure rough similarity. A cross-encoder reads both together and scores relevance directly. It's far too slow to run over a whole corpus, which is why it runs last, over a small candidate set.

Set `USE_MULTI_QUERY=false` or `USE_RERANKING=false` to drop either stage.

---

## Session isolation

Uploads belong to the session that made them. Retrieval is scoped to the caller's own documents plus the shared knowledge base — one user's upload is never retrievable by another.

| Scope | Where it comes from | Who can retrieve it |
|---|---|---|
| Session documents | Uploaded through the UI or `POST /upload` | That session only |
| Shared knowledge base | Files in [`app/kb/`](app/kb/), auto-ingested at startup | Every session |

Sessions expire after `SESSION_TTL` (1h default), refreshed on access. `DELETE /sessions/{id}` clears chat history, dataframes and vectors.

### Durability

The index is written to disk on every change and restored at startup, so uploads survive a restart. Writes go to a temporary file and are renamed into place, so a crash mid-write cannot leave a torn pair; on load, an index whose vector count or dimension disagrees with its document list is discarded rather than trusted.

Two things follow from the index outliving the process:

- **Expired sessions are pruned on startup.** Redis sessions expire while the index does not, so restored vectors belonging to sessions that timed out during downtime are dropped instead of lingering as searchable orphans.
- **Knowledge base files are content-hashed.** Re-ingesting blindly would duplicate the whole corpus on every restart. An unchanged file is skipped; an edited one supersedes its previous chunks.

Set `PERSIST_INDEX=false` to keep everything in memory.

---

## Tools available to the agent

| Tool | Purpose |
|---|---|
| `search_knowledge_base` | Retrieve passages from indexed documents |
| `calculate_statistics` | sum, mean, median, std, min, max, count, describe, correlation — with optional `group_by` |
| `query_data` | Filter, sort and page through table rows |
| `generate_chart` | One Plotly figure — bar / line / pie / scatter / histogram |
| `generate_dashboard` | Up to six related figures from one dataset in a single call |

Tools run **concurrently** — a question needing a lookup and two calculations issues all three at once rather than serially.

Structured data (spreadsheets, PDF tables) is addressed by `table_name`. Figures the model extracts from prose are passed inline as `custom_data`, which lets it chart numbers that were never in a table.

### Charts travel out-of-band

A Plotly figure runs to tens of kilobytes. Returning one as a tool result would put it in the context window on every subsequent turn, and asking the model to copy it into its answer is both expensive and unreliable.

Instead, figures are intercepted when the tool returns: they attach directly to the response, and the model receives a short receipt — titles and summary statistics only. Measured on a four-chart dashboard, the model sees **673 bytes instead of 75,643** — a 99% reduction, repeated on every turn thereafter.

A malformed chart specification is isolated: the remaining charts still render, and the failure is reported alongside them rather than discarding the batch.

### Failure handling

Transient provider faults — HTTP 429 rate limits, 503 overload, timeouts — are retried with exponential backoff and jitter before failing over to the other provider.

Funding failures are deliberately excluded. An exhausted account also returns 429, but no amount of retrying will clear it, so it fails straight through instead of burning billable calls.

---

## Structured output

Final answers conform to a Pydantic schema — `answer`, `visualizations`, `key_insights`, `sources_used` — so the frontend renders them without parsing prose.

The two providers reach this differently, and the differences are instructive:

- **OpenAI** — tool declarations and a `json_schema` response format travel in one request; the model either calls a tool or returns JSON matching the schema. Sent through `create()` rather than the `parse()` helper, because `parse()` requires every tool to be `strict`, and the free-form `custom_data` argument cannot be expressed in OpenAI's strict subset.
- **Gemini** — rejects a response schema alongside tool declarations, so tools run first and a second pass shapes the answer. That pass uses JSON mode rather than a schema: the Developer API also rejects `additionalProperties`, which Pydantic emits for this schema's `Dict[str, Any]` fields. The shape is specified in the system prompt instead.

Both paths converge on the same parser, which falls back through direct JSON, fenced code blocks, and field salvage — so a malformed answer degrades rather than crashes.

---

## Configuration

Everything lives in `.env`; see [`.env.example`](.env.example) for the full set.

```bash
DEFAULT_MODEL=gemini:gemini-2.5-flash     # "gemini:" prefix routes to Google
FALLBACK_MODEL=gemini:gemini-flash-latest # one failover hop, then a clean error
LLM_MAX_RETRIES=2                         # transient faults retried before failover

EMBEDDING_MODEL=BAAI/bge-m3               # 1024-dim, 100+ languages
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

USE_MULTI_QUERY=true
NUM_QUERY_VARIANTS=3                      # total phrasings, original included
RETRIEVAL_CANDIDATES=25                   # pool handed to the reranker
RETRIEVAL_TOP_K=5                         # passages kept
RERANK_THRESHOLD=0.35                     # 0.50 precision · 0.25 recall

BM25_WEIGHT=0.3                           # lexical vs semantic in RRF
SEMANTIC_WEIGHT=0.7

CHUNK_SIZE=1000
CHUNK_OVERLAP=200                         # carried across splits so context survives

PERSIST_INDEX=true                        # index survives restarts
MAX_QUERY_CHARS=4000                      # request bounds
MAX_CUSTOM_DATA_ROWS=5000
```

**Tuning.** Poor recall → raise `NUM_QUERY_VARIANTS` and `RETRIEVAL_CANDIDATES`, lower `RERANK_THRESHOLD`. Irrelevant passages → raise the threshold. Codes and identifiers matter more than phrasing → raise `BM25_WEIGHT`.

---

## Hardware

GPU is detected automatically and used if present; otherwise everything runs on CPU. The default PyTorch wheels are CUDA 13 builds.

The embedder and reranker are both ~570M parameters and load together, needing roughly 4.5GB of VRAM in fp32. On a smaller card, run `USE_RERANKING=false`, or force CPU with `CUDA_VISIBLE_DEVICES=`.

---

## API

| Method | Route | |
|---|---|---|
| `POST` | `/upload` | Ingest a file into a session |
| `POST` | `/chat` | Ask a question |
| `GET` | `/sessions/{id}/history` | Chat history and stats |
| `GET` | `/sessions/{id}/files` | Documents in a session |
| `GET` | `/sessions/{id}/dataframes` | Tables and their columns |
| `DELETE` | `/sessions/{id}/files/{doc_id}` | Remove one document |
| `DELETE` | `/sessions/{id}` | Clear the session entirely |
| `GET` | `/models` · `POST` `/models/select` | List / switch model |
| `GET` | `/status` · `/health` | Index stats · liveness |

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@report.pdf" -F "session_id=demo"

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","query":"What drove the change in Q4?","use_rag":true}'
```

---

## Layout

```
app/
├── main.py              FastAPI routes and the agent loop
├── frontend.py          Streamlit UI
├── core/
│   ├── config.py        Settings, GPU detection
│   ├── vectorstore.py   FAISS index, session-scoped search
│   └── session.py       Redis: history, dataframes, document index
├── services/
│   ├── ingest.py        Parsing, table extraction, chunking
│   ├── retrieval.py     Multi-query, hybrid search, RRF, reranking
│   ├── llm.py           Provider routing, tool calling, structured output
│   ├── tools.py         Tool schemas and execution
│   └── chart.py         Plotly generation
└── utils/
    ├── prompts.py       System prompt
    └── schemas.py       Pydantic request/response models
```

---

## Known limitations

- BM25 is rebuilt per request rather than maintained incrementally — fine at this corpus size, the first thing to change as it grows.
- The index is rebuilt in full when documents are removed. Acceptable at this scale; an IDMap-backed index would avoid it.
- No authentication. Sessions are client-supplied UUIDs, which isolate data but do not authenticate it. Put this behind auth before exposing it.
- Schema enforcement is stronger on OpenAI than on Gemini, for the API reasons above. Gemini relies on JSON mode plus a prompt-specified shape and a tolerant parser.
- `matplotlib` is declared in `requirements.txt` but unused — charts are Plotly only. Safe to drop on the next rebuild.

## License

MIT — see [LICENSE](LICENSE).

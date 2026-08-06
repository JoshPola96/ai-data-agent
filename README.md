# AI Data Agent — Multilingual RAG for Document Q&A and Charting

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![FAISS](https://img.shields.io/badge/vector%20search-FAISS-orange)
![Plotly](https://img.shields.io/badge/charts-Plotly-3F4F75)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

**Ask questions about your PDFs, Word documents, Excel files and CSVs — in any language — and get charts back, not just text.**

A Retrieval-Augmented Generation (RAG) agent that reads documents into both a searchable text index and queryable dataframes, then decides for itself whether a question needs a document lookup, a statistical calculation, a filtered table, a single chart, or a whole dashboard — and can do several of those at once, in parallel.

- **Backend** FastAPI · **Frontend** Streamlit · **Sessions** Redis · **Vector store** FAISS
- **Models** OpenAI and Google Gemini, switchable at runtime, with automatic failover
- **Multilingual** `BAAI/bge-m3` throughout — ask in English, retrieve from Arabic, answer in either

### What you can use it for

- **Chat with PDFs** — question a stack of reports, contracts or research papers at once
- **Spreadsheet analysis** — "revenue by region", "top 10 by margin", correlations, distributions
- **Instant dashboards** — one question, several linked charts
- **Charting figures that were never in a table** — numbers pulled out of a PDF's prose
- **Cross-language search** — a knowledge base in one language, questions in another

---

## Quick start

```bash
cp .env.example .env      # add GEMINI_API_KEY and/or OPENAI_API_KEY
docker compose up --build
```

Then open **http://localhost:8501**. API docs at **http://localhost:8000/docs**.

The first build downloads ~4GB of model weights into the image, so it takes a while. After that, startup is about 45 seconds — the time to load the models into memory.

```bash
docker compose logs -f backend        # watch the agent loop
docker compose down                   # stop
```

Drop files into [`app/kb/`](app/kb/) to share them across every session, or upload through the UI for a single session.

---

## Visual analysis

Charts are a first-class output, not an afterthought. The agent picks the form that fits the question:

| Question shape | Chart |
|---|---|
| Compare categories | `bar` — set `color_column` and series break out side by side |
| Change over time | `line` — dates detected and sorted automatically |
| Share of a whole | `pie` |
| Relationship between two numbers | `scatter` |
| Spread of one number | `histogram` |
| Spread and outliers per category | `box` |
| Which columns move together | `heatmap` — correlation across every numeric column |

Ranking questions ("top 10 products by revenue") trim to `top_n` rather than plotting everything.

**Dashboards in one call.** `generate_dashboard` renders up to six related figures from a single dataset — a trend, a breakdown and a distribution together. A malformed spec is isolated: the other charts still render, and the failure is reported next to them instead of losing the batch.

**Column names are matched fuzzily.** The model asking for `revenue` still finds `Revenue_USD`, via case-insensitive then close-match resolution. Small thing, but it removes the most common cause of a failed chart.

**Charts travel out-of-band.** A Plotly figure runs to tens of kilobytes. Returning one as a tool result would put it in the context window on every later turn, and asking the model to copy it into its answer is expensive and unreliable. Figures are intercepted when the tool returns, attached straight to the response, and the model gets a short receipt — titles and summary statistics only.

> Measured on a four-chart dashboard: the model sees **673 bytes instead of 75,643**, on every turn thereafter.

Every figure is also emitted as standalone interactive HTML, so it can be embedded outside the app.

---

## How retrieval works

A single question fans out into several searches, then narrows back down:

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

**Why both searches.** BM25 matches literal tokens — invoice numbers, product codes, names. Embeddings match meaning — "how much did we make" against a passage about revenue. Each fails where the other works, so results are fused by rank rather than score, which avoids calibrating two incomparable scales.

**Why expand the query.** One phrasing reaches one neighbourhood of the embedding space. Generating alternatives and searching all of them widens recall before the reranker narrows for precision. It is best-effort: if the call fails, retrieval proceeds with the original query alone.

**Why rerank.** Bi-encoders embed query and passage separately, so they can only measure rough similarity. A cross-encoder reads both together. It is far too slow for a whole corpus, which is why it runs last, over a small candidate set.

Set `USE_MULTI_QUERY=false` or `USE_RERANKING=false` to drop either stage.

---

## Session isolation

Uploads belong to the session that made them. Retrieval is scoped to the caller's own documents plus the shared knowledge base — one user's upload is never retrievable by another.

| Scope | Source | Who can retrieve it |
|---|---|---|
| Session documents | UI upload or `POST /upload` | That session only |
| Shared knowledge base | Files in [`app/kb/`](app/kb/) | Every session |

Sessions expire after `SESSION_TTL` (1h default), refreshed on access. `DELETE /sessions/{id}` clears history, dataframes and vectors.

### Durability

The index is written to disk on every change and restored at startup, so uploads survive a restart. Writes go to a temporary file and are renamed into place, so a crash mid-write cannot leave a torn pair. On load, an index whose vector count or dimension disagrees with its document list is discarded rather than trusted.

Two things follow from the index outliving the process:

- **Expired sessions are pruned at startup.** Redis sessions expire, the index does not — so vectors from sessions that timed out during downtime are dropped instead of lingering as searchable orphans.
- **Knowledge base files are content-hashed.** Re-ingesting blindly would duplicate the whole corpus on every restart. Unchanged files are skipped; edited ones supersede their previous chunks.

Set `PERSIST_INDEX=false` to keep everything in memory.

---

## Tools available to the agent

| Tool | Purpose |
|---|---|
| `search_knowledge_base` | Retrieve passages from indexed documents |
| `calculate_statistics` | sum, mean, median, std, min, max, count, describe, correlation — with optional `group_by` |
| `query_data` | Filter, sort and page through rows |
| `generate_chart` | One figure, seven types |
| `generate_dashboard` | Up to six related figures in a single call |

Tools run **concurrently** — a question needing a lookup and two calculations issues all three at once.

Structured data is addressed by `table_name`. Figures the model extracts from prose are passed as `custom_data`, which is what lets it chart numbers that were never tabular.

### Failure handling

Transient provider faults — 429 rate limits, 503 overload, timeouts — retry with exponential backoff and jitter before failing over to the other provider.

Funding failures are deliberately excluded. An exhausted account also returns 429, but no retry will clear it, so it fails straight through instead of burning billable calls. The distinction matters more than it looks: Gemini's *recoverable* rate-limit message reads "check your plan and billing details", so matching on the word "billing" would misclassify it as terminal.

---

## Structured output

Final answers conform to a Pydantic schema — `answer`, `visualizations`, `key_insights`, `sources_used` — so the frontend renders them without parsing prose.

The two providers reach it differently, and the differences are instructive:

- **OpenAI** — tool declarations and a `json_schema` response format travel in one request. Sent through `create()` rather than the `parse()` helper, because `parse()` requires every tool to be `strict`, and the free-form `custom_data` argument cannot be expressed in OpenAI's strict subset.
- **Gemini** — rejects a response schema alongside tool declarations, so tools run first and a second pass shapes the answer. That pass uses JSON mode rather than a schema, because the Developer API also rejects `additionalProperties`, which Pydantic emits for `Dict[str, Any]` fields.

Both converge on the same parser, which falls back through direct JSON, fenced code blocks, then field salvage — so a malformed answer degrades instead of crashing.

---

## Configuration

All settings live in `.env`; see [`.env.example`](.env.example) for the full set, and [DEPLOYMENT.md](DEPLOYMENT.md) for tuning presets by document type, language and priority.

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

BM25_WEIGHT=0.3                           # lexical vs semantic in fusion
SEMANTIC_WEIGHT=0.7

CHUNK_SIZE=1000
CHUNK_OVERLAP=200                         # carried across splits so context survives

PERSIST_INDEX=true                        # index survives restarts
MAX_QUERY_CHARS=4000                      # request bounds
MAX_CUSTOM_DATA_ROWS=5000
```

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
| `DELETE` | `/sessions/{id}` | Clear the session |
| `GET` · `POST` | `/models` · `/models/select` | List / switch model |
| `GET` | `/status` · `/health` | Index stats · liveness |

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@report.pdf" -F "session_id=demo"

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","query":"What drove the change in Q4?","use_rag":true}'
```

---

## Tests

```bash
docker compose exec backend python -m unittest discover -s tests -t .
```

73 tests, none of which touch the network — providers are mocked, charts are verified against real Plotly output, retries against synthetic faults. They cover session isolation, chunk overlap, index persistence and torn-write recovery, chart failure isolation, and the retry classifier.

```bash
docker compose exec backend python tests/e2e.py
```

An end-to-end pass against a running stack: ingestion, analysis, dashboards, cross-language retrieval, isolation, model switching, input bounds and deletion. It makes real LLM calls, so it costs money and is deliberately excluded from discovery.

---

## Layout

```
app/
├── main.py              FastAPI routes and the agent loop
├── frontend.py          Streamlit UI
├── core/
│   ├── config.py        Settings, GPU detection
│   ├── vectorstore.py   FAISS index, session-scoped search, persistence
│   └── session.py       Redis: history, dataframes, document index
├── services/
│   ├── ingest.py        Parsing, table extraction, chunking
│   ├── retrieval.py     Multi-query, hybrid search, fusion, reranking
│   ├── llm.py           Provider routing, tool calling, structured output, retry
│   ├── tools.py         Tool schemas and execution
│   └── chart.py         Plotly generation
└── utils/
    ├── prompts.py       System prompt
    └── schemas.py       Pydantic request/response models
```

---

## Known limitations

- No authentication. Session ids isolate data but do not authenticate it — put this behind auth before exposing it.
- The vector index is per-process, so backend replicas would each hold their own. A shared vector database is the fix for horizontal scaling.
- `llm_service.model` is process-global: switching model affects every user on that replica.
- BM25 is rebuilt per request rather than maintained incrementally — fine at this corpus size, the first thing to change as it grows.
- The index is rebuilt in full when documents are removed. An IDMap-backed index would avoid it.
- Schema enforcement is stronger on OpenAI than on Gemini, for the API reasons above.

## License

MIT — see [LICENSE](LICENSE).

# AI Data Agent — Multilingual RAG for Document Q&A, Charting and Diagrams

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![FAISS](https://img.shields.io/badge/vector%20search-FAISS-orange)
![Plotly](https://img.shields.io/badge/charts-Plotly-3F4F75)
![Mermaid](https://img.shields.io/badge/diagrams-Mermaid-FF3670)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)
![Tests](https://img.shields.io/badge/tests-163%20offline-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

**Ask questions about your PDFs, Word documents, Excel files and CSVs — in any language — and get charts, dashboards and flowcharts back, not just text.**

A Retrieval-Augmented Generation (RAG) agent that reads documents into both a searchable text index and queryable dataframes, then decides for itself whether a question needs a document lookup, a statistical calculation, a filtered table, a chart, a multi-panel dashboard, or a diagram — and can do several of those at once, in parallel.

- **Backend** FastAPI · **Frontend** Streamlit · **Sessions** Redis · **Vector store** FAISS
- **Models** OpenAI and Google Gemini, switchable at runtime, with automatic failover
- **Multilingual** `BAAI/bge-m3` throughout — ask in English, retrieve from Arabic, answer in English

### What you can use it for

- **Chat with PDFs** — question a stack of reports, contracts or research papers at once
- **Spreadsheet analysis** — revenue by region, top 10 by margin, correlations, distributions, rates
- **Instant dashboards** — one question, up to six linked charts
- **Flowcharts from documents** — "explain this process" returns a rendered diagram
- **Charting figures that were never in a table** — numbers pulled out of a PDF's prose
- **Cross-language search** — a knowledge base in one language, questions in another

---

## Quick start

```bash
cp .env.example .env      # add GEMINI_API_KEY and/or OPENAI_API_KEY
docker compose up --build
```

Open **http://localhost:8501**. API docs at **http://localhost:8000/docs**.

The first build downloads ~4GB of model weights into the image, so expect 20–40 minutes depending on your connection. After that, startup is about 45 seconds — the time to load the models into memory.

```bash
docker compose logs -f backend        # watch the agent loop reason
docker compose ps                     # container health
docker compose down                   # stop
```

Drop files into [`app/kb/`](app/kb/) to share them across every session, or upload through the UI for a single session.

### Without a GPU

The default compose file reserves an NVIDIA device. On a machine without one:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

### Developing

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Adds `--reload`. It is not the default because every edit restarts the backend, and the backend reloads ~2GB of models each time.

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

Four options do most of the analytical work:

- **`top_n`** — ranking questions trim instead of plotting all 400 rows.
- **`reference`** — draws a `mean`, `median` or literal baseline. A comparison without one leaves "compared with what" unanswered.
- **`resample`** — buckets a date axis into `D`/`W`/`M`/`Q`/`Y` before aggregating. Daily rows rarely answer a monthly question.
- **`y2_column`** — a second measure on its own right-hand axis. This is the honest way to put revenue beside a margin percentage; sharing one scale flattens the smaller series into the baseline.

**Rates are computed, not looked up.** `y_column` accepts `"returns/units"` to divide two numeric columns, because rates and per-unit values are what people ask about and exist in no source column. A rate aggregates as `sum(numerator) / sum(denominator)` — averaging per-row ratios would weight a 2-unit row the same as a 2000-unit one, a different and usually wrong statistic.

**Labels are formatted by magnitude.** Revenue reads `2.46M`, a return rate reads `3.69%`, a count reads `1029`. SI notation everywhere renders `0.0369` as `36.9m` — milli — beside charts labelled in millions.

**Dashboards in one call.** `generate_dashboard` renders up to six related figures from one dataset. A malformed spec is isolated: the other charts still render and the failure is reported next to them rather than losing the batch.

**Column names are matched fuzzily.** Asking for `revenue` still finds `Revenue_USD`, via case-insensitive then close-match resolution. It removes the most common cause of a failed chart.

### Diagrams

Ask what a document's process looks like and you get a rendered flowchart — sequence diagrams, state machines, entity relationships and timelines too.

The agent writes Mermaid fluently and escapes it badly. An unquoted parenthesis inside a node label ends the node early and aborts the whole diagram:

```
A[GET /v2/auth/agent/phone-status (phone_number, x-agent-secret)]
                                  ^ Mermaid reads this as a new node shape
```

Labels are therefore quoted before rendering, not trusted. A source whose brackets still do not balance is returned as an error naming the line, rather than drawn as something the author never wrote — and a label containing an arrow is deliberately left alone, because that means a bracket was left open and quoting it would hide the mistake.

### Everything renders in place

| Output | Rendered as |
|---|---|
| Chart / dashboard | Interactive Plotly, zoom and hover, PNG export |
| Diagram | Mermaid SVG, with a "Diagram source" expander |
| Table | Real dataframe — sortable, searchable, CSV download |
| Reasoning | Agent trace expander: every tool call and result |
| Cost | `⚡ 9.2s · 2 turns · 2 tool calls · gemini-2.5-flash` |

**Visuals travel out-of-band.** A Plotly figure runs to tens of kilobytes. Returning one as a tool result would put it in the context window on every later turn, and asking the model to copy it into its answer is expensive and unreliable. Figures and diagrams are intercepted when the tool returns, attached straight to the response, and the model receives a short receipt — titles and summary statistics only.

> Measured on a four-chart dashboard: the model sees **673 bytes instead of 75,643**, on every turn thereafter.

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

**Why rerank.** Bi-encoders embed query and passage separately, so they measure only rough similarity. A cross-encoder reads both together. It is far too slow for a whole corpus, which is why it runs last, over a small candidate set.

Set `USE_MULTI_QUERY=false` or `USE_RERANKING=false` to drop either stage.

### Inference is serialised on purpose

The embedder and reranker are ~2GB each. On a 4GB card they do not share the GPU — they evict each other. Measured on five concurrent retrievals, one reranker batch took **51–102 seconds**; the identical batch run alone takes **0.05**. Parallelism was not slow, it was pathological.

Models therefore load in **fp16** (2.7GB resident instead of 3.9GB) and inference is queued through a dedicated pool with `INFERENCE_CONCURRENCY=1`:

| | before | after |
|---|---|---|
| 5 concurrent retrievals | 137–163s **each** | 2.80s for all five |
| 1 retrieval, 3 variants | — | 0.83s |

Raise `INFERENCE_CONCURRENCY` only on a card that fits both models with headroom.

---

## The reply follows the question, not the sources

An English question against an Arabic corpus used to come back in Arabic: a general "match the user's language" rule at the top of the prompt lost to eighty-eight chunks of immediate Arabic context.

The question is now quoted back verbatim as the anchor, and the directive is repeated as the prompt's **last** line, after the retrieved context. No language detection is involved — the question is already written in the language the reply needs.

---

## Session isolation

Uploads belong to the session that made them. Retrieval is scoped to the caller's own documents plus the shared knowledge base — one user's upload is never retrievable by another.

| Scope | Source | Who can retrieve it |
|---|---|---|
| Session documents | UI upload or `POST /upload` | That session only |
| Shared knowledge base | Files in [`app/kb/`](app/kb/) | Every session |

Sessions expire after `SESSION_TTL` (1h default), refreshed on access. Knowledge base keys never expire — the shared corpus is not a user session.

Deleting a document removes **everything it produced**: the registry entry, its vectors, and the tables extracted from it. Leaving the tables behind would keep a deleted file's data queryable, which is wrong for a spreadsheet and worse for a payslip.

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
| `calculate_statistics` | sum, mean, median, std, min, max, count, describe, correlation — with `group_by` and `a/b` rates |
| `query_data` | Filter, sort and page through rows |
| `generate_chart` | One figure, seven types, with reference lines, dual axes and resampling |
| `generate_dashboard` | Up to six related figures in a single call |
| `generate_diagram` | Mermaid flowchart, sequence, state, ER or timeline |

Tools run **concurrently** — a question needing a lookup and two calculations issues all three at once.

Structured data is addressed by `table_name`. Figures the model extracts from prose are passed as `custom_data`, which is what lets it chart numbers that were never tabular.

**Every table is named in the prompt.** With twenty-four one-row payslip tables, describing only the newest five meant the agent could not know the rest existed and burned its whole turn budget probing them with `query_data`. All tables now get a one-line schema; the newest keep full profiles.

**The last turn withholds tools**, so a model that would otherwise keep calling them is forced to synthesise an answer from what it already gathered.

### Failure handling

Transient provider faults — 429 rate limits, 503 overload, timeouts — retry with exponential backoff and jitter before failing over to the other provider.

Funding failures are deliberately excluded. An exhausted account also returns 429, but no retry will clear it, so it fails straight through instead of burning billable calls. The distinction matters more than it looks: Gemini's *recoverable* rate-limit message reads "check your plan and billing details", so matching on the word "billing" misclassifies it as terminal.

---

## Structured output

Final answers conform to a Pydantic schema — `answer`, `visualizations`, `key_insights`, `sources_used` — so the frontend renders them without parsing prose.

The two providers reach it differently, and the differences are instructive:

- **OpenAI** — tool declarations and a `json_schema` response format travel in one request. Sent through `create()` rather than the `parse()` helper, because `parse()` requires every tool to be `strict`, and the free-form `custom_data` argument cannot be expressed in OpenAI's strict subset.
- **Gemini** — rejects a response schema alongside tool declarations, so tools run first and a second pass shapes the answer. That pass uses JSON mode rather than a schema, because the Developer API also rejects `additionalProperties`, which Pydantic emits for `Dict[str, Any]` fields. Replayed function calls must also carry the `thought_signature` Gemini issued, and a signature from one model is rejected by another — so they are tagged with their issuer.

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
USE_FP16_ON_GPU=true                      # halves resident model size
INFERENCE_CONCURRENCY=1                   # see "Inference is serialised on purpose"

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
| `DELETE` | `/sessions/{id}/files/{doc_id}` | Remove one document and its tables |
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

**163 tests, none of which touch the network.** Providers are mocked, charts are verified against real Plotly output, retries against synthetic faults. They cover session isolation, chunk overlap, index persistence and torn-write recovery, chart failure isolation, rate aggregation, Mermaid repair, the retry classifier, request bounds, document deletion, and the `/chat` handler itself.

```bash
docker compose exec backend python tests/e2e.py
```

An end-to-end pass against a running stack: ingestion, analysis, dashboards, cross-language retrieval, isolation, model switching, input bounds and deletion. It makes real LLM calls, so it costs money and is deliberately excluded from discovery.

Several fixes in this repo exist because a real session found them — a chart tool that crashed on every figure built from prose, a delete button that re-uploaded the file it had just deleted, retrieval 2000× slower under concurrency than alone. Unit tests caught none of those.

---

## Layout

```
app/
├── main.py              FastAPI routes, agent loop, out-of-band visual capture
├── frontend.py          Streamlit UI
├── core/
│   ├── config.py        Settings, GPU and dtype detection
│   ├── gpu.py           Serialised inference pool
│   ├── vectorstore.py   FAISS index, session-scoped search, persistence
│   └── session.py       Redis: history, dataframes, document registry
├── services/
│   ├── ingest.py        Parsing, table extraction, chunking
│   ├── retrieval.py     Multi-query, hybrid search, fusion, reranking
│   ├── llm.py           Provider routing, tool calling, structured output, retry
│   ├── tools.py         Tool schemas and execution
│   ├── chart.py         ChartSpec resolution and Plotly generation
│   └── diagram.py       Mermaid validation and label repair
└── utils/
    ├── prompts.py       System prompt
    └── schemas.py       Pydantic request/response models
```

---

## Limitations

This runs well as a single-host application. It is **not** production-hardened, and the gaps are deliberate rather than unknown.

**Would block a real deployment**

- **No authentication.** Session ids are client-supplied. They isolate data but do not authenticate it — anyone who guesses an id inherits that session until it expires. Put this behind auth before exposing it.
- **No rate limiting.** Each request can trigger several LLM calls, so an open endpoint is a billing risk more than a load one.
- **Single-process state.** The vector index lives in the process, so two backend replicas would hold two different indexes. A shared vector database (Qdrant, pgvector, Milvus) is the real fix for horizontal scaling.
- **`llm_service.model` is process-global.** Switching model via `/models/select` affects every user on that replica.

**Known rough edges**

- The image is **21GB**, of which ~2.5GB is avoidable: `bge-m3` downloads twice (sentence-transformers pulls the ST snapshot, then `AutoModel` fetches safetensors separately), and `build-essential` ships at runtime.
- BM25 is rebuilt per request rather than maintained incrementally — fine at this corpus size, the first thing to change as it grows.
- Removing documents rebuilds the whole index; an IDMap-backed index would avoid it.
- Schema enforcement is stronger on OpenAI than on Gemini, for the API reasons above.
- The OpenAI path is implemented and reaches the API, but has not been exercised against a funded account.
- No CI. Tests run locally via the command above.

## License

MIT — see [LICENSE](LICENSE).

# app/main.py

"""
AI Data Agent - Production FastAPI
Complete multi-document RAG with structured JSON outputs
Version 5.0 - Optimized with enhanced debugging and chart support
"""

import asyncio
import contextvars
import hashlib
import re
import logging
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import shutil
import tempfile
import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.utils.schemas import (
    SESSION_ID_PATTERN,
    UploadResponse,
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    FileInfo,
    ModelInfo,
    SessionStats,
    ChatTurn,
    AgentStep,
)

from app.core.config import get_settings
from app.services.tools import (
    get_tool_definitions,
    execute_tool,
    build_table_context,
)
from app.utils.prompts import anchor_language, get_system_prompt
from app.core.dependencies import (
    vector_store,
    session_store,
    llm_service,
    ingestion_service,
    retriever,
)

settings = get_settings()

# Enhanced logging configuration
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Per-request log capture. A streamed answer that says only "turn 3 of 10" tells the reader
# nothing: the interesting narration — which query was expanded, how many candidates the
# reranker kept, what the chart service resolved — is already written to the log by the code
# doing the work. A contextvar sink collects it per request, so concurrent requests never
# see each other's lines, and the agent loop forwards it as it goes.
#
# Work handed to a thread pool (embedding, reranking) does not inherit the context, so those
# lines still go only to the container log.
_LOG_SINK: contextvars.ContextVar = contextvars.ContextVar("log_sink", default=None)


class _SinkHandler(logging.Handler):
    def emit(self, record):
        sink = _LOG_SINK.get()
        if sink is not None:
            try:
                sink.append(f"{record.name.split('.')[-1]}: {record.getMessage()}")
            except Exception:
                pass


logging.getLogger("app").addHandler(_SinkHandler())


def _drain(sink):
    """Take everything logged since the last look, and clear it."""
    lines, sink[:] = list(sink), []
    return lines


# ============================================================
# LIFESPAN & INITIALIZATION
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with comprehensive initialization"""
    logger.info("=" * 80)
    logger.info("🚀 Starting AI Data Agent v5.0...")
    logger.info("=" * 80)

    try:
        await vector_store.initialize()
        logger.info("✅ Vector store initialized")

        await session_store.initialize()
        logger.info("✅ Session store initialized")

        await llm_service.initialize()
        logger.info("✅ LLM service initialized")

        await retriever._initialize_reranker()
        logger.info("✅ Retriever initialized")

        await prune_expired_sessions()

        if settings.AUTO_INGEST_ON_STARTUP:
            await auto_ingest_kb_folder()

        logger.info("=" * 80)
        logger.info("✅ AI Data Agent Ready")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}", exc_info=True)
        raise

    yield

    logger.info("🔄 Shutting down...")
    await session_store.close()
    logger.info("✅ Shutdown complete")


app = FastAPI(
    title="AI Data Agent",
    version="5.0",
    description="Multi-document RAG with structured JSON outputs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_START = datetime.now(timezone.utc)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


async def prune_expired_sessions():
    """
    Drop restored vectors whose session has since expired in Redis.

    The index outlives the process; Redis sessions do not. Without this, vectors
    from sessions that timed out during downtime would linger and stay searchable
    to whoever guessed the id.
    """
    sessions = {d.get("session_id") for d in await vector_store.get_documents()}
    sessions.discard(settings.KB_SESSION_ID)
    sessions.discard(None)

    if not sessions:
        return

    live = {s for s in sessions if await session_store.session_exists(s)}
    expired = sessions - live

    if expired:
        logger.info(f"🧹 {len(expired)} expired session(s) in restored index")
        await vector_store.prune_sessions(
            lambda sid: sid == settings.KB_SESSION_ID or sid in live
        )


async def auto_ingest_kb_folder():
    """Auto-ingest knowledge base folder"""
    kb_path = Path(settings.KB_FOLDER)
    if not kb_path.exists():
        logger.info(f"📁 KB folder not found: {kb_path}")
        return

    files = [
        f
        for f in kb_path.iterdir()
        if f.is_file()
        and f.suffix.lower() in {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt"}
    ]

    if not files:
        logger.info("📁 KB folder is empty")
        return

    logger.info(f"📚 Checking {len(files)} KB files...")

    # The index is restored from disk before this runs, so re-ingesting blindly would duplicate the whole corpus on every restart. The content hash in doc_id makes an unchanged file a no-op and lets an edited one supersede its previous chunks.
    indexed = {d.get("doc_id") for d in await vector_store.get_documents()}

    # Vectors alone are not enough: the document registry is what tells the agent the file exists. If either half is missing the file is re-ingested to restore both.
    registered = {
        d.get("doc_id")
        for d in await session_store.get_session_documents(settings.KB_SESSION_ID)
    }

    for f in files:
        try:
            content = f.read_bytes()
            doc_id = f"kb_{f.stem}_{hashlib.sha256(content).hexdigest()[:8]}"

            if doc_id in indexed and doc_id in registered:
                logger.info(f"  ⏭ {f.name}: unchanged, already indexed")
                continue

            removed = await vector_store.remove_by_source(f.name)
            if removed:
                logger.info(f"  ♻ {f.name}: replacing {removed} stale chunks")

            result = await ingestion_service.ingest_file(
                settings.KB_SESSION_ID, f.name, content, doc_id=doc_id
            )
            logger.info(
                f"  ✓ {f.name}: {result['text_chunks']} chunks, {result['dataframes']} tables"
            )
        except Exception as e:
            logger.error(f"  ✗ {f.name}: {e}")


async def expand_and_retrieve(query: str, session_id: str) -> List[Dict]:
    """Retrieve against LLM-generated query variants, scoped to the session plus shared KB."""
    variants = (
        await llm_service.generate_query_variants(query, settings.NUM_QUERY_VARIANTS)
        if settings.USE_MULTI_QUERY
        else None
    )
    return await retriever.retrieve(
        query=query,
        expanded_queries=variants,
        session_ids={session_id, settings.KB_SESSION_ID},
    )


async def _empty_dict():
    return {}


async def _empty_list():
    return []


def require_supported_model(name):
    """
    Reject an unknown model instead of letting it fail over silently.

    /models/select validated while /chat did not, so a request naming a model that
    does not exist answered anyway — the primary call failed, the fallback succeeded,
    and the reply looked normal while coming from a model the caller never asked for.
    """
    if name and name not in settings.SUPPORTED_MODELS:
        raise HTTPException(
            400,
            f"Unsupported model '{name}'. Available: {settings.SUPPORTED_MODELS}",
        )
    return name


# Tools whose output is rendered rather than reasoned over
VISUAL_TOOLS = {"generate_chart", "generate_dashboard", "generate_diagram"}

# The only block types the model may compose itself. Charts and diagrams come from tools, never from the model's own JSON, and the frontend draws nothing for any other type.
COMPOSABLE_VISUALS = {"table", "text"}


def capture_visuals(raw: str, sink: List[Dict]) -> str:
    """
    Divert renderable payloads into the response and return a compact receipt.

    Plotly figures run to kilobytes. Feeding them back as tool results would burn
    the context window and force the model to copy them verbatim into its answer,
    which it does unreliably. Diagrams are smaller but no more reliable to copy.
    Both travel out-of-band; the model gets titles and summaries only.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return str(raw)[:500]

    if not payload.get("success"):
        return json.dumps({"success": False, "error": payload.get("error")})

    # A diagram is one visual with source rather than a figure
    if payload.get("mermaid"):
        title = payload.get("title") or f"Diagram {len(sink) + 1}"
        sink.append(
            {
                "type": "diagram",
                "mermaid": payload["mermaid"],
                "caption": title,
            }
        )
        logger.info(f"  📐 Captured diagram '{title}' out-of-band")
        return json.dumps(
            {
                "success": True,
                "rendered": [{"title": title, "summary": payload.get("summary")}],
                "note": "This diagram is already attached to the response. Describe what it shows in 'answer'; do not repeat the source.",
            }
        )

    # generate_dashboard returns a list; generate_chart is the single-figure case
    charts = payload.get("charts") or [payload]
    receipts = []

    for chart in charts:
        layout = (chart.get("chart_json") or {}).get("layout") or {}
        title = (
            chart.get("title")
            or (layout.get("title") or {}).get("text")
            or f"Chart {len(sink) + 1}"
        )

        # A dashboard panel and a separately requested chart of the same thing are the
        # same figure twice. Asked for a six-part review, the model called
        # generate_dashboard and then generate_chart for three of its panels, and the
        # reader scrolled past the same bar chart three times.
        # Title included: the same figure under a different heading is being presented as
        # a different thing, and only an exact repeat is worth suppressing.
        digest = hashlib.sha1(
            json.dumps([title, chart.get("chart_json")], sort_keys=True, default=str).encode()
        ).hexdigest()
        if any(v.get("digest") == digest for v in sink):
            logger.info(f"  ⧗ Skipped duplicate figure '{title}'")
            continue

        entry = {
            "type": "chart",
            "chart_data": {
                "chart_json": chart.get("chart_json"),
                "summary": chart.get("summary"),
            },
            "caption": title,
            "digest": digest,
        }

        # Same heading, different bars: the model redrew the figure, and the redraw is the
        # one it goes on to describe. Asked for revenue by region it charted eight bars,
        # noticed the spellings warning, charted four — and shipped both, the wrong
        # figure sitting directly above the right one under an identical title. Keeping
        # the later attempt is what the prose agrees with.
        prior = next(
            (i for i, v in enumerate(sink)
             if v.get("type") == "chart" and v.get("caption") == title),
            None,
        )
        if prior is not None:
            logger.info(f"  ♻ Replaced an earlier '{title}' with the redrawn figure")
            sink[prior] = entry
        else:
            sink.append(entry)
        receipts.append({"title": title, "summary": chart.get("summary")})

    logger.info(f"  🎨 Captured {len(receipts)} chart(s) out-of-band")

    return json.dumps(
        {
            "success": True,
            "rendered": receipts,
            "failures": payload.get("failures", []),
            "note": "These charts are already attached to the response. Describe them in 'answer' and leave 'visualizations' empty.",
        }
    )


def extract_tool_info(tc: Any) -> tuple[str, Dict]:
    """
    Safely extract tool name and arguments from various formats.
    Supports both dict and object representations.
    """
    logger.debug(f"🔍 Extracting tool info from: {type(tc)}")

    # Dict format (Anthropic/Custom)
    if isinstance(tc, dict):
        if "function" in tc:
            fn = tc["function"]
            name = fn.get("name", "unknown")
            args_str = fn.get("arguments", "{}")

            # Parse arguments if string
            if isinstance(args_str, str):
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse tool arguments: {e}")
                    logger.error(f"   Raw arguments: {args_str}")
                    args = {}
            else:
                args = args_str

            logger.debug(f"  📌 Extracted: {name} with args: {args}")
            return name, args

        # Direct format
        name = tc.get("name", "unknown")
        args = tc.get("arguments", {})
        logger.debug(f"  📌 Extracted (direct): {name} with args: {args}")
        return name, args

    # Object format (OpenAI SDK)
    if hasattr(tc, "function"):
        name = tc.function.name
        args_str = tc.function.arguments

        if isinstance(args_str, str):
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
        else:
            args = args_str

        logger.debug(f"  📌 Extracted (object): {name} with args: {args}")
        return name, args

    logger.warning(f"⚠️  Unknown tool call format: {tc}")
    return "unknown_tool", {}


def _decode_json_string(body: str) -> str:
    """Unescape a JSON string body recovered by regex rather than by the parser."""
    try:
        return json.loads(f'"{body}"')
    except json.JSONDecodeError:
        return body.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def extract_structured_response(content: str) -> tuple[str, Dict]:
    """
    Robust JSON extractor with comprehensive fallback handling.
    Returns (answer_text, metadata_dict)
    """
    logger.debug("🔍 Extracting structured response from LLM output")

    def process_data(data):
        """Process parsed data into response format"""
        return (
            data.get("answer", "No answer provided."),
            {
                "visualizations": data.get("visualizations", []),
                "key_insights": data.get("key_insights", []),
                "assumptions": data.get("assumptions", []),
                "sources_from_response": data.get("sources_used", []),
                # Carried through so a drift is visible in the trace rather than only in
                # the prose, where the reader has to notice it themselves.
                "question_language": data.get("question_language", ""),
            },
        )

    # Strategy 1: Direct JSON parse (primary path with Structured Outputs)
    try:
        data = json.loads(content)

        # Handle double-encoded strings
        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, dict) and "answer" in data:
            logger.info("✅ Successfully parsed structured JSON")
            return process_data(data)
    except json.JSONDecodeError:
        logger.debug("  Direct JSON parse failed, trying markdown extraction")

    # Strategy 2: Markdown code block extraction
    json_patterns = [
        r"```json\s*(.*?)```",  # ```json ... ```
        r"```\s*(.*?)```",  # ``` ... ```
    ]

    for pattern in json_patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()

            # Clean common LLM formatting issues
            json_str = (
                json_str.replace("...", "")
                .replace("Ellipsis", "null")
                .replace("None", "null")
                .replace("'", '"')  # Fix single quotes
            )

            # Remove trailing commas before closing brackets
            json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "answer" in data:
                    logger.info(
                        f"✅ Extracted JSON from markdown block ({len(json_str)} chars)"
                    )
                    return process_data(data)
            except json.JSONDecodeError as e:
                logger.debug(f"  Markdown JSON parse failed: {e}")
                continue

    # Strategy 3: Emergency fallback - construct minimal valid response
    logger.warning("⚠️ All JSON parsing failed - constructing emergency response")

    # The fragment is still a JSON string literal, escapes and all. Taking it raw put a
    # literal backslash-n through to the chat and cut the answer at the first quoted
    # word; decoding it as the string it is restores both.
    answer_match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
    if answer_match:
        answer_text = _decode_json_string(answer_match.group(1))
        logger.info(f"  Salvaged answer field: {answer_text[:100]}...")
    else:
        # Use raw content as answer
        answer_text = content[:1000] + ("..." if len(content) > 1000 else "")
        logger.warning("  Using raw content as answer")

    # Same shape as the parsed paths. Every caller reads these with .get today, so an
    # empty dict does no harm now — it just leaves the next one a KeyError to find.
    #
    # `salvaged` is carried so the reader is told. Asked for a hundred rows as a table the
    # model emitted JSON longer than the token limit; it was cut mid-structure, salvage
    # recovered the prose, and the table it referred to was silently gone — leaving an
    # answer that said "here are the first 100 sales figures" above nothing at all.
    return answer_text, {
        "visualizations": [],
        "salvaged": True,
        "key_insights": [],
        "assumptions": [],
        "sources_from_response": [],
    }


# ============================================================
# API ENDPOINTS
# ============================================================


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    session_id: str = Form(..., pattern=SESSION_ID_PATTERN,
                           max_length=settings.MAX_SESSION_ID_CHARS),
):
    """
    Upload document with stream processing to prevent OOM.
    Supports: PDF, DOCX, XLSX, CSV, TXT
    """
    tmp_path = None

    try:
        logger.info("=" * 80)
        logger.info("📤 UPLOAD REQUEST")
        logger.info(f"  Session: {session_id[:8]}...")
        logger.info(f"  File: {file.filename}")
        logger.info(f"  Type: {file.content_type}")

        # Validate size without loading into RAM
        file.file.seek(0, 2)
        size_mb = file.file.tell() / (1024 * 1024)
        file.file.seek(0)

        logger.info(f"  Size: {size_mb:.2f} MB")

        if size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                400,
                f"File too large: {size_mb:.2f}MB (max {settings.MAX_FILE_SIZE_MB}MB)",
            )

        # Stream to temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_{file.filename}"
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logger.info(f"  ✅ Streamed to temp: {tmp_path}")

        # Ingest
        result = await ingestion_service.ingest_file(
            session_id, file.filename, None, file_path=tmp_path
        )

        logger.info(
            f"  ✅ Ingested: {result['text_chunks']} chunks, {result['dataframes']} tables"
        )
        logger.info("=" * 80)

        return UploadResponse(
            # "unchanged" when the same bytes are already here, so the UI can say so
            status=result.get("status", "success"),
            session_id=session_id,
            filename=result["filename"],
            doc_id=result["doc_id"],
            text_chunks=result["text_chunks"],
            dataframes=result["dataframes"],
        )

    except HTTPException:
        raise
    except ValueError as e:
        # A file we cannot read is the caller's problem to fix, not a server fault
        logger.error(f"❌ Upload rejected: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"❌ Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(500, str(e))

    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug(f"  🗑️  Cleaned temp file: {tmp_path}")


async def run_agent(req: ChatRequest):
    """
    The agent loop, as a stream of events ending in one `final` carrying the response.
    
    Written as a generator so /chat and /chat/stream are the same reasoning: one collects
    the events and returns the answer, the other forwards them as they happen. A ten-turn
    question can run for a minute, and a spinner that says nothing for a minute is
    indistinguishable from a hang.
    """
    sid = req.session_id or str(uuid.uuid4())
    require_supported_model(req.model)

    sink = []
    _LOG_SINK.set(sink)

    logger.info("=" * 80)
    logger.info("💬 CHAT REQUEST (PARALLEL)")
    logger.info(f"  Session: {sid[:8]}...")
    logger.info(f"  Query: {req.query[:100]}...")
    logger.info(f"  RAG: {req.use_rag}")
    logger.info(f"  Model: {req.model or 'default'}")
    logger.info("=" * 80)

    try:
        start_time = asyncio.get_event_loop().time()

        # Load context concurrently. The shared knowledge base is retrievable by every session, so it must appear here too — otherwise the prompt reports no files and the agent denies having any documents it can actually search.
        kb = settings.KB_SESSION_ID
        dfs, hist, docs, kb_dfs, kb_docs = await asyncio.gather(
            session_store.get_all_dataframes(sid),
            session_store.get_chat_history(sid),
            session_store.get_session_documents(sid),
            session_store.get_all_dataframes(kb) if sid != kb else _empty_dict(),
            session_store.get_session_documents(kb) if sid != kb else _empty_list(),
        )

        # Session data shadows the knowledge base on a name clash
        dfs = {**kb_dfs, **dfs}
        docs = docs + kb_docs

        yield {"type": "context", "tables": sorted(dfs), "documents": len(docs)}

        logger.info("📊 Context loaded:")
        logger.info(f"  DataFrames: {len(dfs)}")
        logger.info(f"  History: {len(hist)} messages")
        logger.info(f"  Documents: {len(docs)}")

        # File metadata
        file_meta = [
            {
                "name": d["filename"],
                "dataframes": d.get("dataframes", 0),
                "file_type": d.get("file_type", "unknown"),
            }
            for d in docs
        ]

        # Every table is named so the agent never probes to discover one exists
        df_ctx = build_table_context(dfs)

        # Tools
        tools = get_tool_definitions(dfs, req.query)
        logger.info(f"🔧 Tools available: {len(tools)}")

        # Conversation summary
        conv_summary = "\n".join(
            [f"{m['role']}: {m['content'][:200]}" for m in hist[-6:]]
        )

        # Initialize messages
        msgs = hist + [{"role": "user", "content": anchor_language(req.query)}]

        # Agent loop state
        turn = 0
        answering_model = req.model or llm_service.model
        degraded = announced = None
        final_text = ""
        final_meta = {}
        ctx = ""
        srcs = []
        agent_trace = []
        produced_visuals = []

        # Agentic reasoning loop
        while turn < settings.AGENT_MAX_TURNS:
            turn += 1
            logger.info(f"🔄 Turn {turn}/{settings.AGENT_MAX_TURNS}")

            # Withholding tools on the final turn forces a synthesis from what was already gathered. Otherwise the loop can spend its whole budget calling tools and return an apology instead of an answer.
            final_turn = turn == settings.AGENT_MAX_TURNS
            if final_turn:
                logger.info("  ⏹ Final turn: answering without tools")

            yield {
                "type": "turn",
                "turn": turn,
                "of": settings.AGENT_MAX_TURNS,
                "tools_offered": not final_turn,
            }

            # Build system prompt
            sys_prompt = get_system_prompt(
                ctx if req.use_rag else "",
                df_ctx,
                file_meta,
                conv_summary,
                req.query,
            )

            yield {
                "type": "llm_call",
                "model": req.model or llm_service.model,
                "messages": len(msgs),
                "tools": 0 if final_turn else len(tools),
            }
            yield {"type": "log", "lines": _drain(sink)}

            # Call LLM
            logger.info("  🤖 Calling LLM...")
            llm_resp = await llm_service.chat(
                msgs,
                sys_prompt,
                None if final_turn else tools,
                settings.LLM_TEMPERATURE,
                json_mode=True,
                model_override=req.model,
            )

            content = llm_resp.get("content", "")
            tool_calls = llm_resp.get("tool_calls", [])
            answering_model = llm_resp.get("model") or answering_model
            # A silent failover once presented a Gemini answer as gpt-4o's. If the model that answered is not the one asked for, the reader is told why.
            if llm_resp.get("degraded"):
                degraded = llm_resp["degraded"]

            logger.info(f"  📝 Response: {len(content)} chars")
            logger.info(f"  🔧 Tool calls: {len(tool_calls)}")

            yield {"type": "log", "lines": _drain(sink)}

            if content:
                logger.info(f"  💭 Reasoning preview: {content[:500]}...")
                # Reasoning alongside a tool call is the agent thinking aloud; on the last
                # turn the content is the answer being drafted. Gemini in JSON mode often
                # returns neither, which is why the log stream carries the detail.
                yield {
                    "type": "thinking",
                    "text": content[:600],
                    "drafting": not tool_calls,
                }

            # Once per request, not once per turn: `degraded` stays set for the rest of the
            # loop, so a failover on turn one stacked an identical warning on every turn
            # after it.
            if degraded and degraded != announced:
                announced = degraded
                yield {"type": "notice", "text": degraded}

            # Capture reasoning
            if content and (tool_calls or turn < settings.AGENT_MAX_TURNS):
                agent_trace.append(AgentStep(type="reasoning", content=content))

            # HANDLE TOOL CALLS IN PARALLEL
            if tool_calls:
                logger.info(f"  ⚡ Executing {len(tool_calls)} tools in parallel...")

                # Add assistant message with tool calls to history
                msgs.append(
                    {
                        "role": "assistant",
                        "content": content or "",
                        "tool_calls": tool_calls,
                    }
                )

                tasks = []
                call_info = []

                for tc in tool_calls:
                    tool_name, tool_args = extract_tool_info(tc)
                    tool_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")

                    # Store info to map results back
                    call_info.append(
                        {"id": tool_id, "name": tool_name, "args": tool_args}
                    )

                    # Update Trace
                    agent_trace.append(
                        AgentStep(
                            type="tool_call", tool_name=tool_name, tool_args=tool_args
                        )
                    )
                    yield {"type": "tool_call", "name": tool_name, "args": tool_args}

                    # Branch execution: Search vs standard tools
                    if tool_name == "search_knowledge_base":
                        if req.use_rag:
                            tasks.append(
                                expand_and_retrieve(tool_args.get("query", ""), sid)
                            )
                        else:
                            # If RAG is off but called, return info task
                            async def rag_off():
                                return "RAG is disabled."

                            tasks.append(rag_off())
                    else:
                        tasks.append(execute_tool(tool_name, tool_args, dfs))

                # Fire all tasks at once
                raw_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results back into messages
                for i, result in enumerate(raw_results):
                    info = call_info[i]

                    if isinstance(result, Exception):
                        formatted_result = f"Error: {str(result)}"
                        logger.error(f"  ❌ Parallel error in {info['name']}: {result}")
                    elif info["name"] in VISUAL_TOOLS:
                        formatted_result = capture_visuals(result, produced_visuals)
                    elif info["name"] == "search_knowledge_base" and isinstance(
                        result, list
                    ):
                        # Specialized formatting for Search Knowledge Base
                        if result:
                            formatted_result = (
                                f"Found {len(result)} relevant documents."
                            )
                            ctx += "\n\n" + "\n".join(
                                [f"[{d['source']}] {d['content']}" for d in result]
                            )
                            srcs.extend([d["source"] for d in result])
                        else:
                            formatted_result = "No relevant documents found."
                    else:
                        # Standard tool results
                        formatted_result = str(result)

                    # Update trace and message list
                    agent_trace.append(
                        AgentStep(
                            type="tool_result",
                            tool_name=info["name"],
                            tool_result=formatted_result[:500],
                        )
                    )

                    msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": info["id"],
                            "name": info["name"],
                            "content": formatted_result,
                        }
                    )
                    yield {
                        "type": "tool_result",
                        "name": info["name"],
                        "summary": " ".join(formatted_result.split())[:220],
                        "failed": formatted_result.startswith("Error:"),
                    }

                yield {"type": "log", "lines": _drain(sink)}

                logger.info("  ✅ Batch execution complete")
                continue  # Go to next turn to let LLM analyze the parallel results

            # No tool calls - Final Answer handling
            if content and not tool_calls:
                logger.info("  🎯 Final answer detected")
                final_text, final_meta = extract_structured_response(content)
                break

            if not content:
                logger.warning("  ⚠️ Empty response, breaking loop")
                final_text = "I apologize, but I couldn't generate a response."
                break

        logger.info(f"🏁 Agent loop completed: {turn} turns")

        # Visuals the tools produced win over anything the model echoed back, and only# blocks the frontend can draw survive. A model that describes a chart in `visualizations` instead of calling the tool emits a spec, not a figure, and the renderer draws an empty panel for it — silently, which is how it went unnoticed.
        echoed = final_meta.get("visualizations") or []
        composed = [v for v in echoed if v.get("type") in COMPOSABLE_VISUALS]
        if len(composed) != len(echoed):
            logger.warning(
                f"⚠️ Dropped {len(echoed) - len(composed)} unrenderable visual(s) from the model"
            )
        final_meta["visualizations"] = produced_visuals + composed
        if produced_visuals:
            logger.info(f"🎨 Attached {len(produced_visuals)} visual(s) to response")

        # Save to history
        await session_store.add_chat_turn(sid, req.query, final_text or "No response.")

        # Build metadata
        processing_time = asyncio.get_event_loop().time() - start_time
        meta = {
            "processing_time": round(processing_time, 2),
            "turns_taken": turn,
            # The model that produced the answer, which is not the requested one when a provider 429s and the call fails over.
            "model_used": answering_model,
            "degraded": degraded,
            **final_meta,
        }

        # A salvaged answer is a truncated one: whatever it promised beyond the prose —
        # a table, a figure, its insights — did not survive. Saying so is the difference
        # between a short answer and a wrong one.
        if final_meta.get("salvaged"):
            logger.warning("⚠️ Answer salvaged from malformed JSON; structured fields lost")
            meta["degraded"] = (
                (degraded + " ") if degraded else ""
            ) + ("The reply was cut short and had to be recovered from partial output, so "
                 "its insights and sources are missing and anything it promised beyond the "
                 "prose may be too. Charts already drawn are still shown. Ask for fewer rows.")

        # Logged because an answer that drifted into another language is obvious to the
        # reader and invisible in the trace, and the declaration is the only place the
        # model states what it thought the question was written in.
        if meta.get("question_language"):
            logger.info(f"🗣 Answered as: {meta['question_language']}")

        logger.info(f"⏱️ Total time: {processing_time:.2f}s")
        logger.info("=" * 80)

        yield {"type": "log", "lines": _drain(sink)}
        yield {
            "type": "final",
            "response": ChatResponse(
                session_id=sid,
                response=final_text or "No response generated.",
                sources=list(set(srcs)),
                metadata=meta,
                agent_trace=agent_trace,
            ),
        }

    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint with parallel agentic reasoning loop.
    Supports concurrent tool execution, RAG, and structured JSON outputs.
    """
    require_supported_model(req.model)

    async for event in run_agent(req):
        if event["type"] == "final":
            return event["response"]

    raise HTTPException(500, "The agent loop produced no answer")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    The same reasoning, narrated as newline-delimited JSON while it happens.

    Each line is one event: `context`, `turn`, `thinking`, `tool_call`, `tool_result`,
    `notice`, and finally `final` carrying the identical payload /chat returns. Errors
    arrive as an `error` event rather than a torn response, because the status line has
    already been sent by the time anything can fail.
    """
    require_supported_model(req.model)

    async def events():
        try:
            async for event in run_agent(req):
                if event["type"] == "final":
                    event = {"type": "final", "response": event["response"].model_dump()}
                yield json.dumps(event, default=str) + "\n"
        except HTTPException as e:
            yield json.dumps({"type": "error", "detail": e.detail}) + "\n"
        except Exception as e:
            logger.error(f"❌ Streamed chat error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/sessions/{sid}/history", response_model=HistoryResponse)
async def get_history(sid: str):
    """Get chat history with accurate stored timestamps"""
    logger.info(f"📜 History request: {sid[:8]}...")

    # Fetch raw data from Redis (as bytes/strings)
    key = session_store._chat_key(sid)
    raw_history = await session_store.redis_client.lrange(key, 0, -1)

    stats = await session_store.get_session_stats(sid)

    turns = []
    if raw_history:
        for turn_bytes in raw_history:
            # turn is the dict: {"user": "...", "assistant": "...", "timestamp": "..."}
            turn_data = json.loads(turn_bytes.decode("utf-8"))

            turns.append(
                ChatTurn(
                    user=turn_data["user"],
                    assistant=turn_data["assistant"],
                    timestamp=turn_data["timestamp"],  # Using the real stored time
                )
            )

    logger.info(f"  ✅ Retrieved {len(turns)} accurate turns")

    return HistoryResponse(
        session_id=sid,
        history=turns,
        stats=SessionStats(**stats),
    )


@app.delete("/sessions/{sid}/history")
async def delete_history(sid: str):
    """Clear chat history"""
    logger.info(f"🗑️  Deleting history: {sid[:8]}...")
    await session_store.clear_chat_history(sid)
    logger.info("  ✅ History cleared")
    return {"status": "success", "message": "Chat history cleared"}


@app.delete("/sessions/{sid}/files/{doc_id}")
async def delete_file(sid: str, doc_id: str):
    """Delete a specific file from session and vector store"""
    logger.info(f"🗑️ Request to delete file {doc_id} in session {sid}")

    success = await session_store.delete_document(
        sid, doc_id, vector_store=vector_store
    )

    if success:
        return {"status": "success", "message": "File deleted"}
    else:
        raise HTTPException(404, "File not found")


@app.get("/sessions/{sid}/files", response_model=List[FileInfo])
async def list_files(sid: str):
    """List uploaded files"""
    logger.info(f"📂 Files request: {sid[:8]}...")

    docs = await session_store.get_session_documents(sid)

    logger.info(f"  ✅ Found {len(docs)} files")

    return [
        FileInfo(
            filename=d["filename"],
            doc_id=d["doc_id"],
            uploaded_at=d.get("uploaded_at", ""),
            text_chunks=d.get("text_chunks", 0),
            dataframes=d.get("dataframes", 0),
            file_type=d.get("file_type", ""),
        )
        for d in docs
    ]


@app.get("/sessions/{sid}/dataframes")
async def list_dataframes(sid: str):
    """List available dataframes/tables"""
    logger.info(f"📊 DataFrames request: {sid[:8]}...")

    names = await session_store.list_dataframes(sid)
    details = {}

    for name in names:
        df = await session_store.get_dataframe(sid, name)
        if df is not None:
            details[name] = {
                "rows": len(df),
                "columns": df.columns.tolist(),
            }

    logger.info(f"  ✅ Found {len(details)} dataframes")

    return {"session_id": sid, "dataframes": details}


@app.delete("/sessions/{sid}")
async def clear_session(sid: str):
    """Clear entire session across Redis and Vector Store"""
    logger.info(f"🗑️  Clearing session: {sid[:8]}...")

    # Pass vector_store dependency to handle physical deletion
    await session_store.clear_session(sid, vector_store=vector_store)

    return {"status": "success", "message": "Session cleared from memory and disk."}


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available AI models"""
    curr = llm_service.model

    return [
        ModelInfo(
            name=name,
            provider="Google" if name.startswith("gemini") else "OpenAI",
            is_default=(name == curr),
        )
        for name in settings.SUPPORTED_MODELS
    ]


@app.post("/models/select")
async def select_model(model_name: str = Query(...)):
    """Select AI model"""
    require_supported_model(model_name)

    llm_service.model = model_name
    logger.info(f"🤖 Model switched to: {model_name}")

    return {"status": "success", "model": model_name}


@app.get("/status")
async def status():
    """System status"""
    stats = vector_store.get_stats()

    return {
        "status": "ok",
        "total_documents": stats["total_documents"],
        "indexed_vectors": stats["indexed_vectors"],
        "llm_model": llm_service.model,
        "uptime_seconds": (datetime.now(timezone.utc) - _START).total_seconds(),
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/")
async def root():
    """API root"""
    return {
        "name": "AI Data Agent",
        "version": "5.0",
        "description": "Multi-document RAG with structured JSON outputs",
        "docs": "/docs",
    }

# app/main.py

"""
AI Data Agent - Production FastAPI
Complete multi-document RAG with structured JSON outputs
Version 5.0 - Optimized with enhanced debugging and chart support
"""

import asyncio
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
from app.utils.schemas import (
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
from app.utils.prompts import get_system_prompt
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

    # The index is restored from disk before this runs, so re-ingesting blindly would
    # duplicate the whole corpus on every restart. The content hash in doc_id makes an
    # unchanged file a no-op and lets an edited one supersede its previous chunks.
    indexed = {d.get("doc_id") for d in await vector_store.get_documents()}

    # Vectors alone are not enough: the document registry is what tells the agent the
    # file exists. If either half is missing the file is re-ingested to restore both.
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


# Tools whose output is rendered rather than reasoned over
VISUAL_TOOLS = {"generate_chart", "generate_dashboard", "generate_diagram"}


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

        sink.append(
            {
                "type": "chart",
                "chart_data": {
                    "chart_json": chart.get("chart_json"),
                    "summary": chart.get("summary"),
                },
                "caption": title,
            }
        )
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
                "sources_from_response": data.get("sources_used", []),
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

    # Try to extract at least the answer field
    answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', content, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1)
        logger.info(f"  Salvaged answer field: {answer_text[:100]}...")
    else:
        # Use raw content as answer
        answer_text = content[:1000] + ("..." if len(content) > 1000 else "")
        logger.warning("  Using raw content as answer")

    return answer_text, {}


# ============================================================
# API ENDPOINTS
# ============================================================


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), session_id: str = Form(...)):
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
            status="success",
            session_id=session_id,
            filename=result["filename"],
            doc_id=result["doc_id"],
            text_chunks=result["text_chunks"],
            dataframes=result["dataframes"],
        )

    except Exception as e:
        logger.error(f"❌ Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(500, str(e))

    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug(f"  🗑️  Cleaned temp file: {tmp_path}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint with parallel agentic reasoning loop.
    Supports concurrent tool execution, RAG, and structured JSON outputs.
    """
    sid = req.session_id or str(uuid.uuid4())

    logger.info("=" * 80)
    logger.info("💬 CHAT REQUEST (PARALLEL)")
    logger.info(f"  Session: {sid[:8]}...")
    logger.info(f"  Query: {req.query[:100]}...")
    logger.info(f"  RAG: {req.use_rag}")
    logger.info(f"  Model: {req.model or 'default'}")
    logger.info("=" * 80)

    try:
        start_time = asyncio.get_event_loop().time()

        # Load context concurrently. The shared knowledge base is retrievable by every
        # session, so it must appear here too — otherwise the prompt reports no files
        # and the agent denies having any documents it can actually search.
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
        tools = get_tool_definitions(dfs)
        logger.info(f"🔧 Tools available: {len(tools)}")

        # Conversation summary
        conv_summary = "\n".join(
            [f"{m['role']}: {m['content'][:200]}" for m in hist[-6:]]
        )

        # Initialize messages
        msgs = hist + [{"role": "user", "content": req.query}]

        # Agent loop state
        turn = 0
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

            # Withholding tools on the final turn forces a synthesis from what was
            # already gathered. Otherwise the loop can spend its whole budget calling
            # tools and return an apology instead of an answer.
            final_turn = turn == settings.AGENT_MAX_TURNS
            if final_turn:
                logger.info("  ⏹ Final turn: answering without tools")

            # Build system prompt
            sys_prompt = get_system_prompt(
                ctx if req.use_rag else "",
                df_ctx,
                file_meta,
                conv_summary,
            )

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

            logger.info(f"  📝 Response: {len(content)} chars")
            logger.info(f"  🔧 Tool calls: {len(tool_calls)}")

            if content:
                logger.info(f"  💭 Reasoning preview: {content[:500]}...")

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

        # Visuals actually produced by tools win over anything the model echoed back;
        # tables and prose blocks it composed are kept alongside them
        if produced_visuals:
            rendered = {"chart", "diagram"}
            composed = [
                v
                for v in final_meta.get("visualizations", [])
                if v.get("type") not in rendered
            ]
            final_meta["visualizations"] = produced_visuals + composed
            logger.info(f"🎨 Attached {len(produced_visuals)} visual(s) to response")

        # Save to history
        await session_store.add_chat_turn(sid, req.query, final_text or "No response.")

        # Build metadata
        processing_time = asyncio.get_event_loop().time() - start_time
        meta = {
            "processing_time": round(processing_time, 2),
            "turns_taken": turn,
            "model_used": req.model or llm_service.model,
            **final_meta,
        }

        logger.info(f"⏱️ Total time: {processing_time:.2f}s")
        logger.info("=" * 80)

        return ChatResponse(
            session_id=sid,
            response=final_text or "No response generated.",
            sources=list(set(srcs)),
            metadata=meta,
            agent_trace=agent_trace,
        )

    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


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
    if model_name not in settings.SUPPORTED_MODELS:
        raise HTTPException(
            400, f"Invalid model. Valid options: {settings.SUPPORTED_MODELS}"
        )

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

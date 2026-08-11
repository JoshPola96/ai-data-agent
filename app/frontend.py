# app/frontend.py

"""
Streamlit Frontend - Complete API Integration v5.0
All endpoints integrated, enhanced chart rendering, production-ready
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import uuid
import os
import plotly.graph_objects as go
import pandas as pd
import json
from html import escape

# Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="AI Data Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "use_rag" not in st.session_state:
    st.session_state.use_rag = True

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()

if "current_model" not in st.session_state:
    st.session_state.current_model = None


# ============================================================
# API FUNCTIONS
# ============================================================


def upload_file(file):
    """Upload file to the backend"""
    try:
        files = {"file": (file.name, file, file.type)}
        data = {"session_id": st.session_state.session_id}

        resp = requests.post(
            f"{API_URL}/upload",
            files=files,
            data=data,
            timeout=120,
        )

        if resp.status_code == 200:
            return True, resp.json()
        else:
            return False, resp.text

    except requests.ConnectionError:
        return False, "Backend unavailable — it may still be starting (~45s)."
    except Exception as e:
        return False, str(e)


def send_message(query, model=None):
    """Send chat message"""
    try:
        payload = {
            "session_id": st.session_state.session_id,
            "query": query,
            "use_rag": st.session_state.use_rag,
        }

        if model:
            payload["model"] = model

        resp = requests.post(
            f"{API_URL}/chat",
            json=payload,
            timeout=300,
        )

        if resp.status_code == 200:
            return resp.json()
        else:
            return {
                "response": f"Error: {resp.text}",
                "sources": [],
                "metadata": {},
                "agent_trace": [],
            }

    except requests.ConnectionError:
        return {
            "response": (
                "⏳ **Backend unavailable.** It loads ~4GB of models on start, which takes "
                "about 45 seconds. Wait a moment and send the message again — your uploaded "
                "files are safe."
            ),
            "sources": [], "metadata": {}, "agent_trace": [],
        }
    except requests.Timeout:
        return {
            "response": "⏱️ **Request timed out.** Long documents with reranking can exceed the limit; try a narrower question.",
            "sources": [], "metadata": {}, "agent_trace": [],
        }
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "sources": [],
            "metadata": {},
            "agent_trace": [],
        }


def get_session_files():
    """Get list of uploaded files"""
    try:
        resp = requests.get(
            f"{API_URL}/sessions/{st.session_state.session_id}/files",
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
        return []
    except requests.RequestException:
        return []


def get_session_dataframes():
    """Get list of dataframes/tables"""
    try:
        resp = requests.get(
            f"{API_URL}/sessions/{st.session_state.session_id}/dataframes",
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json().get("dataframes", {})
        return {}
    except requests.RequestException:
        return {}


def get_chat_history():
    """Get chat history"""
    try:
        resp = requests.get(
            f"{API_URL}/sessions/{st.session_state.session_id}/history",
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.RequestException:
        return None


def clear_chat_history():
    """Clear chat history"""
    try:
        resp = requests.delete(
            f"{API_URL}/sessions/{st.session_state.session_id}/history",
            timeout=5,
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


def delete_file(doc_id):
    """Call backend to delete file"""
    try:
        resp = requests.delete(
            f"{API_URL}/sessions/{st.session_state.session_id}/files/{doc_id}",
            timeout=10,
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


def clear_session():
    """Clear entire session"""
    try:
        resp = requests.delete(
            f"{API_URL}/sessions/{st.session_state.session_id}",
            timeout=5,
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


def get_models():
    """Get available models"""
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return []
    except requests.RequestException:
        return []


def select_model(model_name):
    """Select AI model"""
    try:
        resp = requests.post(
            f"{API_URL}/models/select",
            params={"model_name": model_name},
            timeout=5,
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


def get_status():
    """Get system status"""
    try:
        resp = requests.get(f"{API_URL}/status", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.RequestException:
        return None


# ============================================================
# UI COMPONENTS
# ============================================================


def render_chart(chart_data, slot=""):
    """
    Render Plotly chart with responsive sizing.
    Supports both old format (chart_json) and new format (full chart_data)
    """
    try:
        # Handle different chart data formats
        if "chart_json" in chart_data:
            chart_json = chart_data["chart_json"]
        else:
            chart_json = chart_data

        # Create figure from JSON
        fig = go.Figure(chart_json)

        # Slot disambiguates identical figures in one dashboard, which would
        # otherwise collide on a content-derived key
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"chart_{slot}_{hash(str(chart_json))}",
        )

    except Exception as e:
        st.error(f"❌ Chart rendering failed: {e}")
        st.code(json.dumps(chart_data, indent=2))


def render_diagram(mermaid, slot=""):
    """
    Render Mermaid client-side; Streamlit has no diagram widget.

    Height is estimated from line count because an iframe cannot size itself to
    its content, and a clipped flowchart is worse than a slightly tall one.
    """
    height = min(1200, 200 + 32 * len(mermaid.splitlines()))

    components.html(
        f"""
        <div class="mermaid" style="font-family:Arial,sans-serif">{escape(mermaid)}</div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
        <script>
          mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
        </script>
        """,
        height=height,
        scrolling=True,
    )

    with st.expander("Diagram source", expanded=False):
        st.code(mermaid, language="mermaid")


def render_visualization(viz, slot=""):
    """Render one visualization: chart, diagram, table or prose."""
    kind = viz.get("type")

    if kind == "chart":
        if viz.get("chart_data"):
            render_chart(viz["chart_data"], slot)

    elif kind == "diagram":
        if viz.get("mermaid"):
            render_diagram(viz["mermaid"], slot)

    elif kind == "table":
        if viz.get("data"):
            # st.dataframe brings sorting, search and CSV download for free
            st.dataframe(pd.DataFrame(viz["data"]), use_container_width=True)

    elif kind == "text":
        if viz.get("content"):
            st.markdown(viz["content"])

    if viz.get("caption") and kind != "text":
        st.caption(viz["caption"])


def render_message(msg):
    """Render a chat message with all its components"""

    with st.chat_message(msg["role"]):
        # Main content
        st.markdown(msg["content"])

        # Key insights
        if msg.get("insights"):
            st.divider()
            st.subheader("💡 Key Insights")
            for insight in msg["insights"]:
                st.markdown(f"- {insight}")

        # Visualizations (new format)
        if msg.get("visualizations"):
            st.divider()
            for i, viz in enumerate(msg["visualizations"]):
                render_visualization(viz, f"hist{i}")
                if i < len(msg["visualizations"]) - 1:
                    st.markdown("")  # Spacing

        # Legacy chart support
        elif msg.get("chart"):
            st.divider()
            render_chart(msg["chart"], "legacy")

        # Sources
        if msg.get("sources"):
            st.divider()
            with st.expander("📚 Sources", expanded=False):
                for src in msg["sources"]:
                    st.caption(f"• {src}")


def render_agent_trace(trace):
    """Render agent reasoning trace for debugging"""

    if not trace:
        return

    with st.expander("🔍 Agent Trace (Debug)", expanded=False):
        for i, step in enumerate(trace, 1):
            if step["type"] == "reasoning":
                st.markdown(f"**{i}. Reasoning**")
                st.text(step.get("content", "")[:500])

            elif step["type"] == "tool_call":
                st.markdown(f"**{i}. Tool Call: {step.get('tool_name')}**")
                st.json(step.get("tool_args", {}))

            elif step["type"] == "tool_result":
                st.markdown(f"**{i}. Tool Result: {step.get('tool_name')}**")
                st.text(step.get("tool_result", "")[:300])

            if i < len(trace):
                st.markdown("---")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📂 Upload Files")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt"],
        accept_multiple_files=True,
        help="Supports: PDF, DOCX, XLSX, CSV, TXT",
    )

    if uploaded_files:
        uploaded_any = False

        for file in uploaded_files:
            # Skip if already uploaded
            if file.name in st.session_state.uploaded_files:
                continue

            with st.spinner(f"Processing {file.name}..."):
                ok, result = upload_file(file)

                if ok:
                    st.success(f"✅ {file.name}")
                    st.session_state.uploaded_files.add(file.name)
                    uploaded_any = True
                else:
                    st.error(f"❌ {file.name}")
                    st.caption(str(result)[:200])

        if uploaded_any:
            st.rerun()

    st.divider()

    # ========================================
    # Settings
    # ========================================

    st.header("⚙️ Settings")

    # RAG Toggle
    new_rag = st.checkbox(
        "Enable RAG",
        value=st.session_state.use_rag,
        help="Use document retrieval for context",
    )

    if new_rag != st.session_state.use_rag:
        st.session_state.use_rag = new_rag
        st.rerun()

    st.divider()

    # ========================================
    # Model Selection
    # ========================================

    st.header("🤖 AI Model")

    models = get_models()

    if models:
        current = next((m["name"] for m in models if m.get("is_default")), None)

        if current:
            st.session_state.current_model = current

        model_options = {
            f"{m['name']} ({m['provider']})"
            + (" ⭐" if m.get("is_default") else ""): m["name"]
            for m in models
        }

        selected_label = st.selectbox(
            "Select Model",
            list(model_options.keys()),
            help="Choose which AI model to use",
        )

        selected_model = model_options[selected_label]

        if st.button("Switch Model", use_container_width=True):
            if select_model(selected_model):
                st.session_state.current_model = selected_model
                st.success(f"✅ Switched to {selected_model}")
                st.rerun()
            else:
                st.error("❌ Failed to switch model")
    else:
        st.warning("⚠️ Models unavailable")

    st.divider()

    # ========================================
    # System Status
    # ========================================

    st.header("📊 System Status")

    status = get_status()

    if status:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Documents", status.get("total_documents", 0))
            st.metric("Vectors", status.get("indexed_vectors", 0))

        with col2:
            uptime_min = int(status.get("uptime_seconds", 0) / 60)
            st.metric("Uptime", f"{uptime_min}m")
            st.metric("Model", status.get("llm_model", "N/A")[:15])
    else:
        st.warning("⚠️ Status unavailable")

    st.divider()

    # ========================================
    # Session Tools
    # ========================================

    st.header("🔧 Session Tools")

    # Files with Delete Option
    with st.expander("📂 View / Manage Files", expanded=False):
        files = get_session_files()
        if files:
            for f in files:
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.write(f"📄 **{f['filename']}**")
                    st.caption(
                        f"{f.get('file_type', '?')} | {f.get('text_chunks', 0)} chunks"
                    )
                with col2:
                    # Unique key for every button is crucial
                    if st.button("🗑️", key=f"del_{f['doc_id']}", help="Delete file"):
                        with st.spinner("Deleting..."):
                            if delete_file(f["doc_id"]):
                                st.success("Deleted")
                                st.session_state.uploaded_files.discard(f["filename"])
                                st.rerun()
                            else:
                                st.error("Failed")

            if st.button("🔄 Refresh List", use_container_width=True):
                st.rerun()
        else:
            st.info("No files uploaded")

    # Dataframes
    if st.button("📊 View Tables", use_container_width=True):
        dfs = get_session_dataframes()
        if dfs:
            st.subheader("Available Tables")
            for name, info in dfs.items():
                st.write(f"📊 **{name}**")
                st.caption(
                    f"Rows: {info.get('rows', 0)} | Columns: {len(info.get('columns', []))}"
                )
        else:
            st.info("No tables available")

    # History
    if st.button("📜 View History", use_container_width=True):
        hist = get_chat_history()
        if hist:
            stats = hist.get("stats", {})
            st.subheader("Chat History")
            st.write(f"**Turns:** {stats.get('chat_turns', 0)}")
            st.write(f"**Documents:** {stats.get('documents', 0)}")
            st.write(f"**Tables:** {stats.get('dataframes', 0)}")
        else:
            st.info("No history available")

    st.divider()

    # ========================================
    # Clear Options
    # ========================================

    st.header("🗑️ Clear Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Chat", use_container_width=True):
            if clear_chat_history():
                st.session_state.messages = []
                st.success("✅ Chat cleared")
                st.rerun()
            else:
                st.error("❌ Failed")

    with col2:
        if st.button("Clear All", use_container_width=True):
            if clear_session():
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.session_state.uploaded_files = set()
                st.success("✅ All cleared")
                st.rerun()
            else:
                st.error("❌ Failed")

    st.divider()

    # ========================================
    # Session Info
    # ========================================

    st.caption(f"**Session:** {st.session_state.session_id[:8]}...")

    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.uploaded_files = set()
        st.rerun()


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================

st.title("🤖 AI Data Agent")
st.caption("Ask questions across your documents and spreadsheets — answers come back with charts")

if status is None:
    st.warning(
        "⏳ Backend not reachable yet. It loads ~4GB of models on start, which takes about "
        "45 seconds. This page will work normally once it is up.",
        icon="⏳",
    )

# Display chat history
for msg in st.session_state.messages:
    render_message(msg)

# Chat input
if prompt := st.chat_input("Ask about your data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            resp = send_message(prompt, model=st.session_state.current_model)

        # Display response
        st.markdown(resp["response"])

        # Get metadata
        meta = resp.get("metadata", {})

        # Display insights
        if meta.get("key_insights"):
            st.divider()
            st.subheader("💡 Key Insights")
            for insight in meta["key_insights"]:
                st.markdown(f"- {insight}")

        # Display visualizations
        vizs = meta.get("visualizations", [])
        if vizs:
            st.divider()
            for i, viz in enumerate(vizs):
                render_visualization(viz, f"live{i}")

        # Legacy chart support
        elif meta.get("chart"):
            st.divider()
            render_chart(meta["chart"], "livelegacy")

        # Display sources
        all_sources = list(
            set(resp.get("sources", []) + meta.get("sources_from_response", []))
        )

        if all_sources:
            st.divider()
            with st.expander("📚 Sources", expanded=False):
                for src in all_sources:
                    st.caption(f"• {src}")

        # What the agent actually did, and what it cost
        if resp.get("agent_trace"):
            render_agent_trace(resp["agent_trace"])

        if meta:
            tools = len([s for s in resp.get("agent_trace", []) if s["type"] == "tool_call"])
            st.caption(
                f"⚡ {meta.get('processing_time', '?')}s · "
                f"{meta.get('turns_taken', '?')} turns · "
                f"{tools} tool calls · "
                f"{meta.get('model_used', 'unknown')}"
            )

    # Save to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": resp["response"],
            "chart": meta.get("chart"),
            "visualizations": vizs,
            "insights": meta.get("key_insights", []),
            "sources": all_sources,
        }
    )


# ============================================================
# EXAMPLE QUERIES
# ============================================================

if not st.session_state.messages:
    st.info("👋 Upload documents to get started!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📄 Document Q&A**")
        st.markdown("""
        - "Summarise all documents"
        - "What does it say about X?"
        - "Compare documents A and B"
        - Ask in any language — answers come back in the same one
        """)

    with col2:
        st.markdown("**📊 Charts & Dashboards**")
        st.markdown("""
        - "Build a dashboard: revenue by region, trend by quarter, units vs revenue"
        - "Show the margin distribution per region"
        - "Which columns correlate?"
        - "Top 10 products by revenue"
        """)

    with col3:
        st.markdown("**🔢 Analysis**")
        st.markdown("""
        - "What's the average revenue by region?"
        - "Describe the dataset"
        - "Filter to Q4 and sort by margin"
        - "Chart the figures in this PDF"
        """)

# Footer
st.divider()
st.caption(
    f"🤖 AI Data Agent v5.0 | Session: {st.session_state.session_id[:12]}... | Model: {st.session_state.current_model or 'Default'}"
)

"""FullRag Streamlit Debug UI — entry point."""

import sys
from pathlib import Path

# Ensure project root on sys.path before any local imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from streamlit_app.utils.cleanup import clean_user_data
from streamlit_app.utils.log_capture import get_log_handler
from streamlit_app.utils.status import (
    check_groq,
    check_db,
    check_gemini,
    check_gpu,
    get_embedding_model_name,
)

# ─── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="FullRag Debug UI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global dark-mode CSS ───────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }

    /* ── Headings ── */
    h1 { color: #58a6ff !important; font-weight: 700; }
    h2 { color: #79c0ff !important; font-weight: 600; }
    h3 { color: #cdd9e5 !important; font-weight: 600; }

    /* ── Cards / containers ── */
    div[data-testid="stExpander"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043, #3fb950);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(46,160,67,0.3);
    }

    /* ── Status badges ── */
    .badge-ok   { background:#1f6feb22; color:#58a6ff; border:1px solid #1f6feb; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
    .badge-warn { background:#bb800022; color:#e3b341; border:1px solid #bb8000; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
    .badge-err  { background:#da363022; color:#f85149; border:1px solid #da3630; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
    .badge-info { background:#8b949e22; color:#8b949e; border:1px solid #444c56; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }

    /* ── Code blocks ── */
    code, pre, .stCode {
        font-family: 'JetBrains Mono', monospace;
        background: #161b22 !important;
        border: 1px solid #30363d;
        border-radius: 6px;
    }

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 6px;
    }

    /* ── Dataframe ── */
    .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }

    /* ── Divider ── */
    hr { border-color: #30363d; }

    /* ── Chat messages ── */
    .stChatMessage { background: #161b22; border: 1px solid #30363d; border-radius: 10px; }

    /* ── Progress bar ── */
    .stProgress > div > div > div > div { background: linear-gradient(90deg, #1f6feb, #58a6ff); }

    /* ── Alerts ── */
    .stAlert { border-radius: 8px; border-left-width: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── One-time startup tasks ──────────────────────────────────────────────────
if "app_initialized" not in st.session_state:
    clean_user_data()
    get_log_handler()  # attach in-memory log handler
    st.session_state["app_initialized"] = True

# ─── Sidebar header ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 16px 0 8px;">
            <span style="font-size:2.4rem;">🧠</span>
            <h2 style="margin:4px 0 0; color:#58a6ff !important; font-size:1.2rem;">FullRag Debug UI</h2>
            <span style="color:#8b949e; font-size:0.78rem;">Developer Debugging Console</span>
        </div>
        <hr style="margin: 8px 0 16px; border-color:#30363d;">
        """,
        unsafe_allow_html=True,
    )

    # ── Live status checks (cached 30 s) ────────────────────────────────────
    @st.cache_data(ttl=30, show_spinner=False)
    def _get_status():
        db_ok, db_msg = check_db()
        gem_ok, gem_msg = check_gemini()
        groq_ok, groq_msg = check_groq()
        gpu_ok, gpu_msg = check_gpu()
        emb_model = get_embedding_model_name()
        return db_ok, db_msg, gem_ok, gem_msg, groq_ok, groq_msg, gpu_ok, gpu_msg, emb_model

    db_ok, db_msg, gem_ok, gem_msg, groq_ok, groq_msg, gpu_ok, gpu_msg, emb_model = _get_status()

    def _badge(ok: bool, label: str, detail: str) -> str:
        icon = "🟢" if ok else "🔴"
        cls = "badge-ok" if ok else "badge-err"
        return (
            f"<div style='margin-bottom:8px;'>"
            f"  {icon} <strong>{label}</strong><br>"
            f"  <span class='{cls}'>{detail[:60]}</span>"
            f"</div>"
        )

    gpu_icon = "🟢" if gpu_ok else "🟡"
    st.markdown(
        _badge(db_ok, "Database", db_msg)
        + _badge(gem_ok, "Gemini", gem_msg)
        + _badge(groq_ok, "Groq", groq_msg)
        + f"<div style='margin-bottom:8px;'>{gpu_icon} <strong>GPU</strong><br>"
        + f"  <span class='badge-{'ok' if gpu_ok else 'warn'}'>{gpu_msg[:60]}</span></div>"
        + f"<div style='margin-bottom:8px;'>🔵 <strong>Embed Model</strong><br>"
        + f"  <span class='badge-info'>{emb_model}</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='margin:12px 0; border-color:#30363d;'>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─── Landing page content ────────────────────────────────────────────────────
st.title("🧠 FullRag Debug UI")
st.markdown(
    """
    Welcome to the **FullRag Developer Debugging Console**.  
    Use the **sidebar** to navigate between modules.

    Each page gives you **live, interactive access** to a specific part of the pipeline:
    """,
    unsafe_allow_html=False,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**📄 Data Pipeline**")
    st.markdown(
        "- 🏠 Overview & Architecture\n"
        "- 📄 Document Ingestion\n"
        "- ✂️ Chunker Playground\n"
        "- ✨ LLM Enrichment\n"
        "- 🧲 Embeddings\n"
        "- 🗄️ Database Browser"
    )
with col2:
    st.markdown("**🤖 RAG & Retrieval**")
    st.markdown(
        "- 🔍 Retrieval (Vector + Keyword)\n"
        "- 🤖 RAG Pipeline Chatbot\n"
        "- 🚀 Full Pipeline Runner"
    )
with col3:
    st.markdown("**🧪 Testing & Evaluation**")
    st.markdown(
        "- 📊 Retrieval Evaluation\n"
        "- 🧑‍⚖️ Generation Judge Eval\n"
        "- 🧪 Test Runner (pytest)\n"
        "- 📡 Observability / Logs\n"
        "- ⚙️ Settings & Config"
    )

st.info("💡 **Tip:** All service dependencies (DB, embeddings) are called directly in-process. Make sure PostgreSQL is running before using retrieval or database pages.")

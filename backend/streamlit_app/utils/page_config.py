"""Shared page setup — call apply_page_config() at the top of every page."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st


DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }

section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

h1 { color: #58a6ff !important; font-weight: 700; }
h2 { color: #79c0ff !important; font-weight: 600; }
h3 { color: #cdd9e5 !important; font-weight: 600; }

div[data-testid="stExpander"] { background: #161b22; border: 1px solid #30363d; border-radius: 8px; }
div[data-testid="stMetric"] { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; }

.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: #ffffff; border: none; border-radius: 6px; font-weight: 600; transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950);
    transform: translateY(-1px); box-shadow: 0 4px 12px rgba(46,160,67,0.3);
}

.badge-ok   { background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb;padding:2px 10px;border-radius:20px;font-size:0.78rem;font-weight:600; }
.badge-warn { background:#bb800022;color:#e3b341;border:1px solid #bb8000;padding:2px 10px;border-radius:20px;font-size:0.78rem;font-weight:600; }
.badge-err  { background:#da363022;color:#f85149;border:1px solid #da3630;padding:2px 10px;border-radius:20px;font-size:0.78rem;font-weight:600; }
.badge-info { background:#8b949e22;color:#8b949e;border:1px solid #444c56;padding:2px 10px;border-radius:20px;font-size:0.78rem;font-weight:600; }

code, pre { font-family: 'JetBrains Mono', monospace; background: #161b22 !important; border: 1px solid #30363d; border-radius: 6px; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea { background: #0d1117 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; border-radius: 6px; }
.stDataFrame { border: 1px solid #30363d; border-radius: 8px; }
hr { border-color: #30363d; }
.stChatMessage { background: #161b22; border: 1px solid #30363d; border-radius: 10px; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #1f6feb, #58a6ff); }
.stAlert { border-radius: 8px; border-left-width: 4px; }
</style>
"""


def apply_page_config() -> None:
    """Inject dark-mode CSS and render the sidebar status bar on every page."""
    # Dark mode
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    # Startup tasks
    if "app_initialized" not in st.session_state:
        from streamlit_app.utils.cleanup import clean_user_data
        from streamlit_app.utils.log_capture import get_log_handler
        clean_user_data()
        get_log_handler()
        st.session_state["app_initialized"] = True

    # Sidebar status bar
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:12px 0 4px;">
                <span style="font-size:2rem;">🧠</span>
                <div style="color:#58a6ff;font-weight:700;font-size:1rem;">FullRag Debug UI</div>
                <div style="color:#8b949e;font-size:0.72rem;">Developer Console</div>
            </div>
            <hr style="margin:8px 0 12px;border-color:#30363d;">
            """,
            unsafe_allow_html=True,
        )

        @st.cache_data(ttl=30, show_spinner=False)
        def _status():
            from streamlit_app.utils.status import (
                check_db, check_gemini, check_groq, check_gpu, get_embedding_model_name
            )
            return (
                check_db(),
                check_gemini(),
                check_groq(),
                check_gpu(),
                get_embedding_model_name(),
            )

        (db_ok, db_msg), (gem_ok, gem_msg), (groq_ok, groq_msg), (gpu_ok, gpu_msg), emb = _status()

        def _row(icon, label, cls, detail):
            return (
                f"<div style='margin-bottom:7px;font-size:0.8rem;'>{icon} "
                f"<strong>{label}</strong><br>"
                f"<span class='{cls}'>{detail[:55]}</span></div>"
            )

        st.markdown(
            _row("🟢" if db_ok else "🔴", "Database", "badge-ok" if db_ok else "badge-err", db_msg)
            + _row("🟢" if gem_ok else "🔴", "Gemini", "badge-ok" if gem_ok else "badge-err", gem_msg)
            + _row("🟢" if groq_ok else "🔴", "Groq", "badge-ok" if groq_ok else "badge-err", groq_msg)
            + _row("🟢" if gpu_ok else "🟡", "GPU", "badge-ok" if gpu_ok else "badge-warn", gpu_msg)
            + _row("🔵", "Embed Model", "badge-info", emb),
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='margin:10px 0;border-color:#30363d;'>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True, key="sidebar_refresh_btn"):
            st.cache_data.clear()
            st.rerun()

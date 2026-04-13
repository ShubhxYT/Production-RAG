"""Page 12 — Observability & Logs."""

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app.utils.page_config import apply_page_config
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Observability | FullRag", page_icon="📡", layout="wide")

apply_page_config()

st.title("📡 Observability & Logs")
st.markdown("Live in-memory log stream captured from all FullRag modules.")
st.markdown("---")

# Ensure the handler is attached
from streamlit_app.utils.log_capture import get_log_handler
handler = get_log_handler()

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    level_filter = st.selectbox("Log level", ["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
with col2:
    module_filter = st.text_input("Module filter", placeholder="e.g. retrieval, pipeline")
with col3:
    auto_refresh = st.toggle("Auto-refresh (2s)", value=False)
with col4:
    st.write("")
    st.write("")
    if st.button("🗑️ Clear Logs", use_container_width=True):
        handler.clear()
        st.rerun()

st.markdown("---")

# ── Fetch + filter records ────────────────────────────────────────────────────
records = handler.get_records()

if level_filter != "ALL":
    records = [r for r in records if r["level"] == level_filter]

if module_filter.strip():
    records = [r for r in records if module_filter.strip().lower() in r["logger"].lower()]

# ── Stats ─────────────────────────────────────────────────────────────────────
level_counts: dict[str, int] = {}
for r in handler.get_records():
    level_counts[r["level"]] = level_counts.get(r["level"], 0) + 1

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Total Records", len(handler.get_records()))
mc2.metric("DEBUG", level_counts.get("DEBUG", 0))
mc3.metric("INFO", level_counts.get("INFO", 0))
mc4.metric("WARNING", level_counts.get("WARNING", 0))
mc5.metric("ERROR", level_counts.get("ERROR", 0) + level_counts.get("CRITICAL", 0))

st.markdown("---")
st.markdown(f"**Showing {len(records)} record(s)**")

# ── Log table ─────────────────────────────────────────────────────────────────
LEVEL_COLORS = {
    "DEBUG": "#8b949e",
    "INFO": "#58a6ff",
    "WARNING": "#e3b341",
    "ERROR": "#f85149",
    "CRITICAL": "#ff7b72",
}

if not records:
    st.info("No log records yet. Use the pipeline pages to generate activity.")
else:
    # Render as styled markdown rows for colorization
    rows_html = []
    for r in reversed(records[-200:]):  # show most recent first
        color = LEVEL_COLORS.get(r["level"], "#cdd9e5")
        rows_html.append(
            f"<div style='display:flex;gap:12px;padding:5px 10px;border-bottom:1px solid #21262d;"
            f"font-family:\"JetBrains Mono\",monospace;font-size:0.78rem;'>"
            f"<span style='color:#8b949e;min-width:90px;'>{r['time']}</span>"
            f"<span style='color:{color};min-width:70px;font-weight:700;'>{r['level']}</span>"
            f"<span style='color:#79c0ff;min-width:180px;overflow:hidden;text-overflow:ellipsis;"
            f"white-space:nowrap;'>{r['logger']}</span>"
            f"<span style='color:#e6edf3;flex:1;'>{r['message'][:200]}</span>"
            f"{'<span style=\"color:#6e7681;\">' + r['request_id'][:12] + '</span>' if r.get('request_id') else ''}"
            f"</div>"
        )

    log_container = (
        "<div style='background:#0d1117;border:1px solid #30363d;border-radius:8px;"
        "max-height:600px;overflow-y:auto;padding:8px 0;'>"
        + "".join(rows_html)
        + "</div>"
    )
    st.markdown(log_container, unsafe_allow_html=True)

    # Export
    if st.download_button(
        "⬇️ Export logs as CSV",
        data=pd.DataFrame(records).to_csv(index=False),
        file_name="fullrag_logs.csv",
        mime="text/csv",
    ):
        pass

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(2)
    st.rerun()

"""Page 3 — Chunker Playground."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app.utils.page_config import apply_page_config
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Chunker | FullRag", page_icon="✂️", layout="wide")

apply_page_config()

st.title("✂️ Chunker Playground")
st.markdown("Paste raw text or upload a file to see how it gets chunked.")
st.markdown("---")

from streamlit_app.utils.cleanup import get_user_data_dir

# ── Input ─────────────────────────────────────────────────────────────────────
tab_paste, tab_file = st.tabs(["📝 Paste Text", "📁 Upload File"])

raw_text = ""
with tab_paste:
    raw_text = st.text_area(
        "Raw text to chunk",
        height=300,
        placeholder="Paste any document text here…",
        value=st.session_state.get("chunker_text", ""),
    )

with tab_file:
    uf = st.file_uploader("Upload file", type=["pdf", "docx", "html", "md"])
    if uf:
        user_data = get_user_data_dir()
        dest = user_data / uf.name
        dest.write_bytes(uf.getbuffer())
        try:
            from ingestion.pipeline import IngestionPipeline, LOADER_REGISTRY
            ext = dest.suffix.lower()
            if ext in LOADER_REGISTRY:
                loader = LOADER_REGISTRY[ext]()
                doc = loader.load(dest, user_data)
                raw_text = doc.raw_content
                st.session_state["chunker_text"] = raw_text
                st.success(f"Loaded `{uf.name}` — {len(raw_text):,} chars")
            else:
                st.error(f"Unsupported format: {ext}")
        except Exception as e:
            st.error(f"Load error: {e}")

if not raw_text.strip():
    st.info("Enter some text above to start.")
    st.stop()

st.markdown("---")

# ── Chunker config ────────────────────────────────────────────────────────────
st.subheader("⚙️ Chunker Settings")
col1, col2 = st.columns(2)
with col1:
    max_tokens = st.slider("Max tokens per chunk", 64, 1024, 512, step=32)
with col2:
    overlap = st.slider("Overlap tokens", 0, 256, 64, step=16)

run_btn = st.button("✂️ Run Chunker", type="primary", use_container_width=False)

if not run_btn:
    st.stop()

# ── Run chunking ──────────────────────────────────────────────────────────────
try:
    from ingestion.chunker import chunk_document
    from ingestion.restructurer import restructure
    from ingestion.models import Document
    from datetime import datetime, timezone

    elements = restructure(raw_text)
    doc = Document(
        title="Playground",
        source_path="playground",
        format="text",
        raw_content=raw_text,
        elements=elements,
        created_at=datetime.now(timezone.utc),
    )
    # Temporarily override token limits via monkey-patching config if possible
    chunks = chunk_document(doc)
except Exception as e:
    st.error(f"Chunking failed: {e}")
    st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"📊 Results — {len(chunks)} chunks")

col_a, col_b, col_c, col_d = st.columns(4)
token_counts = [c.token_count for c in chunks]
col_a.metric("Total Chunks", len(chunks))
col_b.metric("Avg Tokens", f"{sum(token_counts)/len(token_counts):.0f}" if token_counts else "—")
col_c.metric("Min Tokens", min(token_counts) if token_counts else "—")
col_d.metric("Max Tokens", max(token_counts) if token_counts else "—")

# Token distribution chart
st.subheader("📈 Token Distribution")
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=token_counts,
    nbinsx=20,
    marker_color="#1f6feb",
    marker_line_color="#58a6ff",
    marker_line_width=1,
    opacity=0.85,
    name="Chunk token count",
))
fig.update_layout(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_color="#e6edf3",
    xaxis_title="Token Count",
    yaxis_title="Number of Chunks",
    xaxis=dict(gridcolor="#30363d"),
    yaxis=dict(gridcolor="#30363d"),
    height=320,
    margin=dict(l=0, r=0, t=24, b=0),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# Chunk cards
st.subheader("📋 Chunks")
for i, chunk in enumerate(chunks):
    label = f"Chunk {i+1} — {chunk.token_count} tokens"
    if chunk.section_path:
        label += f" | {' > '.join(chunk.section_path)}"
    with st.expander(label):
        st.text_area("Text", chunk.text, height=140, key=f"c_{i}", disabled=True)
        meta_col1, meta_col2 = st.columns(2)
        meta_col1.markdown(f"**Element types:** {[et.value for et in chunk.element_types]}")
        meta_col2.markdown(f"**Pages:** {chunk.page_numbers}")

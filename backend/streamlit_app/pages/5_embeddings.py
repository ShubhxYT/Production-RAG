"""Page 5 — Embeddings."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import time
from streamlit_app.utils.page_config import apply_page_config
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Embeddings | FullRag", page_icon="🧲", layout="wide")

apply_page_config()

st.title("🧲 Embeddings")
st.markdown("Generate embeddings for single or multiple texts and inspect the vectors.")
st.markdown("---")

# ── Config badges ─────────────────────────────────────────────────────────────
try:
    import torch
    from embeddings.models import EmbeddingConfig
    cfg = EmbeddingConfig()
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    st.markdown(
        f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;'>"
        f"<span style='background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb;padding:4px 14px;border-radius:20px;font-size:0.82rem;font-weight:600;'>🧠 Model: {cfg.model_name}</span>"
        f"<span style='background:#{'238636' if device=='CUDA' else '6e768122'}22;color:#{'3fb950' if device=='CUDA' else '8b949e'};border:1px solid #{'2ea043' if device=='CUDA' else '444c56'};padding:4px 14px;border-radius:20px;font-size:0.82rem;font-weight:600;'>⚡ Device: {device} — {gpu_name}</span>"
        f"<span style='background:#8b949e22;color:#cdd9e5;border:1px solid #444c56;padding:4px 14px;border-radius:20px;font-size:0.82rem;font-weight:600;'>📐 Dimensions: {cfg.dimensions}</span>"
        f"<span style='background:#8b949e22;color:#cdd9e5;border:1px solid #444c56;padding:4px 14px;border-radius:20px;font-size:0.82rem;font-weight:600;'>📦 Batch: {cfg.batch_size}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
except Exception as e:
    st.warning(f"Could not read embedding config: {e}")

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["📝 Single Text", "📋 Batch"])

single_text = ""
batch_texts: list[str] = []

with tab_single:
    single_text = st.text_area("Text to embed", height=160, placeholder="Enter any text…")

with tab_batch:
    batch_raw = st.text_area(
        "Texts to embed (one per line)",
        height=200,
        placeholder="Line 1\nLine 2\nLine 3…",
    )
    if batch_raw.strip():
        batch_texts = [l.strip() for l in batch_raw.splitlines() if l.strip()]
        st.info(f"{len(batch_texts)} texts detected.")

run_btn = st.button("🧲 Generate Embeddings", type="primary")
if not run_btn:
    st.stop()

texts_to_embed = batch_texts if batch_texts else ([single_text] if single_text.strip() else [])
if not texts_to_embed:
    st.warning("Please enter some text first.")
    st.stop()

# ── Run embedding ─────────────────────────────────────────────────────────────
with st.spinner("Loading model and generating embeddings…"):
    try:
        from embeddings.service import EmbeddingService
        from embeddings.models import EmbeddingConfig
        svc = EmbeddingService(config=EmbeddingConfig())
        t0 = time.perf_counter()
        result = svc.embed(texts_to_embed)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Embedding Results")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Texts Embedded", len(result.vectors))
m2.metric("Dimensions", result.dimensions)
m3.metric("Model", result.model.split("/")[-1])
m4.metric("Elapsed", f"{elapsed_ms:.1f} ms")

# ── Vector visualization ──────────────────────────────────────────────────────
st.subheader("🔢 Vector Preview (first 48 dimensions)")
for i, (text, vec) in enumerate(zip(texts_to_embed, result.vectors)):
    preview_dims = vec[:48]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(preview_dims))),
        y=preview_dims,
        marker_color=[
            f"rgba(31,111,235,{min(1.0, abs(v)*5+0.2)})" for v in preview_dims
        ],
        name=f"Text {i+1}",
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#e6edf3",
        xaxis_title="Dimension index",
        yaxis_title="Value",
        xaxis=dict(gridcolor="#30363d"),
        yaxis=dict(gridcolor="#30363d"),
        height=240,
        margin=dict(l=0, r=0, t=28, b=0),
        title=dict(text=f"Text {i+1}: {text[:60]}…" if len(text) > 60 else text, font=dict(size=13)),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"Raw vector — Text {i+1} (first 16 values)"):
        st.code(", ".join(f"{v:.6f}" for v in vec[:16]) + ", …", language="text")

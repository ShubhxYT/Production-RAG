"""Page 4 — LLM Enrichment."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app.utils.page_config import apply_page_config
import streamlit as st

st.set_page_config(page_title="Enrichment | FullRag", page_icon="✨", layout="wide")

apply_page_config()

st.title("✨ LLM Enrichment")
st.markdown(
    "Enrich a chunk with Gemini-generated **summary**, **keywords**, and **hypothetical questions**."
)
st.markdown("---")

# ── Model badge ───────────────────────────────────────────────────────────────
try:
    from config.settings import get_generation_model
    model = get_generation_model()
    st.markdown(
        f"<span style='background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb;"
        f"padding:4px 14px;border-radius:20px;font-size:0.82rem;font-weight:600;'>"
        f"🤖 Judge Model: Gemini — <code>{model}</code> (read-only)</span>",
        unsafe_allow_html=True,
    )
except Exception:
    st.warning("Gemini API key not configured.")

st.markdown("")

# ── Input source ──────────────────────────────────────────────────────────────
tab_manual, tab_staged = st.tabs(["📝 Enter Chunk Text", "📂 Pick from Staged Chunks"])

chunk_text = ""

with tab_manual:
    chunk_text_input = st.text_area(
        "Chunk text to enrich",
        height=250,
        placeholder="Paste a chunk of text here…",
        key="enrich_input",
    )
    if chunk_text_input.strip():
        chunk_text = chunk_text_input

with tab_staged:
    staging_dir = _ROOT / "staging"
    staged_files = sorted(staging_dir.glob("*.json")) if staging_dir.exists() else []
    if not staged_files:
        st.info("No staged JSON files found in `staging/`.")
    else:
        sel_file = st.selectbox("Select staged document", [f.name for f in staged_files])
        if sel_file:
            try:
                from ingestion.staging import load_staged_document
                doc = load_staged_document(staging_dir / sel_file)
                chunk_labels = [
                    f"Chunk {i+1} — {c.token_count} tokens — {c.text[:60]}…"
                    for i, c in enumerate(doc.chunks)
                ]
                sel_chunk_idx = st.selectbox("Select chunk", range(len(chunk_labels)), format_func=lambda i: chunk_labels[i])
                chunk_text = doc.chunks[sel_chunk_idx].text
                st.text_area("Chunk preview", chunk_text, height=140, disabled=True)
            except Exception as e:
                st.error(f"Failed to load staged file: {e}")

if not chunk_text.strip():
    st.info("Enter or select a chunk text above to enrich.")
    st.stop()

st.markdown("---")
run_btn = st.button("✨ Enrich with Gemini", type="primary")
if not run_btn:
    st.stop()

# ── Run enrichment ────────────────────────────────────────────────────────────
with st.spinner("Calling Gemini for enrichment…"):
    try:
        from generation.llm_service import GeminiProvider
        from generation.prompts import ENRICHMENT_SYSTEM_PROMPT
        provider = GeminiProvider()
        result = provider.enrich_chunk(chunk_text, ENRICHMENT_SYSTEM_PROMPT)
    except Exception as e:
        st.error(f"Enrichment failed: {e}")
        st.stop()

# ── Display results ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Enrichment Output")

col_sum, col_meta = st.columns([2, 1])

with col_sum:
    st.markdown("#### 📝 Summary")
    st.success(result.summary)

    st.markdown("#### ❓ Hypothetical Questions")
    for i, q in enumerate(result.hypothetical_questions, 1):
        st.markdown(
            f"<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;"
            f"padding:10px 14px;margin-bottom:8px;color:#cdd9e5;'>"
            f"<strong>Q{i}:</strong> {q}</div>",
            unsafe_allow_html=True,
        )

with col_meta:
    st.markdown("#### 🏷️ Keywords")
    for kw in result.keywords:
        st.markdown(
            f"<span style='display:inline-block;background:#1f6feb22;color:#58a6ff;"
            f"border:1px solid #1f6feb;padding:3px 12px;border-radius:20px;"
            f"font-size:0.82rem;margin:3px 2px;'>{kw}</span>",
            unsafe_allow_html=True,
        )

"""Page 7 — Retrieval."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import time
from streamlit_app.utils.page_config import apply_page_config
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Retrieval | FullRag", page_icon="🔍", layout="wide")

apply_page_config()

st.title("🔍 Retrieval")
st.markdown("Query the vector database with semantic search, keyword search, or both.")
st.markdown("---")

# ── Query input ───────────────────────────────────────────────────────────────
query = st.text_input(
    "🔎 Query",
    placeholder="What is the main topic of the document?",
    key="retrieval_query",
)

col1, col2, col3 = st.columns(3)
with col1:
    top_k = st.slider("Top K results", 1, 20, 5)
with col2:
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.0, step=0.05,
                          help="Set to 0 to disable filtering")
with col3:
    search_mode = st.selectbox("Search mode", ["Vector (semantic)", "Keyword (FTS)", "Both"])

threshold_val = threshold if threshold > 0 else None
run_btn = st.button("🔍 Search", type="primary", use_container_width=False)

if not run_btn or not query.strip():
    if run_btn and not query.strip():
        st.warning("Please enter a query.")
    st.stop()

# ── Run retrieval ─────────────────────────────────────────────────────────────
with st.spinner("Retrieving…"):
    try:
        from retrieval.service import RetrievalService
        svc = RetrievalService()
        t0 = time.perf_counter()

        vector_results = []
        keyword_results = []

        if search_mode in ("Vector (semantic)", "Both"):
            resp = svc.retrieve_sync(query, top_k=top_k, threshold=threshold_val)
            vector_results = resp.results
            vec_latency = resp.latency_ms

        if search_mode in ("Keyword (FTS)", "Both"):
            import asyncio
            kw_results = asyncio.run(svc.retrieve_by_keyword(query, top_k=top_k))
            keyword_results = kw_results

        total_ms = (time.perf_counter() - t0) * 1000

    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        st.info("Make sure the database is running and has been seeded.")
        st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("---")

def _score_color(score: float) -> str:
    if score >= 0.8:
        return "#3fb950"
    elif score >= 0.6:
        return "#e3b341"
    else:
        return "#f85149"

def _render_results(results, mode_label: str):
    if not results:
        st.info(f"No {mode_label} results found.")
        return

    st.markdown(f"#### {mode_label} — {len(results)} result(s)")

    # Score bar chart
    fig = go.Figure(go.Bar(
        x=[f"#{i+1}" for i in range(len(results))],
        y=[r.similarity_score for r in results],
        marker_color=[_score_color(r.similarity_score) for r in results],
        text=[f"{r.similarity_score:.3f}" for r in results],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3", height=240,
        xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d", range=[0, 1.1]),
        margin=dict(l=0, r=0, t=8, b=0), showlegend=False,
        yaxis_title="Similarity Score",
    )
    st.plotly_chart(fig, use_container_width=True)

    for i, r in enumerate(results):
        score_clr = _score_color(r.similarity_score)
        with st.expander(
            f"#{i+1}  |  Score: {r.similarity_score:.4f}  |  "
            f"{r.document_title or 'Unknown doc'}  |  {r.token_count} tokens"
        ):
            # Meta row
            meta = []
            if r.source_path:
                meta.append(f"📁 `{Path(r.source_path).name}`")
            if r.page_numbers:
                meta.append(f"📄 Pages {r.page_numbers}")
            if r.section_path:
                meta.append(f"📂 {' > '.join(r.section_path)}")
            if r.keywords:
                meta.append(f"🏷️ {', '.join(r.keywords[:5])}")
            st.markdown("  ·  ".join(meta))

            # Score badge
            st.markdown(
                f"<span style='background:{score_clr}22;color:{score_clr};"
                f"border:1px solid {score_clr};padding:2px 10px;border-radius:16px;"
                f"font-size:0.8rem;font-weight:700;'>Score: {r.similarity_score:.4f}</span>",
                unsafe_allow_html=True,
            )

            if r.summary:
                st.markdown(f"**Summary:** {r.summary}")

            st.text_area("Chunk text", r.text, height=130, key=f"ret_{mode_label}_{i}", disabled=True)

# Latency badge
st.markdown(
    f"<span style='background:#8b949e22;color:#8b949e;border:1px solid #444c56;"
    f"padding:3px 12px;border-radius:16px;font-size:0.8rem;'>⏱ {total_ms:.1f} ms total</span>",
    unsafe_allow_html=True,
)
st.markdown("")

if search_mode == "Both":
    col_v, col_k = st.columns(2)
    with col_v:
        _render_results(vector_results, "🧲 Vector")
    with col_k:
        _render_results(keyword_results, "🔤 Keyword")
elif search_mode == "Vector (semantic)":
    _render_results(vector_results, "🧲 Vector")
else:
    _render_results(keyword_results, "🔤 Keyword")

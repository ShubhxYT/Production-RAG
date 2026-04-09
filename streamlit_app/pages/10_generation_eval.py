"""Page 10 — Generation Evaluation (LLM Judge Panel)."""

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app.utils.page_config import apply_page_config
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Gen Eval | FullRag", page_icon="🧑‍⚖️", layout="wide")

apply_page_config()

st.title("🧑‍⚖️ Generation Evaluation — Judge Panel")
st.markdown("Evaluate a generated answer across **Faithfulness · Relevance · Completeness · Coherence**.")
st.markdown("---")

# ── Judge model badge ─────────────────────────────────────────────────────────
try:
    from config.settings import get_generation_model
    judge_model = get_generation_model()
except Exception:
    judge_model = "gemini-2.5-flash"

st.markdown(
    f"<span style='background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb;"
    f"padding:4px 14px;border-radius:20px;font-size:0.82rem;font-weight:600;'>"
    f"🧑‍⚖️ Judge: Gemini — <code>{judge_model}</code> (read-only)</span>",
    unsafe_allow_html=True,
)
st.markdown("")

# ── Input mode ────────────────────────────────────────────────────────────────
tab_rag, tab_manual = st.tabs(["🤖 Run via RAG Pipeline", "📝 Paste Custom Answer"])

query = ""
answer = ""
context_chunks: list[str] = []

with tab_rag:
    st.markdown("Enter a query — the RAG pipeline will generate an answer and the judge panel will evaluate it.")
    rag_query = st.text_input("Query", placeholder="What is retrieval-augmented generation?", key="gen_eval_rag_q")
    rag_provider = st.selectbox("Generation provider", ["groq", "gemini"], key="gen_eval_provider")
    rag_top_k = st.slider("Top K", 1, 10, 5, key="gen_eval_top_k")

    if st.button("🚀 Generate + Judge", type="primary", key="gen_eval_rag_btn"):
        if not rag_query.strip():
            st.warning("Enter a query first.")
        else:
            query = rag_query
            with st.spinner("Running RAG pipeline…"):
                try:
                    from pipeline.rag import RAGPipeline
                    pipeline = RAGPipeline(provider_name=rag_provider)
                    response = asyncio.run(pipeline.query(query, top_k=rag_top_k))
                    answer = response.answer
                    context_chunks = [s.chunk_summary or "" for s in response.sources if s.chunk_summary]
                    st.success(f"Generated answer ({len(answer)} chars, {response.latency.total_ms:.0f}ms)")
                    with st.expander("📝 Generated Answer"):
                        st.markdown(answer)
                except Exception as e:
                    st.error(f"RAG pipeline error: {e}")
                    st.stop()

with tab_manual:
    manual_query = st.text_input("Query", placeholder="What is retrieval-augmented generation?", key="gen_eval_manual_q")
    manual_answer = st.text_area("Answer to evaluate", height=200, placeholder="The generated answer…")
    manual_context = st.text_area(
        "Context chunks (one per line/paragraph)",
        height=150,
        placeholder="Context chunk 1…\n---\nContext chunk 2…",
    )

    if st.button("🧑‍⚖️ Judge Answer", type="primary", key="gen_eval_manual_btn"):
        if not manual_query.strip() or not manual_answer.strip():
            st.warning("Please enter both query and answer.")
        else:
            query = manual_query
            answer = manual_answer
            if manual_context.strip():
                context_chunks = [p.strip() for p in manual_context.split("---") if p.strip()]
            else:
                context_chunks = []

# ── Run judges ────────────────────────────────────────────────────────────────
if query and answer:
    st.markdown("---")
    st.subheader("🧑‍⚖️ Judge Panel Results")

    with st.spinner("Running Gemini judge panel…"):
        try:
            from evaluation.generation_judges import JudgePanel
            panel = JudgePanel.default_panel(provider_name="gemini")
            scores = panel.evaluate_all(query, answer, context_chunks)
        except Exception as e:
            st.error(f"Judge evaluation failed: {e}")
            st.stop()

    # ── Radar chart ───────────────────────────────────────────────────────────
    dim_names = [s.dimension.value.capitalize() for s in scores]
    score_vals = [s.score for s in scores]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=score_vals + [score_vals[0]],
        theta=dim_names + [dim_names[0]],
        fill="toself",
        fillcolor="rgba(31,111,235,0.2)",
        line=dict(color="#58a6ff", width=2),
        marker=dict(size=8, color="#58a6ff"),
        name="Judge Scores",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 5], gridcolor="#30363d", color="#8b949e"),
            angularaxis=dict(gridcolor="#30363d", color="#cdd9e5"),
        ),
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
        height=380,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Score summary ─────────────────────────────────────────────────────────
    passed_total = sum(1 for s in scores if s.passed)
    avg_score = sum(s.score for s in scores) / len(scores) if scores else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Average Score", f"{avg_score:.2f} / 5")
    m2.metric("Passing Dimensions", f"{passed_total} / {len(scores)}")
    overall_pass = passed_total == len(scores)
    m3.metric(
        "Overall",
        "✅ PASS" if overall_pass else "❌ FAIL",
        delta=None,
    )

    # ── Dimension cards ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Dimension Breakdown")
    cols = st.columns(len(scores))
    for i, (score, col) in enumerate(zip(scores, cols)):
        with col:
            pass_icon = "✅" if score.passed else "❌"
            score_color = "#3fb950" if score.score >= 4 else "#e3b341" if score.score >= 3 else "#f85149"
            col.markdown(
                f"<div style='background:#161b22;border:1px solid {score_color};"
                f"border-radius:10px;padding:14px;text-align:center;'>"
                f"<div style='font-size:0.8rem;color:#8b949e;text-transform:uppercase;"
                f"font-weight:600;letter-spacing:1px;'>{score.dimension.value}</div>"
                f"<div style='font-size:2.4rem;font-weight:700;color:{score_color};margin:6px 0;'>"
                f"{score.score}<span style='font-size:1rem;color:#8b949e;'>/5</span></div>"
                f"<div style='font-size:1.2rem;'>{pass_icon}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Reasoning"):
                st.markdown(score.reasoning)

"""Page 8 — RAG Pipeline Chatbot."""

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app.utils.page_config import apply_page_config
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="RAG Pipeline | FullRag", page_icon="🤖", layout="wide")

apply_page_config()

st.title("🤖 RAG Pipeline")
st.markdown("Full end-to-end retrieval-augmented generation chatbot with source citations.")
st.markdown("---")

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ RAG Settings")
    provider = st.selectbox(
        "Generation Provider",
        ["groq", "gemini"],
        index=0,
        help="Groq: fast inference. Gemini: enrichment/judge model.",
    )
    top_k = st.slider("Top K chunks", 1, 15, 5)
    prompt_variant = st.selectbox(
        "Prompt variant",
        ["auto", "qa", "summarize", "insufficient"],
        index=0,
        help="'auto' lets the pipeline detect the best template.",
    )
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["rag_messages"] = []
        st.session_state["rag_latencies"] = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = []
if "rag_latencies" not in st.session_state:
    st.session_state["rag_latencies"] = []

# ── Display chat history ──────────────────────────────────────────────────────
main_col, meta_col = st.columns([3, 1])

with main_col:
    for msg in st.session_state["rag_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        score_color = "#3fb950" if src["score"] >= 0.7 else "#e3b341"
                        st.markdown(
                            f"**{src['title'] or 'Unknown'}** &nbsp; "
                            f"<span style='background:{score_color}22;color:{score_color};"
                            f"border:1px solid {score_color};padding:1px 8px;"
                            f"border-radius:12px;font-size:0.78rem;'>Score: {src['score']:.3f}</span>",
                            unsafe_allow_html=True,
                        )
                        if src.get("pages"):
                            st.caption(f"Pages: {src['pages']}")
                        if src.get("summary"):
                            st.caption(f"Summary: {src['summary']}")
                        st.divider()

with meta_col:
    if st.session_state["rag_latencies"]:
        last = st.session_state["rag_latencies"][-1]
        st.markdown("**⏱ Last Request Latency**")
        fig = go.Figure(go.Bar(
            x=["Retrieval", "Context", "Generation"],
            y=[last.get("retrieval_ms", 0), last.get("context_ms", 0), last.get("generation_ms", 0)],
            marker_color=["#1f6feb", "#8b949e", "#238636"],
            text=[f"{v:.0f}ms" for v in [last.get("retrieval_ms", 0), last.get("context_ms", 0), last.get("generation_ms", 0)]],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", height=220,
            xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"),
            margin=dict(l=0, r=0, t=8, b=0), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Total", f"{last.get('total_ms', 0):.0f} ms")

        if last.get("token_usage"):
            tu = last["token_usage"]
            st.markdown("**🪙 Token Usage**")
            st.markdown(
                f"- Prompt: `{tu.get('prompt_tokens', 0)}`\n"
                f"- Completion: `{tu.get('completion_tokens', 0)}`\n"
                f"- Total: `{tu.get('total_tokens', 0)}`"
            )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question…"):
    st.session_state["rag_messages"].append({"role": "user", "content": prompt})
    with main_col:
        with st.chat_message("user"):
            st.markdown(prompt)

    with main_col:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⏳ Retrieving and generating…")

            try:
                from pipeline.rag import RAGPipeline

                pipeline = RAGPipeline(provider_name=provider)
                pv = None if prompt_variant == "auto" else prompt_variant
                response = asyncio.run(pipeline.query(prompt, top_k=top_k, prompt_variant=pv))

                placeholder.markdown(response.answer)

                sources = [
                    {
                        "title": s.document_title,
                        "score": s.similarity_score,
                        "pages": s.page_numbers,
                        "summary": s.chunk_summary,
                        "path": s.source_path,
                    }
                    for s in response.sources
                ]

                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for src in sources:
                            score_color = "#3fb950" if src["score"] >= 0.7 else "#e3b341"
                            st.markdown(
                                f"**{src['title'] or 'Unknown'}** &nbsp; "
                                f"<span style='background:{score_color}22;color:{score_color};"
                                f"border:1px solid {score_color};padding:1px 8px;"
                                f"border-radius:12px;font-size:0.78rem;'>Score: {src['score']:.3f}</span>",
                                unsafe_allow_html=True,
                            )
                            if src.get("pages"):
                                st.caption(f"Pages: {src['pages']}")
                            if src.get("summary"):
                                st.caption(f"Summary: {src['summary']}")
                            st.divider()

                # Store
                st.session_state["rag_messages"].append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": sources,
                })
                st.session_state["rag_latencies"].append({
                    "retrieval_ms": response.latency.retrieval_ms,
                    "context_ms": response.latency.context_ms,
                    "generation_ms": response.latency.generation_ms,
                    "total_ms": response.latency.total_ms,
                    "token_usage": response.token_usage.model_dump() if response.token_usage else {},
                })

            except Exception as e:
                placeholder.error(f"RAG pipeline error: {e}")
                st.session_state["rag_messages"].append({
                    "role": "assistant",
                    "content": f"❌ Error: {e}",
                    "sources": [],
                })

    st.rerun()

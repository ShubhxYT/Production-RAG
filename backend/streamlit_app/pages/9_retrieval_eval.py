"""Page 9 — Retrieval Evaluation."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app.utils.page_config import apply_page_config
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Retrieval Eval | FullRag", page_icon="📊", layout="wide")

apply_page_config()

st.title("📊 Retrieval Evaluation")
st.markdown("Run ground-truth retrieval evaluation or compute metrics for a custom query.")
st.markdown("---")

DATASET_PATH = _ROOT / "evaluation" / "datasets" / "retrieval_ground_truth.json"

# ── Config ────────────────────────────────────────────────────────────────────
st.subheader("⚙️ Evaluation Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    top_k = st.slider("Top K", 1, 20, 10)
with col2:
    threshold = st.slider("Threshold", 0.0, 1.0, 0.0, step=0.05,
                           help="0 = no threshold")
with col3:
    k_values = st.multiselect("K values for metrics", [1, 3, 5, 10, 15, 20],
                               default=[1, 3, 5, 10])

threshold_val = threshold if threshold > 0 else None

# Dataset info
if DATASET_PATH.exists():
    try:
        import json
        raw = json.loads(DATASET_PATH.read_text())
        n_annotations = len(raw.get("annotations", []))
        version = raw.get("version", "unknown")
        st.info(f"📋 Dataset v{version} — {n_annotations} annotations at `{DATASET_PATH.relative_to(_ROOT)}`")
    except Exception:
        pass
else:
    st.warning(f"Ground truth dataset not found at `{DATASET_PATH}`")

col_run, col_save = st.columns([1, 1])
run_btn = col_run.button("▶️ Run Evaluation", type="primary", use_container_width=True)
save_enabled = st.checkbox("Save report to `evaluation/results/`", value=True)

if run_btn:
    with st.spinner("Running retrieval evaluation…"):
        try:
            from retrieval.service import RetrievalService
            from evaluation.retrieval_runner import EvaluationRunner

            svc = RetrievalService()
            runner = EvaluationRunner(
                retrieval_service=svc,
                dataset_path=DATASET_PATH,
                top_k=top_k,
                threshold=threshold_val,
                k_values=k_values or [1, 3, 5, 10],
            )
            report = runner.run()

            if save_enabled:
                saved_path = runner.save_report(report)
                st.success(f"Report saved to `{saved_path.relative_to(_ROOT)}`")

        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.stop()

    # ── Aggregate metrics chart ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Aggregate Metrics")

    metric_rows = [
        {"Metric": m.metric_name, "Value": m.value, "k": m.k}
        for m in report.aggregate_metrics
    ]

    # Group by metric base name
    from collections import defaultdict
    by_metric: dict = defaultdict(list)
    for m in report.aggregate_metrics:
        base = m.metric_name.split("@")[0] if "@" in m.metric_name else m.metric_name
        by_metric[base].append(m)

    colors = {"precision": "#1f6feb", "recall": "#3fb950", "mrr": "#e3b341", "ndcg": "#bc8cff"}
    fig = go.Figure()
    for base, metrics in by_metric.items():
        color = colors.get(base.lower(), "#8b949e")
        fig.add_trace(go.Bar(
            name=base.upper(),
            x=[m.metric_name for m in metrics],
            y=[m.value for m in metrics],
            marker_color=color,
            text=[f"{m.value:.4f}" for m in metrics],
            textposition="outside",
        ))

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#e6edf3",
        height=380, barmode="group",
        xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d", range=[0, 1.15]),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        margin=dict(l=0, r=0, t=8, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Aggregate table
    st.dataframe(
        pd.DataFrame(metric_rows).sort_values("Metric"),
        use_container_width=True, hide_index=True,
    )

    # ── Per-query results ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"🔍 Per-Query Results ({len(report.per_query_results)} queries)")

    for i, qr in enumerate(report.per_query_results):
        hit_ratio = len(set(qr.retrieved_ids) & set(qr.relevant_ids)) / max(len(qr.relevant_ids), 1)
        color = "#3fb950" if hit_ratio > 0.5 else "#e3b341" if hit_ratio > 0 else "#f85149"
        label = f"Q{i+1}: {qr.query[:70]}…" if len(qr.query) > 70 else f"Q{i+1}: {qr.query}"
        with st.expander(label):
            st.markdown(f"**Tags:** {', '.join(qr.tags) if qr.tags else 'none'}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Retrieved", len(qr.retrieved_ids))
            m2.metric("Relevant", len(qr.relevant_ids))
            m3.metric("Hits", len(set(qr.retrieved_ids) & set(qr.relevant_ids)))
            mdf = pd.DataFrame([{"Metric": m.metric_name, "Value": f"{m.value:.4f}"} for m in qr.metrics])
            st.dataframe(mdf, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Custom query eval ─────────────────────────────────────────────────────────
st.subheader("🧪 Custom Query Evaluation")
st.markdown("Enter a query and the known relevant chunk IDs to compute metrics.")

cq = st.text_input("Custom query", placeholder="Enter your query…")
relevant_ids_raw = st.text_area(
    "Relevant chunk IDs (one per line)",
    height=100,
    placeholder="uuid-1\nuuid-2\n…",
)
cq_top_k = st.slider("Top K (custom)", 1, 20, 5, key="cq_topk")

if st.button("🔍 Evaluate Custom Query", use_container_width=False):
    if not cq.strip():
        st.warning("Please enter a query.")
    else:
        relevant_ids = [l.strip() for l in relevant_ids_raw.splitlines() if l.strip()]
        with st.spinner("Retrieving…"):
            try:
                from retrieval.service import RetrievalService
                from evaluation.retrieval_metrics import compute_all_metrics
                svc = RetrievalService()
                resp = svc.retrieve_sync(cq, top_k=cq_top_k)
                retrieved_ids = [r.chunk_id for r in resp.results]
                if relevant_ids:
                    metrics = compute_all_metrics(retrieved_ids, set(relevant_ids), cq_top_k)
                    st.success(f"Retrieved {len(retrieved_ids)} chunks in {resp.latency_ms:.1f}ms")
                    mdf = pd.DataFrame([
                        {"Metric": m.metric_name, "Value": f"{m.value:.4f}", "k": m.k}
                        for m in metrics
                    ])
                    st.dataframe(mdf, use_container_width=True, hide_index=True)
                else:
                    st.info(f"Retrieved {len(retrieved_ids)} chunks. (No relevant IDs provided — cannot compute metrics.)")
                    for i, r in enumerate(resp.results):
                        st.markdown(f"**#{i+1}** `{r.chunk_id}` — score: `{r.similarity_score:.4f}` — {r.text[:80]}…")
            except Exception as e:
                st.error(f"Error: {e}")

"""Page 1 — Overview & Architecture."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app.utils.page_config import apply_page_config
import streamlit as st

st.set_page_config(page_title="Overview | FullRag", page_icon="🏠", layout="wide")

apply_page_config()

st.title("🏠 Overview & Architecture")
st.markdown("---")

# ── Architecture diagram ──────────────────────────────────────────────────────
arch_img = _ROOT / "fullrag-architecture.png"
if arch_img.exists():
    st.subheader("System Architecture")
    st.image(str(arch_img), use_container_width=True)
else:
    st.warning("Architecture image not found at project root.")

st.markdown("---")

# ── System summary ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Project Info")
    try:
        import tomllib
        toml_path = _ROOT / "pyproject.toml"
        with open(toml_path, "rb") as f:
            meta = tomllib.load(f)["project"]
        st.markdown(f"""
| Field | Value |
|---|---|
| **Name** | `{meta.get('name', 'fullrag')}` |
| **Version** | `{meta.get('version', 'unknown')}` |
| **Python** | `{sys.version.split()[0]}` |
| **Root** | `{_ROOT}` |
        """)
    except Exception as e:
        st.error(f"Could not read pyproject.toml: {e}")

with col2:
    st.subheader("🔧 Module Directory")
    modules = [
        ("ingestion", "Loaders, restructuring, chunking, staging"),
        ("generation", "LLM config, enrichment prompts & providers"),
        ("embeddings", "Local sentence-transformer embedding + cache"),
        ("database", "SQLAlchemy models, repository, seed"),
        ("retrieval", "Query embed + vector similarity search"),
        ("evaluation", "Retrieval & generation metrics & runners"),
        ("pipeline", "End-to-end RAG orchestrator"),
        ("api", "FastAPI REST endpoints"),
        ("observability", "Structured logging & tracing"),
        ("config", "Environment-based settings loader"),
    ]
    rows = []
    for name, desc in modules:
        mod_path = _ROOT / name
        n_files = len(list(mod_path.glob("*.py"))) if mod_path.exists() else 0
        rows.append({"Module": f"`{name}/`", "Files": n_files, "Description": desc})

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ── Provider config ───────────────────────────────────────────────────────────
st.subheader("⚙️ Active Provider Configuration")
c1, c2, c3, c4 = st.columns(4)

with c1:
    try:
        from config.settings import get_generation_model
        model = get_generation_model()
        st.metric("Gemini Model", model, help="Used for enrichment and generation judging")
    except Exception:
        st.metric("Gemini Model", "Not configured")

with c2:
    try:
        from config.settings import get_groq_base_url, get_groq_model
        url = get_groq_base_url()
        model = get_groq_model()
        st.metric("Groq Endpoint", url.split("/")[2], help=f"Model: {model}")
    except Exception:
        st.metric("Groq Endpoint", "Not configured")

with c3:
    try:
        from embeddings.models import EmbeddingConfig
        cfg = EmbeddingConfig()
        st.metric("Embed Model", cfg.model_name.split("/")[-1], help=cfg.model_name)
        st.metric("Dimensions", cfg.dimensions)
    except Exception:
        st.metric("Embed Model", "Unknown")

with c4:
    import torch
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    st.metric("Compute Device", device)
    st.metric("GPU", gpu_name[:24] if gpu_name != "N/A" else "N/A")

st.markdown("---")
st.subheader("🧑‍⚖️ Judge Panel")
st.info(
    "**Generation evaluation judges** use Gemini (`gemini-2.5-flash`) by default.\n\n"
    "Dimensions evaluated: **Faithfulness · Relevance · Completeness · Coherence** — scored 1–5, passing threshold ≥ 3."
)

# ── Data flow ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔄 Core Data Flow")
st.markdown("""
```
1. Ingest  ──  PDF / DOCX / HTML / MD  →  normalized Document elements
2. Chunk   ──  structure-aware chunking  →  text units with metadata
3. Enrich  ──  Gemini LLM  →  summary + keywords + hypothetical questions
4. Embed   ──  local SentenceTransformer (CUDA/CPU)  →  768-dim vectors
5. Seed    ──  PostgreSQL + pgvector (HNSW index)  →  stored chunks + embeddings
6. Retrieve  ──  cosine similarity / keyword FTS  →  top-K ranked chunks
7. Generate  ──  Cerebras (RAG) / Gemini (enrichment)  →  grounded answer
8. Evaluate  ──  Precision@k · Recall@k · MRR · NDCG  |  Judge scores
```
""")

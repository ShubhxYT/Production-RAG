"""Page 6 — Database Browser."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app.utils.page_config import apply_page_config
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Database | FullRag", page_icon="🗄️", layout="wide")

apply_page_config()

st.title("🗄️ Database Browser")
st.markdown("Inspect the PostgreSQL/pgvector database — stats, documents, and chunks.")
st.markdown("---")

# ── DB URL display ────────────────────────────────────────────────────────────
try:
    from config.settings import get_database_url
    db_url = get_database_url()
    # Mask credentials
    masked = db_url
    if "@" in db_url:
        scheme_and_creds, rest = db_url.split("@", 1)
        if "://" in scheme_and_creds:
            scheme = scheme_and_creds.split("://")[0]
            masked = f"{scheme}://****:****@{rest}"
    st.markdown(
        f"<span style='background:#161b22;color:#8b949e;border:1px solid #30363d;"
        f"padding:4px 14px;border-radius:8px;font-size:0.82rem;font-family:monospace;'>"
        f"🔗 {masked}</span>",
        unsafe_allow_html=True,
    )
except Exception:
    pass

st.markdown("")

# ── Stats ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=10, show_spinner=False)
def _get_stats() -> dict | None:
    try:
        from database.connection import get_session
        from database.models import DocumentModel, ChunkModel, ChunkEmbeddingModel
        from sqlalchemy import func, select
        session = get_session()
        try:
            doc_count = session.execute(select(func.count(DocumentModel.id))).scalar()
            chunk_count = session.execute(select(func.count(ChunkModel.id))).scalar()
            emb_count = session.execute(select(func.count(ChunkEmbeddingModel.id))).scalar()
            return {"docs": doc_count, "chunks": chunk_count, "embeddings": emb_count}
        finally:
            session.close()
    except Exception as e:
        return {"error": str(e)}

col_refresh, _ = st.columns([1, 5])
if col_refresh.button("🔄 Refresh", use_container_width=True):
    st.cache_data.clear()

stats = _get_stats()
if stats and "error" in stats:
    st.error(f"Database error: {stats['error']}")
elif stats:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Documents", stats["docs"])
    c2.metric("✂️ Chunks", stats["chunks"])
    c3.metric("🧲 Embeddings", stats["embeddings"])
    c4.metric("⚠️ Missing Embeddings", max(0, stats["chunks"] - stats["embeddings"]))

st.markdown("---")

# ── Seed from staging ─────────────────────────────────────────────────────────
st.subheader("🌱 Seed from Staging")
staging_dir = _ROOT / "staging"
staged = sorted(staging_dir.glob("*.json")) if staging_dir.exists() else []
st.markdown(f"Found **{len(staged)}** JSON file(s) in `staging/`.")

if st.button("🌱 Seed Database from Staging", type="primary"):
    with st.spinner("Seeding database…"):
        try:
            from database.seed import seed_from_staging
            result = seed_from_staging(str(staging_dir))
            st.success(
                f"✅ Seeded: {result['documents']} docs, {result['chunks']} chunks, "
                f"{result['embeddings']} embeddings | skipped: {result['skipped']} | "
                f"failed: {result.get('failed', 0)} | {result.get('elapsed_seconds', 0):.1f}s"
            )
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Seed failed: {e}")

st.markdown("---")

# ── Document browser ──────────────────────────────────────────────────────────
st.subheader("📄 Document Browser")

try:
    from database.connection import get_session
    from database.models import DocumentModel
    from sqlalchemy import select

    session = get_session()
    try:
        docs = session.execute(select(DocumentModel).order_by(DocumentModel.created_at.desc())).scalars().all()
    finally:
        session.close()

    if not docs:
        st.info("No documents in the database yet.")
    else:
        doc_rows = [
            {
                "ID": str(d.id)[:8] + "…",
                "Title": d.title or "Untitled",
                "Format": d.format,
                "Source": Path(d.source_path).name if d.source_path else "—",
                "Created": d.created_at.strftime("%Y-%m-%d %H:%M") if d.created_at else "—",
            }
            for d in docs
        ]
        st.dataframe(pd.DataFrame(doc_rows), use_container_width=True, hide_index=True)

        # Chunk browser
        st.markdown("---")
        st.subheader("🔍 Chunk Browser")
        doc_titles = [d.title or Path(d.source_path).name for d in docs]
        sel_title = st.selectbox("Select document", doc_titles)
        sel_doc = docs[doc_titles.index(sel_title)]

        kw_filter = st.text_input("Filter chunks by keyword", placeholder="optional keyword filter…")

        from database.models import ChunkModel
        session2 = get_session()
        try:
            chunks = session2.execute(
                select(ChunkModel)
                .where(ChunkModel.document_id == sel_doc.id)
                .order_by(ChunkModel.position)
            ).scalars().all()
        finally:
            session2.close()

        if kw_filter:
            chunks = [c for c in chunks if kw_filter.lower() in (c.text or "").lower()]

        st.markdown(f"**{len(chunks)} chunk(s)**")
        for i, chunk in enumerate(chunks):
            section = " > ".join(chunk.section_path or []) or "—"
            with st.expander(f"Chunk {i+1}  |  {chunk.token_count or 0} tokens  |  {section}"):
                st.markdown(f"**ID:** `{chunk.id}`")
                st.markdown(f"**Keywords:** {', '.join(chunk.keywords or [])}")
                if chunk.summary:
                    st.markdown(f"**Summary:** {chunk.summary}")
                st.text_area("Text", chunk.text or "", height=120, key=f"db_chunk_{i}", disabled=True)

except Exception as e:
    st.error(f"Could not connect to database: {e}")
    st.info("Make sure PostgreSQL is running: `docker compose -f pgvector.yaml up -d`")

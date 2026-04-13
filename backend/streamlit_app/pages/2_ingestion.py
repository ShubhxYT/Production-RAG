"""Page 2 — Document Ingestion."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app.utils.page_config import apply_page_config
import streamlit as st

st.set_page_config(page_title="Ingestion | FullRag", page_icon="📄", layout="wide")

apply_page_config()

st.title("📄 Document Ingestion")
st.markdown("Upload documents to ingest, chunk, and optionally run through the full pipeline.")
st.markdown("---")

from streamlit_app.utils.cleanup import get_user_data_dir

uploaded = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "html", "htm", "md"],
    accept_multiple_files=True,
)

if not uploaded:
    st.info("⬆️ Upload one or more documents to get started.")
    st.stop()

user_data = get_user_data_dir()
saved_paths: list[Path] = []
for uf in uploaded:
    dest = user_data / uf.name
    dest.write_bytes(uf.getbuffer())
    saved_paths.append(dest)

st.success(f"✅ {len(saved_paths)} file(s) ready in `user_data/`")

st.markdown("---")
st.subheader("⚡ Pipeline Actions")
col_a, col_b, col_c = st.columns(3)
run_ingest = col_a.button("📄 Ingest & Chunk", use_container_width=True)
run_enrich = col_b.button("✨ Ingest + Enrich", use_container_width=True)
run_full   = col_c.button("🚀 Full Pipeline (Ingest→Chunk→Embed→Seed DB)", use_container_width=True)
st.markdown("---")


def _run_ingestion(paths: list[Path]) -> list:
    from ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline(output_dir=str(user_data / "results"))
    docs = []
    prog = st.progress(0, text="Starting ingestion…")
    for i, p in enumerate(paths):
        prog.progress(i / len(paths), text=f"Processing {p.name}…")
        doc = pipeline.ingest_file(p)
        if doc is not None:
            docs.append(doc)
    prog.progress(1.0, text="Ingestion done!")
    return docs


if run_ingest or run_enrich or run_full:
    with st.spinner("Running ingestion…"):
        try:
            docs = _run_ingestion(saved_paths)
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            st.stop()

    if not docs:
        st.warning("No documents were successfully ingested.")
        st.stop()

    st.session_state["ingested_docs"] = docs

    if run_enrich or run_full:
        st.write("#### ✨ Enriching chunks with Gemini…")
        from ingestion.enrichment import enrich_document
        ep = st.progress(0, text="Enriching…")
        for i, doc in enumerate(docs):
            ep.progress(i / len(docs), text=f"Enriching {doc.title}…")
            try:
                enrich_document(doc)
            except Exception as e:
                st.warning(f"Enrichment failed for {doc.title}: {e}")
        ep.progress(1.0, text="Enrichment done!")

    if run_full:
        st.write("#### 🧲 Generating embeddings & seeding DB…")
        from embeddings.service import EmbeddingService
        from embeddings.models import EmbeddingConfig
        from database.connection import get_session
        from database.repository import DocumentRepository
        embed_svc = EmbeddingService(config=EmbeddingConfig())
        repo = DocumentRepository()
        ep2 = st.progress(0, text="Embedding…")
        for i, doc in enumerate(docs):
            ep2.progress(i / len(docs), text=f"Embedding {doc.title}…")
            texts = [c.text for c in doc.chunks]
            if texts:
                try:
                    result = embed_svc.embed(texts)
                    session = get_session()
                    try:
                        repo.insert_document(session, doc)
                        pairs = [(c.id, v) for c, v in zip(doc.chunks, result.vectors)]
                        repo.insert_bulk_embeddings(session, pairs, result.model)
                        session.commit()
                    except Exception as e:
                        session.rollback()
                        st.warning(f"DB seed failed for {doc.title}: {e}")
                    finally:
                        session.close()
                except Exception as e:
                    st.warning(f"Embedding error for {doc.title}: {e}")
        ep2.progress(1.0, text="Pipeline complete!")
        st.success("🎉 Full pipeline complete — documents seeded into the database!")

    st.markdown("---")
    st.subheader(f"📊 Ingestion Results — {len(docs)} document(s)")
    import pandas as pd
    rows = []
    for doc in docs:
        rows.append({
            "File": doc.title or Path(doc.source_path).name,
            "Format": doc.format,
            "Elements": len(doc.elements),
            "Chunks": len(doc.chunks),
            "Total Tokens": sum(c.token_count for c in doc.chunks),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔍 Chunk Inspector")
    doc_names = [d.title or Path(d.source_path).name for d in docs]
    sel_name = st.selectbox("Select document", doc_names)
    sel_doc = docs[doc_names.index(sel_name)]

    for i, chunk in enumerate(sel_doc.chunks):
        section = " > ".join(chunk.section_path) if chunk.section_path else "No section"
        with st.expander(f"Chunk {i+1}  |  {chunk.token_count} tokens  |  {section}"):
            st.markdown(f"**ID:** `{chunk.id}`")
            if chunk.page_numbers:
                st.markdown(f"**Pages:** {chunk.page_numbers}")
            if chunk.summary:
                st.markdown(f"**Summary:** {chunk.summary}")
            if chunk.keywords:
                st.markdown("**Keywords:** " + "  ".join([f"`{k}`" for k in chunk.keywords]))
            st.text_area("Text", chunk.text, height=150, key=f"chunk_{i}_{sel_doc.id}", disabled=True)

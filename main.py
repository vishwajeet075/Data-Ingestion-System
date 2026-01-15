# main.py
import streamlit as st
import os
import tempfile
from pathlib import Path
from datetime import datetime

from document_processor import DocumentProcessor
from vector_database import VectorDatabase
from utils import display_debug_info, create_metadata, safe_json_dumps

# =========================
# 🔐 HARDCODED CONFIG
# =========================
VOYAGE_API_KEY = "pa-MrCNniXMkRHY0OzQRsbBMTRRxGl_hHxrf1rVVhyExJ6"
QDRANT_URL = "https://129744c0-79a4-4302-89be-b3978b26fa2c.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.82u-T4-ffPlMwwbqIkA-CXtjMCxBS_wZMvko6Wx68tc"

# =========================
# Streamlit config
# =========================
st.set_page_config(
    page_title="RAG Database Ingestion",
    page_icon="🗄️",
    layout="wide"
)


def main():
    st.title("🗄️ RAG Database Ingestion System")
    st.markdown("### Upload documents to create vector embeddings and store in Qdrant")

    # Session state
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = {}

    if "vector_db" not in st.session_state:
        try:
            with st.spinner("Initializing vector database..."):
                vector_db = VectorDatabase(
                    VOYAGE_API_KEY,
                    QDRANT_URL,
                    QDRANT_API_KEY
                )
                debug_info = vector_db.initialize_collection()
                st.session_state.vector_db = vector_db

            if debug_info.get("error"):
                st.error(f"Database init failed: {debug_info['error']}")
                return
            else:
                st.success("✅ Vector database initialized")

        except Exception as e:
            st.error(f"Initialization error: {e}")
            return

    # =========================
    # Sidebar (cleaned)
    # =========================
    with st.sidebar:
        st.header("⚙️ Processing Settings")

        chunk_size = st.slider(
            "Chunk Size (chars)",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100
        )

        chunk_overlap = st.slider(
            "Chunk Overlap (chars)",
            min_value=0,
            max_value=500,
            value=200,
            step=50
        )

        st.divider()

        if st.button("🔄 Clear Session"):
            st.session_state.clear()
            st.rerun()

    # =========================
    # File Upload
    # =========================
    st.subheader("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Drag and drop or click to browse",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Upload one or more documents to begin")
        return

    st.info(f"📄 {len(uploaded_files)} file(s) ready")

    if st.button("🚀 Start Processing", type="primary", use_container_width=True):
        processor = DocumentProcessor()
        vector_db = st.session_state.vector_db

        total_chunks = 0
        file_debug_info = {}

        for uploaded_file in uploaded_files:
            st.divider()
            st.subheader(f"📄 Processing: {uploaded_file.name}")

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            file_debug = {
                "filename": uploaded_file.name,
                "file_size": uploaded_file.size,
                "processing_start": datetime.now().isoformat(),
                "success": False
            }

            try:
                text, extraction_debug = processor.process_file(
                    tmp_path,
                    uploaded_file.name
                )
                file_debug["extraction"] = extraction_debug

                if not text:
                    raise ValueError("No text extracted")

                metadata = create_metadata(
                    uploaded_file.name,
                    uploaded_file.name.split(".")[-1],
                    uploaded_file.size,
                    extraction_debug.get("method", "unknown")
                )

                with st.spinner("Embedding & storing..."):
                    chunks_uploaded, ingestion_debug = vector_db.ingest_document(
                        text,
                        metadata,
                        chunk_size,
                        chunk_overlap
                    )

                file_debug["chunks_uploaded"] = chunks_uploaded
                file_debug["ingestion"] = ingestion_debug
                total_chunks += chunks_uploaded

                if chunks_uploaded > 0:
                    verify_debug = vector_db.verify_ingestion(uploaded_file.name)
                    file_debug["verification"] = verify_debug

                    if verify_debug.get("verified"):
                        st.success(f"✅ {chunks_uploaded} chunks stored")
                        file_debug["success"] = True
                    else:
                        st.warning("Stored but verification incomplete")
                else:
                    st.error("No chunks uploaded")

            except Exception as e:
                st.error(f"❌ {str(e)[:200]}")
                file_debug["error"] = str(e)

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                file_debug["processing_end"] = datetime.now().isoformat()
                file_debug_info[uploaded_file.name] = file_debug

        # =========================
        # Summary
        # =========================
        st.divider()
        st.subheader("📊 Summary")

        success_files = sum(
            1 for f in file_debug_info.values() if f.get("success")
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Files", len(uploaded_files))
        col2.metric("Successful", success_files)
        col3.metric("Total Chunks", total_chunks)

        st.session_state.debug_info = {
            "summary": {
                "files": len(uploaded_files),
                "successful": success_files,
                "chunks": total_chunks
            },
            "files": file_debug_info
        }

        if success_files:
            st.success("🎉 Processing complete")
            st.balloons()


if __name__ == "__main__":
    main()
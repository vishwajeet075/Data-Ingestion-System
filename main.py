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
    st.markdown("### Upload documents, images, videos, or audio to create vector embeddings")

    # Check for dependencies
    with st.sidebar:
        st.header("🔧 Dependencies Check")
        try:
            import moviepy.editor as mp
            st.success("✓ moviepy installed")
        except ImportError:
            st.warning("✗ moviepy not installed (needed for video)")
            st.info("Install: `pip install moviepy`")

        try:
            import speech_recognition as sr
            st.success("✓ speech_recognition installed")
        except ImportError:
            st.warning("✗ speech_recognition not installed (needed for audio)")
            st.info("Install: `pip install speechrecognition`")

        try:
            import pytesseract
            st.success("✓ pytesseract installed")
        except ImportError:
            st.warning("✗ pytesseract not installed (needed for images)")
            st.info("Install: `pip install pytesseract` and install Tesseract OCR")

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

        st.header("🖼️ Image Processing")
        image_preprocess = st.checkbox(
            "Preprocess images for better OCR",
            value=True,
            help="Enhance contrast and sharpness before OCR"
        )

        st.header("🎵 Audio Processing")
        audio_language = st.selectbox(
            "Speech Language",
            ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "hi-IN"],
            index=0,
            help="Language for speech recognition"
        )

        st.divider()

        if st.button("🔄 Clear Session"):
            st.session_state.clear()
            st.rerun()

    # =========================
    # File Upload
    # =========================
    st.subheader("📁 Upload Files")

    uploaded_files = st.file_uploader(
        "Drag and drop or click to browse",
        type=[
            # Documents
            "pdf", "txt", "docx",
            # Images
            "jpg", "jpeg", "png", "bmp", "tiff", "tif", "gif",
            # Videos
            "mp4", "avi", "mov", "mkv", "flv", "wmv",
            # Audio
            "mp3", "wav", "ogg", "m4a", "flac", "aac"
        ],
        accept_multiple_files=True,
        help="Supported: Documents (PDF, TXT, DOCX), Images (JPG, PNG, etc.), Videos (MP4, AVI, etc.), Audio (MP3, WAV, etc.)"
    )

    if not uploaded_files:
        st.info("Upload one or more files to begin")
        return

    st.info(f"📄 {len(uploaded_files)} file(s) ready")

    if st.button("🚀 Start Processing", type="primary", use_container_width=True):
        processor = DocumentProcessor()
        vector_db = st.session_state.vector_db

        total_chunks = 0
        file_debug_info = {}

        for uploaded_file in uploaded_files:
            st.divider()

            # Show file icon based on type
            file_ext = uploaded_file.name.lower().split('.')[-1]
            if file_ext in ['mp4', 'avi', 'mov', 'mkv']:
                icon = "🎬"
            elif file_ext in ['mp3', 'wav', 'ogg', 'm4a']:
                icon = "🎵"
            elif file_ext in ['jpg', 'jpeg', 'png', 'bmp']:
                icon = "🖼️"
            else:
                icon = "📄"

            st.subheader(f"{icon} Processing: {uploaded_file.name}")

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            file_debug = {
                "filename": uploaded_file.name,
                "file_type": file_ext,
                "file_size": uploaded_file.size,
                "processing_start": datetime.now().isoformat(),
                "success": False
            }

            try:
                # Process based on file type
                if file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif']:
                    st.info("🖼️ Processing image with OCR...")

                elif file_ext in ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']:
                    st.info("🎬 Extracting audio from video...")

                elif file_ext in ['mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac']:
                    st.info("🎵 Transcribing audio...")
                else:
                    st.info("📄 Extracting text...")

                text, extraction_debug = processor.process_file(
                    tmp_path,
                    uploaded_file.name
                )
                file_debug["extraction"] = extraction_debug

                if not text:
                    raise ValueError("No text extracted")

                # Show extraction stats
                word_count = len(text.split())
                char_count = len(text)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words Extracted", word_count)
                with col2:
                    st.metric("Characters", char_count)
                with col3:
                    st.metric("File Type", file_ext.upper())

                # Show preview for text files
                if file_ext in ['txt', 'pdf', 'docx']:
                    with st.expander("📋 Text Preview", expanded=False):
                        preview_text = text[:2000] + ("..." if len(text) > 2000 else "")
                        st.text_area(
                            "Preview",
                            preview_text,
                            height=150,
                            key=f"preview_{uploaded_file.name}"
                        )

                metadata = create_metadata(
                    uploaded_file.name,
                    file_ext,
                    uploaded_file.size,
                    extraction_debug.get("method", "unknown")
                )

                with st.spinner("Creating embeddings and storing..."):
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
        st.subheader("📊 Processing Summary")

        success_files = sum(
            1 for f in file_debug_info.values() if f.get("success")
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Files", len(uploaded_files))
        col2.metric("Successful", success_files)
        col3.metric("Total Chunks", total_chunks)

        # Calculate success rate
        if len(uploaded_files) > 0:
            success_rate = (success_files / len(uploaded_files)) * 100
            col4.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            col4.metric("Success Rate", "0%")

        # File type breakdown
        st.subheader("📁 File Type Breakdown")
        file_types = {}
        for file in uploaded_files:
            ext = file.name.lower().split('.')[-1]
            file_types[ext] = file_types.get(ext, 0) + 1

        if file_types:
            cols = st.columns(len(file_types))
            for idx, (ext, count) in enumerate(file_types.items()):
                with cols[idx % len(cols)]:
                    st.metric(f"{ext.upper()} Files", count)

        st.session_state.debug_info = {
            "summary": {
                "files": len(uploaded_files),
                "successful": success_files,
                "chunks": total_chunks,
                "file_types": file_types
            },
            "files": file_debug_info
        }

        if success_files:
            st.success("🎉 Processing complete!")
            #st.balloons()

            # Show debug info download
            st.divider()
            st.subheader("📥 Export Results")

            if st.session_state.debug_info:
                debug_json = safe_json_dumps(st.session_state.debug_info)
                st.download_button(
                    label="Download Debug Info",
                    data=debug_json,
                    file_name=f"ingestion_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
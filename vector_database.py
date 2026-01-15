# vector_database.py
from typing import List, Dict, Any, Tuple
import uuid
from datetime import datetime
import streamlit as st
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
import voyageai


class SimpleTextSplitter:
    """Simple text splitter with smart boundary detection"""

    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """Split text into chunks of specified size with overlap"""

        if not text or len(text.strip()) == 0:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size

            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            split_pos = end

            # Priority 1: paragraph boundary
            last_paragraph = text.rfind("\n\n", start, end)
            if (
                last_paragraph != -1
                and last_paragraph > start + chunk_size // 2
            ):
                split_pos = last_paragraph + 2

            # Priority 2: sentence boundary
            elif ". " in text[start:end]:
                last_sentence = text.rfind(". ", start, end)
                if (
                    last_sentence != -1
                    and last_sentence > start + chunk_size // 2
                ):
                    split_pos = last_sentence + 2

            # Priority 3: line break
            elif "\n" in text[start:end]:
                last_newline = text.rfind("\n", start, end)
                if (
                    last_newline != -1
                    and last_newline > start + chunk_size // 2
                ):
                    split_pos = last_newline + 1

            # Priority 4: word boundary
            elif " " in text[start:end]:
                last_space = text.rfind(" ", start, end)
                if (
                    last_space != -1
                    and last_space > start + chunk_size // 2
                ):
                    split_pos = last_space + 1

            chunk = text[start:split_pos].strip()
            if chunk:
                chunks.append(chunk)

            start = max(start + 1, split_pos - chunk_overlap)

        if len(chunks) > 1:
            chunks = [
                ch for i, ch in enumerate(chunks)
                if len(ch) > 50 or i == len(chunks) - 1
            ]

        return chunks

class VectorDatabase:
    def __init__(self, voyage_api_key: str, qdrant_url: str, qdrant_api_key: str):
        self.voyage_client = voyageai.Client(api_key=voyage_api_key)
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )
        self.collection_name = "documents"
        self.vector_size = 1024
        self.text_splitter = SimpleTextSplitter()
        self._collection_initialized = False

    def initialize_collection(self, force_recreate: bool = False) -> Dict[str, Any]:
        """Initialize collection only once per session"""
        debug_info = {
            "collection_exists": False,
            "collection_created": False,
            "error": None
        }

        if self._collection_initialized and not force_recreate:
            debug_info["collection_exists"] = True
            debug_info["cached"] = True
            return debug_info

        try:
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                debug_info["collection_exists"] = True
                points_count = getattr(collection_info, "points_count", None)
                debug_info["collection_info"] = {
                    "points_count": points_count,
                    "status": str(collection_info.status)
                    }
                self._collection_initialized = True
                st.success(
                    f"✓ Collection '{self.collection_name}' exists "
                    f"({points_count} points)"
                )

            except Exception as e:
                if "not found" in str(e).lower() or "404" in str(e):
                    try:
                        self.qdrant_client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(
                                size=self.vector_size,
                                distance=Distance.COSINE
                            )
                        )
                        debug_info["collection_created"] = True
                        self._collection_initialized = True
                        st.success(f"✓ Created collection '{self.collection_name}'")
                    except Exception as create_error:
                        debug_info["error"] = f"Creation failed: {str(create_error)}"
                        st.error(f"Collection creation failed: {create_error}")
                else:
                    debug_info["error"] = f"Check failed: {str(e)}"
                    st.error(f"Collection check failed: {e}")

        except Exception as e:
            debug_info["error"] = str(e)
            st.error(f"Database connection error: {e}")

        return debug_info

    def create_embeddings(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """Create embeddings with retry logic"""
        debug_info = {
            "total_texts": len(texts),
            "embeddings_created": 0,
            "model_used": "voyage-2",
            "errors": []
        }

        all_embeddings = []

        try:
            batch_size = min(128, len(texts))

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1

                try:
                    result = self.voyage_client.embed(
                        texts=batch,
                        model="voyage-2",
                        input_type="document"
                    )
                    all_embeddings.extend(result.embeddings)
                    debug_info["embeddings_created"] += len(result.embeddings)
                    st.info(f"✓ Batch {batch_num}: Created {len(batch)} embeddings")

                except Exception as batch_error:
                    debug_info["errors"].append(
                        f"Batch {batch_num}: {str(batch_error)}"
                    )
                    st.warning(
                        f"Batch {batch_num} failed: "
                        f"{str(batch_error)[:100]}"
                    )

                    for text in batch:
                        try:
                            result = self.voyage_client.embed(
                                texts=[text],
                                model="voyage-2",
                                input_type="document"
                            )
                            all_embeddings.extend(result.embeddings)
                            debug_info["embeddings_created"] += 1
                        except Exception:
                            debug_info["errors"].append(
                                f"Failed individual text: {text[:50]}..."
                            )

            if all_embeddings:
                debug_info["dimensions"] = len(all_embeddings[0])
                st.success(
                    f"✓ Created {debug_info['embeddings_created']}/"
                    f"{debug_info['total_texts']} embeddings"
                )

            return all_embeddings, debug_info

        except Exception as e:
            debug_info["errors"].append(f"Overall error: {str(e)}")
            st.error(f"Embedding creation failed: {e}")
            return [], debug_info

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Split text into chunks with detailed metrics"""
        debug_info = {
            "original_length": len(text),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": 0,
            "chunk_lengths": []
        }

        if not text or len(text.strip()) == 0:
            st.warning("No text to chunk")
            return [], debug_info

        chunks = self.text_splitter.split_text(
            text, chunk_size, chunk_overlap
        )
        debug_info["total_chunks"] = len(chunks)

        if chunks:
            debug_info["chunk_lengths"] = [len(c) for c in chunks]
            debug_info["avg_chunk_length"] = (
                sum(debug_info["chunk_lengths"]) // len(chunks)
            )
            debug_info["min_chunk_length"] = min(debug_info["chunk_lengths"])
            debug_info["max_chunk_length"] = max(debug_info["chunk_lengths"])

            st.info(
                f"✓ Created {len(chunks)} chunks "
                f"(avg {debug_info['avg_chunk_length']} chars)"
            )

            with st.expander("📊 Chunk Statistics", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Chunks", len(chunks))
                with col2:
                    st.metric(
                        "Avg Length",
                        f"{debug_info['avg_chunk_length']} chars"
                    )
                with col3:
                    st.metric(
                        "Size Range",
                        f"{debug_info['min_chunk_length']}-"
                        f"{debug_info['max_chunk_length']}"
                    )

                for i, chunk in enumerate(chunks[:3]):
                    st.write(f"**Chunk {i + 1}:** {chunk[:150]}...")

        else:
            st.warning("No text chunks created")

        return chunks, debug_info

    def ingest_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Tuple[int, Dict[str, Any]]:
        """Ingest document with progress tracking"""
        debug_info = {
            "ingestion_start": datetime.now().isoformat(),
            "total_points": 0,
            "success": False
        }

        init_result = self.initialize_collection()
        if init_result.get("error"):
            debug_info["error"] = (
                f"Collection initialization failed: {init_result['error']}"
            )
            st.error(debug_info["error"])
            return 0, debug_info

        chunks, chunking_info = self.chunk_text(
            text, chunk_size, chunk_overlap
        )
        debug_info["chunking_info"] = chunking_info

        if not chunks:
            st.warning("No text chunks created from document")
            return 0, debug_info

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text(
                f"Creating embeddings for {len(chunks)} chunks..."
            )
            progress_bar.progress(30)

            embeddings, embedding_info = self.create_embeddings(chunks)
            debug_info["embedding_info"] = embedding_info

            if not embeddings:
                st.error("No embeddings created")
                return 0, debug_info

            progress_bar.progress(60)
            status_text.text(
                f"Preparing {len(embeddings)} points for upload..."
            )

            points = []

            for i, (chunk, embedding) in enumerate(
                zip(chunks[:len(embeddings)], embeddings)
            ):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            **metadata,
                            "chunk_id": i,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "chunk_text": chunk[:1000],
                            "chunk_hash": hashlib.md5(
                                chunk.encode()
                            ).hexdigest(),
                            "chunk_length": len(chunk),
                            "word_count": len(chunk.split()),
                            "ingestion_timestamp": datetime.now().isoformat()
                        }
                    )
                )

            progress_bar.progress(80)
            status_text.text(
                f"Uploading {len(points)} points to Qdrant..."
            )

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

            debug_info["total_points"] = len(points)
            debug_info["success"] = True

            st.success(f"✅ Uploaded {len(points)} points to Qdrant")

            progress_bar.progress(100)
            progress_bar.empty()
            status_text.empty()

            return len(points), debug_info

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            debug_info["error"] = str(e)
            st.error(f"Ingestion failed: {e}")
            return 0, debug_info

    def verify_ingestion(self, filename: str) -> Dict[str, Any]:
        """Verify ingestion was successful"""
        debug_info = {
            "verified": False,
            "timestamp": datetime.now().isoformat()
        }

        try:
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "filename",
                            "match": {"value": filename}
                        }
                    ]
                },
                limit=5,
                with_payload=True
            )

            if search_result and search_result[0]:
                points = search_result[0]
                debug_info["verified"] = True
                debug_info["points_found"] = len(points)

                st.success(
                    f"✓ Verified: Found {len(points)} chunks for '{filename}'"
                )
            else:
                st.warning(f"No documents found for '{filename}'")
                debug_info["error"] = "No documents found"

        except Exception as e:
            debug_info["error"] = str(e)
            st.error(f"Verification failed: {e}")

        return debug_info
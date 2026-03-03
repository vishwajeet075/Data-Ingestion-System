from typing import List, Dict, Any, Tuple
import uuid
from datetime import datetime
import streamlit as st
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import voyageai


class SimpleTextSplitter:
    """Simple text splitter with smart boundary detection - COMPLETE CHUNKS"""

    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 20  # Made configurable
    ) -> List[str]:

        if not text or not text.strip():
            return []

        chunks: List[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size

            if end >= text_length:
                # FIXED: Include final chunk regardless of size
                final_chunk = text[start:].strip()
                if final_chunk:
                    chunks.append(final_chunk)
                break

            split_pos = end

            # Paragraph boundary
            last_paragraph = text.rfind("\n\n", start, end)
            if last_paragraph != -1 and last_paragraph > start + chunk_size // 2:
                split_pos = last_paragraph + 2

            # Sentence boundary
            elif ". " in text[start:end]:
                last_sentence = text.rfind(". ", start, end)
                if last_sentence != -1 and last_sentence > start + chunk_size // 2:
                    split_pos = last_sentence + 2

            # Line break
            elif "\n" in text[start:end]:
                last_newline = text.rfind("\n", start, end)
                if last_newline != -1 and last_newline > start + chunk_size // 2:
                    split_pos = last_newline + 1

            # Word boundary
            elif " " in text[start:end]:
                last_space = text.rfind(" ", start, end)
                if last_space != -1 and last_space > start + chunk_size // 2:
                    split_pos = last_space + 1

            chunk = text[start:split_pos].strip()
            if chunk:
                chunks.append(chunk)

            start = max(start + 1, split_pos - chunk_overlap)

        # FIXED: Only filter out VERY small chunks (configurable), keep most chunks
        if len(chunks) > 1:
            filtered_chunks = []
            dropped_count = 0

            for i, ch in enumerate(chunks):
                # Keep last chunk OR chunks above minimum size
                if i == len(chunks) - 1 or len(ch) >= min_chunk_size:
                    filtered_chunks.append(ch)
                else:
                    dropped_count += 1
                    st.warning(f"⚠️ Dropped very small chunk ({len(ch)} chars): '{ch[:50]}...'")

            if dropped_count > 0:
                st.info(f"ℹ️ Dropped {dropped_count} chunks smaller than {min_chunk_size} chars")

            chunks = filtered_chunks

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
            # Check if collection exists
            self.qdrant_client.get_collection(self.collection_name)
            debug_info["collection_exists"] = True
            st.success(f"✓ Collection '{self.collection_name}' exists")

        except Exception:
            # Collection does not exist → create
            try:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                debug_info["collection_created"] = True
                st.success(f"✓ Created collection '{self.collection_name}'")

            except Exception as e:
                debug_info["error"] = str(e)
                st.error(f"Collection initialization failed: {e}")
                return debug_info

        # ✅ Always ensure payload index exists
        from qdrant_client.http import models

        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="filename",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        except Exception:
            # Index already exists → safe to ignore
            pass

        self._collection_initialized = True
        return debug_info

    def create_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], Dict[str, Any]]:

        debug_info = {
            "total_texts": len(texts),
            "embeddings_created": 0,
            "model": "voyage-2",
            "errors": [],
            "batches_processed": 0
        }

        embeddings: List[List[float]] = []

        try:
            batch_size = min(128, len(texts))
            total_batches = (len(texts) + batch_size - 1) // batch_size

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1

                # Progress indicator for large documents
                if total_batches > 1:
                    st.info(f"Creating embeddings: batch {batch_num}/{total_batches}")

                result = self.voyage_client.embed(
                    texts=batch,
                    model="voyage-2",
                    input_type="document"
                )

                embeddings.extend(result.embeddings)
                debug_info["embeddings_created"] += len(result.embeddings)
                debug_info["batches_processed"] += 1

            st.success(f"✓ Created {len(embeddings)} embeddings in {debug_info['batches_processed']} batches")
            return embeddings, debug_info

        except Exception as e:
            debug_info["errors"].append(str(e))
            st.error(f"Embedding creation failed: {e}")
            return [], debug_info

    def chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_size: int = 20
    ) -> Tuple[List[str], Dict[str, Any]]:

        chunks = self.text_splitter.split_text(
            text,
            chunk_size,
            chunk_overlap,
            min_chunk_size
        )

        debug_info = {
            "original_length": len(text),
            "original_words": len(text.split()),
            "total_chunks": len(chunks),
            "chunk_lengths": [len(c) for c in chunks],
            "avg_chunk_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
            "min_chunk_length": min(len(c) for c in chunks) if chunks else 0,
            "max_chunk_length": max(len(c) for c in chunks) if chunks else 0
        }

        st.info(f"📊 Chunking: {debug_info['original_words']} words → {len(chunks)} chunks "
                f"(avg: {int(debug_info['avg_chunk_length'])} chars)")

        return chunks, debug_info

    def ingest_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        store_full_chunk: bool = True  # NEW: Option to store full chunks
    ) -> Tuple[int, Dict[str, Any]]:

        debug_info = {
            "start_time": datetime.now().isoformat(),
            "success": False,
            "points_uploaded": 0,
            "original_text_length": len(text),
            "original_word_count": len(text.split())
        }

        # Initialize collection
        init_result = self.initialize_collection()
        if init_result.get("error"):
            debug_info["error"] = init_result["error"]
            return 0, debug_info

        # Chunk the text
        st.info(f"📝 Processing {len(text)} characters ({len(text.split())} words)...")
        chunks, chunk_info = self.chunk_text(text, chunk_size, chunk_overlap)

        if not chunks:
            st.error("❌ No chunks created from text")
            debug_info["error"] = "No chunks created"
            return 0, debug_info

        debug_info["chunking"] = chunk_info

        # Create embeddings
        st.info(f"🔄 Creating embeddings for {len(chunks)} chunks...")
        embeddings, embed_info = self.create_embeddings(chunks)

        if not embeddings:
            st.error("❌ Failed to create embeddings")
            debug_info["error"] = "Embedding creation failed"
            return 0, debug_info

        debug_info["embeddings"] = embed_info

        # Verify counts match
        if len(chunks) != len(embeddings):
            st.error(f"❌ Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
            debug_info["error"] = "Chunk/embedding count mismatch"
            return 0, debug_info

        # Create points for Qdrant
        points: List[PointStruct] = []
        total_stored_chars = 0

        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            # CRITICAL FIX: Store FULL chunk text, not truncated
            chunk_text_to_store = chunk if store_full_chunk else chunk[:5000]
            total_stored_chars += len(chunk_text_to_store)

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        **metadata,
                        "chunk_index": idx,
                        "chunk_text": chunk_text_to_store,  # FIXED: Full chunk
                        "chunk_hash": hashlib.md5(chunk.encode()).hexdigest(),
                        "chunk_length": len(chunk),  # Original length
                        "stored_length": len(chunk_text_to_store),  # What we stored
                        "word_count": len(chunk.split()),
                        "timestamp": datetime.now().isoformat()
                    }
                )
            )

        debug_info["total_stored_chars"] = total_stored_chars
        debug_info["storage_efficiency"] = round(
            (total_stored_chars / len(text)) * 100, 2
        ) if len(text) > 0 else 0

        # Upload to Qdrant
        st.info(f"⬆️ Uploading {len(points)} points to Qdrant...")
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
            debug_info["error"] = f"Upload failed: {str(e)}"
            return 0, debug_info

        debug_info["success"] = True
        debug_info["points_uploaded"] = len(points)
        debug_info["end_time"] = datetime.now().isoformat()

        st.success(f"✅ Successfully uploaded {len(points)} chunks "
                f"({debug_info['storage_efficiency']}% of original text)")

        return len(points), debug_info

    def verify_ingestion(self, filename: str,
      expected_chunks: int = None
      ) -> Dict[str, Any]:
        """Enhanced verification with full document stats"""
        debug_info = {
            "verified": False,
            "timestamp": datetime.now().isoformat()
        }
        try:
            # Get collection info
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            total_points = collection_info.points_count

            # Get sample points for this file
            result, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {"key": "filename", "match": {"value": filename}}
                    ]
                },
                limit=100,  # Increased from 5
                with_payload=True
            )

            if result:
                debug_info["verified"] = True
                debug_info["points_found"] = len(result)
                debug_info["total_points_in_collection"] = total_points

                # Calculate stats
                chunk_lengths = [p.payload.get("chunk_length", 0) for p in result]
                stored_lengths = [p.payload.get("stored_length", 0) for p in result]

                debug_info["stats"] = {
                    "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
                    "avg_stored_length": sum(stored_lengths) / len(stored_lengths) if stored_lengths else 0,
                    "total_chars_in_sample": sum(stored_lengths),
                    "min_chunk": min(chunk_lengths) if chunk_lengths else 0,
                    "max_chunk": max(chunk_lengths) if chunk_lengths else 0
                }

                # Check if expected count matches
                if expected_chunks and len(result) != expected_chunks:
                    st.warning(f"⚠️ Expected {expected_chunks} chunks but found {len(result)}")
                    debug_info["count_mismatch"] = True
                else:
                    st.success(f"✓ Verification successful ({len(result)} chunks found in '{filename}')")

                # Show sample chunk
                if result:
                    sample = result[0].payload
                    st.info(f"Sample chunk: {sample.get('chunk_text', '')[:200]}...")

            else:
                debug_info["error"] = f"No points found for file '{filename}'"
                st.error(f"❌ No points found for file '{filename}'")

        except Exception as e:
            debug_info["error"] = str(e)
            st.error(f"Verification failed: {e}")

        return debug_info

    def get_document_stats(self, filename: str) -> Dict[str, Any]:
        """Get complete statistics for an ingested document"""
        stats = {
            "filename": filename,
            "total_chunks": 0,
            "total_characters": 0,
            "total_words": 0,
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Scroll through all points for this file
            all_points = []
            offset = None

            while True:
                result, next_offset = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter={
                        "must": [
                            {"key": "filename", "match": {"value": filename}}
                        ]
                    },
                    limit=100,
                    offset=offset,
                    with_payload=True
                )
    
                all_points.extend(result)
    
                if next_offset is None:
                    break
                offset = next_offset
    
            stats["total_chunks"] = len(all_points)
            stats["total_characters"] = sum(
                p.payload.get("chunk_length", 0) for p in all_points
            )
            stats["total_words"] = sum(
                p.payload.get("word_count", 0) for p in all_points
            )
    
            st.success(f"📊 Document '{filename}': {stats['total_chunks']} chunks, "
                    f"{stats['total_characters']:,} chars, {stats['total_words']:,} words")
    
        except Exception as e:
            stats["error"] = str(e)
            st.error(f"Failed to get stats: {e}")
    
        return stats
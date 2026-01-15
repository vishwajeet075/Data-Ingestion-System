# document_processor.py
import os
import tempfile
from typing import Tuple, Dict, Any
from pathlib import Path
import streamlit as st

# Document processing
import PyPDF2
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.layout import LAParams
import docx
from multimedia_processor import MultimediaProcessor

class DocumentProcessor:
    """Optimized document processing with better text extraction"""

    @staticmethod
    def _read_file_in_chunks(file_path: str, chunk_size: int = 8192) -> bytes:
        """Read file in chunks to handle large files"""
        content = b""
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                content += chunk
        return content

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF using multiple methods"""
        debug_info = {
            "methods_tried": [],
            "methods_success": {},
            "total_pages": 0,
            "total_words": 0,
            "total_chars": 0
        }

        all_text = ""

        # Method 1: pdfplumber (most reliable)
        try:
            with pdfplumber.open(file_path) as pdf:
                debug_info["total_pages"] = len(pdf.pages)
                pages_text = []

                for i, page in enumerate(pdf.pages):
                    # Extract text with multiple strategies
                    page_text = ""

                    # Strategy 1: Standard extraction
                    page_text = page.extract_text()

                    # Strategy 2: Extract with x_tolerance for better spacing
                    if not page_text or len(page_text.strip()) < 10:
                        page_text = page.extract_text(x_tolerance=3, y_tolerance=3)

                    # Strategy 3: Extract by words and join
                    if not page_text or len(page_text.strip()) < 10:
                        words = page.extract_words()
                        if words:
                            page_text = " ".join([word['text'] for word in words])

                    if page_text and len(page_text.strip()) > 5:
                        pages_text.append(page_text)
                        debug_info[f"page_{i+1}_words"] = len(page_text.split())
                        debug_info[f"page_{i+1}_chars"] = len(page_text)

                if pages_text:
                    all_text = "\n".join(pages_text)
                    debug_info["methods_tried"].append("pdfplumber")
                    debug_info["methods_success"]["pdfplumber"] = True
                    debug_info["total_words"] = len(all_text.split())
                    debug_info["total_chars"] = len(all_text)
                    st.success(f"✓ pdfplumber: Extracted {debug_info['total_words']} words from {debug_info['total_pages']} pages")
                    return all_text.strip(), debug_info

        except Exception as e:
            debug_info["methods_success"]["pdfplumber"] = False
            st.warning(f"pdfplumber failed: {str(e)[:100]}")

        # Method 2: PyPDF2 (fallback)
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                debug_info["total_pages"] = len(pdf_reader.pages)

                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                if text.strip():
                    all_text = text
                    debug_info["methods_tried"].append("PyPDF2")
                    debug_info["methods_success"]["PyPDF2"] = True
                    debug_info["total_words"] = len(all_text.split())
                    debug_info["total_chars"] = len(all_text)
                    st.success(f"✓ PyPDF2: Extracted {debug_info['total_words']} words")
                    return all_text.strip(), debug_info

        except Exception as e:
            debug_info["methods_success"]["PyPDF2"] = False
            st.warning(f"PyPDF2 failed: {str(e)[:100]}")

        # Method 3: pdfminer (for complex layouts)
        try:
            # Configure for better extraction
            laparams = LAParams(
                line_margin=0.5,
                word_margin=0.1,
                boxes_flow=0.5
            )

            text = pdfminer_extract(file_path, laparams=laparams)
            if text.strip():
                all_text = text
                debug_info["methods_tried"].append("pdfminer")
                debug_info["methods_success"]["pdfminer"] = True
                debug_info["total_words"] = len(all_text.split())
                debug_info["total_chars"] = len(all_text)
                st.success(f"✓ pdfminer: Extracted {debug_info['total_words']} words")
                return all_text.strip(), debug_info

        except Exception as e:
            debug_info["methods_success"]["pdfminer"] = False
            st.warning(f"pdfminer failed: {str(e)[:100]}")

        st.error("❌ All PDF extraction methods failed")
        return "", debug_info

    @staticmethod
    def extract_text_from_docx(file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract ALL text from DOCX files"""
        debug_info = {"success": False}
        try:
            doc = docx.Document(file_path)

            # Extract from paragraphs
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)

            # Extract from tables
            tables_text = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text)
                    if row_text:
                        tables_text.append(" | ".join(row_text))

            # Extract from headers/footers
            sections_text = []
            for section in doc.sections:
                if section.header:
                    sections_text.append("Header: " + section.header.paragraphs[0].text)
                if section.footer:
                    sections_text.append("Footer: " + section.footer.paragraphs[0].text)

            # Combine all text
            all_text_parts = []
            if paragraphs:
                all_text_parts.append("\n".join(paragraphs))
            if tables_text:
                all_text_parts.append("\n--- Tables ---\n" + "\n".join(tables_text))
            if sections_text:
                all_text_parts.append("\n--- Sections ---\n" + "\n".join(sections_text))

            text = "\n".join(all_text_parts)

            if text.strip():
                debug_info.update({
                    "success": True,
                    "paragraphs": len(paragraphs),
                    "tables": len(doc.tables),
                    "words": len(text.split()),
                    "chars": len(text)
                })
                st.success(f"✓ DOCX: {debug_info['paragraphs']} paragraphs, {debug_info['words']} words")
                return text.strip(), debug_info
            else:
                st.error("No text found in DOCX file")
                return "", debug_info

        except Exception as e:
            st.error(f"DOCX extraction error: {str(e)[:200]}")
            debug_info["error"] = str(e)
            return "", debug_info

    @staticmethod
    def extract_text_from_txt(file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract ALL text from TXT files with robust encoding detection"""
        debug_info = {"success": False}

        # Try multiple encodings in order of likelihood
        encodings = [
            'utf-8',
            'utf-8-sig',  # UTF-8 with BOM
            'latin-1',
            'cp1252',     # Windows
            'iso-8859-1',
            'ascii',
            'utf-16',
            'utf-16le',
            'utf-16be'
        ]

        file_size = os.path.getsize(file_path)
        debug_info["file_size_bytes"] = file_size

        # For very large files, read in chunks
        max_size = 10 * 1024 * 1024  # 10MB limit
        if file_size > max_size:
            st.warning(f"Large file ({file_size:,} bytes). Processing in chunks...")

        for encoding in encodings:
            try:
                # Read entire file if small, chunks if large
                if file_size <= max_size:
                    with open(file_path, 'r', encoding=encoding, errors='strict') as f:
                        text = f.read()
                else:
                    # Read in chunks
                    text = ""
                    with open(file_path, 'r', encoding=encoding, errors='strict') as f:
                        while True:
                            chunk = f.read(1024 * 1024)  # 1MB chunks
                            if not chunk:
                                break
                            text += chunk

                # Verify we got actual text
                if text.strip():
                    # Check if text looks reasonable (not just garbage)
                    if len(text.split()) > 0:
                        debug_info.update({
                            "success": True,
                            "encoding": encoding,
                            "words": len(text.split()),
                            "chars": len(text),
                            "lines": len(text.splitlines())
                        })
                        st.success(f"✓ TXT [{encoding}]: {debug_info['words']} words, {debug_info['lines']} lines")
                        return text.strip(), debug_info

            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.warning(f"Encoding {encoding} failed: {str(e)[:100]}")
                continue

        # Last resort: binary read with ignore errors
        try:
            with open(file_path, 'rb') as f:
                binary_data = f.read()
                text = binary_data.decode('utf-8', errors='ignore')

                if text.strip():
                    debug_info.update({
                        "success": True,
                        "encoding": "utf-8 (forced with ignore errors)",
                        "words": len(text.split()),
                        "chars": len(text),
                        "lines": len(text.splitlines())
                    })
                    st.warning(f"⚠️ TXT [fallback]: Used error ignoring, {debug_info['words']} words")
                    return text.strip(), debug_info

        except Exception as e:
            st.error(f"All TXT extraction methods failed: {str(e)[:200]}")

        debug_info["error"] = "Could not decode with any encoding"
        return "", debug_info

    @classmethod
    def process_file(cls, file_path: str, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Process file based on its extension with progress tracking"""
        file_ext = filename.lower().split('.')[-1]

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text(f"Processing {filename}...")
            progress_bar.progress(20)

            text = ""
            debug_info = {}

            if file_ext == 'pdf':
                text, debug_info = cls.extract_text_from_pdf(file_path)
            elif file_ext == 'docx':
                text, debug_info = cls.extract_text_from_docx(file_path)
            elif file_ext == 'txt':
                text, debug_info = cls.extract_text_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            progress_bar.progress(80)

            if text:
                # Verify extraction quality
                word_count = len(text.split())
                char_count = len(text)

                if word_count == 0:
                    st.error(f"No words extracted from {filename}")
                elif word_count < 10:
                    st.warning(f"Only {word_count} words extracted from {filename}")
                else:
                    st.success(f"Extracted {word_count:,} words, {char_count:,} characters")

            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

            return text.strip(), debug_info

        except Exception as e:
            status_text.empty()
            progress_bar.empty()
            st.error(f"Error processing {filename}: {str(e)[:200]}")
            return "", {"error": str(e), "success": False}
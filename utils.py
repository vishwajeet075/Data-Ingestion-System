# utils.py
import json
import hashlib
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import streamlit as st
import time

def display_debug_info(debug_data: Dict[str, Any], title: str = "Debug Information"):
    """Display debug information in a structured way"""
    with st.expander(f"🔍 {title}", expanded=False):
        st.json(debug_data)

def create_metadata(filename: str, file_type: str, file_size: int,
                    extraction_method: str) -> Dict[str, Any]:
    """Create standardized metadata for documents"""
    return {
        "filename": filename,
        "file_type": file_type,
        "file_size": file_size,
        "extraction_method": extraction_method,
        "upload_timestamp": datetime.now().isoformat(),
        "processing_timestamp": datetime.now().isoformat(),
        "unique_id": f"{filename}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    }

def safe_json_dumps(data: Any) -> str:
    """Safely convert data to JSON string"""
    def default_serializer(obj):
        if isinstance(obj, (datetime)):
            return obj.isoformat()
        return str(obj)

    return json.dumps(data, indent=2, default=default_serializer)
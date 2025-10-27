"""Utility functions for data ingestion"""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any


def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def get_file_hash(file_path: Path) -> str:
    """Generate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def clean_text(text: str) -> str:
    """Clean extracted text"""
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove special characters that might cause issues
    text = text.replace("\x00", "")
    
    return text.strip()


def create_metadata(
    file_name: str,
    file_hash: str,
    chunk_index: int,
    total_chunks: int,
    **kwargs
) -> Dict[str, Any]:
    """Create metadata dictionary for a chunk"""
    metadata = {
        "file_name": file_name,
        "file_hash": file_hash,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    metadata.update(kwargs)
    return metadata


def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split a list into batches"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

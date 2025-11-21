"""Utility functions for data ingestion"""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any

from .constants import MAX_FILENAME_LENGTH


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration with standardized format.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
               Defaults to INFO level.
    
    Returns:
        Logger instance configured with the specified level.
    """
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
    **kwargs
) -> Dict[str, Any]:
    """Create metadata dictionary for a chunk"""
    metadata = {
        "file_name": file_name,
        "file_hash": file_hash,
        "chunk_index": chunk_index,
    }
    metadata.update(kwargs)
    return metadata


def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split a list into batches"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def sanitize_filename(filename: str) -> str:
    """Sanitize filename and prevent directory traversal attacks
    
    Args:
        filename: Input filename to sanitize
        
    Returns:
        Safe filename with invalid characters replaced and path separators removed
    """
    import os
    
    # Remove path separators to prevent directory traversal
    filename = os.path.basename(filename)
    
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Prevent hidden files and special names
    if filename.startswith('.'):
        filename = '_' + filename[1:]
    
    # Limit length to filesystem maximum
    if len(filename) > MAX_FILENAME_LENGTH:
        name, ext = os.path.splitext(filename)
        filename = name[:MAX_FILENAME_LENGTH - len(ext)] + ext
    
    return filename

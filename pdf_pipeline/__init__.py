"""PDF Pipeline Package

This package provides a complete pipeline for ingesting PDF documents from Google Cloud Storage,
processing them using RAG-Anything with Google's Generative AI, and storing embeddings in Milvus.
"""

__version__ = "0.1.0"

from .config import Config
from .gcs_downloader import GCSDownloader, download_pdfs_from_gcs
from .rag_processor import RAGProcessor
from .vector_store import MilvusVectorStore
from .utils import sanitize_filename

__all__ = [
    "Config",
    "GCSDownloader",
    "download_pdfs_from_gcs",
    "RAGProcessor",
    "MilvusVectorStore",
    "sanitize_filename",
]

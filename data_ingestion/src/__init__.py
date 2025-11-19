"""Data Ingestion Package

This package provides tools for ingesting PDF documents from Google Cloud Storage,
extracting text using MinerU, generating embeddings with Gemini API, and storing
them in Milvus vector database for semantic search.
"""

__version__ = "1.0.0"

from .config import Config
from .chunker import TextChunker
from .embedder import TextEmbedder
from .vector_store import MilvusVectorStore
from .pdf_extractor import PDFExtractor
from .gcs_downloader import GCSDownloader
from .exceptions import (
    IngestionError,
    PDFExtractionError,
    EmbeddingError,
    VectorStoreError,
    ConfigurationError,
    GCSError,
    ValidationError
)

__all__ = [
    # Main classes
    'Config',
    'TextChunker',
    'TextEmbedder',
    'MilvusVectorStore',
    'PDFExtractor',
    'GCSDownloader',
    # Exceptions
    'IngestionError',
    'PDFExtractionError',
    'EmbeddingError',
    'VectorStoreError',
    'ConfigurationError',
    'GCSError',
    'ValidationError',
    # Version
    '__version__',
]

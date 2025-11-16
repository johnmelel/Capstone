"""Type definitions for the data ingestion pipeline"""

from typing import TypedDict


class PDFMetadata(TypedDict):
    """Metadata extracted from a PDF document"""
    title: str
    author: str
    subject: str
    creator: str
    producer: str
    creation_date: str
    modification_date: str
    num_pages: int
    file_name: str


class PDFExtractionResult(TypedDict):
    """Result of PDF text extraction with metadata"""
    text: str
    metadata: PDFMetadata


class ChunkMetadata(TypedDict):
    """Metadata for a text chunk"""
    file_name: str
    file_hash: str
    chunk_index: int
    total_chunks: int


class SearchResult(TypedDict):
    """Result from vector store search"""
    id: str
    distance: float
    entity: dict

"""Type definitions for the data ingestion pipeline"""

from typing import TypedDict, List, Optional
from pathlib import Path


class ImageData(TypedDict):
    """Data for an extracted image"""
    path: Path  # Local temporary file path
    bytes: bytes  # Raw image data
    page_num: int  # Page number where image appears
    image_index: int  # Index of image on the page
    bbox: Optional[dict]  # Bounding box {x0, y0, x1, y1}
    size: tuple  # (width, height) in pixels
    gcs_path: Optional[str]  # GCS URI after upload


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


class PageData(TypedDict):
    """Data for a single page"""
    text: str
    page_num: int


class PDFExtractionResult(TypedDict):
    """Result of PDF text extraction with metadata"""
    text: str  # Full text (concatenated)
    pages: List[PageData]  # Text per page
    metadata: PDFMetadata


class MultimodalPDFExtractionResult(TypedDict):
    """Result of PDF extraction with text, images, and metadata"""
    text: str  # Full text (concatenated)
    pages: List[PageData]  # Text per page
    images: List[ImageData]
    metadata: PDFMetadata


class ChunkMetadata(TypedDict):
    """Metadata for a text chunk"""
    file_name: str
    file_hash: str
    chunk_index: int
    total_chunks: int


class MultimodalChunkMetadata(TypedDict):
    """Metadata for a multimodal chunk with images"""
    file_name: str
    file_hash: str
    chunk_index: int
    total_chunks: int
    has_image: bool
    image_count: int
    embedding_type: str  # "text", "image", "multimodal"
    image_gcs_paths: List[str]  # List of GCS URIs
    image_metadata: str  # JSON string with image details


class SearchResult(TypedDict):
    """Result from vector store search"""
    id: str
    distance: float
    entity: dict

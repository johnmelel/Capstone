"""Custom exceptions for data ingestion pipeline"""


class IngestionError(Exception):
    """Base exception for ingestion pipeline errors"""
    pass


class PDFExtractionError(IngestionError):
    """Error during PDF text extraction"""
    pass


class EmbeddingError(IngestionError):
    """Error during embedding generation"""
    pass


class VectorStoreError(IngestionError):
    """Error during vector store operations"""
    pass


class ConfigurationError(IngestionError):
    """Invalid configuration"""
    pass


class GCSError(IngestionError):
    """Google Cloud Storage related errors"""
    pass


class ValidationError(IngestionError):
    """Input validation error"""
    pass

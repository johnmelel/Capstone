import os
from pathlib import Path

class Config:
    # Google Cloud Storage
    GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "path/to/your/service-account.json")
    GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "your-gcs-bucket-name")
    GCS_BUCKET_PREFIX = os.environ.get("GCS_BUCKET_PREFIX", "your-prefix/")
    DOWNLOAD_DIR = Path(os.environ.get("DOWNLOAD_DIR", "/tmp/pdf_downloads"))
    GCS_RECURSIVE = os.environ.get("GCS_RECURSIVE", "True").lower() in ("true", "1", "t")

    # Milvus
    MILVUS_URI = os.environ.get("MILVUS_URI", "your-milvus-uri")
    MILVUS_API_KEY = os.environ.get("MILVUS_API_KEY", "your-milvus-api-key")
    MILVUS_COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION_NAME", "pdf_embeddings")

    # Google Generative AI
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "your-google-api-key")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/embedding-001")
    LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-1.5-pro")

    # RAG-Anything
    MINERU_DEVICE = os.environ.get("MINERU_DEVICE", "cuda") # or "cpu"

    # Logging
    LOG_FILE = os.environ.get("LOG_FILE", "pipeline.log")
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

    # Intermediate data path
    PROCESSED_DATA_DIR = Path(os.environ.get("PROCESSED_DATA_DIR", "/tmp/processed_data"))

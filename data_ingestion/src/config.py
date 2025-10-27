"""Configuration module for data ingestion app"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the data ingestion app"""
    
    # Google Cloud Storage Configuration
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    GCS_BUCKET_PREFIX = os.getenv("GCS_BUCKET_PREFIX", "")  # Folder path in bucket (optional)
    GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")
    GCS_RECURSIVE = os.getenv("GCS_RECURSIVE", "true").lower() in ("true", "1", "yes")
    
    # Milvus Configuration
    MILVUS_URI = os.getenv("MILVUS_URI")
    MILVUS_API_KEY = os.getenv("MILVUS_API_KEY")
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "pdf_embeddings")
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    
    # Local Storage (removed - no longer needed for stream processing)
    # DOWNLOAD_DIR = Path(os.getenv("DOWNLOAD_DIR", "./downloads"))
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required_fields = [
            ("GCS_BUCKET_NAME", cls.GCS_BUCKET_NAME),
            ("MILVUS_URI", cls.MILVUS_URI),
            ("MILVUS_API_KEY", cls.MILVUS_API_KEY),
        ]
        
        missing_fields = [field for field, value in required_fields if not value]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")
        
        # Check if service account file exists
        if not Path(cls.GOOGLE_SERVICE_ACCOUNT_JSON).exists():
            raise FileNotFoundError(
                f"Service account JSON file not found: {cls.GOOGLE_SERVICE_ACCOUNT_JSON}"
            )
        
        return True

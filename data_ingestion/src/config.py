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
    
    # Gemini API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "2048"))
    
    # Gemini Vision Configuration
    GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash-exp")
    IMAGE_DESCRIPTION_PROMPT = os.getenv(
        "IMAGE_DESCRIPTION_PROMPT",
        "Describe this image in detail. Focus on medical/scientific content, "
        "figures, charts, diagrams, or any text visible in the image. "
        "Be concise but comprehensive."
    )
    
    # Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1792"))  # Characters per chunk
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))  # Overlap between chunks
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    
    # Output directories
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./extracted_content"))
    UPLOAD_OUTPUT_TO_GCS = os.getenv("UPLOAD_OUTPUT_TO_GCS", "true").lower() in ("true", "1", "yes")
    GCS_OUTPUT_PREFIX = os.getenv("GCS_OUTPUT_PREFIX", "extracted_content")  # Folder in bucket for outputs
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required_fields = [
            ("GCS_BUCKET_NAME", cls.GCS_BUCKET_NAME),
            ("MILVUS_URI", cls.MILVUS_URI),
            ("MILVUS_API_KEY", cls.MILVUS_API_KEY),
            ("GEMINI_API_KEY", cls.GEMINI_API_KEY),
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

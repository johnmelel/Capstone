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
    GCS_IMAGES_PREFIX = os.getenv("GCS_IMAGES_PREFIX", "images")  # Folder for extracted images
    GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")
    GCS_RECURSIVE = os.getenv("GCS_RECURSIVE", "true").lower() in ("true", "1", "yes")
    
    # Milvus Configuration
    MILVUS_URI = os.getenv("MILVUS_URI")
    MILVUS_API_KEY = os.getenv("MILVUS_API_KEY")
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "pdf_embeddings")
    MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "COSINE").upper()
    
    # Embedding Configuration
    # Choose between "gemini" or "huggingface" embedding backend
    EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "huggingface")  # "gemini" or "huggingface"
    
    # Gemini Embedding Configuration (Legacy - for backward compatibility)
    # The new google-genai SDK supports models like:
    # - gemini-embedding-001 (768 dimensions)
    # - text-embedding-004 (768 dimensions)
    # - text-embedding-005 (768 dimensions)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    
    # HuggingFace Embedding Configuration
    # BiomedCLIP model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    # Outputs 512-dimensional embeddings optimized for biomedical text
    EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8000")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "512"))  # 512 for BiomedCLIP, 768 for Gemini
    MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "2048"))
    
    # Processing Configuration (TOKEN-BASED, not characters)
    # CHUNK_SIZE: Target tokens per chunk (default 1792 leaves 256 token buffer)
    # CHUNK_OVERLAP: Tokens to overlap between chunks (default 64 tokens ≈ 16 words)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1792"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    
    # Multimodal Configuration
    ENABLE_MULTIMODAL = os.getenv("ENABLE_MULTIMODAL", "false").lower() in ("true", "1", "yes")
    IMAGE_FORMAT = os.getenv("IMAGE_FORMAT", "PNG").upper()  # PNG or JPEG
    IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "95"))  # For JPEG compression
    MAX_IMAGE_SIZE_KB = int(os.getenv("MAX_IMAGE_SIZE_KB", "1000"))  # Max size before compression
    EMBEDDING_FUSION_METHOD = os.getenv("EMBEDDING_FUSION_METHOD", "average")  # How to fuse text + image embeddings
    CLEANUP_TEMP_IMAGES = os.getenv("CLEANUP_TEMP_IMAGES", "true").lower() in ("true", "1", "yes")
    
    # MinerU PDF Extraction Configuration
    MINERU_BACKEND = os.getenv("MINERU_BACKEND", "pipeline")  # pipeline, vlm-transformers, vlm-vllm-engine
    MINERU_MODEL_SOURCE = os.getenv("MINERU_MODEL_SOURCE", "huggingface")  # huggingface, modelscope, local
    MINERU_LANG = os.getenv("MINERU_LANG", "en")  # en, ch, auto, etc.
    PDF_EXTRACTION_TIMEOUT = int(os.getenv("PDF_EXTRACTION_TIMEOUT", "3600"))  # 1 hour default
    MINERU_DEBUG_MODE = os.getenv("MINERU_DEBUG_MODE", "false").lower() in ("true", "1", "yes")
    MINERU_ENABLE_TABLES = os.getenv("MINERU_ENABLE_TABLES", "false").lower() in ("true", "1", "yes")
    MINERU_ENABLE_FORMULAS = os.getenv("MINERU_ENABLE_FORMULAS", "false").lower() in ("true", "1", "yes")
    
    # Local Storage (removed - no longer needed for stream processing)
    # DOWNLOAD_DIR = Path(os.getenv("DOWNLOAD_DIR", "./downloads"))

    # Retrieval Test Configuration
    DEFAULT_RETRIEVAL_PROMPT = os.getenv("DEFAULT_RETRIEVAL_PROMPT", "What is the future of AI?")
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required_fields = [
            ("GCS_BUCKET_NAME", cls.GCS_BUCKET_NAME),
            ("MILVUS_URI", cls.MILVUS_URI),
            ("MILVUS_API_KEY", cls.MILVUS_API_KEY),
        ]
        
        # Add backend-specific validation
        if cls.EMBEDDING_BACKEND == "gemini":
            required_fields.append(("GEMINI_API_KEY", cls.GEMINI_API_KEY))
        elif cls.EMBEDDING_BACKEND == "huggingface":
            required_fields.append(("EMBEDDING_SERVICE_URL", cls.EMBEDDING_SERVICE_URL))
        else:
            raise ValueError(
                f"Invalid EMBEDDING_BACKEND: {cls.EMBEDDING_BACKEND}. "
                f"Must be 'gemini' or 'huggingface'"
            )
        
        missing_fields = [field for field, value in required_fields if not value]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")
        
        # Check if service account file exists
        if not Path(cls.GOOGLE_SERVICE_ACCOUNT_JSON).exists():
            raise FileNotFoundError(
                f"Service account JSON file not found: {cls.GOOGLE_SERVICE_ACCOUNT_JSON}"
            )
        
        return True

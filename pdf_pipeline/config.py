import os
from pathlib import Path

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip
    pass


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


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

    @classmethod
    def validate(cls):
        """Validate that all required configuration values are set.
        
        Raises:
            ConfigError: If any required configuration is missing or invalid
        """
        errors = []
        
        # Check Google Cloud Storage settings
        if cls.GOOGLE_SERVICE_ACCOUNT_JSON == "path/to/your/service-account.json":
            errors.append("GOOGLE_SERVICE_ACCOUNT_JSON is not set. Please set this environment variable.")
        elif not Path(cls.GOOGLE_SERVICE_ACCOUNT_JSON).exists():
            errors.append(f"Service account file not found: {cls.GOOGLE_SERVICE_ACCOUNT_JSON}")
        
        if cls.GCS_BUCKET_NAME == "your-gcs-bucket-name":
            errors.append("GCS_BUCKET_NAME is not set. Please set this environment variable.")
        
        # Check Milvus settings
        if cls.MILVUS_URI == "your-milvus-uri":
            errors.append("MILVUS_URI is not set. Please set this environment variable.")
        
        if cls.MILVUS_API_KEY == "your-milvus-api-key":
            errors.append("MILVUS_API_KEY is not set. Please set this environment variable.")
        
        # Check Google API settings
        if cls.GOOGLE_API_KEY == "your-google-api-key":
            errors.append("GOOGLE_API_KEY is not set. Please set this environment variable.")
        
        # Check device setting
        if cls.MINERU_DEVICE not in ["cuda", "cpu", "mps"]:
            errors.append(f"MINERU_DEVICE must be 'cuda', 'cpu', or 'mps', got: {cls.MINERU_DEVICE}")
        
        # Check log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if cls.LOG_LEVEL.upper() not in valid_log_levels:
            errors.append(f"LOG_LEVEL must be one of {valid_log_levels}, got: {cls.LOG_LEVEL}")
        
        if errors:
            error_message = "\n❌ Configuration Errors:\n" + "\n".join(f"  • {error}" for error in errors)
            error_message += "\n\n💡 Please create a .env file or set environment variables. See .env.example for reference."
            raise ConfigError(error_message)
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration (masking sensitive values)."""
        print("\n📋 Current Configuration:")
        print(f"  GCS Bucket: {cls.GCS_BUCKET_NAME}")
        print(f"  GCS Prefix: {cls.GCS_BUCKET_PREFIX}")
        print(f"  Download Dir: {cls.DOWNLOAD_DIR}")
        print(f"  Processed Data Dir: {cls.PROCESSED_DATA_DIR}")
        print(f"  Milvus URI: {cls.MILVUS_URI}")
        print(f"  Milvus Collection: {cls.MILVUS_COLLECTION_NAME}")
        print(f"  Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"  LLM Model: {cls.LLM_MODEL}")
        print(f"  Device: {cls.MINERU_DEVICE}")
        print(f"  Log Level: {cls.LOG_LEVEL}")
        print(f"  Log File: {cls.LOG_FILE}")
        
        # Mask sensitive keys
        if cls.GOOGLE_API_KEY != "your-google-api-key":
            print(f"  Google API Key: {'*' * 8}{cls.GOOGLE_API_KEY[-4:]}")
        if cls.MILVUS_API_KEY != "your-milvus-api-key":
            print(f"  Milvus API Key: {'*' * 8}{cls.MILVUS_API_KEY[-4:]}")
        print()

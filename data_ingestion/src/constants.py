"""Constants used throughout the data ingestion pipeline"""

# Token estimation
CHARS_PER_TOKEN_ESTIMATE = 4  # Rough estimation: 1 token ≈ 4 characters for English text

# Rate limiting
EMBEDDING_RATE_LIMIT_DELAY_SECONDS = 0.1  # Delay between embedding API requests
EMBEDDING_BATCH_DELAY_SECONDS = 1.0  # Delay between batches

# File size limits
MAX_PDF_SIZE_BYTES = 100 * 1024 * 1024  # 100MB maximum PDF file size

# Filename safety
MAX_FILENAME_LENGTH = 255  # Maximum filename length for most filesystems

# MD5 hash validation
MD5_HASH_LENGTH = 32  # Length of MD5 hash in hex format
MD5_HASH_PATTERN = r'^[a-fA-F0-9]{32}$'  # Regex pattern for MD5 hash validation

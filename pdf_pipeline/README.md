# PDF Ingestion Pipeline

This pipeline ingests PDF documents from a Google Cloud Storage (GCS) bucket, processes them using the `rag-anything` library with Google's Generative AI, and stores the resulting embeddings in a Milvus vector database.

## Features

- ✅ Downloads PDFs from Google Cloud Storage
- ✅ Processes PDFs using RAG-Anything with MinerU
- ✅ Generates embeddings using Google's Generative AI (embedding-001)
- ✅ Stores embeddings in Milvus vector database
- ✅ Robust error handling and logging
- ✅ Configuration validation
- ✅ Resume capability for failed files

## Setup

### 1. Install Dependencies

All dependencies including MinerU are installed together using `uv` (a fast Python package installer).

```bash
# Install uv
pip install --upgrade pip
pip install uv

# Install all dependencies from requirements.txt
uv pip install -r requirements.txt
```

This installs:
- `raganything` - Core RAG library
- `mineru[core]` - High-quality PDF parsing
- `google-cloud-storage`, `pymilvus`, `google-generativeai`
- All other required packages

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

### 2. Configure Environment Variables

Create a `.env` file in the `pdf_pipeline` directory. You can copy the example file:

```bash
cp .env.example .env
```

Then edit `.env` and add your actual values:

```env
# Google Cloud Storage
GOOGLE_SERVICE_ACCOUNT_JSON="path/to/your/service-account.json"
GCS_BUCKET_NAME="your-gcs-bucket-name"
GCS_BUCKET_PREFIX="documents/"  # Optional prefix/folder

# Milvus Vector Database
MILVUS_URI="https://your-cluster.milvus.io:19530"
MILVUS_API_KEY="your-milvus-api-key"
MILVUS_COLLECTION_NAME="pdf_embeddings"

# Google Generative AI
GOOGLE_API_KEY="your-google-api-key"
EMBEDDING_MODEL="models/embedding-001"
LLM_MODEL="gemini-1.5-pro"
EMBEDDING_DIM="3072"

# RAG Processing Options
MINERU_DEVICE="cuda"  # or "cpu" or "mps"
RAG_CLEAR_CACHE="True"  # Clear cache for fresh processing
LOG_LEVEL="INFO"
```

### 3. Verify Configuration

The pipeline will automatically validate your configuration on startup and provide helpful error messages if anything is missing or invalid.

## Usage

To run the entire pipeline, execute the `main.py` script from the `pdf_pipeline` directory:

```bash
cd pdf_pipeline
python main.py
```

The script will:

1. **Validate Configuration**: Check MinerU installation, API keys, and settings
2. **Clear Cache** (if enabled): Remove stale RAGAnything cache for fresh processing
3. **Download PDFs**: Download all PDF files from the configured GCS bucket
4. **Process Documents**: Process each PDF using RAGAnything + MinerU for high-quality extraction
5. **Generate Embeddings**: Generate embeddings using Google's Generative AI
6. **Store in Milvus**: Store the embeddings and metadata in the Milvus vector database

### Pipeline Flow

```
Configuration Validation → Cache Cleanup → GCS Download → RAG+MinerU Processing → Embedding Generation → Milvus Storage
```

### Important Notes

- **No Fallback Methods**: The pipeline will fail loudly if RAGAnything/MinerU cannot process a document. This ensures only high-quality data is inserted.
- **Cache Clearing**: By default, the cache is cleared before processing to ensure fresh results. Set `RAG_CLEAR_CACHE=False` to reuse cache.
- **Error Handling**: Failed documents are logged but don't stop the pipeline. Check logs for details.

### Output

- Logs are printed to console and saved to `pipeline.log` (configurable)
- Processed data is temporarily stored in parquet files
- Final embeddings are stored in Milvus collection

## Configuration Options

All configuration is done via environment variables. See `.env.example` for all available options:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Path to GCS service account JSON | Required |
| `GCS_BUCKET_NAME` | GCS bucket name | Required |
| `GCS_BUCKET_PREFIX` | Prefix/folder in bucket | `""` |
| `GCS_RECURSIVE` | Download from subfolders | `True` |
| `DOWNLOAD_DIR` | Local download directory | `/tmp/pdf_downloads` |
| `MILVUS_URI` | Milvus server URI | Required |
| `MILVUS_API_KEY` | Milvus API key | Required |
| `MILVUS_COLLECTION_NAME` | Collection name | `pdf_embeddings` |
| `GOOGLE_API_KEY` | Google API key | Required |
| `EMBEDDING_MODEL` | Embedding model | `models/embedding-001` |
| `EMBEDDING_DIM` | Embedding dimensions | `3072` |
| `LLM_MODEL` | LLM model | `gemini-1.5-pro` |
| `MINERU_DEVICE` | Processing device | `cuda` |
| `RAG_CLEAR_CACHE` | Clear cache before processing | `True` |
| `PROCESSED_DATA_DIR` | Temp data directory | `/tmp/processed_data` |
| `LOG_FILE` | Log file path | `pipeline.log` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Troubleshooting

### Installation Issues

If `uv pip install -r requirements.txt` fails:

1. Ensure `uv` is installed: `pip install uv`
2. Try with verbose output: `uv pip install -r requirements.txt -v`
3. Check Python version (3.8+ required)
4. See [INSTALLATION.md](INSTALLATION.md) for detailed troubleshooting

### MinerU Installation Issues

If MinerU installation fails specifically:

1. Verify correct syntax in requirements.txt: `mineru[core]`
2. Try installing manually: `uv pip install "mineru[core]"`
3. Ensure Python development headers are installed

### Configuration Errors

If you see configuration validation errors on startup:

1. Check that all required environment variables are set in your `.env` file
2. Verify that file paths (service account JSON) are correct and files exist
3. Ensure API keys are valid
4. Verify MinerU is installed: `python -c "import mineru; print('MinerU OK')"`

### Import Errors

If you encounter import errors:

```bash
uv pip install -r requirements.txt --force-reinstall
```

### CUDA/GPU Issues

If you don't have CUDA available, set:

```env
MINERU_DEVICE="cpu"
```

Note: CPU processing will be significantly slower than GPU.

### Processing Failures

The pipeline now **fails loudly** instead of using low-quality fallbacks:

- If a document cannot be processed by RAGAnything/MinerU, it will be skipped with an error
- Check `pipeline.log` for detailed error messages
- Ensure your PDFs are valid and not corrupted
- For persistent issues, try clearing cache: `RAG_CLEAR_CACHE=True`

### Cache Issues

If documents aren't being processed fresh:

1. Set `RAG_CLEAR_CACHE=True` in your `.env` file
2. Manually delete the `rag_storage` directory
3. Restart the pipeline

## Architecture

```
pdf_pipeline/
├── __init__.py           # Package initialization
├── main.py               # Main pipeline orchestrator
├── config.py             # Configuration management
├── gcs_downloader.py     # GCS download functionality
├── rag_processor.py      # RAG-Anything processing
├── vector_store.py       # Milvus vector store operations
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment configuration
└── README.md            # This file
```

## Notes

- **High-Quality Processing**: Uses MinerU for superior PDF parsing (better than PyMuPDF)
- **No Fallback Methods**: Only high-quality data is inserted; low-quality fallbacks have been removed
- **Cache Management**: Cache is cleared by default to ensure fresh processing
- The pipeline skips files that have already been downloaded (based on filename)
- Embeddings use Google's `embedding-001` model (3072 dimensions)
- Milvus collection is created automatically on first run
- Primary keys are generated deterministically based on file hash and chunk index
- Failed documents are logged but don't stop the entire pipeline

## Recent Updates

### Critical Fixes (v2.0)

1. **Removed Fallback Methods**: Eliminated PyMuPDF and raw byte reading fallbacks that were inserting garbage data
2. **Fixed Entity Extraction**: Corrected RAGAnything chunk extraction logic to avoid loading cache files as content
3. **Added Cache Cleanup**: Implemented automatic cache clearing to prevent stale data reuse
4. **Proper MinerU Integration**: Added MinerU validation and installation checks

These fixes ensure the pipeline only processes and inserts high-quality data.

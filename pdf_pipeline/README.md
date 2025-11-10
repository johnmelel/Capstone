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

```bash
pip install -r requirements.txt
```

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

# Processing Options
MINERU_DEVICE="cuda"  # or "cpu" or "mps"
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

1. **Download PDFs**: Download all PDF files from the configured GCS bucket
2. **Process Documents**: Process each PDF using RAG-Anything to extract text, tables, and other content
3. **Generate Embeddings**: Generate embeddings using Google's Generative AI
4. **Store in Milvus**: Store the embeddings and metadata in the Milvus vector database

### Pipeline Flow

```
GCS Bucket → Download → RAG Processing → Embedding Generation → Milvus Storage
```

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
| `LLM_MODEL` | LLM model | `gemini-1.5-pro` |
| `MINERU_DEVICE` | Processing device | `cuda` |
| `PROCESSED_DATA_DIR` | Temp data directory | `/tmp/processed_data` |
| `LOG_FILE` | Log file path | `pipeline.log` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Troubleshooting

### Configuration Errors

If you see configuration validation errors on startup:

1. Check that all required environment variables are set in your `.env` file
2. Verify that file paths (service account JSON) are correct and files exist
3. Ensure API keys are valid

### Import Errors

If you encounter import errors with `raganything`, `google-generativeai`, or `pymilvus`:

```bash
pip install -r requirements.txt --upgrade
```

### CUDA/GPU Issues

If you don't have CUDA available, set:

```env
MINERU_DEVICE="cpu"
```

### Processing Failures

- The pipeline will continue processing other files if one fails
- Check `pipeline.log` for detailed error messages
- Failed files can be reprocessed by running the pipeline again

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

- The pipeline skips files that have already been downloaded (based on filename)
- Embeddings use Google's `embedding-001` model (768 dimensions)
- Milvus collection is created automatically on first run
- Primary keys are generated deterministically based on file hash and chunk index

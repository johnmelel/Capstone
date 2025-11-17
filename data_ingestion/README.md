# Data Ingestion App

A Python application for processing PDFs from Google Cloud Storage, extracting text, generating embeddings, and storing them in Milvus vector database.

## Features

- ☁️ **Direct GCS Processing** - Process PDFs directly from Google Cloud Storage without local downloads
- 📁 **Recursive folder support** - Process from all subfolders/prefixes automatically
- 📄 **Advanced PDF Parsing** - Extract text from PDFs using MinerU (GPU/CPU compatible, OCR support)
- ✂️ Split text into token-aware chunks (max 2048 tokens)
- 🧠 **Flexible Embeddings** - Choose between:
  - **BiomedCLIP** (HuggingFace) - Optimized for medical/scientific text (512 dimensions)
  - **Gemini API** - General-purpose embeddings (768 dimensions)
- 🗄️ Store embeddings in Milvus vector database
- 🚀 **GPU Acceleration** - Optional GPU support for faster PDF processing
- 🐳 Docker support for easy deployment
- ✅ Comprehensive test suite

## Project Structure

```
data_ingestion/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── gcs_downloader.py      # Google Cloud Storage client (for listing blobs)
│   ├── pdf_extractor.py       # PDF text extraction
│   ├── chunker.py             # Text chunking
│   ├── embedder.py            # Gemini embedding client (legacy)
│   ├── hf_embedder.py         # HuggingFace embedding client
│   ├── vector_store.py        # Milvus vector store interface
│   ├── ingest.py              # Main ingestion pipeline
│   └── utils.py               # Utility functions
├── embedding_service/          # Standalone embedding service
│   ├── app.py                 # FastAPI service
│   ├── requirements.txt       # Service dependencies
│   ├── Dockerfile             # Service container
│   ├── start_service.sh       # Linux/Mac startup script
│   └── start_service.bat      # Windows startup script
├── tests/                      # Test suite
├── .env                        # Environment variables (create from .env.example)
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Container orchestration
├── QUICKSTART_EMBEDDINGS.md   # Quick start guide for embeddings
├── EMBEDDING_MIGRATION_GUIDE.md # Detailed migration guide
└── README.md                  # This file
```

## Prerequisites

- Python 3.11+
- Google Cloud service account with Storage access
- Milvus instance (cloud or self-hosted)
- Docker (optional, for containerized deployment)

## Embedding Options

This project supports two embedding backends:

### Option 1: BiomedCLIP (Recommended for Medical/Scientific Text)

**Advantages:**
- ✅ Optimized for biomedical and scientific text
- ✅ No API costs (self-hosted)
- ✅ Full data privacy
- ✅ Works offline
- ✅ 512 dimensions

**Quick Start:**
```bash
# Start embedding service
cd embedding_service
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000

# Configure in .env
EMBEDDING_BACKEND=huggingface
EMBEDDING_SERVICE_URL=http://localhost:8000
EMBEDDING_DIMENSION=512
```

📖 See [QUICKSTART_EMBEDDINGS.md](QUICKSTART_EMBEDDINGS.md) for detailed setup.

### Option 2: Google Gemini API (Legacy)

**Advantages:**
- ✅ No local infrastructure needed
- ✅ Always up-to-date models
- ✅ 768 dimensions

**Setup:**
```bash
# Configure in .env
EMBEDDING_BACKEND=gemini
GEMINI_API_KEY=your_api_key
EMBEDDING_DIMENSION=768
```

📖 See [GEMINI_SETUP.md](GEMINI_SETUP.md) for API key setup.

## Setup

### 1. Clone and Navigate

```bash
cd data_ingestion
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: MinerU is a powerful PDF parsing library that supports both CPU and GPU processing. For detailed installation instructions including GPU setup, see [MINERU_SETUP.md](MINERU_SETUP.md).

**Quick GPU Check** (optional):
```bash
# Check if GPU is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 4. Google Cloud Storage Setup

#### Create a Service Account:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Cloud Storage API:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Cloud Storage API"
   - Click "Enable"
4. Create a service account:
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Fill in the details and click "Create"
   - Grant the "Storage Object Viewer" role (or "Storage Admin" if you need write access)
   - Click "Continue" and then "Done"
5. Download the JSON key:
   - Click on the created service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose JSON format
   - Save the file as `service_account.json` in the project root

#### Grant Bucket Access:

1. Go to [Cloud Storage](https://console.cloud.google.com/storage/browser)
2. Find your bucket or create a new one
3. Click on the bucket name
4. Go to the "Permissions" tab
5. Click "Grant Access"
6. Add the service account email (found in the JSON file) with "Storage Object Viewer" role
7. Save the bucket name (e.g., `my-pdf-bucket`)
8. (Optional) Note the folder/prefix path if your PDFs are in a subfolder (e.g., `documents/pdfs`)

**Note**: The app will automatically download from all subfolders/prefixes by default. You can disable this by setting `GCS_RECURSIVE=false` in your `.env` file.

### 5. Milvus Setup

#### Option A: Use Milvus Cloud (Zilliz)

1. Sign up at [Zilliz Cloud](https://cloud.zilliz.com/)
2. Create a cluster
3. Get your URI and API token from the cluster details

#### Option B: Self-Hosted Milvus

```bash
# Using Docker Compose
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d
```

### 6. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Google Cloud Storage Configuration
GCS_BUCKET_NAME=my-pdf-bucket
GCS_BUCKET_PREFIX=documents/pdfs  # Optional: folder path in bucket (leave empty for root)
GOOGLE_SERVICE_ACCOUNT_JSON=service_account.json
GCS_RECURSIVE=true  # Set to false to only download from prefix folder

# Milvus Configuration
MILVUS_URI=your-milvus-uri
MILVUS_API_KEY=your-api-key
MILVUS_COLLECTION_NAME=pdf_embeddings

# Embedding Configuration
EMBEDDING_BACKEND=huggingface  # Options: "huggingface" or "gemini"

# HuggingFace BiomedCLIP Configuration (when EMBEDDING_BACKEND=huggingface)
EMBEDDING_SERVICE_URL=http://localhost:8000
EMBEDDING_DIMENSION=512

# Gemini Configuration (when EMBEDDING_BACKEND=gemini)
# GEMINI_API_KEY=your-gemini-api-key  # Get from https://makersuite.google.com/app/apikey
# EMBEDDING_MODEL=text-embedding-004
# EMBEDDING_DIMENSION=768

# Processing Configuration
MAX_TOKENS_PER_CHUNK=2048
CHUNK_SIZE=1500  # Characters (validated against token limit)
CHUNK_OVERLAP=150
BATCH_SIZE=100
```

**Important**: 
- For BiomedCLIP: See [QUICKSTART_EMBEDDINGS.md](QUICKSTART_EMBEDDINGS.md)
- For Gemini: See [GEMINI_SETUP.md](GEMINI_SETUP.md)

## Usage

### Run Locally

```bash
python src/ingest.py
```

The pipeline will:
1. Connect directly to Google Cloud Storage bucket
2. List all PDFs from the specified bucket (and all subfolders/prefixes if recursive mode is enabled)
3. Process each PDF in memory without local storage
4. Extract text from each PDF
5. Split text into chunks
6. Generate embeddings for each chunk
7. Store embeddings in Milvus with file path metadata

### Run with Docker

Build the Docker image:

```bash
docker build -t data-ingestion-app .
```

Run the container:

```bash
docker run --env-file .env data-ingestion-app
```

Or with docker-compose:

```yaml
version: '3.8'
services:
  data-ingestion:
    build: .
    env_file:
      - .env
    volumes:
      - ./downloads:/app/downloads
      - ./service_account.json:/app/service_account.json
```

```bash
docker-compose up
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_gcs_downloader.py

# Run with verbose output
pytest -v
```

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `GCS_BUCKET_NAME` | Google Cloud Storage bucket name | Required |
| `GCS_BUCKET_PREFIX` | Folder/prefix path in bucket (e.g., 'documents/') | `""` (empty, root) |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Path to service account JSON file | `service_account.json` |
| `GCS_RECURSIVE` | Process PDFs from all subfolders/prefixes recursively | `true` |
| `MILVUS_URI` | Milvus server URI | Required |
| `MILVUS_API_KEY` | Milvus API key/token | Required |
| `MILVUS_COLLECTION_NAME` | Name of Milvus collection | `pdf_embeddings` |
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `EMBEDDING_MODEL` | Gemini embedding model name | `text-embedding-005` |
| `MAX_TOKENS_PER_CHUNK` | Maximum tokens per chunk | `2048` |
| `CHUNK_SIZE` | Maximum characters per chunk | `1500` |
| `CHUNK_OVERLAP` | Overlap between chunks | `150` |
| `BATCH_SIZE` | Batch size for processing | `100` |
| `MINERU_BACKEND` | MinerU backend (pipeline, vlm-transformers, vlm-vllm-engine) | `pipeline` |
| `MINERU_MODEL_SOURCE` | Model source (huggingface, modelscope, local) | `huggingface` |
| `MINERU_LANG` | OCR language (en, ch, auto, etc.) | `en` |
| `PDF_EXTRACTION_TIMEOUT` | Timeout for PDF processing in seconds | `900` (15 min) |
| `MINERU_DEBUG_MODE` | Keep temporary files for debugging | `false` |
| `MINERU_ENABLE_TABLES` | Enable table extraction (Phase 2) | `false` |
| `MINERU_ENABLE_FORMULAS` | Enable formula extraction (Phase 2) | `false` |

## Architecture

### Workflow

```
Google Cloud Storage → Direct Stream Processing → Extract Text → Chunk Text → Generate Embeddings → Store in Milvus
```

### Components

1. **GCS Client**: Authenticates with GCS and lists PDF blobs from bucket
2. **PDFExtractor**: Extracts text from PDFs using MinerU (GPU/CPU compatible, supports OCR for scanned documents)
3. **TextChunker**: Splits text into token-aware overlapping chunks (max 2048 tokens)
4. **TextEmbedder**: Generates 768-dim embeddings using Google Gemini text-embedding-005 API
5. **MilvusVectorStore**: Manages Milvus collection and operations
6. **IngestionPipeline**: Orchestrates the entire workflow

## Troubleshooting

### Google Cloud Storage Authentication Issues

- Ensure the service account JSON file is valid and in the correct location
- Verify the service account has "Storage Object Viewer" role on the bucket
- Check that the Cloud Storage API is enabled in your project
- Confirm the bucket name is correct (no `gs://` prefix needed)

### Bucket Access Issues

- Make sure the bucket exists and is accessible
- Verify the service account has permissions on the bucket
- Check if the `GCS_BUCKET_PREFIX` path exists in the bucket
- Ensure you're using the correct GCP project

### PDF Extraction Errors

- **Scanned PDFs**: MinerU automatically handles OCR for scanned documents (no additional setup needed)
- **Password-protected PDFs**: Will fail - ensure PDFs are unlocked before processing
- **GPU Out of Memory**: Set `PDF_EXTRACTION_TIMEOUT` to a higher value or process fewer PDFs simultaneously
- **Slow Processing**: MinerU is more thorough than basic extractors - expect 15-30 seconds per PDF
- **CUDA Errors**: If GPU errors occur, MinerU will automatically fall back to CPU processing
- **Debug Mode**: Set `MINERU_DEBUG_MODE=true` to keep temporary files for troubleshooting

### Milvus Connection Issues

- Verify your Milvus URI and API key are correct
- Check network connectivity to Milvus server
- Ensure the Milvus server is running and accessible

### Memory Issues

- Reduce `BATCH_SIZE` if running out of memory
- Process PDFs in smaller batches
- Consider using a smaller embedding model

## Advanced Usage

### Bucket Prefix (Folder Path)

If your PDFs are organized in a specific folder within the bucket, use the prefix:

```env
GCS_BUCKET_NAME=my-bucket
GCS_BUCKET_PREFIX=documents/medical/reports
```

This will only download PDFs from `gs://my-bucket/documents/medical/reports/` and its subfolders.

### Recursive vs Non-Recursive Processing

By default, the app processes PDFs from all subfolders/prefixes. To process only from the specified prefix:

```python
from src.ingest import IngestionPipeline

# The pipeline automatically respects the GCS_RECURSIVE setting
pipeline = IngestionPipeline()
pipeline.run()  # Will use recursive=True by default
```

Or set in `.env`:
```env
GCS_RECURSIVE=false
```

### Gemini API Configuration

The pipeline uses Google's Gemini `text-embedding-005` model:

- **Embedding Dimension**: 768
- **Max Tokens**: 2048 per request
- **Pricing**: ~$0.000025 per 1,000 characters (very cheap)
- **Free Tier**: 1,500 requests per day

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

See [GEMINI_SETUP.md](GEMINI_SETUP.md) for complete setup instructions.

### Custom Chunking Strategy

```python
from src.chunker import TextChunker

chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
chunks = chunker.chunk_text(text)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

MIT License

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check existing issues for solutions
- Review the troubleshooting section

## Acknowledgments

- [Google Gemini API](https://ai.google.dev/) for high-quality text embeddings
- [Milvus](https://milvus.io/) for vector database
- [Google Cloud Storage](https://cloud.google.com/storage) for file storage
- [MinerU](https://github.com/opendatalab/MinerU) for advanced PDF parsing with OCR support

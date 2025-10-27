# Data Ingestion App

A Python application for processing PDFs from Google Cloud Storage, extracting text, generating embeddings, and storing them in Milvus vector database.

## Features

- ☁️ **Direct GCS Processing** - Process PDFs directly from Google Cloud Storage without local downloads
- 📁 **Recursive folder support** - Process from all subfolders/prefixes automatically
- 📄 Extract text from PDFs using PyMuPDF (fast and accurate)
- ✂️ Split text into overlapping chunks for better semantic search
- 🧠 Generate embeddings using sentence-transformers
- 🗄️ Store embeddings in Milvus vector database
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
│   ├── embedder.py            # Text embedding generation
│   ├── vector_store.py        # Milvus vector store interface
│   ├── ingest.py              # Main ingestion pipeline
│   └── utils.py               # Utility functions
├── tests/                      # Test suite
├── .env                        # Environment variables (create from .env.example)
├── .env.example               # Example environment variables
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
└── README.md                  # This file
```

## Prerequisites

- Python 3.11+
- Google Cloud service account with Storage access
- Milvus instance (cloud or self-hosted)
- Docker (optional, for containerized deployment)

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
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Processing Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
BATCH_SIZE=100

# Local Storage
DOWNLOAD_DIR=./downloads
```

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
| `EMBEDDING_MODEL` | Sentence-transformers model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Maximum characters per chunk | `512` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `BATCH_SIZE` | Batch size for processing | `100` |

## Architecture

### Workflow

```
Google Cloud Storage → Direct Stream Processing → Extract Text → Chunk Text → Generate Embeddings → Store in Milvus
```

### Components

1. **GCS Client**: Authenticates with GCS and lists PDF blobs from bucket
2. **PDFExtractor**: Extracts text from PDFs using PyMuPDF (fitz) - fast, accurate, and reliable
3. **TextChunker**: Splits text into overlapping chunks with metadata
4. **TextEmbedder**: Generates embeddings using sentence-transformers
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

- Some PDFs may be scanned images without text - consider adding OCR support
- Password-protected PDFs will fail - ensure PDFs are unlocked
- PyMuPDF handles most PDF formats well, including complex layouts
- For scanned documents, consider enabling OCR (pytesseract integration available)

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

### Custom Embedding Model

You can use any model from [Hugging Face sentence-transformers](https://huggingface.co/sentence-transformers):

```env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

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

- [sentence-transformers](https://www.sbert.net/) for embedding models
- [Milvus](https://milvus.io/) for vector database
- [Google Cloud Storage](https://cloud.google.com/storage) for file storage

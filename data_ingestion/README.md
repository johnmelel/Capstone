# Data Ingestion Pipeline

A Python application for processing PDFs from Google Cloud Storage using **MinerU** for advanced extraction, **Gemini Vision** for image descriptions, and **Gemini embeddings** for vector storage in Milvus.

## Pipeline Architecture

```
GCS Bucket → MinerU PDF Extraction → Gemini Vision (images→text) → Text Chunking → Gemini Embeddings → Milvus Vector Store
```

## Key Features

- 🚀 **Advanced PDF Processing** - MinerU extracts text, tables, and images with high accuracy
- 👁️ **Intelligent Image Handling** - Gemini Vision converts images to detailed text descriptions
- ☁️ **Direct GCS Processing** - Stream PDFs from Google Cloud Storage without local storage
- 📁 **Recursive folder support** - Process all PDFs from bucket subfolders automatically
- ✂️ **Smart Text Chunking** - Token-aware chunking optimized for Gemini (max 2048 tokens)
- 🧠 **High-Quality Embeddings** - Gemini text embeddings (768 dimensions)
- 🗄️ **Vector Storage** - Milvus/Zilliz for efficient similarity search
- � **Output to GCS** - Save extracted content (images, metadata) back to bucket
- ✅ **Comprehensive Testing** - Full test suite for all components

## Project Structure

```
data_ingestion/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration from environment variables
│   ├── mineru_parser.py       # MinerU PDF extraction + Gemini Vision
│   ├── gcs_downloader.py      # Google Cloud Storage client
│   ├── chunker.py             # Text chunking with token awareness
│   ├── embedder.py            # Gemini text embedding generation
│   ├── vector_store.py        # Milvus vector store interface
│   ├── ingest.py              # Main ingestion pipeline orchestrator
│   └── utils.py               # Utility functions
├── tests/                      # Test suite
│   ├── test_mineru_parser.py  # MinerU parser tests
│   ├── test_chunker.py        # Text chunking tests
│   ├── test_embedder.py       # Embedding generation tests
│   ├── test_gcs_downloader.py # GCS client tests
│   └── test_vector_store.py   # Vector store tests
├── .env                        # Environment variables (create from .env.example)
├── .env.example               # Example environment variables
├── requirements.txt           # Python dependencies (includes mineru==0.2.6)
├── test_setup.py              # Setup validation script
├── test_mineru_pipeline.py    # Quick pipeline test
├── INTEGRATION_README.md      # Detailed integration documentation
├── TESTING_GUIDE.md           # Testing instructions
└── README.md                  # This file
```

## Prerequisites

- Python 3.11+
- **MinerU** - Advanced PDF extraction library
- Google Cloud service account with Storage access
- Gemini API key (for embeddings and vision)
- Milvus/Zilliz instance (cloud or self-hosted)

## Setup

### 1. Install Dependencies

```bash
cd data_ingestion
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This will install all required packages including:
- `mineru==0.2.6` - PDF extraction
- `google-genai` - Gemini API (embeddings + vision)
- `pymilvus` - Vector database
- `google-cloud-storage` - GCS integration

### 2. Install MinerU

MinerU may require additional system dependencies. If the pip install doesn't work fully:

```bash
# See MinerU installation guide
https://github.com/opendatalab/MinerU#installation
```

Verify installation:
```bash
magic-pdf --version
```

### 3. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Google Cloud Storage Configuration
GCS_BUCKET_NAME=your-bucket-name
GCS_BUCKET_PREFIX=Data/  # Optional: folder path in bucket
GOOGLE_SERVICE_ACCOUNT_JSON=path/to/service_account.json
GCS_RECURSIVE=true  # Process all subfolders

# Milvus/Zilliz Configuration
MILVUS_URI=your-milvus-uri
MILVUS_API_KEY=your-api-key
MILVUS_COLLECTION_NAME=your_collection_name

# Gemini API Configuration
GEMINI_API_KEY=your-gemini-api-key
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=768

# Gemini Vision Configuration (for image descriptions)
GEMINI_VISION_MODEL=gemini-2.0-flash-exp
IMAGE_DESCRIPTION_PROMPT=Describe this image in detail. Focus on medical/scientific content, figures, charts, diagrams, or any text visible in the image. Be concise but comprehensive.

# Output Configuration
OUTPUT_DIR=./extracted_content
UPLOAD_OUTPUT_TO_GCS=true
GCS_OUTPUT_PREFIX=extracted_content

# Processing Configuration
CHUNK_SIZE=1792
CHUNK_OVERLAP=64
BATCH_SIZE=100
MAX_TOKENS_PER_CHUNK=2048
```

**Get API Keys:**
- Gemini API: https://makersuite.google.com/app/apikey
- Milvus/Zilliz: https://cloud.zilliz.com/

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

# Gemini Embedding Configuration
GEMINI_API_KEY=your-gemini-api-key  # Get from https://makersuite.google.com/app/apikey
EMBEDDING_MODEL=text-embedding-005
MAX_TOKENS_PER_CHUNK=2048

# Processing Configuration
CHUNK_SIZE=1500  # Characters (validated against token limit)
CHUNK_OVERLAP=150
BATCH_SIZE=100
```

**Important**: See [GEMINI_SETUP.md](GEMINI_SETUP.md) for detailed Gemini API configuration.

## Testing

### Quick Setup Validation

Run the setup test to verify all components:

```bash
python test_setup.py
```

This validates:
- ✅ Environment variables and configuration
- ✅ Gemini API connection (embeddings + vision)
- ✅ Google Cloud Storage access
- ✅ Milvus/Zilliz database connection
- ✅ MinerU installation
- ✅ All pipeline components

### Test Single PDF

Test with one PDF before running the full pipeline:

```bash
python test_mineru_pipeline.py
```

This will:
1. Download one PDF from your bucket
2. Extract text and images with MinerU
3. Generate image descriptions with Gemini Vision
4. Create embeddings with Gemini
5. Store in Milvus
6. Upload extracted content to GCS (if enabled)

### Run Full Test Suite

```bash
# Run all unit tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_mineru_parser.py -v

# Run with verbose output
pytest -v
```

Test coverage:
- ✅ MinerU parser with Gemini Vision integration
- ✅ Text chunking with token awareness
- ✅ Gemini embedding generation
- ✅ GCS bucket operations
- ✅ Milvus vector store operations

## Usage

### Run the Pipeline

Once tests pass, run the full ingestion pipeline:

```bash
python -m src.ingest
```

The pipeline will:
1. ☁️ Connect to your GCS bucket
2. 📋 List all PDFs (recursively if enabled)
3. 📄 For each PDF:
   - Extract with MinerU (text, tables, images)
   - Convert images to text using Gemini Vision
   - Chunk all text content
   - Generate Gemini embeddings (768-dim)
   - Store in Milvus with metadata
   - Upload extracted content to GCS (optional)
4. ⏭️ Skip already-processed PDFs (deduplication)

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

## Architecture

### Workflow

```
Google Cloud Storage → Direct Stream Processing → Extract Text → Chunk Text → Generate Embeddings → Store in Milvus
```

### Components

1. **GCS Client**: Authenticates with GCS and lists PDF blobs from bucket
2. **PDFExtractor**: Extracts text from PDFs using PyMuPDF (fitz) - fast, accurate, and reliable
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
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing

# Data Ingestion Pipeline - MinerU + Vertex AI Integration

## Overview

This pipeline ingests PDF documents from Google Cloud Storage, extracts multimodal content (text, images, tables) using MinerU, generates embeddings using Vertex AI's multimodal model, and stores everything in Milvus (Zilliz) vector database.

## Features

- ✅ **Advanced PDF Extraction**: Uses MinerU CLI for high-quality text, image, and table extraction
- ✅ **Multimodal Embeddings**: Vertex AI multimodal model (1408 dimensions) handles text AND images
- ✅ **GCS Integration**: Directly streams PDFs from Google Cloud Storage buckets
- ✅ **Milvus Vector Store**: Stores embeddings in Zilliz cloud database with JSON metadata support
- ✅ **Deduplication**: Automatically skips already-processed files
- ✅ **Batch Processing**: Efficient batch embedding and insertion

## Architecture

```
GCS Bucket → MinerU Parser → Vertex AI Embedder → Milvus/Zilliz DB
   (PDFs)      (Extract)        (Embed)            (Store)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Important**: MinerU requires additional setup. Follow instructions at: https://github.com/opendatalab/MinerU

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:

```bash
# Google Cloud Storage
GCS_BUCKET_NAME=your-bucket-name
GCS_BUCKET_PREFIX=documents  # Optional folder path
GOOGLE_SERVICE_ACCOUNT_JSON=service_account.json

# Google Cloud Vertex AI
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# Milvus/Zilliz
MILVUS_URI=your-milvus-uri
MILVUS_API_KEY=your-api-key
MILVUS_COLLECTION_NAME=pdf_embeddings

# Processing
CHUNK_SIZE=800
CHUNK_OVERLAP=150
BATCH_SIZE=100
```

### 3. Service Account Setup

Place your Google Cloud service account JSON file in the project root and update `GOOGLE_SERVICE_ACCOUNT_JSON` path.

Required permissions:
- Storage Object Viewer (for GCS)
- Vertex AI User (for embeddings)

## Usage

### Run the Pipeline

```bash
python -m src.ingest
```

The pipeline will:
1. Connect to your GCS bucket
2. List all PDF files (respects `GCS_BUCKET_PREFIX` and `GCS_RECURSIVE` settings)
3. Download and process each PDF with MinerU
4. Extract text chunks and images
5. Generate embeddings for all content
6. Store in Milvus with rich metadata

### Test Setup

Verify your configuration:

```bash
python test_setup.py
```

## Output

### Extracted Content

Content is saved to `./extracted_content/`:
- `images/` - Extracted images from PDFs
- `temp/` - Temporary MinerU processing files

### Milvus Schema

Each vector in Milvus contains:
- `vector` (FLOAT_VECTOR, 1408 dim) - Multimodal embedding
- `text` (VARCHAR) - Text content or image caption
- `file_name` (VARCHAR) - Source PDF name
- `content_type` (VARCHAR) - "text", "image", or "table"
- `metadata_json` (VARCHAR) - Full metadata as JSON including:
  - `pdf_name`
  - `chunk_index`
  - `char_count`, `word_count` (for text)
  - `figure_refs`, `table_refs` (references found in text)

## How It Works

### 1. MinerU Parser (`src/mineru_parser.py`)

- Runs MinerU CLI on each PDF
- Extracts markdown text and images
- Cleans and chunks text with overlap
- Handles image extraction and copying

### 2. Vertex AI Embedder (`src/vertex_embedder.py`)

- Initializes Vertex AI multimodal model
- Embeds text chunks (max 1000 chars)
- Embeds images with optional captions
- Returns normalized 1408-dim vectors

### 3. Vector Store (`src/vector_store.py`)

- Connects to Milvus/Zilliz cloud
- Creates collection with multimodal schema
- Inserts embeddings in batches
- Supports JSON metadata storage

### 4. Main Pipeline (`src/ingest.py`)

- Orchestrates all components
- Downloads PDFs from GCS to temp files
- Manages batch processing
- Handles deduplication
- Provides progress tracking

## Configuration Options

### Chunking

- `CHUNK_SIZE`: Characters per text chunk (default: 800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 150)

### Processing

- `BATCH_SIZE`: Number of embeddings per batch (default: 100)
- `GCS_RECURSIVE`: Process subfolders recursively (default: true)

### Embeddings

- Fixed at 1408 dimensions (Vertex AI multimodal model)
- Automatic L2 normalization
- Handles text truncation for long content

## Troubleshooting

### MinerU Installation Issues

MinerU requires specific dependencies. See: https://github.com/opendatalab/MinerU#installation

### Vertex AI Authentication

Ensure your service account has Vertex AI User role:
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT@PROJECT.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### Milvus Connection Errors

Verify your Milvus URI and API key. For Zilliz cloud, format is:
```
MILVUS_URI=https://your-cluster.api.gcp-us-west1.zillizcloud.com
```

## Performance

- **Embedding Speed**: ~10 chunks/second (text), ~5 images/second
- **Memory**: ~2GB for typical workload
- **MinerU Processing**: ~10-30 seconds per PDF depending on size

## What Changed from Original

### Removed
- ❌ `pdfplumber` PDF extractor
- ❌ Gemini text embeddings
- ❌ Simple text-only chunker

### Added
- ✅ MinerU parser (advanced extraction)
- ✅ Vertex AI multimodal embeddings
- ✅ Image and table support
- ✅ JSON metadata storage
- ✅ Temporary file handling for GCS → MinerU

### Updated
- ✅ Vector store schema (added `content_type`, `metadata_json`)
- ✅ Configuration (added Vertex AI settings)
- ✅ Main pipeline (unified multimodal processing)

## Next Steps

1. **Query Interface**: Build a query system using the stored embeddings
2. **RAG Integration**: Connect to LLM for retrieval-augmented generation
3. **Monitoring**: Add metrics and logging for production use
4. **Scaling**: Consider parallel processing for large document sets

## Credits

- **MinerU**: https://github.com/opendatalab/MinerU
- **Vertex AI**: Google Cloud multimodal embeddings
- **Milvus**: Open-source vector database

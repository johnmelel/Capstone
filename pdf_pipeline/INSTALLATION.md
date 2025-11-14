# PDF Pipeline Installation Guide

## Quick Start

```bash
# 1. Install uv
pip install --upgrade pip
pip install uv

# 2. Install all dependencies
uv pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your credentials

# 4. Verify installation
python -c "from config import Config; Config.validate(); Config.print_config()"

# 5. Run pipeline
python main.py
```

## Detailed Installation

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation Steps

### 1. Upgrade pip and Install uv

`uv` is a fast Python package installer that we'll use for all dependencies.

```bash
pip install --upgrade pip
pip install uv
```

### 2. Install All Dependencies with uv

All dependencies including MinerU are listed in `requirements.txt`. Install them all at once:

```bash
uv pip install -r requirements.txt
```

This will install:
- `raganything` - Core RAG library
- `mineru[core]` - High-quality PDF parsing engine
- `google-cloud-storage` - For GCS access
- `pymilvus` - Vector database client
- `google-generativeai` - For embeddings and LLM
- All other dependencies

### 3. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Google Cloud Storage
GOOGLE_SERVICE_ACCOUNT_JSON=/path/to/service-account.json
GCS_BUCKET_NAME=your-bucket-name
GCS_BUCKET_PREFIX=your-prefix/
DOWNLOAD_DIR=/tmp/pdf_downloads
GCS_RECURSIVE=True

# Milvus Vector Database
MILVUS_URI=your-milvus-uri
MILVUS_API_KEY=your-milvus-api-key
MILVUS_COLLECTION_NAME=pdf_embeddings

# Google Generative AI
GOOGLE_API_KEY=your-google-api-key
EMBEDDING_MODEL=models/embedding-001
LLM_MODEL=gemini-1.5-pro
EMBEDDING_DIM=3072

# MinerU Configuration
MINERU_DEVICE=cuda  # Options: cuda, cpu, mps (for Apple Silicon)

# RAG Processing
RAG_CLEAR_CACHE=True  # Clear cache before processing (recommended)

# Logging
LOG_FILE=pipeline.log
LOG_LEVEL=INFO

# Data Paths
PROCESSED_DATA_DIR=/tmp/processed_data
```

### 4. Verify Installation

Run the configuration validator:

```bash
python -c "from config import Config; Config.validate(); Config.print_config()"
```

This will check that:
- All required packages are installed
- MinerU is available
- Environment variables are set correctly
- Device configuration is valid (CUDA/CPU/MPS)

## Device Configuration

### CUDA (NVIDIA GPU)

If you have an NVIDIA GPU and want to use CUDA acceleration:

```bash
MINERU_DEVICE=cuda
```

Ensure you have:
- CUDA-compatible GPU
- CUDA toolkit installed
- PyTorch with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### CPU

For CPU-only processing:

```bash
MINERU_DEVICE=cpu
```

Note: CPU processing will be slower than GPU.

### MPS (Apple Silicon)

For Apple M1/M2/M3 Macs:

```bash
MINERU_DEVICE=mps
```

Requires PyTorch with MPS support: `pip install torch torchvision`

## Troubleshooting

### Installation Issues

If `uv pip install -r requirements.txt` fails:

1. Ensure `uv` is installed: `pip install uv`
2. Try installing with verbose output: `uv pip install -r requirements.txt -v`
3. Check Python version compatibility (Python 3.8+)
4. If specific packages fail, try installing them individually:
   ```bash
   uv pip install raganything
   uv pip install "mineru[core]"
   ```

### MinerU Installation Issues

If MinerU installation fails specifically:

1. Ensure you have the correct syntax: `mineru[core]` (with brackets)
2. Try upgrading pip and uv: `pip install --upgrade pip uv`
3. Check that Python development headers are installed (required for some dependencies)

### Import Errors

If you get import errors when running the pipeline:

1. Verify packages are installed: `uv pip list`
2. Check that MinerU is installed: `python -c "import mineru; print(mineru.__version__)"`
3. Re-run installation: `uv pip install -r requirements.txt --force-reinstall`

### CUDA Not Available

If CUDA is configured but not detected:

1. Check NVIDIA drivers: `nvidia-smi`
2. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Consider using `MINERU_DEVICE=cpu` as fallback

## Running the Pipeline

Once installation is complete:

```bash
python main.py
```

The pipeline will:
1. Validate configuration (including MinerU)
2. Clear cache (if `RAG_CLEAR_CACHE=True`)
3. Download PDFs from GCS
4. Process PDFs with RAGAnything + MinerU
5. Insert embeddings into Milvus

Check `pipeline.log` for detailed execution logs.

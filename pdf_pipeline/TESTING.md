# PDF Pipeline Testing Guide

This guide provides comprehensive instructions for testing the PDF ingestion pipeline with **RAGAnything**.

## Prerequisites

Before testing, ensure you have:

1. **Python Environment**: Python 3.9+ installed
2. **Dependencies**: All packages from `requirements.txt` installed (including RAGAnything)
3. **Credentials**: 
   - Google Cloud service account JSON
   - Google API key (for Generative AI)
   - Milvus cluster credentials
4. **Test Data**: At least one PDF file in your GCS bucket
5. **GPU (Optional)**: CUDA-capable GPU for faster processing with MinerU

## Understanding RAGAnything

The pipeline now uses **RAGAnything's `process_document_complete` method**, which:
- Uses MinerU parser for advanced PDF extraction
- Extracts text, tables, images, and equations
- Stores data in LightRAG's knowledge graph
- Supports multimodal processing

## Step 1: Environment Setup

### 1.1 Verify RAGAnything Installation

```bash
python -c "from raganything import RAGAnything; print('RAGAnything installed:', RAGAnything.__module__)"
```

### 1.2 Check Available Methods

```bash
python test_raganything_api.py
```

Should show methods including:
- `process_document_complete` ✓
- `process_documents_batch`
- `aquery`
- `lightrag`

### 1.3 Configure Environment

Edit `.env` with your actual credentials:

```env
# Required
GOOGLE_SERVICE_ACCOUNT_JSON="path/to/service-account.json"
GCS_BUCKET_NAME="your-bucket-name"
GCS_BUCKET_PREFIX="test_file"
MILVUS_URI="https://your-milvus-uri"
MILVUS_API_KEY="your-milvus-key"
GOOGLE_API_KEY="your-google-api-key"

# RAGAnything/MinerU
MINERU_DEVICE="cuda"  # or "cpu"
EMBEDDING_MODEL="models/embedding-001"
LLM_MODEL="gemini-1.5-pro"
```

## Step 2: Test RAGAnything Processing

### 2.1 Test Single Document Processing

Create `test_rag_processing.py`:

```python
import asyncio
import logging
from pathlib import Path
from rag_processor import RAGProcessor

logging.basicConfig(level=logging.INFO)

async def test_process():
    processor = RAGProcessor()
    
    # Use a downloaded PDF
    test_pdf = Path("/tmp/pdf_downloads/your-test-file.pdf")
    
    if test_pdf.exists():
        result = await processor.process_document(test_pdf)
        print(f"✓ Processed: {result}")
    else:
        print(f"✗ Test file not found: {test_pdf}")

asyncio.run(test_process())
```

Run: `python test_rag_processing.py`

**Expected Output:**
```
INFO:rag_processor:Initializing RAGAnything with MinerU parser...
INFO:rag_processor:RAGAnything initialized successfully
INFO:rag_processor:Parser: mineru
INFO:rag_processor:Processing document: /tmp/pdf_downloads/test.pdf
INFO:rag_processor:Using process_document_complete method
INFO: Processing PDF with MinerU...
INFO:rag_processor:Document processed, extracting data from LightRAG storage...
INFO:rag_processor:Processing 48 extracted entities...
INFO:rag_processor:Successfully processed test.pdf
✓ Processed: /tmp/processed_data/test.parquet
```

### 2.2 Inspect Processed Data

```python
import pandas as pd

df = pd.read_parquet("/tmp/processed_data/your-file.parquet")
print(f"Chunks: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Embedding dimension: {len(df['vector'].iloc[0])}")
print(f"\nFirst chunk text:\n{df['text'].iloc[0][:200]}...")
```

## Step 3: Test Complete Pipeline

### 3.1 Run Full Pipeline

```bash
cd ~/Capstone/pdf_pipeline
python main.py
```

**Monitor for:**

1. ✅ Configuration validation passes
2. ✅ RAGAnything initializes with MinerU parser
3. ✅ GCS download completes
4. ✅ `process_document_complete` executes successfully
5. ✅ Entities extracted from LightRAG
6. ✅ Embeddings generated (if not in RAGAnything storage)
7. ✅ Data inserted into Milvus

### 3.2 Expected Log Flow

```
INFO:__main__:PDF Ingestion Pipeline - Starting
...
INFO:rag_processor:Initializing RAGAnything with MinerU parser...
INFO:rag_processor:RAGAnything initialized successfully
INFO:rag_processor:Parser: mineru
...
INFO:__main__:Step 2/3: Processing PDFs with RAG-Anything...
INFO:rag_processor:Processing document: /tmp/pdf_downloads/Research Capstone Syllabus-25.pdf
INFO:rag_processor:Using process_document_complete method
INFO: Processing PDF with MinerU...
INFO:rag_processor:Document processed, extracting data from LightRAG storage...
INFO:rag_processor:LightRAG type: <class 'lightrag.lightrag.LightRAG'>
INFO:rag_processor:Processing 48 extracted entities...
INFO:rag_processor:Successfully processed Research Capstone Syllabus-25.pdf
INFO:rag_processor:Extracted 48 chunks from document
...
INFO:vector_store:Inserting 48 entities into Milvus
INFO:vector_store:Successfully inserted 48 entities
...
INFO:__main__:Pipeline completed successfully!
```

## Step 4: Verify RAGAnything Storage

### 4.1 Check Working Directory

```bash
ls -la ./rag_storage/
```

Should contain:
- LightRAG storage files
- Processed document metadata
- Knowledge graph data

### 4.2 Query via RAGAnything

```python
import asyncio
from rag_processor import RAGProcessor

async def test_query():
    processor = RAGProcessor()
    
    # Query the processed documents
    result = await processor.rag.aquery("What is this document about?")
    print(f"Query result: {result}")

asyncio.run(test_query())
```

## Step 5: Troubleshooting

### Issue: MinerU Not Working

**Error**: `MinerU parser not available`

**Solution**:
```bash
# Install MinerU dependencies
pip install magic-pdf
pip install paddlepaddle  # for GPU: paddlepaddle-gpu
```

### Issue: No Entities Extracted

**Error**: `No entities extracted from RAGAnything`

**Check**:
1. Verify document was actually processed:
   ```python
   status = processor.rag.get_document_processing_status(str(file_path))
   print(status)
   ```

2. Check LightRAG storage:
   ```python
   if hasattr(processor.rag, 'lightrag'):
       print(f"LightRAG working dir: {processor.rag.lightrag.working_dir}")
   ```

3. **Fallback**: If entities can't be extracted, the code automatically falls back to direct text extraction

### Issue: CUDA Out of Memory

**Solution**:
```env
MINERU_DEVICE="cpu"
```

### Issue: Embedding Dimension Mismatch

**Check**: Ensure `EMBEDDING_MODEL` matches the dimension:
- `models/embedding-001`: 768 dimensions
- `models/text-embedding-004`: 768 dimensions

## Step 6: Advanced Testing

### 6.1 Batch Processing

Test with multiple files:

```python
import asyncio
from rag_processor import RAGProcessor
from pathlib import Path

async def batch_test():
    processor = RAGProcessor()
    pdf_files = list(Path("/tmp/pdf_downloads").glob("*.pdf"))
    
    for pdf in pdf_files[:3]:  # Test first 3
        print(f"Processing: {pdf.name}")
        result = await processor.process_document(pdf)
        print(f"✓ Result: {result}\n")

asyncio.run(batch_test())
```

### 6.2 Performance Benchmarking

```python
import time
import asyncio
from rag_processor import RAGProcessor

async def benchmark():
    processor = RAGProcessor()
    pdf_path = Path("/tmp/pdf_downloads/test.pdf")
    
    start = time.time()
    result = await processor.process_document(pdf_path)
    elapsed = time.time() - start
    
    print(f"Processing time: {elapsed:.2f}s")
    print(f"Result: {result}")

asyncio.run(benchmark())
```

## Success Criteria

Your pipeline is working correctly with RAGAnything if:

- ✅ RAGAnything initializes with MinerU parser
- ✅ `process_document_complete` executes without errors
- ✅ Entities are extracted from LightRAG storage
- ✅ Embeddings have correct dimension (768)
- ✅ Data is inserted into Milvus successfully
- ✅ You can query both via RAGAnything and Milvus
- ✅ Multimodal content (tables, images) is processed

## Next Steps

After successful testing:

1. **Optimize MinerU**: Tune parser settings for your PDFs
2. **Scale Up**: Process larger batches
3. **Monitor**: Add performance metrics
4. **Query Testing**: Test RAG queries via both RAGAnything and direct Milvus
5. **Production**: Deploy with proper error handling and monitoring

## RAGAnything-Specific Notes

- **Parser**: MinerU provides better extraction than basic PDF parsers
- **Storage**: Data is stored in LightRAG's knowledge graph
- **Multimodal**: Can process images, tables, and equations
- **Query**: Can query via `rag.aquery()` or directly from Milvus
- **Fallback**: Code includes fallback for cases where entity extraction fails

The pipeline is now **fully integrated with RAGAnything** using the correct `process_document_complete` API! 🚀


## Prerequisites

Before testing, ensure you have:

1. **Python Environment**: Python 3.9+ installed
2. **Dependencies**: All packages from `requirements.txt` installed
3. **Credentials**: 
   - Google Cloud service account JSON
   - Google API key (for Generative AI)
   - Milvus cluster credentials
4. **Test Data**: At least one PDF file in your GCS bucket

## Step 1: Environment Setup

### 1.1 Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 1.2 Install Dependencies

```bash
cd pdf_pipeline
pip install -r requirements.txt
```

### 1.3 Configure Environment

Copy and configure the environment file:

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials. **Required fields:**

```env
GOOGLE_SERVICE_ACCOUNT_JSON="path/to/service-account.json"
GCS_BUCKET_NAME="your-bucket-name"
GCS_BUCKET_PREFIX="test_file"  # or your folder name
MILVUS_URI="https://your-milvus-uri"
MILVUS_API_KEY="your-milvus-key"
GOOGLE_API_KEY="your-google-api-key"
```

## Step 2: Test RAGAnything API

Since RAGAnything API can vary by version, first test what methods are available:

```bash
python test_raganything_api.py
```

This will output:
- Available methods in RAGAnything
- Available attributes
- Which expected methods exist

**Expected Output:**
```
RAGAnything Public Methods and Attributes:
...
Methods:
  - insert_file()  (or similar)
  - process_document()
...
```

## Step 3: Test Individual Components

### 3.1 Test Configuration

Create `test_config.py`:

```python
from config import Config

try:
    Config.validate()
    print("✓ Configuration is valid!")
    Config.print_config()
except Exception as e:
    print(f"✗ Configuration error: {e}")
```

Run: `python test_config.py`

### 3.2 Test GCS Download

Create `test_gcs.py`:

```python
import logging
from gcs_downloader import download_pdfs_from_gcs

logging.basicConfig(level=logging.INFO)

try:
    files = download_pdfs_from_gcs()
    print(f"\n✓ Successfully downloaded {len(files)} files:")
    for f in files:
        print(f"  - {f}")
except Exception as e:
    print(f"\n✗ GCS download failed: {e}")
```

Run: `python test_gcs.py`

### 3.3 Test Milvus Connection

Create `test_milvus.py`:

```python
import logging
from vector_store import MilvusVectorStore

logging.basicConfig(level=logging.INFO)

try:
    vs = MilvusVectorStore()
    stats = vs.get_stats()
    print(f"\n✓ Connected to Milvus!")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Entities: {stats['num_entities']}")
    vs.close()
except Exception as e:
    print(f"\n✗ Milvus connection failed: {e}")
```

Run: `python test_milvus.py`

### 3.4 Test Embeddings

Create `test_embeddings.py`:

```python
import logging
from rag_processor import google_embedding_func

logging.basicConfig(level=logging.INFO)

try:
    test_texts = ["Hello world", "Testing embeddings"]
    embeddings = google_embedding_func(test_texts)
    print(f"\n✓ Embeddings generated successfully!")
    print(f"  Number of embeddings: {len(embeddings)}")
    print(f"  Embedding dimension: {len(embeddings[0])}")
except Exception as e:
    print(f"\n✗ Embedding generation failed: {e}")
```

Run: `python test_embeddings.py`

## Step 4: Run Full Pipeline

### 4.1 Test Run with Small Dataset

Start with a small test (1-2 PDFs):

```bash
python main.py
```

**Monitor the output for:**

1. ✓ Configuration validation passes
2. ✓ GCS connection successful
3. ✓ Files downloaded
4. ✓ RAG processing completes
5. ✓ Embeddings generated
6. ✓ Data inserted into Milvus

### 4.2 Check Logs

```bash
# View recent logs
tail -f pipeline.log

# Search for errors
grep ERROR pipeline.log

# Search for warnings
grep WARNING pipeline.log
```

## Step 5: Verify Results

### 5.1 Check Downloaded Files

```bash
ls -la /tmp/pdf_downloads/
# or on Windows:
# dir C:\tmp\pdf_downloads\
```

### 5.2 Check Processed Data

```bash
ls -la /tmp/processed_data/
# Should see .parquet files
```

### 5.3 Query Milvus

Create `test_query.py`:

```python
import logging
from vector_store import MilvusVectorStore
from rag_processor import google_embedding_func

logging.basicConfig(level=logging.INFO)

vs = MilvusVectorStore()

# Generate query embedding
query_text = "What is this document about?"
query_embedding = google_embedding_func([query_text])[0]

# Search
results = vs.search(query_embedding, top_k=5)

print(f"\nFound {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Distance: {result['distance']:.4f}")
    print(f"   Text: {result['entity']['text'][:200]}...")
    print(f"   File: {result['entity']['file_name']}")

vs.close()
```

Run: `python test_query.py`

## Step 6: Troubleshooting

### Common Issues and Solutions

#### Issue 1: RAGAnything API Errors

**Error**: `'RAGAnything' object has no attribute 'insert'`

**Solution**: Run `test_raganything_api.py` to see available methods, then update `rag_processor.py` to use the correct method name.

#### Issue 2: Import Errors

**Error**: `ModuleNotFoundError: No module named 'raganything'`

**Solution**: 
```bash
pip install raganything --upgrade
pip install -r requirements.txt
```

#### Issue 3: Google API Errors

**Error**: `google.api_core.exceptions.PermissionDenied`

**Solution**: 
- Verify `GOOGLE_API_KEY` is correct
- Check API is enabled in Google Cloud Console
- Verify billing is enabled

#### Issue 4: Milvus Connection Errors

**Error**: `Failed to connect to Milvus`

**Solution**:
- Verify `MILVUS_URI` and `MILVUS_API_KEY`
- Check network connectivity
- Ensure Milvus cluster is running

#### Issue 5: CUDA/GPU Errors

**Error**: `CUDA not available`

**Solution**:
```env
MINERU_DEVICE="cpu"
```

## Step 7: Performance Testing

### 7.1 Test with Multiple Files

Gradually increase the number of PDFs:
- Start: 1-2 files
- Then: 5-10 files
- Finally: Full dataset

Monitor:
- Processing time per file
- Memory usage
- Error rates

### 7.2 Benchmark Script

Create `benchmark.py`:

```python
import time
import asyncio
from pathlib import Path
from gcs_downloader import download_pdfs_from_gcs
from rag_processor import RAGProcessor

async def benchmark():
    start = time.time()
    
    # Download
    dl_start = time.time()
    files = download_pdfs_from_gcs()
    dl_time = time.time() - dl_start
    
    # Process
    proc_start = time.time()
    processor = RAGProcessor()
    for f in files[:5]:  # Test first 5 files
        await processor.process_document(f)
    proc_time = time.time() - proc_start
    
    total_time = time.time() - start
    
    print(f"\n📊 Benchmark Results:")
    print(f"  Download time: {dl_time:.2f}s")
    print(f"  Processing time: {proc_time:.2f}s")
    print(f"  Avg per file: {proc_time/min(len(files), 5):.2f}s")
    print(f"  Total time: {total_time:.2f}s")

asyncio.run(benchmark())
```

## Step 8: Automated Testing (Optional)

### Create Unit Tests

Create `tests/test_pipeline.py`:

```python
import pytest
from pathlib import Path
from config import Config
from gcs_downloader import GCSDownloader
from vector_store import MilvusVectorStore

def test_config_validation():
    """Test configuration validation"""
    try:
        Config.validate()
        assert True
    except Exception as e:
        pytest.fail(f"Config validation failed: {e}")

def test_milvus_connection():
    """Test Milvus connection"""
    vs = MilvusVectorStore()
    stats = vs.get_stats()
    assert 'collection_name' in stats
    vs.close()

# Add more tests...
```

Run: `pytest tests/`

## Success Criteria

Your pipeline is working correctly if:

- ✅ All configuration validates without errors
- ✅ PDFs download successfully from GCS
- ✅ Documents are processed without crashes
- ✅ Embeddings are generated (768-dimensional vectors)
- ✅ Data is inserted into Milvus
- ✅ You can query and retrieve results from Milvus
- ✅ Logs show no critical errors
- ✅ Process completes end-to-end

## Next Steps

After successful testing:

1. **Production Deployment**: Set up on production environment
2. **Monitoring**: Add monitoring and alerting
3. **Scheduling**: Set up automated runs (cron/Airflow)
4. **Scaling**: Test with larger datasets
5. **Optimization**: Profile and optimize slow components

## Support

If you encounter issues:

1. Check `pipeline.log` for detailed errors
2. Review the troubleshooting section above
3. Verify all prerequisites are met
4. Test components individually before full pipeline

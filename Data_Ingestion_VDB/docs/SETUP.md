# Setup Instructions

## 1. Install Dependencies
```bash
pip install -r requirements.txt

# Option A: Docker (recommended)
docker pull milvusdb/milvus:latest
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest

# Option B: Milvus Lite (embedded, for testing)
pip install milvus

# Create .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# Test mode (1 PDF, first 5 pages)
python run_pipeline.py --test

# Full run
python run_pipeline.py --full
```
### `config/config.yaml`

```yaml
# Gemini Embedding Settings
embedding:
  model: "models/embedding-001"
  dimensions: 768
  batch_size: 100
  task_type: "retrieval_document"
  normalize: true

# Text Chunking
chunking:
  method: "semantic"
  chunk_size: 800  # characters
  overlap: 100

# Image Settings
images:
  min_width: 100
  min_height: 100
  extract_captions: true
  caption_search_radius: 150  # pixels

# Table Settings
tables:
  detection_method: "text_based"
  min_rows: 3
  min_columns: 2

# Milvus Vector Store
vector_store:
  host: "localhost"
  port: 19530
  collection_name: "medical_textbooks"
  index_type: "IVF_FLAT"  # or HNSW for larger datasets
  metric_type: "COSINE"
  nlist: 1024  # for IVF_FLAT

# Processing
processing:
  test_mode: false
  max_test_pages: 5
  skip_existing: true
  max_workers: 4

# API Settings
api:
  retry_attempts: 3
  retry_delay: 2
  timeout: 30
  requests_per_minute: 1000

```
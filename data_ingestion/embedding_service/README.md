# BiomedCLIP Embedding Service

This is a standalone FastAPI service that provides text embeddings using the Microsoft BiomedCLIP model (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`). The service is designed to run independently and can be used by the data ingestion pipeline and retrieval systems.

## Model Information

- **Model**: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- **Type**: PubMedBERT-based text encoder from BiomedCLIP
- **Output Dimension**: 512
- **Max Sequence Length**: 256 tokens
- **Optimization**: Fine-tuned for biomedical text and concepts
- **Paper**: [BiomedCLIP: Biomedical Vision-Language Foundation Model](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

## Architecture

The embedding service follows a microservices architecture:

```
┌─────────────────────────────────────┐
│   Data Ingestion Pipeline          │
│   - PDF Extraction                  │
│   - Text Chunking                   │
│   - Metadata Management             │
└──────────────┬──────────────────────┘
               │ HTTP Requests
               ↓
┌─────────────────────────────────────┐
│   Embedding Service (Port 8000)    │
│   - BiomedCLIP Model Loading       │
│   - Text Embedding Generation       │
│   - Batch Processing                │
└──────────────┬──────────────────────┘
               │ Embeddings (512-dim)
               ↓
┌─────────────────────────────────────┐
│   Milvus Vector Store               │
│   - Storage & Indexing              │
│   - Similarity Search               │
└─────────────────────────────────────┘
               ↑
               │ Query Embeddings
┌──────────────┴──────────────────────┐
│   Retrieval System                  │
│   - Query Processing                │
│   - Result Ranking                  │
└─────────────────────────────────────┘
```

## API Endpoints

### Health Check
```bash
GET /health
```
Returns the service status and model information.

**Response:**
```json
{
  "status": "healthy",
  "model": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
  "device": "cuda:0",
  "embedding_dimension": 512
}
```

### Generate Embeddings
```bash
POST /embed
```
Generate embeddings for a list of texts.

**Request Body:**
```json
{
  "texts": ["What are the symptoms of diabetes?", "Explain MRI imaging"],
  "normalize": true
}
```

**Response:**
```json
{
  "embeddings": [[0.123, -0.456, ...], [0.789, 0.234, ...]],
  "dimension": 512,
  "model": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
  "processing_time": 0.234
}
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Installation & Setup

### Option 1: Manual Setup (Recommended for Development)

1. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
cd embedding_service
pip install -r requirements.txt
```

3. **Start the service:**
```bash
# Linux/Mac
./start_service.sh

# Windows
start_service.bat

# Or manually
uvicorn app:app --host 0.0.0.0 --port 8000
```

4. **Verify the service:**
```bash
curl http://localhost:8000/health
```

### Option 2: Docker (Recommended for Production)

1. **Build and start with Docker Compose:**
```bash
docker-compose up -d
```

2. **View logs:**
```bash
docker-compose logs -f embedding-service
```

3. **Stop the service:**
```bash
docker-compose down
```

### Option 3: Docker with GPU Support

1. **Install NVIDIA Docker runtime:**
```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. **Uncomment GPU configuration in docker-compose.yml:**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

3. **Start the service:**
```bash
docker-compose up -d
```

## Configuration

The service can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSFORMERS_CACHE` | `/models` | Directory to cache downloaded models |
| `HOST` | `0.0.0.0` | Host to bind the service |
| `PORT` | `8000` | Port to bind the service |

## Performance Considerations

### CPU vs GPU
- **CPU**: Suitable for development and low-throughput scenarios (2-5 seconds per batch)
- **GPU**: Recommended for production (0.1-0.5 seconds per batch)

### Batch Sizes
- **Recommended**: 10-50 texts per request
- **Maximum**: 100 texts per request
- Larger batches improve throughput but increase latency

### Model Caching
On first run, the service downloads ~400MB of model files. These are cached in:
- Local: `~/.cache/huggingface/`
- Docker: `/models` volume

## Testing the Service

### Test with cURL
```bash
# Health check
curl http://localhost:8000/health

# Generate embeddings
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Symptoms of COVID-19 include fever and cough"],
    "normalize": true
  }'
```

### Test with Python
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Generate embeddings
response = requests.post(
    "http://localhost:8000/embed",
    json={
        "texts": ["What is machine learning?"],
        "normalize": True
    }
)
embeddings = response.json()["embeddings"]
print(f"Embedding dimension: {len(embeddings[0])}")
```

## Troubleshooting

### Service won't start
1. Check if port 8000 is already in use:
   ```bash
   netstat -an | grep 8000
   ```
2. Check logs for errors:
   ```bash
   docker-compose logs embedding-service
   ```

### Model download fails
1. Check internet connection
2. Manually download model:
   ```python
   from transformers import AutoModel, AutoTokenizer
   model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
   AutoTokenizer.from_pretrained(model_name)
   AutoModel.from_pretrained(model_name)
   ```

### Out of memory errors
1. Reduce batch size in requests
2. Use CPU instead of GPU
3. Increase Docker memory limit

## Integration with Data Ingestion

The data ingestion pipeline automatically uses the embedding service when configured:

1. **Set environment variable:**
```bash
export EMBEDDING_BACKEND=huggingface
export EMBEDDING_SERVICE_URL=http://localhost:8000
export EMBEDDING_DIMENSION=512
```

2. **Run ingestion:**
```bash
python -m src.ingest
```

## Monitoring

### Health Monitoring
The service includes a health check endpoint that can be used by orchestration systems:
```bash
curl http://localhost:8000/health
```

### Logging
Logs include:
- Model loading time
- Request processing times
- Error traces
- Device information (CPU/GPU)

### Metrics
Consider adding Prometheus metrics for production:
- Request count
- Processing latency
- Error rate
- Batch size distribution

## Security Considerations

1. **Network Security**: In production, bind to internal network only
2. **Authentication**: Consider adding API key authentication
3. **Rate Limiting**: Add rate limiting for public deployments
4. **Input Validation**: Service validates input size and format

## License

This service uses the BiomedCLIP model which is released under the MIT License.

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for MinerU and PyTorch
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download MinerU models (optional - can be done at runtime)
# Uncomment to pre-download models during build (increases image size)
# RUN python -c "from magic_pdf.model.model_manager import ModelManager; ModelManager().download_models()"

# Copy application code
COPY src/ ./src/

# NOTE: DO NOT copy .env or service_account.json into the image!
# Mount these as volumes at runtime:
# docker run -v $(pwd)/.env:/app/.env \
#           -v $(pwd)/service_account.json:/app/service_account.json \
#           data-ingestion-app

# Create directories
RUN mkdir -p ./downloads

# Set environment variables for MinerU
ENV PYTHONPATH=/app
ENV MINERU_MODEL_SOURCE=huggingface
ENV MINERU_BACKEND=pipeline
ENV MINERU_LANG=en
ENV PDF_EXTRACTION_TIMEOUT=900
ENV MINERU_DEBUG_MODE=false
ENV MINERU_ENABLE_TABLES=false
ENV MINERU_ENABLE_FORMULAS=false

# Run the ingestion script
CMD ["python", "src/ingest.py"]

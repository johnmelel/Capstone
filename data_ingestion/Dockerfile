FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Copy environment file and service account JSON
COPY .env .env
COPY service_account.json service_account.json

# Create downloads directory
RUN mkdir -p ./downloads

# Set Python path
ENV PYTHONPATH=/app

# Run the ingestion script
CMD ["python", "src/ingest.py"]

#!/bin/bash
# Start the embedding service

echo "Starting BiomedCLIP Embedding Service..."
echo "This may take a few minutes on first run to download the model."

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r embedding_service/requirements.txt

# Start the service
echo "Starting service on http://localhost:8000"
cd embedding_service
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

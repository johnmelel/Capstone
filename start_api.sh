#!/bin/bash
# Start FastAPI Backend for Iris AI Medical Assistant
# Connects React frontend to LangGraph multi-agent backend

echo "======================================"
echo "Starting FastAPI Backend"
echo "======================================"

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d "agents/venv_studio" ]; then
    echo "Activating agents virtual environment..."
    source agents/venv_studio/bin/activate
fi

# Check if required packages are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Error: FastAPI not installed. Installing requirements..."
    pip install -r front_end/api/requirements.txt
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found in project root"
    echo "Make sure environment variables are configured"
fi

echo ""
echo "Starting FastAPI server on http://localhost:8000"
echo "API endpoints:"
echo "  - GET  /         - API info"
echo "  - GET  /health   - Health check"
echo "  - POST /chat     - Process medical queries"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start uvicorn server
python -m uvicorn front_end.api.main:app --reload --port 8000 --host 0.0.0.0

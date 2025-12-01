#!/bin/bash
# Start all A2A workers for the clinical retrieval system

set -e  # Exit on error

echo "Starting A2A Clinical Retrieval System Workers..."
echo "=================================================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: Virtual environment not detected. Make sure to run: source venv/bin/activate"
fi

# Kill any existing processes on these ports
echo "Cleaning up existing processes..."
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
lsof -ti:8002 | xargs kill -9 2>/dev/null || true

sleep 2

# Start structured worker in background
echo "Starting Structured Worker (port 8001)..."
cd "$(dirname "$0")/.."
python3 agents/structured_worker/main.py &
STRUCTURED_PID=$!

# Wait a moment for startup
sleep 3

# Start unstructured worker in background  
echo "Starting Unstructured Worker (port 8002)..."
python3 agents/unstructured_worker/main.py &
UNSTRUCTURED_PID=$!

# Wait for workers to start up
sleep 5

# Check if workers are running
echo "Checking worker status..."

if curl -s http://localhost:8001/health >/dev/null; then
    echo "✓ Structured Worker (port 8001) is running"
else
    echo "✗ Structured Worker (port 8001) failed to start"
    exit 1
fi

if curl -s http://localhost:8002/health >/dev/null; then
    echo "✓ Unstructured Worker (port 8002) is running"
else
    echo "✗ Unstructured Worker (port 8002) failed to start"
    exit 1
fi

echo ""
echo "Workers are ready! You can now run queries:"
echo "  python3 scripts/query_system.py \"Get patients who took amphetamine in last 24 hours\""
echo ""
echo "To stop workers: kill $STRUCTURED_PID $UNSTRUCTURED_PID"
echo "Or use Ctrl+C to stop this script and all workers"

# Keep script running and handle shutdown
trap "echo 'Shutting down workers...'; kill $STRUCTURED_PID $UNSTRUCTURED_PID 2>/dev/null || true; exit 0" INT TERM

# Wait for workers
wait $STRUCTURED_PID $UNSTRUCTURED_PID

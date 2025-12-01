#!/bin/bash
# Start React Frontend for Iris AI Medical Assistant

echo "======================================"
echo "Starting React Frontend"
echo "======================================"

cd "$(dirname "$0")/front_end/client"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "node_modules not found. Installing dependencies..."
    npm install
fi

echo ""
echo "Starting Vite dev server..."
echo "Frontend will be available at http://localhost:5173"
echo ""
echo "Make sure the FastAPI backend is running on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# Start Vite dev server
npm run dev

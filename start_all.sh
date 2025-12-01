#!/bin/bash
# Master startup script for Iris AI Medical Assistant
# Starts all three components in separate terminal tabs/windows

echo "=============================================="
echo "Iris AI Medical Assistant - Complete Startup"
echo "=============================================="
echo ""
echo "This script will start:"
echo "  1. FastAPI Backend (port 8000)"
echo "  2. React Frontend (port 5173)"
echo ""
echo "Note: MCP server runs on stdio and is called by the backend"
echo ""

cd "$(dirname "$0")"

# Make scripts executable
chmod +x start_api.sh start_frontend.sh

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please ensure .env exists in the project root with:"
    echo "  - GOOGLE_API_KEY"
    echo "  - LANGSMITH_API_KEY"
    echo "  - MILVUS_URI"
    echo "  - MILVUS_API_KEY"
    exit 1
fi

# Detect OS and terminal
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Starting on macOS..."
    echo ""
    
    # Start API backend in new terminal
    osascript -e 'tell application "Terminal" to do script "cd '"$(pwd)"' && ./start_api.sh"'
    echo "✓ FastAPI backend starting in new terminal..."
    
    sleep 2
    
    # Start frontend in new terminal
    osascript -e 'tell application "Terminal" to do script "cd '"$(pwd)"' && ./start_frontend.sh"'
    echo "✓ React frontend starting in new terminal..."
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Starting on Linux..."
    echo ""
    
    # Try different terminal emulators
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "cd $(pwd) && ./start_api.sh; exec bash"
        gnome-terminal -- bash -c "cd $(pwd) && ./start_frontend.sh; exec bash"
    elif command -v xterm &> /dev/null; then
        xterm -e "cd $(pwd) && ./start_api.sh" &
        xterm -e "cd $(pwd) && ./start_frontend.sh" &
    else
        echo "No suitable terminal found. Please run these commands manually:"
        echo "  Terminal 1: ./start_api.sh"
        echo "  Terminal 2: ./start_frontend.sh"
        exit 1
    fi
    
    echo "✓ FastAPI backend starting..."
    echo "✓ React frontend starting..."
else
    echo "Unsupported OS. Please run these commands manually:"
    echo "  Terminal 1: ./start_api.sh"
    echo "  Terminal 2: ./start_frontend.sh"
    exit 1
fi

echo ""
echo "=============================================="
echo "Startup initiated!"
echo "=============================================="
echo ""
echo "Wait 10-15 seconds for services to start, then:"
echo ""
echo "  Frontend:  http://localhost:5173"
echo "  API:       http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo ""
echo "To stop: Close the terminal windows or press Ctrl+C in each"
echo ""

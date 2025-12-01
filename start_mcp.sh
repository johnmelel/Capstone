#!/bin/bash
# Start MCP Server for Medical Assistant
# This provides EMR database and vector search access

echo "======================================"
echo "Starting MCP Server"
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

# Check if MCP server requirements are installed
if ! python -c "import mcp" 2>/dev/null; then
    echo "Error: MCP not installed. Installing requirements..."
    pip install -r mcp-servers/requirements.txt
fi

echo ""
echo "Starting MCP server on stdio..."
echo "MCP server provides:"
echo "  - EMR database queries (SQLite)"
echo "  - Vector search (Milvus)"
echo "  - Query rewriting"
echo ""

cd mcp-servers
python server.py

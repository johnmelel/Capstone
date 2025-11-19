@echo off
REM Start the embedding service on Windows

echo Starting BiomedCLIP Embedding Service...
echo This may take a few minutes on first run to download the model.

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r embedding_service\requirements.txt

REM Start the service
echo Starting service on http://localhost:8000
cd embedding_service
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

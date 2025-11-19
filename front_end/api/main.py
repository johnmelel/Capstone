import sys
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the project root to python path to allow imports from medical-agent
# Assuming this file is in front_end/api/main.py and we need to reach medical-agent/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(project_root, "medical-agent"))

try:
    from orchestrator.main import Orchestrator
except ImportError as e:
    print(f"Error importing Orchestrator: {e}")
    # Create a dummy class if import fails to allow server to start for testing
    class Orchestrator:
        def __init__(self, *args, **kwargs):
            pass
        def query(self, q):
            return {"answer": "Orchestrator not available", "error": str(e)}

app = FastAPI(title="Medical Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In production, specify the React app URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Orchestrator
# We need to handle the service account path. 
# We'll look for it in the root or use an environment variable.
SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "adsp-34002-ip09-team-2-a02bd14c0e77.json")
SERVICE_ACCOUNT_PATH = os.path.join(project_root, SERVICE_ACCOUNT_FILE)

# Global orchestrator instance
orchestrator = None

@app.on_event("startup")
async def startup_event():
    global orchestrator
    print(f"Initializing Orchestrator with service account: {SERVICE_ACCOUNT_PATH}")
    # We pass the path, the Orchestrator class handles the rest.
    # Note: The Orchestrator expects the workers to be running on localhost:8001 and localhost:8002
    orchestrator = Orchestrator(service_account_path=SERVICE_ACCOUNT_PATH)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    structured_source: Optional[Any] = None
    unstructured_source: Optional[Any] = None
    reasoning_steps: Optional[int] = None
    error: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        # The orchestrator.query method returns a dict
        result = orchestrator.query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "orchestrator_initialized": orchestrator is not None}

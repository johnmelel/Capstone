"""
FastAPI Backend for Iris AI Medical Assistant
Integrates React frontend with LangGraph multi-agent backend
"""
import sys
import os
import re
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

# Add agents directory to path
agents_path = os.path.join(project_root, "agents")
sys.path.insert(0, project_root)
sys.path.insert(0, agents_path)

# Import LangGraph multi-agent system
try:
    from agents.graph import run_medical_query
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"Error importing LangGraph backend: {e}")
    BACKEND_AVAILABLE = False

app = FastAPI(title="Iris AI Medical Assistant API")

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - specify React URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    structured_source: Optional[Any] = None
    unstructured_source: Optional[Any] = None
    reasoning_steps: Optional[int] = None
    error: Optional[str] = None


def extract_patient_id(query: str) -> Optional[str]:
    """
    Extract patient ID from query text.
    
    Looks for patterns like:
    - "patient 10000032"
    - "patient ID 10000032"
    - "subject_id 10000032"
    
    Args:
        query: User's query text
    
    Returns:
        Patient ID string if found, None otherwise
    """
    patterns = [
        r'patient\s+(?:id\s+)?(\d+)',
        r'subject[_\s]?id\s+(\d+)',
        r'pt\s+(\d+)',
    ]
    
    query_lower = query.lower()
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            return match.group(1)
    
    return None


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - processes user queries through multi-agent system.
    
    Frontend sends:
        {"query": "What are lab results for patient 10000032?"}
    
    Backend processes through:
        User Query → Planner → Router → {EMR, Research} → Aggregator → Response
    
    Returns:
        {
            "answer": "Clinical report with findings",
            "structured_source": {...EMR data if used...},
            "unstructured_source": {...research data if used...},
            "reasoning_steps": 4  # Number of agent coordination steps
        }
    """
    if not BACKEND_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Multi-agent backend not available. Check agents/ installation."
        )
    
    try:
        # Extract patient ID from query if present
        patient_id = extract_patient_id(request.query)
        
        # Run query through LangGraph multi-agent system
        result = await run_medical_query(
            query=request.query,
            patient_id=patient_id,
            user_id=None  # TODO: Add auth and extract user_id
        )
        
        # Map backend state to frontend expected format
        response = ChatResponse(
            answer=result.get("final_response", "No response generated"),
            structured_source=result.get("emr_result"),
            unstructured_source=result.get("research_result"),
            reasoning_steps=len(result.get("messages", []))
        )
        
        return response
        
    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "backend_available": BACKEND_AVAILABLE,
        "service": "Iris AI Medical Assistant API"
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Iris AI Medical Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat - Process medical queries",
            "health": "GET /health - Health check"
        },
        "backend_status": "available" if BACKEND_AVAILABLE else "unavailable"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

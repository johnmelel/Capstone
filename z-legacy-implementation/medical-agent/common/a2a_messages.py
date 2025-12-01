"""
A2A (Agent-to-Agent) Message Protocol
Minimal implementation built as we discover requirements
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid


class A2ATask(BaseModel):
    """A2A Task message sent from orchestrator to workers"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str  # "structured_query" or "unstructured_search"
    query: str  # The actual query/search terms
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class A2AArtifact(BaseModel):
    """A2A Artifact response from workers to orchestrator"""
    task_id: str
    success: bool
    answer: str  # Summary/result
    evidence: List[str] = Field(default_factory=list)  # Source tables/docs
    raw_data: Optional[Dict[str, Any]] = None  # Raw query results
    error_message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MCPRequest(BaseModel):
    """MCP protocol request wrapper"""
    method: str  # "call_tool"
    tool_name: str  # "sql.query" or "vector.search"
    arguments: Dict[str, Any]


class MCPResponse(BaseModel):
    """MCP protocol response wrapper"""
    success: bool
    result: Any = None
    error: Optional[str] = None

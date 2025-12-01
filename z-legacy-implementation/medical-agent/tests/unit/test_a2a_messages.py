"""
Tests for A2A messaging protocol
"""
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.a2a_messages import A2ATask, A2AArtifact, MCPRequest, MCPResponse
from datetime import datetime


def test_a2a_task_creation():
    """Test A2A task creation and serialization"""
    task = A2ATask(
        task_type="structured_query",
        query="Get patients who took Amphetamine in last 24 hours",
        parameters={"hours_back": 24}
    )
    
    assert task.task_type == "structured_query"
    assert task.query == "Get patients who took Amphetamine in last 24 hours"
    assert task.parameters["hours_back"] == 24
    assert task.task_id is not None
    assert isinstance(task.timestamp, str)  # Timestamp is stored as ISO string


def test_a2a_artifact_creation():
    """Test A2A artifact creation and serialization"""
    artifact = A2AArtifact(
        task_id="test-123",
        success=True,
        answer="Found 3 patients",
        evidence=["prescriptions", "emar"],
        raw_data={"patient_count": 3}
    )
    
    assert artifact.task_id == "test-123"
    assert artifact.success is True
    assert artifact.answer == "Found 3 patients"
    assert "prescriptions" in artifact.evidence
    assert artifact.raw_data["patient_count"] == 3


def test_a2a_task_serialization():
    """Test task can be serialized and deserialized"""
    original_task = A2ATask(
        task_type="structured_query",
        query="Test query"
    )
    
    # Test model_dump and reconstruction
    task_data = original_task.model_dump()
    reconstructed_task = A2ATask(**task_data)
    
    assert reconstructed_task.task_type == original_task.task_type
    assert reconstructed_task.query == original_task.query
    assert reconstructed_task.task_id == original_task.task_id


def test_mcp_request_response():
    """Test MCP request/response structures"""
    request = MCPRequest(
        method="call_tool",
        tool_name="sql.query",
        arguments={"sql": "SELECT * FROM patients LIMIT 5"}
    )
    
    response = MCPResponse(
        success=True,
        result={"rows": [], "row_count": 0}
    )
    
    assert request.method == "call_tool"
    assert request.tool_name == "sql.query"
    assert response.success is True
    assert response.result["row_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__])

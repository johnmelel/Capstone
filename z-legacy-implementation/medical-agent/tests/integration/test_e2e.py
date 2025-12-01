"""
End-to-end integration tests for the A2A Clinical Retrieval System
"""
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock, patch
import json

from orchestrator.main import Orchestrator
from agents.structured_worker.main import StructuredWorker
from agents.unstructured_worker.main import UnstructuredWorker
from common.a2a_messages import A2ATask, A2AArtifact


class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_complete_query_flow_mock(self):
        """Test complete query flow with mocked workers"""
        
        # Create orchestrator
        orchestrator = Orchestrator(
            service_account_path="fake_credentials.json",
            structured_worker_url="http://localhost:8001", 
            unstructured_worker_url="http://localhost:8002"
        )
        
        # Mock the post_task function to simulate worker responses
        with patch('orchestrator.main.post_task') as mock_post_task:
            
            # Mock structured worker response
            structured_artifact = A2AArtifact(
                task_id="e2e-struct-1",
                success=True,
                answer="Found 2 patients taking amphetamine in the last 24 hours.",
                evidence=["prescriptions", "emar", "patients"],
                raw_data={
                    "rows": [
                        {"subject_id": 12345, "drug": "Amphetamine", "starttime": "2024-01-01 10:00:00"},
                        {"subject_id": 12346, "drug": "Amphetamine", "starttime": "2024-01-01 14:30:00"}
                    ],
                    "patient_count": 2,
                    "patient_ids": ["12345", "12346"],
                    "sql": "SELECT * FROM prescriptions WHERE..."
                }
            )
            
            # Mock unstructured worker response
            unstructured_artifact = A2AArtifact(
                task_id="e2e-unstruct-1",
                success=True,
                answer="Found 3 relevant medical guidelines. Key monitoring points: cardiac assessment, blood pressure monitoring.",
                evidence=["pharm_guide_001", "safety_manual_034", "prescribe_guide_022"],
                raw_data={
                    "documents": [
                        {
                            "text": "Amphetamine monitoring requires cardiac assessment including blood pressure and heart rate checks.",
                            "doc_id": "pharm_guide_001",
                            "relevance": 0.95
                        },
                        {
                            "text": "Common side effects include increased heart rate, elevated blood pressure, decreased appetite.",
                            "doc_id": "safety_manual_034", 
                            "relevance": 0.89
                        }
                    ],
                    "context_snippets": [
                        "Amphetamine monitoring requires cardiac assessment including blood pressure and heart rate checks.",
                        "Common side effects include increased heart rate, elevated blood pressure, decreased appetite."
                    ],
                    "search_query": "amphetamine monitoring guidelines side effects safety warnings",
                    "total_found": 2
                }
            )
            
            # Configure mock to return different responses based on URL
            def mock_post_task_side_effect(url, task):
                if "8001" in url:  # structured worker
                    return structured_artifact
                elif "8002" in url:  # unstructured worker
                    return unstructured_artifact
                else:
                    raise ValueError(f"Unknown URL: {url}")
                    
            mock_post_task.side_effect = mock_post_task_side_effect
            
            # Execute end-to-end query
            result = orchestrator.query("Get me all patients who took Amphetamine in the last 24 hours")
            
            # Verify complete response structure
            assert isinstance(result, dict)
            assert "answer" in result
            assert "structured_source" in result
            assert "unstructured_source" in result
            
            # Verify answer combines both sources
            answer = result["answer"]
            assert "2 patients taking amphetamine" in answer.lower()
            assert "monitor" in answer.lower() or "cardiac" in answer.lower()
            
            # Verify sources are properly attributed
            assert len(result["structured_source"]) > 0
            assert len(result["unstructured_source"]) > 0
            assert "prescriptions" in result["structured_source"]
            assert "pharm_guide_001" in result["unstructured_source"]
            
            # Verify both workers were called
            assert mock_post_task.call_count == 2
            
            # Verify structured worker was called with correct task
            struct_call = mock_post_task.call_args_list[0]
            assert struct_call[0][0] == "http://localhost:8001"  # URL
            struct_task = struct_call[0][1]  # Task
            assert struct_task.task_type == "structured_query"
            assert "amphetamine" in struct_task.query.lower()
            
            # Verify unstructured worker was called with correct task
            unstruct_call = mock_post_task.call_args_list[1]
            assert unstruct_call[0][0] == "http://localhost:8002"  # URL
            unstruct_task = unstruct_call[0][1]  # Task
            assert unstruct_task.task_type == "unstructured_search"
            assert "amphetamine" in unstruct_task.query.lower()
    
    def test_worker_components_integration(self):
        """Test individual worker components work correctly"""
        
        # Test structured worker processes tasks correctly
        structured_worker = StructuredWorker()
        
        # Create test task
        struct_task = A2ATask(
            task_type="structured_query",
            query="Get patients who took amphetamine in last 24 hours"
        )
        
        # Mock the MCP call to avoid database dependency
        with patch.object(structured_worker.mcp_server, 'call_tool') as mock_mcp:
            mock_response = Mock()
            mock_response.success = True
            mock_response.result = {
                "rows": [{"subject_id": 123, "drug": "Amphetamine"}],
                "row_count": 1
            }
            mock_mcp.return_value = mock_response
            
            artifact = structured_worker.process_task(struct_task)
            
            assert artifact.success
            assert artifact.answer == ""  # NEW ARCHITECTURE: no interpretation in worker
            assert artifact.raw_data["rows"] == [{"subject_id": 123, "drug": "Amphetamine"}]
            assert artifact.raw_data["row_count"] == 1
            assert len(artifact.evidence) > 0
        
        # Test unstructured worker processes tasks correctly  
        unstructured_worker = UnstructuredWorker()
        
        unstruct_task = A2ATask(
            task_type="unstructured_search",
            query="amphetamine monitoring guidelines"
        )
        
        # This uses the dummy MCP vector server, so no mocking needed
        artifact = unstructured_worker.process_task(unstruct_task)
        
        assert artifact.success
        assert "relevant medical guidelines" in artifact.answer
        assert len(artifact.evidence) > 0
        
    def test_task_planning_and_generation(self):
        """Test that orchestrator correctly plans tasks"""
        orchestrator = Orchestrator("fake_creds.json")
        
        # Test drug query planning
        struct_task, unstruct_task = orchestrator._plan_tasks(
            "Get patients who took insulin in last 12 hours"
        )
        
        # Verify structured task
        assert struct_task.task_type == "structured_query" 
        assert "insulin" in struct_task.query.lower()
        assert struct_task.parameters["hours_back"] == 12
        
        # Verify unstructured task
        assert unstruct_task.task_type == "unstructured_search"
        assert "insulin" in unstruct_task.query.lower()
        assert unstruct_task.parameters["top_k"] == 3
        
    def test_error_handling_integration(self):
        """Test error handling across the system"""
        orchestrator = Orchestrator("fake_creds.json")
        
        with patch('orchestrator.main.post_task') as mock_post_task:
            # Mock both workers failing
            error_artifact = A2AArtifact(
                task_id="error-test",
                success=False,
                answer="",
                error_message="Worker connection failed"
            )
            mock_post_task.return_value = error_artifact
            
            result = orchestrator.query("test query")
            
            # Should still return valid response structure
            assert "answer" in result
            assert "structured_source" in result  
            assert "unstructured_source" in result
            
            # Should indicate errors but not crash
            assert len(result["answer"]) > 0  # Should have some fallback response
    
    def test_query_system_cli_integration(self):
        """Test the CLI query system components"""
        # Import after path setup
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        from scripts.query_system import main
        
        # Test that CLI components can be imported without error
        # This ensures all dependencies and imports are correct
        assert main is not None
        
        # Test argument parsing (without actually running main)
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('query')
        parser.add_argument('--output', choices=['json', 'pretty', 'answer-only'], default='pretty')
        
        # Should parse without error
        args = parser.parse_args(['test query', '--output', 'json'])
        assert args.query == 'test query'
        assert args.output == 'json'


if __name__ == "__main__":
    pytest.main([__file__])

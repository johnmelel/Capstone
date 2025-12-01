"""
Tests for Unstructured Worker
"""
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.a2a_messages import A2ATask, A2AArtifact
from unittest.mock import Mock, patch


# Import unstructured worker after path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents', 'unstructured_worker'))
from agents.unstructured_worker.main import UnstructuredWorker


class TestUnstructuredWorker:
    """Test unstructured worker functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.worker = UnstructuredWorker()
        
    def test_unsupported_task_type(self):
        """Test handling of unsupported task types"""
        task = A2ATask(
            task_type="unsupported_type",
            query="test query"
        )
        
        artifact = self.worker.process_task(task)
        
        assert not artifact.success
        assert "Unsupported task type" in artifact.error_message
        assert artifact.task_id == task.task_id
        
    def test_generate_search_query_drug_monitoring(self):
        """Test search query generation for drug monitoring"""
        query = "What are the monitoring requirements for amphetamine?"
        search_query = self.worker._generate_search_query(query, {})
        
        assert "amphetamine" in search_query
        assert "monitoring" in search_query
        
    def test_generate_search_query_drug_safety(self):
        """Test search query generation for drug safety"""
        query = "What are the safety warnings for insulin?"
        search_query = self.worker._generate_search_query(query, {})
        
        assert "insulin" in search_query
        assert "safety" in search_query
        
    def test_generate_search_query_general_admission(self):
        """Test search query generation for hospital admissions"""
        query = "What are the protocols for patient admission?"
        search_query = self.worker._generate_search_query(query, {})
        
        assert "hospital admission" in search_query
        assert "protocols" in search_query
        
    def test_generate_search_query_fallback(self):
        """Test search query generation fallback to original query"""
        query = "Some random medical question"
        search_query = self.worker._generate_search_query(query, {})
        
        assert search_query == query
        
    def test_format_results_no_documents(self):
        """Test result formatting with no documents"""
        search_result = {"documents": [], "total_found": 0}
        artifact = self.worker._format_results_to_artifact("test-id", search_result, "test query")
        
        assert artifact.success
        assert "No relevant medical context found" in artifact.answer
        assert "medical_knowledge_base" in artifact.evidence
        
    def test_format_results_with_documents(self):
        """Test result formatting with medical documents"""
        search_result = {
            "documents": [
                {
                    "text": "Amphetamine monitoring requires cardiac assessment including blood pressure checks.",
                    "doc_id": "doc_001",
                    "relevance": 0.95
                },
                {
                    "text": "Monitor for arrhythmias and hypertension in amphetamine patients.",
                    "doc_id": "doc_002", 
                    "relevance": 0.89
                }
            ],
            "total_found": 2
        }
        
        artifact = self.worker._format_results_to_artifact("test-id", search_result, "amphetamine monitoring")
        
        assert artifact.success
        assert "Found 2 relevant medical guidelines" in artifact.answer
        assert "cardiac assessment" in artifact.answer
        assert "doc_001" in artifact.evidence
        assert "doc_002" in artifact.evidence
        assert len(artifact.raw_data["context_snippets"]) == 2
        
    def test_format_results_side_effects_context(self):
        """Test result formatting with side effects context"""
        search_result = {
            "documents": [
                {
                    "text": "Common side effects of amphetamine include increased heart rate and elevated blood pressure.",
                    "doc_id": "side_effects_001",
                    "relevance": 0.92
                }
            ],
            "total_found": 1
        }
        
        artifact = self.worker._format_results_to_artifact("test-id", search_result, "amphetamine side effects")
        
        assert artifact.success
        assert "side effects noted" in artifact.answer
        
    def test_format_results_contraindications_context(self):
        """Test result formatting with contraindications context"""
        search_result = {
            "documents": [
                {
                    "text": "Amphetamine contraindications include severe cardiovascular disease.",
                    "doc_id": "contraind_001",
                    "relevance": 0.87
                }
            ],
            "total_found": 1
        }
        
        artifact = self.worker._format_results_to_artifact("test-id", search_result, "amphetamine contraindications")
        
        assert artifact.success
        assert "contraindications and safety warnings" in artifact.answer
        
    @patch('common.mcp_vector_server.MCPVectorServer.call_tool')
    def test_process_task_mcp_error(self, mock_mcp_call):
        """Test task processing when MCP call fails"""
        # Mock MCP failure
        mock_response = Mock()
        mock_response.success = False
        mock_response.error = "Vector database connection failed"
        mock_mcp_call.return_value = mock_response
        
        task = A2ATask(
            task_type="unstructured_search",
            query="amphetamine monitoring guidelines"
        )
        
        artifact = self.worker.process_task(task)
        
        assert not artifact.success
        assert "Vector search failed" in artifact.error_message
        assert "Vector database connection failed" in artifact.error_message
        
    @patch('common.mcp_vector_server.MCPVectorServer.call_tool')
    def test_process_task_success(self, mock_mcp_call):
        """Test successful task processing"""
        # Mock successful MCP response
        mock_response = Mock()
        mock_response.success = True
        mock_response.result = {
            "documents": [
                {
                    "text": "Amphetamine monitoring requires cardiac assessment.",
                    "doc_id": "guide_001",
                    "relevance": 0.95
                }
            ],
            "total_found": 1,
            "query": "amphetamine monitoring guidelines"
        }
        mock_mcp_call.return_value = mock_response
        
        task = A2ATask(
            task_type="unstructured_search",
            query="amphetamine monitoring guidelines"
        )
        
        artifact = self.worker.process_task(task)
        
        assert artifact.success
        assert "Found 1 relevant medical guidelines" in artifact.answer
        assert artifact.task_id == task.task_id
        assert len(artifact.evidence) > 0
        assert "guide_001" in artifact.evidence
        
    def test_process_task_custom_top_k(self):
        """Test task processing with custom top_k parameter"""
        task = A2ATask(
            task_type="unstructured_search",
            query="amphetamine monitoring",
            parameters={"top_k": 5}
        )
        
        # This should not raise an error and should use the custom top_k
        # We can't easily test the MCP call here without mocking, but we can test
        # that the parameter is extracted correctly
        with patch.object(self.worker.mcp_server, 'call_tool') as mock_call:
            mock_response = Mock()
            mock_response.success = True
            mock_response.result = {"documents": [], "total_found": 0}
            mock_call.return_value = mock_response
            
            artifact = self.worker.process_task(task)
            
            # Check that call_tool was called with correct top_k
            args, kwargs = mock_call.call_args
            assert kwargs == {}  # No kwargs passed
            assert args[0] == "vector.search"
            assert args[1]["top_k"] == 5


if __name__ == "__main__":
    pytest.main([__file__])

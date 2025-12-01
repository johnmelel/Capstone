"""
Tests for Orchestrator
"""
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock, patch, MagicMock
import json

from common.a2a_messages import A2ATask, A2AArtifact


# Import orchestrator after path setup  
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'orchestrator'))
from orchestrator.main import Orchestrator


class TestOrchestrator:
    """Test orchestrator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create orchestrator with mocked Gemini (will fail gracefully)
        self.orchestrator = Orchestrator(
            service_account_path="fake_service_account.json",
            structured_worker_url="http://localhost:8001",
            unstructured_worker_url="http://localhost:8002"
        )
        
    def test_extract_time_period_default(self):
        """Test time period extraction defaults to 24 hours"""
        period = self.orchestrator._extract_time_period("Get patients who took amphetamine")
        assert period == 24
        
    def test_extract_time_period_12_hours(self):
        """Test extracting 12 hour period"""
        period1 = self.orchestrator._extract_time_period("patients in last 12 hours")
        period2 = self.orchestrator._extract_time_period("patients in past 12 hours") 
        assert period1 == 12
        assert period2 == 12
        
    def test_extract_time_period_48_hours(self):
        """Test extracting 48 hour period"""
        period = self.orchestrator._extract_time_period("medications in last 48 hours")
        assert period == 48
        
    def test_extract_time_period_72_hours(self):
        """Test extracting 72 hour period"""
        period = self.orchestrator._extract_time_period("admissions in past 72 hours")
        assert period == 72
        
    def test_generate_context_query_drug(self):
        """Test context query generation for drug queries"""
        context = self.orchestrator._generate_context_query("patients who took amphetamine")
        assert "amphetamine" in context
        assert "monitoring" in context
        assert "guidelines" in context
        
    def test_generate_context_query_admission(self):
        """Test context query generation for admission queries"""
        context = self.orchestrator._generate_context_query("recent patient admissions")
        assert "patient care" in context
        assert "admission" in context
        
    def test_generate_context_query_fallback(self):
        """Test context query generation fallback"""
        query = "some other medical question"
        context = self.orchestrator._generate_context_query(query)
        assert context == query
        
    def test_plan_tasks(self):
        """Test task planning creates correct A2A tasks"""
        user_question = "Get patients who took amphetamine in last 12 hours"
        
        structured_task, unstructured_task = self.orchestrator._plan_tasks(user_question)
        
        # Check structured task
        assert structured_task.task_type == "structured_query"
        assert structured_task.query == user_question
        assert structured_task.parameters["hours_back"] == 12
        
        # Check unstructured task  
        assert unstructured_task.task_type == "unstructured_search"
        assert "amphetamine" in unstructured_task.query
        assert unstructured_task.parameters["top_k"] == 3
        
    def test_prepare_reasoning_context_success(self):
        """Test reasoning context preparation with successful results"""
        user_question = "Get amphetamine patients"
        
        structured_result = A2AArtifact(
            task_id="test-1",
            success=True,
            answer="Found 3 patients",
            evidence=["prescriptions", "emar"],
            raw_data={"patient_count": 3}
        )
        
        unstructured_result = A2AArtifact(
            task_id="test-2", 
            success=True,
            answer="Monitor for cardiac effects",
            evidence=["clinical_guidelines"],
            raw_data={"context_snippets": ["Monitor blood pressure", "Check for arrhythmias"]}
        )
        
        context = self.orchestrator._prepare_reasoning_context(
            user_question, structured_result, unstructured_result
        )
        
        assert "Get amphetamine patients" in context
        # NEW ARCHITECTURE: orchestrator gets raw database results
        assert "Get amphetamine patients" in context
        assert "Monitor for cardiac effects" in context  
        assert "Row Count: 0" in context  # Because no raw rows provided in this test
        assert "Monitor blood pressure" in context
        
    def test_prepare_reasoning_context_errors(self):
        """Test reasoning context preparation with error results"""
        user_question = "Test query"
        
        structured_result = A2AArtifact(
            task_id="test-1",
            success=False,
            answer="",
            error_message="Database connection failed"
        )
        
        unstructured_result = A2AArtifact(
            task_id="test-2",
            success=False, 
            answer="",
            error_message="Vector search failed"
        )
        
        context = self.orchestrator._prepare_reasoning_context(
            user_question, structured_result, unstructured_result
        )
        
        assert "Database connection failed" in context
        assert "Vector search failed" in context
        
    def test_generate_fallback_response_success(self):
        """Test fallback response generation with successful results"""
        structured_result = A2AArtifact(
            task_id="test-1",
            success=True,
            answer="Found 2 patients with amphetamine.",
            evidence=["prescriptions"]
        )
        
        unstructured_result = A2AArtifact(
            task_id="test-2",
            success=True,
            answer="Monitor cardiac function.",
            evidence=["guidelines"]
        )
        
        response = self.orchestrator._generate_fallback_response(structured_result, unstructured_result)
        
        assert "Found 2 patients with amphetamine" in response["answer"]
        assert "Monitor cardiac function" in response["answer"] 
        assert "prescriptions" in response["structured_source"]
        assert "guidelines" in response["unstructured_source"]
        
    def test_generate_fallback_response_errors(self):
        """Test fallback response with errors"""
        structured_result = A2AArtifact(
            task_id="test-1",
            success=False,
            answer="",
            error_message="SQL error"
        )
        
        unstructured_result = A2AArtifact(
            task_id="test-2",
            success=False,
            answer="",
            error_message="Vector error"
        )
        
        response = self.orchestrator._generate_fallback_response(structured_result, unstructured_result)
        
        assert "Unable to process query due to worker errors" in response["answer"]
        assert response["structured_source"] == []
        assert response["unstructured_source"] == []
        
    def test_generate_fallback_response_with_error_message(self):
        """Test fallback response includes error message"""
        structured_result = A2AArtifact(task_id="test-1", success=True, answer="Success", evidence=[])
        unstructured_result = A2AArtifact(task_id="test-2", success=True, answer="Success", evidence=[])
        
        response = self.orchestrator._generate_fallback_response(
            structured_result, unstructured_result, error="Gemini API failed"
        )
        
        assert "Advanced reasoning unavailable: Gemini API failed" in response["answer"]
        
    @patch('orchestrator.main.post_task')
    def test_send_structured_task_success(self, mock_post_task):
        """Test successful structured task sending"""
        task = A2ATask(task_type="structured_query", query="test")
        expected_artifact = A2AArtifact(task_id=task.task_id, success=True, answer="Success", evidence=[])
        mock_post_task.return_value = expected_artifact
        
        result = self.orchestrator._send_structured_task(task)
        
        assert result == expected_artifact
        mock_post_task.assert_called_once_with("http://localhost:8001", task)
        
    @patch('orchestrator.main.post_task')
    def test_send_structured_task_error(self, mock_post_task):
        """Test structured task sending with error"""
        task = A2ATask(task_type="structured_query", query="test")
        mock_post_task.side_effect = Exception("Connection failed")
        
        result = self.orchestrator._send_structured_task(task)
        
        assert not result.success
        assert "Structured worker error: Connection failed" in result.error_message
        
    @patch('orchestrator.main.post_task')
    def test_send_unstructured_task_success(self, mock_post_task):
        """Test successful unstructured task sending"""
        task = A2ATask(task_type="unstructured_search", query="test")
        expected_artifact = A2AArtifact(task_id=task.task_id, success=True, answer="Success", evidence=[])
        mock_post_task.return_value = expected_artifact
        
        result = self.orchestrator._send_unstructured_task(task)
        
        assert result == expected_artifact
        mock_post_task.assert_called_once_with("http://localhost:8002", task)
        
    @patch('orchestrator.main.post_task')
    def test_send_unstructured_task_error(self, mock_post_task):
        """Test unstructured task sending with error"""
        task = A2ATask(task_type="unstructured_search", query="test")
        mock_post_task.side_effect = Exception("Connection failed")
        
        result = self.orchestrator._send_unstructured_task(task)
        
        assert not result.success
        assert "Unstructured worker error: Connection failed" in result.error_message
        
    @patch('orchestrator.main.post_task')
    def test_query_end_to_end_mock(self, mock_post_task):
        """Test end-to-end query processing with mocked workers"""
        # Mock worker responses
        structured_artifact = A2AArtifact(
            task_id="struct-1",
            success=True,
            answer="Found 2 patients taking amphetamine",
            evidence=["prescriptions", "emar"],
            raw_data={"patient_count": 2}
        )
        
        unstructured_artifact = A2AArtifact(
            task_id="unstruct-1", 
            success=True,
            answer="Monitor cardiac function and blood pressure",
            evidence=["clinical_guidelines"],
            raw_data={"context_snippets": ["Check BP regularly"]}
        )
        
        # Mock post_task to return different results based on URL
        def mock_post_task_side_effect(url, task):
            if "8001" in url:  # structured worker
                return structured_artifact
            else:  # unstructured worker
                return unstructured_artifact
                
        mock_post_task.side_effect = mock_post_task_side_effect
        
        # Test query
        result = self.orchestrator.query("Get patients who took amphetamine in last 24 hours")
        
        # Verify result structure
        assert "answer" in result
        assert "structured_source" in result
        assert "unstructured_source" in result
        
        # Should contain both worker results (since Gemini will fallback)
        assert "Found 2 patients taking amphetamine" in result["answer"]
        assert "Monitor cardiac function" in result["answer"]
        
    def test_query_with_exception(self):
        """Test query handling when exception occurs"""
        # Force an exception during planning
        with patch.object(self.orchestrator, '_plan_tasks', side_effect=Exception("Planning failed")):
            result = self.orchestrator.query("test query")
            
            assert "error" in result
            assert "Error processing query: Planning failed" in result["answer"]
            assert result["structured_source"] == []
            assert result["unstructured_source"] == []


if __name__ == "__main__":
    pytest.main([__file__])

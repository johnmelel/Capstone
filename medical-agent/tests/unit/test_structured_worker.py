"""
Tests for Structured Worker
"""
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.a2a_messages import A2ATask, A2AArtifact
from unittest.mock import Mock, patch


# Import structured worker after path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents', 'structured_worker'))
from agents.structured_worker.main import StructuredWorker


class TestStructuredWorker:
    """Test structured worker functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.worker = StructuredWorker()
        
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
        
    def test_generate_sql_from_query_amphetamine(self):
        """Test SQL generation for amphetamine query"""
        query = "Get me all patients who took Amphetamine in the last 24 hours"
        sql = self.worker._generate_sql_from_query(query, {})
        
        assert sql is not None
        assert "amphetamine" in sql.lower()
        assert "24" in sql
        assert "select" in sql.lower()
        assert "patients" in sql.lower()
        
    def test_generate_sql_from_query_different_hours(self):
        """Test SQL generation with different time periods"""
        query1 = "patients who took aspirin in the last 12 hours"
        sql1 = self.worker._generate_sql_from_query(query1, {})
        assert "12" in sql1
        
        query2 = "patients who took insulin in the past 48 hours"
        sql2 = self.worker._generate_sql_from_query(query2, {})
        assert "48" in sql2
        
    def test_generate_sql_from_query_recent_admissions(self):
        """Test SQL generation for recent admissions"""
        query = "Show me recent admissions in the last 24 hours"
        sql = self.worker._generate_sql_from_query(query, {})
        
        assert sql is not None
        assert "admissions" in sql.lower()
        assert "24" in sql
        
    def test_generate_sql_from_query_unsupported(self):
        """Test SQL generation for unsupported queries"""
        query = "What is the weather like today?"
        sql = self.worker._generate_sql_from_query(query, {})
        
        assert sql is None
        
    def test_format_results_no_data(self):
        """Test result formatting with no data - NEW ARCHITECTURE: raw data only"""
        sql_result = {"rows": [], "row_count": 0}
        artifact = self.worker._format_results_to_artifact("test-id", sql_result, "SELECT * FROM patients")
        
        assert artifact.success
        assert artifact.answer == ""  # Empty - orchestrator handles interpretation
        assert "prescriptions" in artifact.evidence
        assert artifact.raw_data["rows"] == []
        assert artifact.raw_data["row_count"] == 0
        assert "SELECT * FROM patients" in artifact.raw_data["sql_query"]
        
    def test_format_results_with_data(self):
        """Test result formatting with patient data - NEW ARCHITECTURE: raw data only"""
        sql_result = {
            "rows": [
                {"subject_id": 123, "drug": "Amphetamine", "gender": "M"},
                {"subject_id": 456, "drug": "Amphetamine", "gender": "F"}
            ],
            "row_count": 2
        }
        
        artifact = self.worker._format_results_to_artifact("test-id", sql_result, "SELECT * FROM patients")
        
        assert artifact.success
        assert artifact.answer == ""  # Empty - orchestrator handles interpretation
        assert artifact.raw_data["rows"] == sql_result["rows"]
        assert artifact.raw_data["row_count"] == 2
        assert "SELECT * FROM patients" in artifact.raw_data["sql_query"]
        # Raw data contains the actual rows for orchestrator to interpret
        assert artifact.raw_data["rows"][0]["subject_id"] == 123
        assert artifact.raw_data["rows"][0]["drug"] == "Amphetamine"
        
    @patch('agents.structured_worker.main.StructuredWorker._generate_sql_from_query')
    def test_process_task_no_sql_generated(self, mock_sql_gen):
        """Test task processing when SQL generation fails"""
        mock_sql_gen.return_value = None
        
        task = A2ATask(
            task_type="structured_query",
            query="unsupported query"
        )
        
        artifact = self.worker.process_task(task)
        
        assert not artifact.success
        assert "Could not generate SQL query" in artifact.error_message
        
    @patch('common.mcp_sql_server.MCPSQLServer.call_tool')
    def test_process_task_mcp_error(self, mock_mcp_call):
        """Test task processing when MCP call fails"""
        # Mock MCP failure
        mock_response = Mock()
        mock_response.success = False
        mock_response.error = "Database connection failed"
        mock_mcp_call.return_value = mock_response
        
        task = A2ATask(
            task_type="structured_query",
            query="Get patients who took amphetamine"
        )
        
        artifact = self.worker.process_task(task)
        
        assert not artifact.success
        assert "SQL execution failed" in artifact.error_message
        assert "Database connection failed" in artifact.error_message
        
    @patch('common.mcp_sql_server.MCPSQLServer.call_tool')
    def test_process_task_success(self, mock_mcp_call):
        """Test successful task processing - NEW ARCHITECTURE: raw data only"""
        # Mock successful MCP response
        mock_response = Mock()
        mock_response.success = True
        mock_response.result = {
            "rows": [{"subject_id": 123, "drug": "Amphetamine"}],
            "row_count": 1
        }
        mock_mcp_call.return_value = mock_response
        
        task = A2ATask(
            task_type="structured_query",
            query="Get patients who took amphetamine in last 24 hours"
        )
        
        artifact = self.worker.process_task(task)
        
        assert artifact.success
        assert artifact.answer == ""  # Empty - orchestrator handles interpretation
        assert artifact.task_id == task.task_id
        assert len(artifact.evidence) > 0
        # Verify raw data is preserved for orchestrator
        assert artifact.raw_data["rows"] == [{"subject_id": 123, "drug": "Amphetamine"}]
        assert artifact.raw_data["row_count"] == 1
        assert "sql_query" in artifact.raw_data


if __name__ == "__main__":
    pytest.main([__file__])

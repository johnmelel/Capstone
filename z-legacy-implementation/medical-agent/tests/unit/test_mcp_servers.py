"""
Tests for MCP server implementations
"""
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.mcp_sql_server import MCPSQLServer
from common.mcp_vector_server import MCPVectorServer, MedicalContextBuilder


class TestMCPSQLServer:
    """Test SQL MCP server (without actual database connection)"""
    
    def test_get_available_tools(self):
        """Test SQL server tool listing"""
        server = MCPSQLServer()
        tools = server.get_available_tools()
        assert "sql.query" in tools
        
    def test_unknown_tool_call(self):
        """Test handling of unknown tool calls"""
        server = MCPSQLServer()
        response = server.call_tool("unknown.tool", {})
        
        assert not response.success
        assert "Unknown tool" in response.error
        


class TestMCPVectorServer:
    """Test Vector MCP server (with dummy data)"""
    
    def test_get_available_tools(self):
        """Test vector server tool listing"""
        server = MCPVectorServer()
        tools = server.get_available_tools()
        assert "vector.search" in tools
        
    def test_unknown_tool_call(self):
        """Test handling of unknown tool calls"""
        server = MCPVectorServer()
        response = server.call_tool("unknown.tool", {})
        
        assert not response.success
        assert "Unknown tool" in response.error
        
    def test_vector_search_amphetamine(self):
        """Test vector search for amphetamine"""
        server = MCPVectorServer()
        response = server.call_tool("vector.search", {
            "query": "amphetamine monitoring guidelines",
            "top_k": 2
        })
        
        assert response.success
        assert "documents" in response.result
        docs = response.result["documents"]
        assert len(docs) <= 2
        assert len(docs) > 0
        
        # Check that amphetamine-specific knowledge is returned
        found_amphetamine_content = False
        for doc in docs:
            if "amphetamine" in doc["text"].lower():
                found_amphetamine_content = True
                break
        assert found_amphetamine_content
        
    def test_vector_search_default_content(self):
        """Test vector search for unknown drug returns default content"""
        server = MCPVectorServer()
        response = server.call_tool("vector.search", {
            "query": "unknown_drug_xyz monitoring",
            "top_k": 3
        })
        
        assert response.success
        docs = response.result["documents"]
        assert len(docs) > 0
        
        # Should return general medical content
        general_content_found = False
        for doc in docs:
            if "medication monitoring" in doc["text"].lower():
                general_content_found = True
                break
        assert general_content_found
        
    def test_vector_search_top_k_limit(self):
        """Test that top_k parameter limits results"""
        server = MCPVectorServer()
        response = server.call_tool("vector.search", {
            "query": "amphetamine",
            "top_k": 1
        })
        
        assert response.success
        docs = response.result["documents"]
        assert len(docs) == 1
        
    def test_medical_context_builder(self):
        """Test medical context query builders"""
        query1 = MedicalContextBuilder.build_drug_monitoring_query("amphetamine")
        assert "amphetamine" in query1
        assert "monitoring" in query1
        
        query2 = MedicalContextBuilder.build_safety_query("insulin")
        assert "insulin" in query2
        assert "safety" in query2
        
        query3 = MedicalContextBuilder.build_general_query("diabetes")
        assert "diabetes" in query3
        assert "clinical" in query3


if __name__ == "__main__":
    pytest.main([__file__])

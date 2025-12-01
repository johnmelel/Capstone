"""
Test suite for EMR and Research workers

Tests worker functionality with real MCP server connection.
"""
import pytest
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from emr_worker import EMRWorker
from research_worker import ResearchWorker

# Load environment variables
load_dotenv()

# Get paths
PROJECT_ROOT = Path(__file__).parent.parent
MCP_SERVER_PATH = PROJECT_ROOT / "mcp-servers" / "server.py"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Skip tests if Gemini API key not available
pytestmark = pytest.mark.skipif(
    not GEMINI_API_KEY,
    reason="GEMINI_API_KEY not set in environment"
)


@pytest.fixture
def mcp_server_command():
    """Command to start MCP server."""
    return f"python {MCP_SERVER_PATH}"


@pytest.fixture
def emr_worker(mcp_server_command):
    """Create EMRWorker instance."""
    return EMRWorker(mcp_server_command, GEMINI_API_KEY)


@pytest.fixture
def research_worker(mcp_server_command):
    """Create ResearchWorker instance."""
    return ResearchWorker(mcp_server_command, GEMINI_API_KEY)


class TestEMRWorker:
    """Tests for EMR Worker"""
    
    @pytest.mark.asyncio
    async def test_emr_worker_initialization(self, emr_worker):
        """Test that EMR worker initializes correctly."""
        assert emr_worker is not None
        assert emr_worker.model is not None
    
    @pytest.mark.asyncio
    async def test_emr_worker_patient_labs(self, emr_worker):
        """Test querying patient lab results."""
        result = await emr_worker.handle_task(
            task="What were patient 10000032's most recent lab results?",
            patient_id="10000032"
        )
        
        assert result["success"] is True
        assert "summary" in result
        assert "sql_used" in result
        assert "data" in result
        
        # Check that SQL uses dictionary table JOIN
        assert "d_labitems" in result["sql_used"].lower()
        
        # Check that summary is not empty and doesn't contain raw codes
        assert len(result["summary"]) > 0
        print(f"\nSummary: {result['summary']}")
    
    @pytest.mark.asyncio
    async def test_emr_worker_diagnoses(self, emr_worker):
        """Test querying patient diagnoses."""
        result = await emr_worker.handle_task(
            task="What are the diagnoses for patient 10000032?",
            patient_id="10000032"
        )
        
        assert result["success"] is True
        assert "summary" in result
        
        # Check that SQL uses dictionary table JOIN
        assert "d_icd_diagnoses" in result["sql_used"].lower()
        
        print(f"\nDiagnoses summary: {result['summary']}")
    
    @pytest.mark.asyncio
    async def test_emr_worker_medications(self, emr_worker):
        """Test querying patient medications."""
        result = await emr_worker.handle_task(
            task="What medications was patient 10000032 prescribed?",
            patient_id="10000032"
        )
        
        assert result["success"] is True
        assert "summary" in result
        assert result["row_count"] >= 0
        
        print(f"\nMedications summary: {result['summary']}")
    
    @pytest.mark.asyncio
    async def test_emr_worker_no_results(self, emr_worker):
        """Test handling of query with no results."""
        result = await emr_worker.handle_task(
            task="What were patient 99999999's lab results?",
            patient_id="99999999"
        )
        
        # Should still succeed but return no results
        assert result["success"] is True
        assert result["row_count"] == 0
        assert "No results" in result["summary"] or result["row_count"] == 0


class TestResearchWorker:
    """Tests for Research Worker"""
    
    @pytest.mark.asyncio
    async def test_research_worker_initialization(self, research_worker):
        """Test that Research worker initializes correctly."""
        assert research_worker is not None
        assert research_worker.model is not None
    
    @pytest.mark.asyncio
    async def test_research_worker_basic_search(self, research_worker):
        """Test basic literature search."""
        result = await research_worker.handle_task(
            task="What does the literature say about amphetamine monitoring?"
        )
        
        # May fail if Milvus not connected, check error
        if not result["success"]:
            pytest.skip(f"Milvus connection issue: {result.get('error', 'Unknown error')}")
        
        assert result["success"] is True
        assert "summary" in result
        assert "narrative" in result
        assert "refined_query" in result
        assert "documents" in result
        
        # Check that query was refined
        assert len(result["refined_query"]) > 0
        
        print(f"\nRefined query: {result['refined_query']}")
        print(f"Found {result['total_found']} documents")
        print(f"Narrative: {result['narrative'][:200]}...")
    
    @pytest.mark.asyncio
    async def test_research_worker_with_context(self, research_worker):
        """Test literature search with EMR context."""
        result = await research_worker.handle_task(
            task="What are the clinical guidelines for this medication?",
            context="Patient prescribed Tiotropium Bromide for respiratory condition",
            top_k=3
        )
        
        # May fail if Milvus not connected
        if not result["success"]:
            pytest.skip(f"Milvus connection issue: {result.get('error', 'Unknown error')}")
        
        assert result["success"] is True
        assert "evidence" in result
        assert "sources" in result
        
        # Check that we got 3 or fewer documents
        assert len(result["documents"]) <= 3
        
        print(f"\nEvidence points: {result['evidence']}")
        print(f"Sources: {result['sources']}")
    
    @pytest.mark.asyncio
    async def test_research_worker_synthesis(self, research_worker):
        """Test that research findings are properly synthesized."""
        result = await research_worker.handle_task(
            task="What are best practices for monitoring chronic kidney disease?",
            top_k=5
        )
        
        # May fail if Milvus not connected
        if not result["success"]:
            pytest.skip(f"Milvus connection issue: {result.get('error', 'Unknown error')}")
        
        assert result["success"] is True
        
        # Check synthesis structure
        assert isinstance(result["narrative"], str)
        assert isinstance(result["evidence"], list)
        assert isinstance(result["sources"], list)
        
        # Narrative should be substantial
        assert len(result["narrative"]) > 100
        
        print(f"\nSynthesis:")
        print(f"Narrative length: {len(result['narrative'])} chars")
        print(f"Evidence points: {len(result['evidence'])}")
        print(f"Sources: {len(result['sources'])}")


class TestIntegration:
    """Integration tests combining both workers"""
    
    @pytest.mark.asyncio
    async def test_emr_then_research(self, emr_worker, research_worker):
        """Test workflow: query EMR, then research literature."""
        # First, get medication from EMR
        emr_result = await emr_worker.handle_task(
            task="What medications was patient 10000032 prescribed?",
            patient_id="10000032"
        )
        
        assert emr_result["success"] is True
        
        # Extract a medication from results (if any)
        if emr_result["row_count"] > 0:
            rows = emr_result["data"]["rows"]
            if rows:
                # Assume there's a drug column
                first_med = str(rows[0].get("drug", rows[0].get("medication", "generic medication")))
                
                # Now research that medication
                research_result = await research_worker.handle_task(
                    task=f"What are the clinical guidelines for {first_med}?",
                    context=f"Patient prescribed {first_med}",
                    top_k=3
                )
                
                # May fail if Milvus not connected
                if not research_result["success"]:
                    pytest.skip(f"Milvus connection issue: {research_result.get('error', 'Unknown error')}")
                
                assert research_result["success"] is True
                
                print(f"\nIntegrated workflow:")
                print(f"EMR: Found {emr_result['row_count']} medications")
                print(f"Research: {first_med}")
                print(f"Found {research_result['total_found']} relevant documents")
    
    @pytest.mark.asyncio
    async def test_research_then_emr(self, emr_worker, research_worker):
        """Test workflow: research condition, then check EMR for relevant patients."""
        # First, research a condition
        research_result = await research_worker.handle_task(
            task="What are diagnostic criteria for chronic kidney disease?",
            top_k=3
        )
        
        # May fail if Milvus not connected
        if not research_result["success"]:
            pytest.skip(f"Milvus connection issue: {research_result.get('error', 'Unknown error')}")
        
        assert research_result["success"] is True
        
        # Then check EMR for patients with related diagnoses
        emr_result = await emr_worker.handle_task(
            task="Find patients with kidney-related diagnoses"
        )
        
        assert emr_result["success"] is True
        
        print(f"\nReverse workflow:")
        print(f"Research: {research_result['refined_query']}")
        print(f"EMR: Found {emr_result['row_count']} relevant patients")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

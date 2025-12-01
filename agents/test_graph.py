"""
Tests for LangGraph Multi-Agent Coordination

Tests the complete multi-agent flow including planner, router, and workers.
"""
import os
import pytest
from dotenv import load_dotenv

from agents.graph import create_medical_agent_graph, run_medical_query

# Load environment variables
load_dotenv()


@pytest.mark.asyncio
async def test_graph_emr_only():
    """Test routing to EMR worker only"""
    app = create_medical_agent_graph()
    
    result = await app.ainvoke({
        "query": "What are the lab results for patient 10000032?",
        "patient_id": "10000032",
        "messages": []
    })
    
    # Check planner routing
    assert result["need_emr"] == True, "Should route to EMR"
    assert result["need_research"] == False or result.get("research_result") is None, "Should not route to research"
    
    # Check EMR worker was called
    assert result.get("emr_result") is not None, "EMR result should be present"
    assert result["emr_result"]["success"] == True, "EMR query should succeed"
    
    # Check final response exists
    assert result.get("final_response"), "Final response should be present"
    assert len(result["final_response"]) > 50, "Final response should have content"
    assert "EMR" in result["final_response"], "Response should mention EMR data"


@pytest.mark.asyncio
async def test_graph_research_only():
    """Test routing to research worker only"""
    app = create_medical_agent_graph()
    
    result = await app.ainvoke({
        "query": "What are the treatment guidelines for hypertension?",
        "messages": []
    })
    
    # Check planner routing
    assert result["need_research"] == True, "Should route to research"
    assert result["need_emr"] == False or result.get("emr_result") is None, "Should not route to EMR"
    
    # Check research worker was called
    assert result.get("research_result") is not None, "Research result should be present"
    
    # Check final response exists
    assert result.get("final_response"), "Final response should be present"
    assert len(result["final_response"]) > 50, "Final response should have content"
    assert "Research" in result["final_response"], "Response should mention research findings"


@pytest.mark.asyncio
async def test_graph_both_workers():
    """Test routing to both EMR and research workers"""
    app = create_medical_agent_graph()
    
    result = await app.ainvoke({
        "query": "Patient 10000032 has high creatinine. What does the literature say about this?",
        "patient_id": "10000032",
        "messages": []
    })
    
    # Check planner routing
    assert result["need_emr"] == True, "Should route to EMR"
    assert result["need_research"] == True, "Should route to research"
    
    # Check both workers were called
    assert result.get("emr_result") is not None, "EMR result should be present"
    assert result.get("research_result") is not None, "Research result should be present"
    
    # Check final response combines both
    assert result.get("final_response"), "Final response should be present"
    assert "EMR" in result["final_response"], "Response should mention EMR data"
    assert "Research" in result["final_response"], "Response should mention research findings"


@pytest.mark.asyncio
async def test_run_medical_query_convenience():
    """Test the convenience function run_medical_query()"""
    result = await run_medical_query(
        "What are patient 10000032's medications?",
        patient_id="10000032"
    )
    
    # Check all components ran
    assert result.get("plan"), "Plan should be created"
    assert result.get("final_response"), "Final response should be present"
    assert result.get("messages"), "Messages should track execution"
    
    # Should have routed to EMR
    assert result["need_emr"] == True, "Should identify EMR need"
    assert result.get("emr_result"), "EMR result should be present"


@pytest.mark.asyncio
async def test_planner_decision_making():
    """Test that planner makes correct routing decisions"""
    app = create_medical_agent_graph()
    
    # Test case 1: Patient-specific query → EMR only
    result1 = await app.ainvoke({
        "query": "Show me patient 10000032's diagnoses",
        "patient_id": "10000032",
        "messages": []
    })
    assert result1["need_emr"] == True, "Patient-specific query should need EMR"
    
    # Test case 2: General medical knowledge → Research only
    result2 = await app.ainvoke({
        "query": "What causes diabetes?",
        "messages": []
    })
    assert result2["need_research"] == True, "General knowledge query should need research"
    
    # Test case 3: Patient data + literature context → Both
    result3 = await app.ainvoke({
        "query": "Patient 10000032 is on amphetamine. What are the monitoring guidelines?",
        "patient_id": "10000032",
        "messages": []
    })
    assert result3["need_emr"] == True or result3["need_research"] == True, "Should need at least one source"


@pytest.mark.asyncio
async def test_aggregator_combines_results():
    """Test that aggregator properly combines EMR and research results"""
    app = create_medical_agent_graph()
    
    result = await app.ainvoke({
        "query": "Patient 10000032 has kidney issues. What does research say about treatment?",
        "patient_id": "10000032",
        "messages": []
    })
    
    final = result.get("final_response", "")
    
    # Check both sections are present if both workers ran
    if result.get("emr_result") and result["emr_result"].get("success"):
        assert "EMR" in final, "EMR section should be in final response"
    
    if result.get("research_result") and result["research_result"].get("success"):
        assert "Research" in final, "Research section should be in final response"
    
    # Check formatting
    assert "##" in final, "Response should have markdown headers"


@pytest.mark.asyncio
async def test_message_tracking():
    """Test that execution messages are tracked throughout the graph"""
    result = await run_medical_query(
        "What are the latest lab results for patient 10000032?",
        patient_id="10000032"
    )
    
    messages = result.get("messages", [])
    
    # Should have messages from multiple nodes
    assert len(messages) >= 2, "Should have messages from multiple nodes"
    
    # Check for planner message
    planner_messages = [m for m in messages if "Planner" in m]
    assert len(planner_messages) > 0, "Should have planner message"
    
    # Check for worker messages (EMR or Research)
    worker_messages = [m for m in messages if "EMR" in m or "Research" in m]
    assert len(worker_messages) > 0, "Should have worker messages"
    
    # Check for aggregator message
    aggregator_messages = [m for m in messages if "Aggregator" in m]
    assert len(aggregator_messages) > 0, "Should have aggregator message"


@pytest.mark.asyncio
async def test_error_handling_missing_patient():
    """Test graceful handling when patient_id is missing but needed"""
    app = create_medical_agent_graph()
    
    result = await app.ainvoke({
        "query": "What are the lab results for patient 10000032?",
        # Note: patient_id not provided
        "messages": []
    })
    
    # Query should still complete
    assert result.get("final_response"), "Should have final response even with missing patient_id"
    
    # EMR query might extract patient ID from query text, or might fail gracefully
    if result.get("emr_result"):
        # Either succeeds (if worker extracted ID from query) or has error message
        assert result["emr_result"].get("success") is not None, "Should have success flag"


@pytest.mark.asyncio
async def test_end_to_end_realistic_query():
    """Test a realistic end-to-end clinical query"""
    result = await run_medical_query(
        "Patient 10000032 has elevated white blood cell count. What are the possible causes and what additional tests should be ordered?",
        patient_id="10000032"
    )
    
    # Verify complete execution
    assert result.get("plan"), "Should have a plan"
    assert result.get("final_response"), "Should have final response"
    
    # Should use both sources for this type of query
    # (EMR for current WBC, research for causes and testing guidelines)
    assert result.get("emr_result") or result.get("research_result"), "Should query at least one source"
    
    final = result["final_response"]
    assert len(final) > 100, "Should have substantial response"
    
    # Response should be well-formatted
    assert "##" in final or "\n" in final, "Response should have structure"

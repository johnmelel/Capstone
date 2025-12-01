"""
LangGraph Node Implementations for Multi-Agent Medical Assistant

Each node is a @traceable function that takes AgentState and returns updated state.
Nodes wrap existing worker logic for EMR and research queries.
"""
import os
import json
import logging
import time
from typing import Dict, Any, Optional
import google.generativeai as genai
from langsmith import traceable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_json_from_text(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks"""
    text = text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text.split("```json")[1].split("```")[0].strip()
    elif text.startswith("```"):
        text = text.split("```")[1].split("```")[0].strip()
    
    return json.loads(text)


@traceable(name="planner_node")
async def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes user query and decides which workers to call.
    
    Uses Gemini to make intelligent routing decisions based on query content.
    
    Args:
        state: Current agent state with 'query' and optional 'patient_id'
    
    Returns:
        Updated state with 'plan', 'need_emr', 'need_research' fields
    """
    query = state["query"]
    patient_id = state.get("patient_id")
    
    logger.info(f"Planner analyzing query: {query}")
    
    # Configure Gemini 2.5 Flash (latest stable model)
    # Free tier: 10 RPM, 250K TPM, 250 RPD
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""You are a medical query router. Analyze this query and decide what data sources are needed.

Query: "{query}"
Patient ID: {patient_id if patient_id else "Not specified"}

Decide:
1. need_emr: Does this require patient-specific EMR data? (lab results, medications, diagnoses, procedures, vitals, etc.)
2. need_research: Does this require medical literature, guidelines, or general medical knowledge?

Rules:
- If query asks about a specific patient's data → need_emr=true
- If query asks about general medical knowledge, treatment guidelines, literature → need_research=true
- Some queries may need both (e.g., "Patient X has condition Y, what does literature say about treatment?")
- If no specific data source is clear, default to research for general questions

Respond ONLY with valid JSON (no markdown, no explanation):
{{"need_emr": true/false, "need_research": true/false, "reasoning": "1-2 sentence explanation"}}"""
    
    try:
        # Add retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                decision = parse_json_from_text(response.text)
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    # Rate limited, wait and retry
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    raise
        
        logger.info(f"Planner decision: EMR={decision['need_emr']}, Research={decision['need_research']}")
        logger.info(f"Reasoning: {decision['reasoning']}")
        
        return {
            **state,
            "plan": decision["reasoning"],
            "need_emr": decision["need_emr"],
            "need_research": decision["need_research"],
            "messages": state.get("messages", []) + [f"Planner: {decision['reasoning']}"]
        }
    
    except Exception as e:
        logger.error(f"Planner error: {e}")
        # Default to conservative routing (use research)
        return {
            **state,
            "plan": f"Error in planning: {str(e)}. Defaulting to research.",
            "need_emr": False,
            "need_research": True,
            "messages": state.get("messages", []) + [f"Planner error: {str(e)}"]
        }


@traceable(name="emr_node")
async def emr_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Queries EMR database via EMRWorker.
    
    Args:
        state: Current agent state with 'query' and optional 'patient_id'
    
    Returns:
        Updated state with 'emr_result' field containing worker response
    """
    from agents.emr_worker import EMRWorker
    
    query = state["query"]
    patient_id = state.get("patient_id")
    
    logger.info(f"EMR node processing query for patient {patient_id}")
    
    # Get API key and MCP command from environment
    api_key = os.getenv("GEMINI_API_KEY")
    mcp_command = os.getenv("MCP_SERVER_COMMAND", "python ../mcp-servers/server.py")
    
    if not api_key:
        return {
            "emr_result": {
                "success": False,
                "error": "GEMINI_API_KEY not set",
                "summary": "Failed to query EMR: API key not configured"
            },
            "messages": ["EMR: Error - API key not set"]
        }
    
    try:
        # Create worker and execute query
        worker = EMRWorker(
            mcp_server_command=mcp_command,
            gemini_api_key=api_key
        )
        
        result = await worker.handle_task(task=query, patient_id=patient_id)
        
        row_count = result.get("row_count", 0)
        logger.info(f"EMR node completed: {row_count} rows retrieved")
        
        return {
            "emr_result": result,
            "messages": [f"EMR: Retrieved {row_count} rows"]
        }
    
    except Exception as e:
        logger.error(f"EMR node error: {e}")
        return {
            "emr_result": {
                "success": False,
                "error": str(e),
                "summary": f"Failed to query EMR: {str(e)}"
            },
            "messages": [f"EMR: Error - {str(e)}"]
        }


@traceable(name="research_node")
async def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Searches medical literature via ResearchWorker.
    
    Args:
        state: Current agent state with 'query' and optional 'emr_result' for context
    
    Returns:
        Updated state with 'research_result' field containing worker response
    """
    from agents.research_worker import ResearchWorker
    
    query = state["query"]
    
    logger.info(f"Research node processing query: {query}")
    
    # Get API key and MCP command from environment
    api_key = os.getenv("GEMINI_API_KEY")
    mcp_command = os.getenv("MCP_SERVER_COMMAND", "python ../mcp-servers/server.py")
    
    if not api_key:
        return {
            "research_result": {
                "success": False,
                "error": "GEMINI_API_KEY not set",
                "summary": "Failed to search literature: API key not configured"
            },
            "messages": ["Research: Error - API key not set"]
        }
    
    try:
        # Create worker and execute search
        worker = ResearchWorker(
            mcp_server_command=mcp_command,
            gemini_api_key=api_key
        )
        
        # Use EMR context if available
        context = ""
        if state.get("emr_result") and state["emr_result"].get("success"):
            context = state["emr_result"].get("summary", "")
        
        result = await worker.handle_task(
            task=query,
            context=context,
            top_k=5
        )
        
        doc_count = result.get("total_found", 0)
        logger.info(f"Research node completed: {doc_count} documents found")
        
        return {
            "research_result": result,
            "messages": [f"Research: Found {doc_count} documents"]
        }
    
    except Exception as e:
        logger.error(f"Research node error: {e}")
        return {
            "research_result": {
                "success": False,
                "error": str(e),
                "summary": f"Failed to search literature: {str(e)}"
            },
            "messages": [f"Research: Error - {str(e)}"]
        }


@traceable(name="aggregator_node")
async def aggregator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combines results from EMR and/or research workers into final clinician report.
    
    Uses Gemini to transform raw data into actionable clinical format.
    
    Args:
        state: Current agent state with optional 'emr_result' and 'research_result'
    
    Returns:
        Updated state with 'final_response' field containing professional clinical report
    """
    logger.info("Aggregator combining results and generating clinical report")
    
    # Collect raw data from workers
    raw_parts = []
    
    # Add EMR results if present and successful
    if state.get("emr_result"):
        emr = state["emr_result"]
        if emr.get("success"):
            raw_parts.append(f"EMR DATA:\n{emr['summary']}")
        else:
            raw_parts.append(f"EMR DATA ERROR: {emr.get('error', 'Unknown error')}")
    
    # Add research results if present and successful
    if state.get("research_result"):
        research = state["research_result"]
        if research.get("success"):
            narrative = research.get("narrative", research.get("summary", ""))
            raw_parts.append(f"RESEARCH FINDINGS:\n{narrative}")
            
            # Add evidence bullets if available
            if research.get("evidence"):
                evidence_text = "\n".join([f"- {ev}" for ev in research["evidence"]])
                raw_parts.append(f"KEY EVIDENCE:\n{evidence_text}")
        else:
            raw_parts.append(f"RESEARCH ERROR: {research.get('error', 'Unknown error')}")
    
    # Combine raw data
    if not raw_parts:
        final_response = "No results were generated. Please check if the query was processed correctly."
        logger.warning("No results to aggregate")
        return {
            "final_response": final_response,
            "messages": ["Aggregator: No results to process"]
        }
    
    raw_combined = "\n\n".join(raw_parts)
    
    # Generate professional clinical report using Gemini
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Fallback: return raw combined data if no API key
            logger.warning("GEMINI_API_KEY not set, returning raw data")
            return {
                "final_response": raw_combined,
                "messages": ["Aggregator: No API key, returned raw data"]
            }
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        query = state.get("query", "")
        patient_id = state.get("patient_id", "")
        
        prompt = f"""You are a clinical assistant writing a professional report for a clinician.

Original Query: "{query}"
Patient ID: {patient_id if patient_id else "Not specified"}

Raw Data to Transform:
{raw_combined}

Create a professional clinical report with these sections:

1. CLINICAL SUMMARY
   - 2-3 sentence overview answering the query
   - Focus on what matters clinically

2. KEY FINDINGS
   - Bullet points of important data
   - Include relevant lab values, diagnoses, medications
   - Cite evidence-based information if available

3. CLINICAL RECOMMENDATIONS
   - Actionable next steps for the clinician
   - Consider follow-up, monitoring, or treatment adjustments
   - Base recommendations on the data provided

Keep it professional, concise, and actionable. Use medical terminology appropriately.
Format with markdown headers (##) and bullet points."""
        
        # Add retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                final_response = response.text.strip()
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    raise
        
        logger.info(f"Aggregator generated clinical report: {len(final_response)} characters")
        
        return {
            "final_response": final_response,
            "messages": ["Aggregator: Generated clinical report"]
        }
    
    except Exception as e:
        logger.error(f"Error generating clinical report: {e}")
        # Fallback: return structured raw data
        fallback = f"## Clinical Data\n\n{raw_combined}\n\n*Note: Unable to generate formatted report due to error: {str(e)}*"
        return {
            "final_response": fallback,
            "messages": [f"Aggregator: Error - {str(e)}, returned raw data"]
        }

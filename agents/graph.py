"""
LangGraph Multi-Agent Medical Assistant

Constructs the agent workflow graph with conditional routing.

Flow:
    User Query → Planner → Router (conditional) → {EMR, Research} → Aggregator → Response
"""
import logging
from typing import Dict, Any, List, Literal, Annotated
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from operator import add

from graph_nodes import planner_node, emr_node, research_node, aggregator_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """
    Shared state across all nodes in the graph.
    
    TypedDict with total=False allows optional fields.
    Annotated[List[str], add] means messages from multiple nodes will be concatenated.
    """
    # Input fields
    query: str
    patient_id: str
    user_id: str
    
    # Planning fields
    plan: str
    need_emr: bool
    need_research: bool
    
    # Worker result fields
    emr_result: Dict[str, Any]
    research_result: Dict[str, Any]
    
    # Output fields
    final_response: str
    messages: Annotated[List[str], add]  # Use add operator to concatenate messages from parallel nodes


def route_after_planner(state: AgentState) -> List[str]:
    """
    Conditional routing function after planner node.
    
    Determines which worker nodes to call based on planner's decision.
    
    Args:
        state: Current agent state with 'need_emr' and 'need_research' flags
    
    Returns:
        List of node names to route to (can be empty, one, or both workers)
    """
    routes = []
    
    if state.get("need_emr"):
        routes.append("emr")
    
    if state.get("need_research"):
        routes.append("research")
    
    # If no workers needed (shouldn't happen, but handle it), go to aggregator
    if not routes:
        logger.warning("Planner routed to no workers, defaulting to aggregator")
        routes.append("aggregator")
    
    logger.info(f"Router directing to: {routes}")
    return routes


def create_medical_agent_graph():
    """
    Build and compile the multi-agent medical assistant workflow.
    
    The graph structure:
    - START → planner
    - planner → (conditional) → {emr, research, both}
    - emr → aggregator
    - research → aggregator
    - aggregator → END
    
    Returns:
        Compiled LangGraph application ready for execution
    """
    logger.info("Creating medical agent graph")
    
    # Initialize graph with state schema
    graph = StateGraph(AgentState)
    
    # Add all nodes
    graph.add_node("planner", planner_node)
    graph.add_node("emr", emr_node)
    graph.add_node("research", research_node)
    graph.add_node("aggregator", aggregator_node)
    
    # Set entry point
    graph.set_entry_point("planner")
    
    # Add conditional edges from planner
    # This routes to appropriate workers based on need_emr and need_research flags
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "emr": "emr",
            "research": "research",
            "aggregator": "aggregator"
        }
    )
    
    # Both workers route to aggregator
    graph.add_edge("emr", "aggregator")
    graph.add_edge("research", "aggregator")
    
    # Aggregator is the final node
    graph.add_edge("aggregator", END)
    
    # Compile the graph
    app = graph.compile()
    
    logger.info("Medical agent graph compiled successfully")
    return app


async def run_medical_query(
    query: str,
    patient_id: str = None,
    user_id: str = None
) -> Dict[str, Any]:
    """
    Convenience function to run a medical query through the agent graph.
    
    Args:
        query: Natural language medical query
        patient_id: Optional patient identifier for EMR queries
        user_id: Optional user identifier for personalization
    
    Returns:
        Dict containing final_response and full state
    
    Example:
        result = await run_medical_query(
            "What are the latest lab results for patient 10000032?",
            patient_id="10000032"
        )
        print(result["final_response"])
    """
    # Create graph
    app = create_medical_agent_graph()
    
    # Initialize state
    initial_state = {
        "query": query,
        "messages": []
    }
    
    if patient_id:
        initial_state["patient_id"] = patient_id
    
    if user_id:
        initial_state["user_id"] = user_id
    
    # Run the graph
    logger.info(f"Running medical query: {query}")
    final_state = await app.ainvoke(initial_state)
    
    logger.info("Query completed successfully")
    return final_state


# Example usage
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    async def main():
        # Example 1: EMR-only query
        print("=" * 80)
        print("Example 1: EMR Query")
        print("=" * 80)
        result = await run_medical_query(
            "What are the most recent lab results for patient 10000032?",
            patient_id="10000032"
        )
        print(f"\nFinal Response:\n{result['final_response']}\n")
        print(f"Messages: {result['messages']}\n")
        
        # Example 2: Research-only query
        print("=" * 80)
        print("Example 2: Research Query")
        print("=" * 80)
        result = await run_medical_query(
            "What are the treatment guidelines for hypertension?"
        )
        print(f"\nFinal Response:\n{result['final_response']}\n")
        print(f"Messages: {result['messages']}\n")
        
        # Example 3: Both EMR and research
        print("=" * 80)
        print("Example 3: Combined Query")
        print("=" * 80)
        result = await run_medical_query(
            "Patient 10000032 has elevated creatinine. What does the literature say about treatment?",
            patient_id="10000032"
        )
        print(f"\nFinal Response:\n{result['final_response']}\n")
        print(f"Messages: {result['messages']}\n")
    
    asyncio.run(main())

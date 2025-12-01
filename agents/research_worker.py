"""
Research Worker Agent

Handles medical literature search via MCP server.
Uses Gemini for query refinement and result synthesis.
"""
import json
import logging
from typing import Dict, Any, List
import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchWorker:
    """
    Worker agent that searches medical literature via MCP server.
    
    Responsibilities:
    - Parse natural language research queries
    - Execute semantic search via MCP server
    - Synthesize findings into clinician-friendly summaries
    - Provide evidence-based recommendations with citations
    """
    
    def __init__(self, mcp_server_command: str, gemini_api_key: str):
        """
        Initialize Research Worker.
        
        Args:
            mcp_server_command: Command to start MCP server (e.g., "python mcp-servers/server.py")
            gemini_api_key: Google Gemini API key for LLM reasoning
        """
        self.mcp_server_command = mcp_server_command
        self.gemini_api_key = gemini_api_key
        
        # Configure Gemini 2.5 Flash (latest stable model)
        # Free tier: 10 RPM, 250K TPM, 250 RPD
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        logger.info("ResearchWorker initialized")
    
    async def _call_mcp_tool(self, session: ClientSession, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool and return the result."""
        result = await session.call_tool(tool_name, arguments)
        
        # Parse the result - MCP returns list of TextContent
        if result and len(result.content) > 0:
            content = result.content[0].text
            return json.loads(content)
        
        raise Exception(f"No result from tool {tool_name}")
    
    def _refine_query(self, task: str, context: str = "") -> str:
        """
        Use Gemini to refine the search query for better retrieval.
        
        This acts as a simplified query rewriter (the RL-tuned version will come later).
        
        Args:
            task: Natural language task description
            context: Optional context from EMR data or previous queries
        
        Returns:
            Refined search query optimized for medical literature search
        """
        prompt = f"""You are a medical research librarian. Refine this query for searching medical literature.

Task: {task}
{f"Context: {context}" if context else ""}

Create a focused search query that:
1. Uses medical terminology and standard nomenclature
2. Focuses on key clinical concepts
3. Removes colloquialisms and conversational language
4. Is 1-2 sentences maximum
5. Prioritizes specificity for better retrieval

Provide ONLY the refined query, no explanation."""

        response = self.model.generate_content(prompt)
        refined_query = response.text.strip()
        
        # Remove quotes if present
        if refined_query.startswith('"') and refined_query.endswith('"'):
            refined_query = refined_query[1:-1]
        
        return refined_query
    
    def _synthesize_findings(self, task: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use Gemini to synthesize research findings into a clinical summary.
        
        Args:
            task: Original task description
            documents: List of retrieved documents from semantic search
        
        Returns:
            Dict containing:
                - narrative: Prose summary
                - evidence: Key evidence points
                - sources: Source citations
        """
        if not documents:
            return {
                "narrative": "No relevant literature found for this query.",
                "evidence": [],
                "sources": []
            }
        
        # Format documents for LLM
        docs_text = ""
        for i, doc in enumerate(documents[:5], 1):  # Limit to top 5 for context
            docs_text += f"\n[Document {i}] (Relevance: {doc.get('relevance', 0):.3f})\n"
            docs_text += f"Source: {doc.get('source', 'Unknown')}\n"
            docs_text += f"Text: {doc.get('text', '')[:500]}...\n"  # First 500 chars
        
        prompt = f"""You are a medical research assistant synthesizing literature findings for a clinician.

Research Query: {task}

Retrieved Documents:
{docs_text}

Provide a clinical summary that includes:

1. NARRATIVE (2-3 paragraphs):
   - Key findings from the literature
   - Clinical implications
   - Relevant guidelines or recommendations
   - Any notable controversies or limitations

2. EVIDENCE (3-5 bullet points):
   - Specific evidence-based points
   - Each should be concise and actionable
   - Include strength of evidence when apparent

3. CITATIONS:
   - List the sources used
   - Format: [Document #] Source name

Format your response as JSON:
{{
  "narrative": "Your prose summary here",
  "evidence": ["Point 1", "Point 2", ...],
  "sources": ["[1] Source name", "[2] Source name", ...]
}}"""

        response = self.model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse JSON from response
        try:
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            synthesis = json.loads(response_text)
            return synthesis
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse synthesis JSON: {e}")
            # Fallback to basic structure
            return {
                "narrative": response_text,
                "evidence": [],
                "sources": [f"[{i+1}] {doc.get('source', 'Unknown')}" for i, doc in enumerate(documents[:5])]
            }
    
    async def handle_task(
        self,
        task: str,
        context: str = "",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Handle a medical literature research task.
        
        Args:
            task: Natural language description of the research query
            context: Optional context from EMR data or previous steps
            top_k: Number of documents to retrieve (default: 5)
        
        Returns:
            Dict containing:
                - summary: Human-readable summary
                - narrative: Detailed prose explanation
                - evidence: List of key evidence points
                - sources: List of source citations
                - documents: Raw retrieved documents
                - refined_query: The query used for search
                - success: Boolean indicating success
        """
        try:
            # Parse command for MCP server
            command_parts = self.mcp_server_command.split()
            
            server_params = StdioServerParameters(
                command=command_parts[0],
                args=command_parts[1:],
                env=None
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize session
                    await session.initialize()
                    
                    # Refine the query
                    logger.info(f"Refining query for task: {task}")
                    refined_query = self._refine_query(task, context)
                    logger.info(f"Refined query: {refined_query}")
                    
                    # Execute semantic search via MCP
                    logger.info(f"Searching literature (top_k={top_k})...")
                    search_results = await self._call_mcp_tool(
                        session,
                        "semantic_search",
                        {"query": refined_query, "top_k": top_k}
                    )
                    
                    documents = search_results.get("documents", [])
                    logger.info(f"Retrieved {len(documents)} documents")
                    
                    # Synthesize findings
                    logger.info("Synthesizing findings...")
                    synthesis = self._synthesize_findings(task, documents)
                    
                    return {
                        "success": True,
                        "summary": synthesis.get("narrative", ""),
                        "narrative": synthesis.get("narrative", ""),
                        "evidence": synthesis.get("evidence", []),
                        "sources": synthesis.get("sources", []),
                        "documents": documents,
                        "refined_query": refined_query,
                        "total_found": len(documents)
                    }
        
        except Exception as e:
            logger.error(f"Error handling task: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": f"Failed to search literature: {str(e)}",
                "narrative": "",
                "evidence": [],
                "sources": []
            }

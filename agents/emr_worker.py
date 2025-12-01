"""
EMR Worker Agent

Handles queries to the EMR database via MCP server.
Uses Gemini for natural language to SQL conversion and result summarization.
"""
import json
import logging
from typing import Dict, Any, Optional
import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EMRWorker:
    """
    Worker agent that queries EMR database via MCP server.
    
    Responsibilities:
    - Parse natural language queries for patient_id and intent
    - Generate appropriate SQL queries using Gemini
    - Execute queries via MCP server
    - Summarize results for clinical context
    """
    
    def __init__(self, mcp_server_command: str, gemini_api_key: str):
        """
        Initialize EMR Worker.
        
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
        
        logger.info("EMRWorker initialized")
    
    async def _call_mcp_tool(self, session: ClientSession, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool and return the result."""
        result = await session.call_tool(tool_name, arguments)
        
        # Parse the result - MCP returns list of TextContent
        if result and len(result.content) > 0:
            content = result.content[0].text
            return json.loads(content)
        
        raise Exception(f"No result from tool {tool_name}")
    
    async def _get_schema_info(self, session: ClientSession) -> str:
        """Get database schema information for prompt context."""
        # List all tables
        tables_result = await self._call_mcp_tool(session, "list_tables", {})
        tables = tables_result.get("tables", [])
        
        # Get schema for key tables (patients, admissions, labevents, prescriptions, diagnoses_icd)
        schema_info = "DATABASE SCHEMA:\n\n"
        key_tables = ["patients", "admissions", "labevents", "prescriptions", "diagnoses_icd", 
                      "d_labitems", "d_icd_diagnoses", "d_icd_procedures"]
        
        for table in key_tables:
            if table in tables:
                schema = await self._call_mcp_tool(session, "get_schema", {"table_name": table})
                schema_info += f"\nTable: {table}\n"
                schema_info += "Columns:\n"
                for col in schema.get("columns", []):
                    schema_info += f"  - {col['name']} ({col['type']})\n"
        
        return schema_info
    
    def _generate_sql_query(self, task: str, schema_info: str, patient_id: Optional[str] = None) -> str:
        """
        Use Gemini to generate SQL query from natural language.
        
        Args:
            task: Natural language task description
            schema_info: Database schema information
            patient_id: Optional specific patient ID to filter on
        
        Returns:
            SQL query string
        """
        prompt = f"""You are a SQL expert for a medical EMR database. Generate a SQL query to answer this task.

{schema_info}

CRITICAL RULES:
1. ALWAYS JOIN with dictionary tables (d_labitems, d_icd_diagnoses, d_icd_procedures) to get human-readable names, NOT codes
2. Use SELECT statements only (read-only)
3. Limit results to avoid overwhelming output (use LIMIT clause)
4. For lab results, JOIN labevents with d_labitems on itemid
5. For diagnoses, JOIN diagnoses_icd with d_icd_diagnoses on icd_code
6. For procedures, JOIN procedures_icd with d_icd_procedures on icd_code
7. Order by most recent (charttime DESC, admittime DESC) when applicable

Task: {task}
{f"Patient ID: {patient_id}" if patient_id else ""}

Generate ONLY the SQL query, no explanation. Start with SELECT."""

        response = self.model.generate_content(prompt)
        query = response.text.strip()
        
        # Clean up the query (remove markdown formatting if present)
        if query.startswith("```sql"):
            query = query.split("```sql")[1].split("```")[0].strip()
        elif query.startswith("```"):
            query = query.split("```")[1].split("```")[0].strip()
        
        return query
    
    def _summarize_results(self, task: str, sql_query: str, results: Dict[str, Any]) -> str:
        """
        Use Gemini to summarize SQL results for clinical context.
        
        Args:
            task: Original task description
            sql_query: SQL query that was executed
            results: Query results from MCP server
        
        Returns:
            Human-readable summary
        """
        rows = results.get("rows", [])
        row_count = results.get("row_count", 0)
        
        if row_count == 0:
            return "No results found for this query."
        
        # Format results for LLM
        results_text = json.dumps(rows[:10], indent=2)  # Limit to first 10 rows for context
        
        prompt = f"""You are a medical assistant summarizing EMR query results for a clinician.

Task: {task}

SQL Query Executed:
{sql_query}

Results ({row_count} rows total, showing first 10):
{results_text}

Provide a clear, concise summary suitable for a clinician. Focus on:
1. Key findings (lab values, diagnoses, medications)
2. Temporal information if present (most recent, dates)
3. Clinical significance
4. Any notable patterns or concerns

Keep it professional and concise (2-3 paragraphs maximum)."""

        response = self.model.generate_content(prompt)
        return response.text.strip()
    
    async def handle_task(self, task: str, patient_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle an EMR data query task.
        
        Args:
            task: Natural language description of the query
            patient_id: Optional specific patient ID
        
        Returns:
            Dict containing:
                - summary: Human-readable summary
                - data: Raw query results
                - sql_used: SQL query that was executed
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
                    
                    # Get schema information
                    logger.info("Retrieving database schema...")
                    schema_info = await self._get_schema_info(session)
                    
                    # Generate SQL query
                    logger.info(f"Generating SQL for task: {task}")
                    sql_query = self._generate_sql_query(task, schema_info, patient_id)
                    logger.info(f"Generated SQL: {sql_query}")
                    
                    # Execute query via MCP
                    logger.info("Executing SQL query via MCP...")
                    results = await self._call_mcp_tool(session, "run_sql", {"query": sql_query})
                    
                    # Summarize results
                    logger.info("Summarizing results...")
                    summary = self._summarize_results(task, sql_query, results)
                    
                    return {
                        "success": True,
                        "summary": summary,
                        "data": results,
                        "sql_used": sql_query,
                        "row_count": results.get("row_count", 0)
                    }
        
        except Exception as e:
            logger.error(f"Error handling task: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": f"Failed to process query: {str(e)}"
            }

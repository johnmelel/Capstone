"""
BASELINE 2: Single-Agent Orchestrator
Monolithic orchestrator that does everything internally without A2A communication.
This demonstrates the value of agent specialization and separation of concerns.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import vertexai
from vertexai.generative_models import GenerativeModel
from typing import Dict, Any, Optional
import json
import re

from common.mcp_sql_server import MCPSQLServer
from common.mcp_vector_server import MCPVectorServer
from common.llm_logging import llm_logger


class SingleAgentOrchestrator:
    """Monolithic orchestrator that handles both SQL and vector search internally"""
    
    def __init__(self, service_account_path: str):
        # Initialize MCP servers directly
        self.mcp_sql = MCPSQLServer()
        self.mcp_vector = MCPVectorServer()
        
        # Initialize Gemini
        self._setup_gemini(service_account_path)
        
        # System prompt for reasoning
        self.system_prompt = """You are a medical data analysis assistant that helps interpret clinical queries using both structured EMR data and unstructured medical knowledge.

Your role is to:
1. Analyze user queries about patients, medications, and medical conditions
2. Reason over results from both structured database queries and medical literature/guidelines
3. Provide clear, concise answers that combine both data sources appropriately
4. Always cite your sources and distinguish between EMR data findings and medical guidance

When responding:
- Be precise and factual
- Highlight key clinical insights
- Note any important monitoring requirements or safety considerations
- Keep responses focused and actionable
- Count your reasoning steps as you work through the problem
- Format as JSON with 'answer', 'structured_source', 'unstructured_source', and 'reasoning_steps' fields

IMPORTANT: Please count and include the number of distinct reasoning steps you took to arrive at your answer. Include this as 'reasoning_steps' (integer) in your JSON response.
"""
        
        # Schema context for SQL generation (embedded from structured worker)
        self._setup_schema_context()
        
    def _setup_gemini(self, service_account_path: str):
        """Setup Gemini API with Vertex AI"""
        try:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
            
            try:
                import json as json_lib
                with open(service_account_path, 'r') as f:
                    creds = json_lib.load(f)
                    project_id = creds.get('project_id')
                    
                if project_id:
                    vertexai.init(project=project_id, location="us-central1")
                    self.model = GenerativeModel("gemini-2.5-flash")
                    print(f"Single-Agent Orchestrator: Gemini initialized", file=sys.stderr)
                else:
                    raise ValueError("No project_id found in service account")
                    
            except Exception as e:
                print(f"Warning: Could not initialize Vertex AI: {e}")
                self.model = None
            
        except Exception as e:
            print(f"Warning: Gemini setup failed: {e}")
            self.model = None
    
    def _setup_schema_context(self):
        """Set up database schema for SQL generation"""
        self.schema_info = """
DATABASE SCHEMA (MIMIC-IV SQLite):
patients table: subject_id, gender, anchor_age, anchor_year, dod
admissions table: subject_id, hadm_id, admittime, dischtime, admission_type, admission_location
prescriptions table: subject_id, hadm_id, starttime, stoptime, drug, dose_val_rx, route
emar table: subject_id, hadm_id, charttime, medication, event_txt
pharmacy table: subject_id, hadm_id, starttime, medication, route, frequency
labevents table: labevent_id, subject_id, hadm_id, itemid, charttime, value, valuenum, valueuom, flag
diagnoses_icd table: subject_id, hadm_id, icd_code, icd_version, seq_num
procedures_icd table: subject_id, hadm_id, icd_code, icd_version, chartdate
transfers table: subject_id, hadm_id, careunit, intime, outtime, eventtype
"""
    
    def query(self, user_question: str) -> Dict[str, Any]:
        """Main entry point - handle everything internally"""
        try:
            # Step 1: Generate and execute SQL query internally
            structured_data = self._handle_structured_query(user_question)
            
            # Step 2: Perform vector search internally
            unstructured_data = self._handle_unstructured_search(user_question)
            
            # Step 3: Reason over results with Gemini
            final_answer = self._reason_over_results(user_question, structured_data, unstructured_data)
            
            return final_answer
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "structured_source": [],
                "unstructured_source": [],
                "error": str(e)
            }
    
    def _handle_structured_query(self, query: str) -> Dict[str, Any]:
        """Handle SQL generation and execution internally"""
        try:
            # Generate SQL using Gemini
            sql_query = self._generate_sql(query)
            
            if not sql_query:
                return {"success": False, "error": "Could not generate SQL", "rows": [], "row_count": 0}
            
            # Execute via MCP
            mcp_response = self.mcp_sql.call_tool("sql.query", {"sql": sql_query})
            
            if not mcp_response.success:
                return {"success": False, "error": mcp_response.error, "rows": [], "row_count": 0}
            
            return {
                "success": True,
                "sql_query": sql_query,
                "rows": mcp_response.result.get('rows', []),
                "row_count": mcp_response.result.get('row_count', 0)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "rows": [], "row_count": 0}
    
    def _generate_sql(self, query: str) -> Optional[str]:
        """Generate SQL query using Gemini"""
        if self.model is None:
            return None
        
        try:
            prompt = f"""You are a SQL expert generating queries for a MIMIC-IV medical database.

{self.schema_info}

REQUIREMENTS:
- Generate SQLite-compatible SQL queries only
- Use datetime('now', '-X hours') for time filtering  
- Always include LIMIT to prevent large result sets (max 50)
- Use LIKE '%term%' for medication/drug name matching
- Join patients table for demographic information when possible

Generate a SQL query for: {query}

Return only the SQL query, no explanation:"""

            response = self.model.generate_content(prompt)
            sql_text = response.text.strip()
            
            # Log the LLM interaction
            llm_logger.log_interaction(
                component="single_agent_sql",
                prompt=prompt,
                response=sql_text,
                context=f"SQL generation for query: {query}"
            )
            
            # Clean up markdown formatting
            if sql_text.startswith('```sql'):
                sql_text = sql_text[6:]
            if sql_text.startswith('```'):
                sql_text = sql_text[3:]
            if sql_text.endswith('```'):
                sql_text = sql_text[:-3]
            
            sql_text = sql_text.strip()
            
            # Basic safety validation
            if self._is_sql_safe(sql_text):
                return sql_text
            
            return None
            
        except Exception as e:
            print(f"SQL generation error: {e}", file=sys.stderr)
            return None
    
    def _is_sql_safe(self, sql: str) -> bool:
        """Basic safety validation for generated SQL"""
        sql_upper = sql.upper()
        
        if not sql_upper.strip().startswith('SELECT'):
            return False
        
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        if any(keyword in sql_upper for keyword in dangerous_keywords):
            return False
        
        if 'LIMIT' not in sql_upper:
            return False
        
        return True
    
    def _handle_unstructured_search(self, query: str) -> Dict[str, Any]:
        """Handle vector search internally"""
        try:
            # Generate search query
            search_query = self._generate_context_query(query)
            
            # Execute via MCP
            mcp_response = self.mcp_vector.call_tool("vector.search", {
                "query": search_query,
                "top_k": 3
            })
            
            if not mcp_response.success:
                return {"success": False, "error": mcp_response.error, "documents": []}
            
            documents = mcp_response.result.get('documents', [])
            
            return {
                "success": True,
                "documents": documents,
                "total_found": mcp_response.result.get('total_found', 0)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "documents": []}
    
    def _generate_context_query(self, user_question: str) -> str:
        """Generate medical context query for unstructured search"""
        query_lower = user_question.lower()
        
        drug_match = re.search(r'(amphetamine|amoxicillin|aspirin|metformin|insulin|warfarin|lisinopril|atorvastatin)', query_lower)
        
        if drug_match:
            drug_name = drug_match.group(1)
            return f"{drug_name} monitoring guidelines side effects safety warnings"
        
        if any(word in query_lower for word in ['admission', 'patient', 'hospital']):
            return "patient care protocols hospital admission guidelines"
        
        return user_question
    
    def _reason_over_results(self, user_question: str, structured_data: Dict[str, Any], 
                           unstructured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use Gemini to reason over both results"""
        
        context = self._prepare_reasoning_context(user_question, structured_data, unstructured_data)
        
        if self.model is None:
            return self._generate_fallback_response(structured_data, unstructured_data)
        
        try:
            prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nPlease provide a JSON response combining both sources."
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Log the LLM interaction
            llm_logger.log_interaction(
                component="single_agent_orchestrator",
                prompt=prompt,
                response=response_text,
                context="Final reasoning"
            )
            
            # Try to parse JSON response
            try:
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                response_text = response_text.strip()
                result = json.loads(response_text)
                return result
                
            except json.JSONDecodeError:
                return {
                    "answer": response.text,
                    "structured_source": ["database"],
                    "unstructured_source": ["medical_literature"]
                }
                
        except Exception as e:
            return self._generate_fallback_response(structured_data, unstructured_data, error=str(e))
    
    def _prepare_reasoning_context(self, user_question: str, structured_data: Dict[str, Any], 
                                 unstructured_data: Dict[str, Any]) -> str:
        """Prepare context for Gemini reasoning"""
        context = f"User Question: {user_question}\n\n"
        
        context += "STRUCTURED DATABASE RESULTS:\n"
        if structured_data.get('success'):
            context += f"SQL Query: {structured_data.get('sql_query', 'N/A')}\n"
            context += f"Row Count: {structured_data.get('row_count', 0)}\n"
            
            rows = structured_data.get('rows', [])
            if rows:
                context += "Raw Data (first 5 rows):\n"
                for i, row in enumerate(rows[:5]):
                    context += f"Row {i+1}: {str(row)}\n"
                if len(rows) > 5:
                    context += f"... and {len(rows) - 5} more rows\n"
            else:
                context += "No data rows returned\n"
        else:
            context += f"Database Error: {structured_data.get('error', 'Unknown')}\n"
        
        context += "\nUNSTRUCTURED MEDICAL KNOWLEDGE:\n"
        if unstructured_data.get('success'):
            documents = unstructured_data.get('documents', [])
            context += f"Found {len(documents)} relevant medical documents\n"
            for i, doc in enumerate(documents[:2]):
                context += f"Context {i+1}: {doc.get('text', '')[:200]}...\n"
        else:
            context += f"Knowledge Error: {unstructured_data.get('error', 'Unknown')}\n"
        
        return context
    
    def _generate_fallback_response(self, structured_data: Dict[str, Any], 
                                  unstructured_data: Dict[str, Any], error: str = None) -> Dict[str, Any]:
        """Generate fallback response when Gemini is not available"""
        
        answer_parts = []
        
        if structured_data.get('success'):
            answer_parts.append(f"Found {structured_data.get('row_count', 0)} database records.")
        
        if unstructured_data.get('success'):
            answer_parts.append(f"Found {len(unstructured_data.get('documents', []))} medical documents.")
        
        if not answer_parts:
            answer_parts.append("Unable to process query due to errors.")
        
        answer = " ".join(answer_parts)
        
        if error:
            answer += f" (Note: Advanced reasoning unavailable: {error})"
        
        return {
            "answer": answer,
            "structured_source": ["database"] if structured_data.get('success') else [],
            "unstructured_source": ["medical_literature"] if unstructured_data.get('success') else []
        }


# CLI interface for testing
def main():
    """Simple CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Single-Agent Baseline System')
    parser.add_argument('query', help='Medical query to process')
    parser.add_argument('--service-account', default='adsp-34002-ip09-team-2-e0cca2d396a9.json',
                       help='Path to service account JSON file')
    
    args = parser.parse_args()
    
    orchestrator = SingleAgentOrchestrator(service_account_path=args.service_account)
    
    result = orchestrator.query(args.query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

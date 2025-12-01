"""
Structured Data Worker - Handles A2A tasks for querying MIMIC database
Uses intelligent SQL generation via Gemini instead of pattern matching
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import re
import json
from typing import Dict, Any, Optional

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from common.a2a_messages import A2ATask, A2AArtifact
from common.mcp_sql_server import MCPSQLServer
from common.llm_logging import llm_logger


class StructuredWorker:
    """Worker that handles structured medical data queries"""
    
    def __init__(self):
        self.mcp_server = MCPSQLServer()
        self._setup_gemini()
        self._setup_schema_context()
    
    def _setup_gemini(self):
        """Initialize Gemini for intelligent SQL generation"""
        try:
            # Try to use same service account as orchestrator
            service_account_path = 'adsp-34002-ip09-team-2-e0cca2d396a9.json'
            if os.path.exists(service_account_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
                
                with open(service_account_path, 'r') as f:
                    creds = json.load(f)
                    project_id = creds.get('project_id')
                    
                if project_id:
                    vertexai.init(project=project_id, location="us-central1")
                    self.model = GenerativeModel("gemini-2.5-flash")
                    print(f"✓ Structured worker Gemini initialized", file=sys.stderr)
                else:
                    self.model = None
            else:
                self.model = None
        except Exception as e:
            print(f"Warning: Gemini setup failed in structured worker: {e}", file=sys.stderr)
            self.model = None
    
    def _setup_schema_context(self):
        """Set up database schema and few-shot examples for SQL generation"""
        self.schema_info = """
DATABASE SCHEMA (MIMIC-IV SQLite) - Updated with full dataset and 2025 dates:

patients table:
- subject_id (INTEGER, Primary Key): Unique patient identifier
- gender (TEXT): Patient gender
- anchor_age (INTEGER): Patient age at anchor year
- anchor_year (INTEGER): Shifted year for privacy
- dod (TEXT): Date of death (if applicable)

admissions table:
- subject_id (INTEGER): Patient identifier
- hadm_id (INTEGER): Admission identifier (Primary Key)
- admittime (TEXT): Admission timestamp (2025 dates)
- dischtime (TEXT): Discharge timestamp
- deathtime (TEXT): Death timestamp
- admission_type (TEXT): Type of admission
- admission_location (TEXT): Where admitted from
- discharge_location (TEXT): Where discharged to
- insurance (TEXT): Insurance type
- race (TEXT): Patient race

prescriptions table:
- subject_id (INTEGER): Patient identifier
- hadm_id (INTEGER): Hospital admission identifier  
- starttime (TEXT): When prescription started (2025 dates)
- stoptime (TEXT): When prescription stopped
- drug (TEXT): Medication name
- dose_val_rx (TEXT): Prescribed dose
- route (TEXT): Route of administration

emar table:
- subject_id (INTEGER): Patient identifier
- hadm_id (INTEGER): Hospital admission identifier
- emar_id (TEXT): Unique medication administration record
- charttime (TEXT): When medication was administered (2025 dates)
- medication (TEXT): Medication name administered
- event_txt (TEXT): Administration event type

pharmacy table:
- subject_id (INTEGER): Patient identifier
- hadm_id (INTEGER): Hospital admission identifier
- pharmacy_id (INTEGER): Unique pharmacy record
- starttime (TEXT): Medication start time
- medication (TEXT): Medication name
- route (TEXT): Administration route
- frequency (TEXT): Dosing frequency

labevents table:
- labevent_id (INTEGER): Unique lab event identifier
- subject_id (INTEGER): Patient identifier
- hadm_id (INTEGER): Hospital admission identifier
- itemid (INTEGER): Lab test identifier
- charttime (TEXT): When lab was taken (2025 dates)
- value (TEXT): Lab result value
- valuenum (REAL): Numeric lab value
- valueuom (TEXT): Units of measurement
- flag (TEXT): Abnormal flag (e.g., 'abnormal', 'critical')
- ref_range_lower (REAL): Normal range lower bound
- ref_range_upper (REAL): Normal range upper bound

diagnoses_icd table:
- subject_id (INTEGER): Patient identifier
- hadm_id (INTEGER): Hospital admission identifier
- icd_code (TEXT): ICD diagnosis code
- icd_version (INTEGER): ICD version (9 or 10)
- seq_num (INTEGER): Sequence number (1 = primary diagnosis)

procedures_icd table:
- subject_id (INTEGER): Patient identifier
- hadm_id (INTEGER): Hospital admission identifier
- icd_code (TEXT): ICD procedure code
- icd_version (INTEGER): ICD version (9 or 10)
- chartdate (TEXT): Date procedure was performed

transfers table:
- subject_id (INTEGER): Patient identifier
- hadm_id (INTEGER): Hospital admission identifier
- careunit (TEXT): Care unit name (e.g., ICU, Surgery)
- intime (TEXT): Transfer in time (2025 dates)
- outtime (TEXT): Transfer out time
- eventtype (TEXT): Transfer type

microbiologyevents table:
- subject_id (INTEGER): Patient identifier
- hadm_id (INTEGER): Hospital admission identifier
- charttime (TEXT): When specimen collected (2025 dates)
- spec_type_desc (TEXT): Specimen type (blood, urine, etc.)
- test_name (TEXT): Microbiology test name
- org_name (TEXT): Organism found
- interpretation (TEXT): Susceptibility result
"""

        self.few_shot_examples = [
            {
                "query": "Get patients who took insulin in last 24 hours",
                "sql": """SELECT DISTINCT 
    p.subject_id,
    p.gender,
    p.anchor_age,
    pr.drug,
    pr.starttime,
    e.charttime as admin_time
FROM patients p
JOIN prescriptions pr ON p.subject_id = pr.subject_id
LEFT JOIN emar e ON p.subject_id = e.subject_id 
    AND e.medication LIKE '%insulin%'
WHERE pr.drug LIKE '%insulin%'
    AND (
        pr.starttime >= datetime('now', '-24 hours')
        OR e.charttime >= datetime('now', '-24 hours')
    )
ORDER BY pr.starttime DESC
LIMIT 50;"""
            },
            {
                "query": "Recent admissions in last 24 hours", 
                "sql": """SELECT 
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.admission_type,
    p.gender,
    p.anchor_age
FROM admissions a
JOIN patients p ON a.subject_id = p.subject_id
WHERE a.admittime >= datetime('now', '-24 hours')
ORDER BY a.admittime DESC
LIMIT 50;"""
            },
            {
                "query": "Show patients with abnormal lab values in last 48 hours",
                "sql": """SELECT 
    p.subject_id,
    p.gender,
    l.charttime,
    l.itemid,
    l.value,
    l.valuenum,
    l.valueuom,
    l.flag,
    l.ref_range_lower,
    l.ref_range_upper
FROM patients p
JOIN labevents l ON p.subject_id = l.subject_id
WHERE l.flag IN ('abnormal', 'critical')
    AND l.charttime >= datetime('now', '-48 hours')
ORDER BY l.charttime DESC
LIMIT 50;"""
            },
            {
                "query": "Find patients diagnosed with diabetes",
                "sql": """SELECT DISTINCT
    p.subject_id,
    p.gender,
    p.anchor_age,
    d.icd_code,
    d.seq_num,
    a.admittime
FROM patients p
JOIN diagnoses_icd d ON p.subject_id = d.subject_id
JOIN admissions a ON d.hadm_id = a.hadm_id
WHERE d.icd_code LIKE 'E10%' OR d.icd_code LIKE 'E11%'
    OR d.icd_code LIKE '250%'
ORDER BY a.admittime DESC
LIMIT 50;"""
            },
            {
                "query": "ICU transfers in last week",
                "sql": """SELECT 
    p.subject_id,
    t.careunit,
    t.intime,
    t.outtime,
    p.gender,
    p.anchor_age
FROM patients p
JOIN transfers t ON p.subject_id = t.subject_id
WHERE t.careunit LIKE '%ICU%'
    AND t.intime >= datetime('now', '-7 days')
ORDER BY t.intime DESC
LIMIT 50;"""
            }
        ]
        
    def process_task(self, task: A2ATask) -> A2AArtifact:
        """Process A2A task and return artifact with database results"""
        try:
            if task.task_type != "structured_query":
                return A2AArtifact(
                    task_id=task.task_id,
                    success=False,
                    answer="",
                    error_message=f"Unsupported task type: {task.task_type}"
                )
            
            # Parse the query to determine what SQL to generate
            sql_query = self._generate_sql_from_query(task.query, task.parameters)
            
            if not sql_query:
                return A2AArtifact(
                    task_id=task.task_id,
                    success=False,
                    answer="",
                    error_message="Could not generate SQL query from request"
                )
            
            # Execute query via MCP
            mcp_response = self.mcp_server.call_tool("sql.query", {"sql": sql_query})
            
            if not mcp_response.success:
                return A2AArtifact(
                    task_id=task.task_id,
                    success=False,
                    answer="",
                    error_message=f"SQL execution failed: {mcp_response.error}"
                )
            
            # Format results into artifact
            return self._format_results_to_artifact(task.task_id, mcp_response.result, sql_query)
            
        except Exception as e:
            return A2AArtifact(
                task_id=task.task_id,
                success=False,
                answer="",
                error_message=f"Processing error: {str(e)}"
            )
    
    def _generate_sql_from_query(self, query: str, parameters: Dict[str, Any]) -> Optional[str]:
        """Generate SQL query from natural language using Gemini"""
        if self.model is not None:
            try:
                return self._generate_sql_with_gemini(query)
            except Exception as e:
                print(f"Warning: Gemini SQL generation failed: {e}", file=sys.stderr)
                print("No fallback available - Gemini required for SQL generation", file=sys.stderr)
        else:
            print("Warning: Gemini not available for SQL generation", file=sys.stderr)
        
        # Return None if Gemini unavailable - no fallback needed since Gemini works reliably
        return None
    
    def _generate_sql_with_gemini(self, query: str) -> Optional[str]:
        """Generate SQL using Gemini with schema context and few-shot examples"""
        
        # Build the prompt with schema and examples
        examples_text = ""
        for i, example in enumerate(self.few_shot_examples):
            examples_text += f"\nExample {i+1}:\n"
            examples_text += f"Query: {example['query']}\n"
            examples_text += f"SQL: {example['sql']}\n"
        
        prompt = f"""You are a SQL expert generating queries for a MIMIC-IV medical database. 

{self.schema_info}

REQUIREMENTS:
- Generate SQLite-compatible SQL queries only
- Use datetime('now', '-X hours') for time filtering  
- Always include LIMIT to prevent large result sets (max 50)
- Use LIKE '%term%' for medication/drug name matching (case-insensitive)
- Join patients table for demographic information when possible
- Handle both prescriptions and emar tables for medication queries
- Use DISTINCT when finding unique patients

EXAMPLES:{examples_text}

Generate a SQL query for the following request:
Query: {query}

Return only the SQL query, no explanation or markdown formatting:"""

        # Configure generation with timeout and limits
        generation_config = GenerationConfig(
            temperature=0.1,  # Low temperature for consistent SQL
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024,  # SQL queries shouldn't be too long
        )
        
        response = self.model.generate_content(prompt, generation_config=generation_config)
        sql_text = response.text.strip()
        
        # Log the LLM interaction
        llm_logger.log_interaction(
            component="structured_worker",
            prompt=prompt,
            response=sql_text,
            context=f"SQL generation for query: {query}"
        )
        
        # Clean up any markdown formatting
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
        else:
            print(f"Generated SQL failed safety check: {sql_text}", file=sys.stderr)
            return None
    
    def _is_sql_safe(self, sql: str) -> bool:
        """Basic safety validation for generated SQL"""
        sql_upper = sql.upper()
        
        # Must be a SELECT query
        if not sql_upper.strip().startswith('SELECT'):
            return False
        
        # No dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        if any(keyword in sql_upper for keyword in dangerous_keywords):
            return False
        
        # Should have LIMIT clause for safety
        if 'LIMIT' not in sql_upper:
            return False
        
        return True
    
    
    def _format_results_to_artifact(self, task_id: str, sql_result: Dict[str, Any], sql_query: str) -> A2AArtifact:
        """Format SQL results into A2A artifact - return raw data only"""
        rows = sql_result.get('rows', [])
        row_count = sql_result.get('row_count', 0)
        
        # Simple success/failure response with raw data only
        # Let orchestrator handle all interpretation
        return A2AArtifact(
            task_id=task_id,
            success=True,
            answer="",  # Empty - orchestrator handles formatting
            evidence=["prescriptions", "emar", "patients", "admissions"],
            raw_data={
                "sql_query": sql_query,
                "rows": rows,
                "row_count": row_count
            }
        )


# FastAPI app setup
app = FastAPI(title="Structured Worker", version="1.0.0")
worker = StructuredWorker()


@app.post("/a2a/task")
async def handle_a2a_task(task_data: dict) -> dict:
    """Handle incoming A2A tasks"""
    try:
        task = A2ATask(**task_data)
        artifact = worker.process_task(task)
        return artifact.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "worker": "structured"}


@app.get("/tools")
async def list_tools():
    """List available MCP tools"""
    return {"tools": worker.mcp_server.get_available_tools()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

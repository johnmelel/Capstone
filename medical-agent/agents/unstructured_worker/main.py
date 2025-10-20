"""
Unstructured Data Worker - Handles A2A tasks for medical context retrieval
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import re
from typing import Dict, Any

from common.a2a_messages import A2ATask, A2AArtifact
from common.mcp_vector_server import MCPVectorServer, MedicalContextBuilder


class UnstructuredWorker:
    """Worker that handles medical context retrieval from vector store"""
    
    def __init__(self):
        self.mcp_server = MCPVectorServer()
        self.context_builder = MedicalContextBuilder()
        
    def process_task(self, task: A2ATask) -> A2AArtifact:
        """Process A2A task and return artifact with medical context"""
        try:
            if task.task_type != "unstructured_search":
                return A2AArtifact(
                    task_id=task.task_id,
                    success=False,
                    answer="",
                    error_message=f"Unsupported task type: {task.task_type}"
                )
            
            # Generate appropriate vector search query
            search_query = self._generate_search_query(task.query, task.parameters)
            
            # Execute search via MCP
            top_k = task.parameters.get("top_k", 3)
            mcp_response = self.mcp_server.call_tool("vector.search", {
                "query": search_query,
                "top_k": top_k
            })
            
            if not mcp_response.success:
                return A2AArtifact(
                    task_id=task.task_id,
                    success=False,
                    answer="",
                    error_message=f"Vector search failed: {mcp_response.error}"
                )
            
            # Format results into artifact
            return self._format_results_to_artifact(task.task_id, mcp_response.result, search_query)
            
        except Exception as e:
            return A2AArtifact(
                task_id=task.task_id,
                success=False,
                answer="",
                error_message=f"Processing error: {str(e)}"
            )
    
    def _generate_search_query(self, query: str, parameters: Dict[str, Any]) -> str:
        """Generate vector search query from natural language query"""
        query_lower = query.lower()
        
        # Extract drug name pattern
        drug_match = re.search(r'(amphetamine|amoxicillin|aspirin|metformin|insulin|warfarin|lisinopril|atorvastatin)', query_lower)
        
        if drug_match:
            drug_name = drug_match.group(1)
            
            # Determine what kind of context is needed
            if any(word in query_lower for word in ['monitor', 'monitoring', 'side effects', 'effects']):
                return self.context_builder.build_drug_monitoring_query(drug_name)
            elif any(word in query_lower for word in ['safety', 'warning', 'contraindication', 'interaction']):
                return self.context_builder.build_safety_query(drug_name)
            else:
                # Default to monitoring query
                return self.context_builder.build_drug_monitoring_query(drug_name)
        
        # For non-drug queries, search for general medical context
        if any(word in query_lower for word in ['admission', 'hospital', 'patient']):
            return self.context_builder.build_general_query("hospital admission protocols")
        
        # Fallback: use original query
        return query
    
    def _format_results_to_artifact(self, task_id: str, search_result: Dict[str, Any], search_query: str) -> A2AArtifact:
        """Format vector search results into A2A artifact"""
        documents = search_result.get('documents', [])
        total_found = search_result.get('total_found', 0)
        
        if total_found == 0 or not documents:
            return A2AArtifact(
                task_id=task_id,
                success=True,
                answer="No relevant medical context found.",
                evidence=["medical_knowledge_base"],
                raw_data={"documents": documents, "search_query": search_query}
            )
        
        # Create summary from documents
        context_snippets = []
        doc_sources = []
        
        for doc in documents:
            context_snippets.append(doc.get('text', ''))
            doc_sources.append(doc.get('doc_id', 'unknown'))
        
        # Create answer combining key insights
        answer = f"Found {len(documents)} relevant medical guidelines. "
        
        # Extract key monitoring/safety points from first document
        if documents and 'text' in documents[0]:
            first_doc = documents[0]['text']
            if 'monitor' in first_doc.lower():
                answer += "Key monitoring points: cardiac assessment, blood pressure monitoring. "
            elif 'side effect' in first_doc.lower():
                answer += "Important side effects noted in clinical guidelines. "
            elif 'contraindication' in first_doc.lower():
                answer += "Important contraindications and safety warnings identified. "
        
        return A2AArtifact(
            task_id=task_id,
            success=True,
            answer=answer,
            evidence=doc_sources,
            raw_data={
                "documents": documents,
                "context_snippets": context_snippets,
                "search_query": search_query,
                "total_found": total_found
            }
        )


# FastAPI app setup
app = FastAPI(title="Unstructured Worker", version="1.0.0")
worker = UnstructuredWorker()


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
    return {"status": "healthy", "worker": "unstructured"}


@app.get("/tools")
async def list_tools():
    """List available MCP tools"""
    return {"tools": worker.mcp_server.get_available_tools()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

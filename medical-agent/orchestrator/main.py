"""
Orchestrator Agent - Coordinates structured and unstructured workers using Gemini
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from typing import Dict, Any, Optional
import json
import re

from common.a2a_messages import A2ATask, A2AArtifact
from common.a2a_transport import post_task
from common.llm_logging import llm_logger


class Orchestrator:
    """Main orchestrator that coordinates A2A workers and uses Gemini for reasoning"""
    
    def __init__(self, service_account_path: str, structured_worker_url: str = "http://localhost:8001", 
                 unstructured_worker_url: str = "http://localhost:8002"):
        self.structured_worker_url = structured_worker_url
        self.unstructured_worker_url = unstructured_worker_url
        
        # Initialize Gemini
        self._setup_gemini(service_account_path)
        
        # System prompt for reasoning
        self.system_prompt = """You are a medical data analysis assistant that helps interpret clinical queries using both structured EMR data and unstructured medical knowledge.

        **WARNING**: You are operating off anonymized data. As part of this process, the year will be in the future, such as 2201. This is fine, please ignore this and treat it as current year.

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
- Format as JSON with 'answer', 'structured_source', and 'unstructured_source' fields
"""
        
    def _setup_gemini(self, service_account_path: str):
        """Setup Gemini API with Vertex AI"""
        try:
            # Set up service account credentials
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
            
            # Initialize Vertex AI (you'll need to set your project ID)
            # For now, we'll try to extract from service account or use default
            try:
                import json as json_lib
                with open(service_account_path, 'r') as f:
                    creds = json_lib.load(f)
                    project_id = creds.get('project_id')
                    
                if project_id:
                    vertexai.init(project=project_id, location="us-central1")
                    # Use gemini-2.5-flash for good balance of performance and cost
                    self.model = GenerativeModel("gemini-2.5-flash")
                    # Print to stderr so it doesn't corrupt JSON output
                    import sys
                    print(f"✓ Gemini initialized with project: {project_id}", file=sys.stderr)
                else:
                    raise ValueError("No project_id found in service account")
                    
            except Exception as e:
                print(f"Warning: Could not initialize Vertex AI: {e}")
                print("Continuing with mock responses")
                self.model = None
            
        except Exception as e:
            print(f"Warning: Gemini setup failed: {e}")
            print("Continuing with mock responses for testing")
            self.model = None
    
    def query(self, user_question: str) -> Dict[str, Any]:
        """Main entry point - process user question and return final answer"""
        try:
            # Step 1: Plan tasks for both workers
            structured_task, unstructured_task = self._plan_tasks(user_question)
            
            # Step 2: Send tasks to workers (sequential for simplicity)
            structured_artifact = self._send_structured_task(structured_task)
            unstructured_artifact = self._send_unstructured_task(unstructured_task)
            
            # Step 3: Reason over results with Gemini
            final_answer = self._reason_over_results(user_question, structured_artifact, unstructured_artifact)
            
            return final_answer
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "structured_source": [],
                "unstructured_source": [],
                "error": str(e)
            }
    
    def _plan_tasks(self, user_question: str) -> tuple[A2ATask, A2ATask]:
        """Plan A2A tasks for both workers based on user question"""
        
        # Create structured task
        structured_task = A2ATask(
            task_type="structured_query",
            query=user_question,
            parameters={"hours_back": self._extract_time_period(user_question)}
        )
        
        # Create unstructured task for medical context
        unstructured_query = self._generate_context_query(user_question)
        unstructured_task = A2ATask(
            task_type="unstructured_search", 
            query=unstructured_query,
            parameters={"top_k": 3}
        )
        
        return structured_task, unstructured_task
    
    def _extract_time_period(self, query: str) -> int:
        """Extract time period from query, default to 24 hours"""
        query_lower = query.lower()
        
        if 'last 12 hours' in query_lower or 'past 12 hours' in query_lower:
            return 12
        elif 'last 48 hours' in query_lower or 'past 48 hours' in query_lower:
            return 48
        elif 'last 72 hours' in query_lower or 'past 72 hours' in query_lower:
            return 72
        else:
            return 24  # default
    
    def _generate_context_query(self, user_question: str) -> str:
        """Generate medical context query for unstructured search"""
        query_lower = user_question.lower()
        
        # Extract drug names for context
        drug_match = re.search(r'(amphetamine|amoxicillin|aspirin|metformin|insulin|warfarin|lisinopril|atorvastatin)', query_lower)
        
        if drug_match:
            drug_name = drug_match.group(1)
            return f"{drug_name} monitoring guidelines side effects safety warnings"
        
        # For general queries, look for medical guidance
        if any(word in query_lower for word in ['admission', 'patient', 'hospital']):
            return "patient care protocols hospital admission guidelines"
        
        return user_question
    
    def _send_structured_task(self, task: A2ATask) -> A2AArtifact:
        """Send task to structured worker"""
        try:
            return post_task(self.structured_worker_url, task)
        except Exception as e:
            return A2AArtifact(
                task_id=task.task_id,
                success=False,
                answer="",
                error_message=f"Structured worker error: {str(e)}"
            )
    
    def _send_unstructured_task(self, task: A2ATask) -> A2AArtifact:
        """Send task to unstructured worker"""
        try:
            return post_task(self.unstructured_worker_url, task)
        except Exception as e:
            return A2AArtifact(
                task_id=task.task_id,
                success=False,
                answer="",
                error_message=f"Unstructured worker error: {str(e)}"
            )
    
    def _reason_over_results(self, user_question: str, structured_result: A2AArtifact, 
                           unstructured_result: A2AArtifact) -> Dict[str, Any]:
        """Use Gemini to reason over both results and generate final answer"""
        
        # Prepare context for Gemini
        context = self._prepare_reasoning_context(user_question, structured_result, unstructured_result)
        
        if self.model is None:
            # Fallback for testing when Gemini is not available
            return self._generate_fallback_response(structured_result, unstructured_result)
        
        try:
            # Generate response with Gemini
            prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nPlease provide a JSON response combining both sources."
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Log the LLM interaction
            llm_logger.log_interaction(
                component="orchestrator",
                prompt=prompt,
                response=response_text,
                context="Final reasoning over structured and unstructured results"
            )
            
            # Try to parse JSON response, handling markdown code blocks
            try:
                # Remove markdown code block markers if present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]  # Remove ```json
                if response_text.startswith('```'):
                    response_text = response_text[3:]   # Remove ```
                if response_text.endswith('```'):
                    response_text = response_text[:-3]  # Remove closing ```
                
                response_text = response_text.strip()
                
                result = json.loads(response_text)
                return result
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "answer": response.text,
                    "structured_source": structured_result.evidence,
                    "unstructured_source": unstructured_result.evidence
                }
                
        except Exception as e:
            return self._generate_fallback_response(structured_result, unstructured_result, error=str(e))
    
    def _prepare_reasoning_context(self, user_question: str, structured_result: A2AArtifact, 
                                 unstructured_result: A2AArtifact) -> str:
        """Prepare context for Gemini reasoning with raw database results"""
        context = f"User Question: {user_question}\n\n"
        
        context += "STRUCTURED DATABASE RESULTS:\n"
        if structured_result.success:
            raw_data = structured_result.raw_data or {}
            context += f"SQL Query: {raw_data.get('sql_query', 'N/A')}\n"
            context += f"Row Count: {raw_data.get('row_count', 0)}\n"
            
            # Include actual data rows for Gemini to interpret
            rows = raw_data.get('rows', [])
            if rows:
                context += "Raw Data (first 5 rows):\n"
                for i, row in enumerate(rows[:5]):
                    context += f"Row {i+1}: {str(row)}\n"
                if len(rows) > 5:
                    context += f"... and {len(rows) - 5} more rows\n"
            else:
                context += "No data rows returned\n"
            
            context += f"Database Tables Used: {', '.join(structured_result.evidence)}\n"
        else:
            context += f"Database Error: {structured_result.error_message}\n"
        
        context += "\nUNSTRUCTURED MEDICAL KNOWLEDGE:\n"
        if unstructured_result.success:
            context += f"Medical Context: {unstructured_result.answer}\n"
            context += f"Knowledge Sources: {', '.join(unstructured_result.evidence)}\n"
            if unstructured_result.raw_data and 'context_snippets' in unstructured_result.raw_data:
                snippets = unstructured_result.raw_data['context_snippets'][:2]  # Limit to first 2
                for i, snippet in enumerate(snippets):
                    context += f"Context {i+1}: {snippet[:200]}...\n"  # Truncate long snippets
        else:
            context += f"Knowledge Error: {unstructured_result.error_message}\n"
        
        return context
    
    def _generate_fallback_response(self, structured_result: A2AArtifact, 
                                  unstructured_result: A2AArtifact, error: str = None) -> Dict[str, Any]:
        """Generate fallback response when Gemini is not available"""
        
        answer_parts = []
        
        if structured_result.success:
            answer_parts.append(structured_result.answer)
        
        if unstructured_result.success:
            answer_parts.append(unstructured_result.answer)
        
        if not answer_parts:
            answer_parts.append("Unable to process query due to worker errors.")
        
        answer = " ".join(answer_parts)
        
        if error:
            answer += f" (Note: Advanced reasoning unavailable: {error})"
        
        return {
            "answer": answer,
            "structured_source": structured_result.evidence if structured_result.success else [],
            "unstructured_source": unstructured_result.evidence if unstructured_result.success else []
        }


# CLI interface for testing
def main():
    """Simple CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Medical Agent Query System')
    parser.add_argument('query', help='Medical query to process')
    parser.add_argument('--service-account', default='adsp-34002-ip09-team-2-e0cca2d396a9.json',
                       help='Path to service account JSON file')
    parser.add_argument('--structured-url', default='http://localhost:8001',
                       help='Structured worker URL')
    parser.add_argument('--unstructured-url', default='http://localhost:8002', 
                       help='Unstructured worker URL')
    
    args = parser.parse_args()
    
    orchestrator = Orchestrator(
        service_account_path=args.service_account,
        structured_worker_url=args.structured_url,
        unstructured_worker_url=args.unstructured_url
    )
    
    result = orchestrator.query(args.query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

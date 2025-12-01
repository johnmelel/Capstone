"""
BASELINE 3: Vector-Only Orchestrator
Only uses unstructured medical literature, no structured EMR data.
This demonstrates the value of multi-modal data integration.
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


class VectorOnlyOrchestrator:
    """Baseline orchestrator that only uses vector search, no structured data"""
    
    def __init__(self, service_account_path: str, 
                 unstructured_worker_url: str = "http://localhost:8002"):
        self.unstructured_worker_url = unstructured_worker_url
        
        # Initialize Gemini
        self._setup_gemini(service_account_path)
        
        # Simplified system prompt since we only have one data source
        self.system_prompt = """You are a medical knowledge assistant that helps answer clinical queries using medical literature and guidelines.

Your role is to:
1. Analyze user queries about medical topics
2. Use medical literature and guidelines to provide answers
3. Provide clear, evidence-based recommendations
4. Always cite your sources

When responding:
- Be precise and factual based on medical literature
- Note any important clinical considerations
- Keep responses focused and actionable
- Count your reasoning steps as you work through the problem
- Format as JSON with 'answer', 'sources', and 'reasoning_steps' fields

IMPORTANT: Please count and include the number of distinct reasoning steps you took to arrive at your answer. Include this as 'reasoning_steps' (integer) in your JSON response.
"""
        
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
                    print(f"Vector-Only Orchestrator: Gemini initialized", file=sys.stderr)
                else:
                    raise ValueError("No project_id found in service account")
                    
            except Exception as e:
                print(f"Warning: Could not initialize Vertex AI: {e}")
                self.model = None
            
        except Exception as e:
            print(f"Warning: Gemini setup failed: {e}")
            self.model = None
    
    def query(self, user_question: str) -> Dict[str, Any]:
        """Main entry point - process user question using ONLY vector search"""
        try:
            # Only send to unstructured worker
            unstructured_task = self._create_unstructured_task(user_question)
            unstructured_artifact = self._send_unstructured_task(unstructured_task)
            
            # Reason over results with Gemini
            final_answer = self._reason_over_results(user_question, unstructured_artifact)
            
            return final_answer
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def _create_unstructured_task(self, user_question: str) -> A2ATask:
        """Create unstructured task for medical context"""
        unstructured_query = self._generate_context_query(user_question)
        return A2ATask(
            task_type="unstructured_search", 
            query=unstructured_query,
            parameters={"top_k": 3}
        )
    
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
    
    def _reason_over_results(self, user_question: str, unstructured_result: A2AArtifact) -> Dict[str, Any]:
        """Use Gemini to reason over vector search results only"""
        
        # Prepare context for Gemini
        context = self._prepare_reasoning_context(user_question, unstructured_result)
        
        if self.model is None:
            return self._generate_fallback_response(unstructured_result)
        
        try:
            prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nPlease provide a JSON response based on medical literature."
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Log the LLM interaction
            llm_logger.log_interaction(
                component="vector_only_orchestrator",
                prompt=prompt,
                response=response_text,
                context="Vector-only reasoning"
            )
            
            # Try to parse JSON response
            try:
                # Remove markdown code block markers if present
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
                    "sources": unstructured_result.evidence
                }
                
        except Exception as e:
            return self._generate_fallback_response(unstructured_result, error=str(e))
    
    def _prepare_reasoning_context(self, user_question: str, unstructured_result: A2AArtifact) -> str:
        """Prepare context for Gemini reasoning with vector search results only"""
        context = f"User Question: {user_question}\n\n"
        
        context += "MEDICAL LITERATURE:\n"
        if unstructured_result.success:
            context += f"Medical Context: {unstructured_result.answer}\n"
            context += f"Knowledge Sources: {', '.join(unstructured_result.evidence)}\n"
            if unstructured_result.raw_data and 'context_snippets' in unstructured_result.raw_data:
                snippets = unstructured_result.raw_data['context_snippets'][:2]
                for i, snippet in enumerate(snippets):
                    context += f"Context {i+1}: {snippet[:200]}...\n"
        else:
            context += f"Knowledge Error: {unstructured_result.error_message}\n"
        
        context += "\nNOTE: No patient-specific EMR data is available. Provide general medical guidance based on literature only.\n"
        
        return context
    
    def _generate_fallback_response(self, unstructured_result: A2AArtifact, error: str = None) -> Dict[str, Any]:
        """Generate fallback response when Gemini is not available"""
        
        answer = unstructured_result.answer if unstructured_result.success else "Unable to retrieve medical literature."
        
        if error:
            answer += f" (Note: Advanced reasoning unavailable: {error})"
        
        return {
            "answer": answer,
            "sources": unstructured_result.evidence if unstructured_result.success else []
        }


# CLI interface for testing
def main():
    """Simple CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vector-Only Baseline System')
    parser.add_argument('query', help='Medical query to process')
    parser.add_argument('--service-account', default='adsp-34002-ip09-team-2-e0cca2d396a9.json',
                       help='Path to service account JSON file')
    parser.add_argument('--unstructured-url', default='http://localhost:8002', 
                       help='Unstructured worker URL')
    
    args = parser.parse_args()
    
    orchestrator = VectorOnlyOrchestrator(
        service_account_path=args.service_account,
        unstructured_worker_url=args.unstructured_url
    )
    
    result = orchestrator.query(args.query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

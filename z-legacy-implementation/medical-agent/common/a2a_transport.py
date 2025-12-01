"""
A2A Transport Layer - HTTP-based communication between agents
"""
import httpx
import json
from typing import Optional
from .a2a_messages import A2ATask, A2AArtifact


class A2ATransport:
    """Simple HTTP transport for A2A messaging"""
    
    def __init__(self, timeout: int = 120):
        self.client = httpx.Client(timeout=timeout)
        
    def send_task(self, worker_url: str, task: A2ATask) -> Optional[A2AArtifact]:
        """Send A2A task to worker and return artifact response"""
        try:
            response = self.client.post(
                f"{worker_url}/a2a/task",
                json=task.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            artifact_data = response.json()
            return A2AArtifact(**artifact_data)
            
        except Exception as e:
            # Return error artifact
            return A2AArtifact(
                task_id=task.task_id,
                success=False,
                answer="",
                error_message=f"Transport error: {str(e)}"
            )
    
    def close(self):
        """Close HTTP client"""
        self.client.close()


def post_task(url: str, task: A2ATask) -> A2AArtifact:
    """Convenience function for sending single task"""
    transport = A2ATransport()
    try:
        return transport.send_task(url, task)
    finally:
        transport.close()

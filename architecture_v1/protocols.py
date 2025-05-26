from typing import Any, Dict

class A2AMessage:
    def __init__(self, sender: str, receiver: str, content: Dict[str, Any]):
        self.sender = sender
        self.receiver = receiver
        self.content = content

class MCPRequest:
    def __init__(self, model: str, prompt: str, context: Any = None):
        self.model = model
        self.prompt = prompt
        self.context = context

class MCPResponse:
    def __init__(self, result: Any, metadata: Dict[str, Any] = None):
        self.result = result
        self.metadata = metadata or {}
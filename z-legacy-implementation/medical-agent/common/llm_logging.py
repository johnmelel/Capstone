"""
LLM Logging utilities for capturing prompts and responses in verbose mode
"""
import sys
import os
from typing import Optional

class LLMLogger:
    """Logger for capturing LLM prompts and responses"""
    
    def __init__(self):
        self.enabled = os.getenv("LLM_VERBOSE_LOGGING", "false").lower() == "true"
        self.logs = []
    
    def log_interaction(self, component: str, prompt: str, response: str, context: str = ""):
        """Log an LLM interaction"""
        if not self.enabled:
            return
            
        log_entry = {
            "component": component,
            "context": context,
            "prompt": prompt,
            "response": response
        }
        self.logs.append(log_entry)
        
        # Print to stderr so it doesn't interfere with JSON output
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"LLM INTERACTION - {component.upper()}", file=sys.stderr)
        if context:
            print(f"Context: {context}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"PROMPT:", file=sys.stderr)
        print(prompt, file=sys.stderr)
        print(f"\n{'-'*40}", file=sys.stderr)
        print(f"RESPONSE:", file=sys.stderr)
        print(response, file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
    
    def get_logs(self):
        """Get all logged interactions"""
        return self.logs
    
    def clear_logs(self):
        """Clear all logs"""
        self.logs = []

# Global logger instance
llm_logger = LLMLogger()

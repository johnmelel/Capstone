"""
Medical Agent Workers

This package contains worker agents that handle specific data access tasks:
- EMRWorker: Queries EMR database via MCP
- ResearchWorker: Searches medical literature via MCP
"""

from .emr_worker import EMRWorker
from .research_worker import ResearchWorker

__all__ = ['EMRWorker', 'ResearchWorker']

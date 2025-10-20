"""
Dummy MCP-Vector Server Implementation
Returns mock medical context data until real Milvus cloud is available
"""
import json
from typing import Dict, Any, List, Optional
from .a2a_messages import MCPRequest, MCPResponse


class MCPVectorServer:
    """Dummy MCP server for vector operations - returns mock medical data"""
    
    def __init__(self):
        # Mock medical knowledge base
        self.medical_knowledge = {
            "amphetamine": [
                {
                    "text": "Amphetamine monitoring requires cardiac assessment including blood pressure and heart rate checks. Monitor for arrhythmias and hypertension.",
                    "source": "Clinical Pharmacology Guidelines",
                    "doc_id": "pharm_guide_001",
                    "relevance": 0.95
                },
                {
                    "text": "Common side effects of amphetamine include increased heart rate, elevated blood pressure, decreased appetite, and insomnia. Patients should be monitored for cardiovascular complications.",
                    "source": "Drug Safety Manual", 
                    "doc_id": "safety_manual_034",
                    "relevance": 0.89
                },
                {
                    "text": "Amphetamine contraindications include severe cardiovascular disease, hyperthyroidism, and concurrent MAO inhibitor use. Dose adjustments may be needed in elderly patients.",
                    "source": "Prescribing Guidelines",
                    "doc_id": "prescribe_guide_022",
                    "relevance": 0.87
                }
            ],
            "default": [
                {
                    "text": "General medication monitoring includes assessment of therapeutic effectiveness, adverse reactions, drug interactions, and patient adherence to treatment regimens.",
                    "source": "Clinical Practice Standards",
                    "doc_id": "practice_std_001",
                    "relevance": 0.75
                },
                {
                    "text": "Patient safety protocols require regular review of medication lists, allergy history, and contraindications before prescribing or administering drugs.",
                    "source": "Patient Safety Guidelines",
                    "doc_id": "safety_guide_002", 
                    "relevance": 0.70
                }
            ]
        }
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResponse:
        """Handle MCP tool calls"""
        if tool_name == "vector.search":
            return self._search_vectors(
                arguments.get("query", ""),
                arguments.get("top_k", 3)
            )
        else:
            return MCPResponse(
                success=False,
                error=f"Unknown tool: {tool_name}"
            )
    
    def _search_vectors(self, query: str, top_k: int = 3) -> MCPResponse:
        """Mock vector search - returns relevant medical context"""
        try:
            query_lower = query.lower()
            
            # Find relevant knowledge based on query content
            knowledge_key = "default"
            for drug in self.medical_knowledge.keys():
                if drug in query_lower and drug != "default":
                    knowledge_key = drug
                    break
            
            # Get relevant documents
            docs = self.medical_knowledge.get(knowledge_key, self.medical_knowledge["default"])
            
            # Return top_k results
            results = docs[:min(top_k, len(docs))]
            
            return MCPResponse(
                success=True,
                result={
                    "documents": results,
                    "total_found": len(results),
                    "query": query
                }
            )
            
        except Exception as e:
            return MCPResponse(
                success=False,
                error=f"Vector search error: {str(e)}"
            )
    
    def get_available_tools(self) -> List[str]:
        """Return list of available tools"""
        return ["vector.search"]
    
    def add_medical_knowledge(self, drug_name: str, knowledge_items: List[Dict[str, Any]]):
        """Add new medical knowledge (for future expansion)"""
        self.medical_knowledge[drug_name.lower()] = knowledge_items


class MedicalContextBuilder:
    """Helper class for building medical context queries"""
    
    @staticmethod
    def build_drug_monitoring_query(drug_name: str) -> str:
        """Build query for drug monitoring guidelines"""
        return f"{drug_name} monitoring guidelines side effects contraindications dosing"
    
    @staticmethod
    def build_safety_query(drug_name: str) -> str:
        """Build query for drug safety information"""
        return f"{drug_name} safety warnings adverse reactions drug interactions"
    
    @staticmethod
    def build_general_query(medical_condition: str) -> str:
        """Build general medical query"""
        return f"{medical_condition} clinical guidelines treatment protocols"

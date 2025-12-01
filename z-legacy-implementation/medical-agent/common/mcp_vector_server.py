"""
Real MCP-Vector Server Implementation using Milvus Cloud
Performs actual vector searches against medical knowledge base
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from pymilvus import connections, Collection, utility
from .a2a_messages import MCPRequest, MCPResponse


# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPVectorServer:
    """Real MCP server for vector operations using Milvus Cloud"""
    
    def __init__(self):
        self.milvus_connected = False
        self.collection = None
        self.embedding_model = None
        
        # Configuration from .env
        self.milvus_uri = os.getenv('MILVUS_URI')
        self.milvus_api_key = os.getenv('MILVUS_API_KEY')
        self.collection_name = os.getenv('MILVUS_COLLECTION_NAME', 'capstone_group_2')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.embedding_dimension = int(os.getenv('EMBEDDING_DIMENSION', '768'))
        
        # Initialize connections
        self._initialize_gemini()
        self._initialize_milvus()
    
    def _initialize_gemini(self):
        """Initialize Gemini for embeddings"""
        try:
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                self.embedding_model = 'models/text-embedding-004'
                logger.info("✓ Gemini embeddings initialized")
            else:
                logger.warning("Gemini API key not found - falling back to mock embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini embeddings: {e}")
            self.embedding_model = None
    
    def _initialize_milvus(self):
        """Initialize connection to Milvus Cloud"""
        try:
            if not self.milvus_uri or not self.milvus_api_key:
                logger.warning("Milvus credentials not found - using fallback mock data")
                return
            
            # Connect to Milvus Cloud
            connections.connect(
                alias="default",
                uri=self.milvus_uri,
                token=self.milvus_api_key
            )
            
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                self.milvus_connected = True
                logger.info(f"✓ Connected to Milvus collection: {self.collection_name}")
                
                # Get collection info
                num_entities = self.collection.num_entities
                logger.info(f"Collection contains {num_entities} documents")
            else:
                logger.warning(f"Collection '{self.collection_name}' not found - using fallback mock data")
                
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.milvus_connected = False
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Gemini"""
        try:
            if self.embedding_model:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_query"
                )
                return result['embedding']
            else:
                # Mock embedding for fallback
                import random
                return [random.random() for _ in range(self.embedding_dimension)]
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
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
        """Perform vector search against Milvus or fallback to mock data"""
        try:
            if self.milvus_connected and self.collection:
                return self._search_milvus(query, top_k)
            else:
                return self._search_fallback(query, top_k)
                
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return self._search_fallback(query, top_k)
    
    def _search_milvus(self, query: str, top_k: int) -> MCPResponse:
        """Search real Milvus database"""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                raise Exception("Failed to generate query embedding")
            
            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}
            }
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="vector",  # Vector field per schema
                param=search_params,
                limit=top_k,
                output_fields=["text", "file_name", "file_hash", "chunk_index", "total_chunks"]
            )
            
            # Format results
            documents = []
            if results and len(results) > 0:
                for hit in results[0]:
                    doc = {
                        "text": hit.entity.get("text", ""),
                        "source": hit.entity.get("file_name", "Unknown"),
                        "relevance": float(hit.score),
                        "doc_id": str(hit.id),
                        "file_hash": hit.entity.get("file_hash", ""),
                        "chunk_index": hit.entity.get("chunk_index", 0),
                        "total_chunks": hit.entity.get("total_chunks", 0)
                    }
                    documents.append(doc)
            
            return MCPResponse(
                success=True,
                result={
                    "documents": documents,
                    "total_found": len(documents),
                    "query": query,
                    "source": "milvus_cloud"
                }
            )
            
        except Exception as e:
            logger.error(f"Milvus search error: {e}")
            return self._search_fallback(query, top_k)
    
    def _search_fallback(self, query: str, top_k: int) -> MCPResponse:
        """Fallback mock search when Milvus is unavailable"""
        logger.info("Using fallback mock medical knowledge")
        
        # Enhanced mock medical knowledge base
        medical_knowledge = {
            "amphetamine": [
                {
                    "text": "Amphetamine monitoring requires cardiac assessment including blood pressure and heart rate checks. Monitor for arrhythmias and hypertension. Regular ECGs may be warranted in patients with risk factors.",
                    "source": "Clinical Pharmacology Guidelines",
                    "doc_id": "pharm_guide_001",
                    "relevance": 0.95
                },
                {
                    "text": "Common side effects of amphetamine include increased heart rate, elevated blood pressure, decreased appetite, and insomnia. Patients should be monitored for cardiovascular complications and psychiatric symptoms.",
                    "source": "Drug Safety Manual", 
                    "doc_id": "safety_manual_034",
                    "relevance": 0.89
                }
            ],
            "troponin": [
                {
                    "text": "Elevated troponin levels indicate myocardial injury and are highly sensitive markers for acute coronary syndrome. Serial measurements help assess ongoing cardiac damage.",
                    "source": "Cardiac Biomarkers Clinical Guide",
                    "doc_id": "cardiac_bio_001",
                    "relevance": 0.96
                },
                {
                    "text": "Troponin elevation can occur in non-ACS conditions including myocarditis, pulmonary embolism, sepsis, and chronic kidney disease. Clinical context is essential for interpretation.",
                    "source": "Emergency Medicine Handbook",
                    "doc_id": "em_handbook_045",
                    "relevance": 0.91
                }
            ],
            "chest pain": [
                {
                    "text": "Acute chest pain evaluation requires systematic assessment including ECG, cardiac biomarkers, and risk stratification. Consider ACS, aortic dissection, and pulmonary embolism as high-risk diagnoses.",
                    "source": "Emergency Cardiology Protocols",
                    "doc_id": "cardio_protocol_023",
                    "relevance": 0.94
                }
            ],
            "insulin": [
                {
                    "text": "Insulin therapy monitoring includes regular blood glucose checking, HbA1c assessment, and screening for hypoglycemic episodes. Adjust dosing based on carbohydrate intake and activity levels.",
                    "source": "Diabetes Management Guidelines",
                    "doc_id": "diabetes_guide_012",
                    "relevance": 0.93
                }
            ],
            "warfarin": [
                {
                    "text": "Warfarin requires regular INR monitoring with target range typically 2.0-3.0 for most indications. Monitor for bleeding complications and drug-drug interactions that affect metabolism.",
                    "source": "Anticoagulation Clinical Guidelines",
                    "doc_id": "anticoag_guide_007",
                    "relevance": 0.92
                }
            ],
            "default": [
                {
                    "text": "General medication monitoring includes assessment of therapeutic effectiveness, adverse reactions, drug interactions, and patient adherence to treatment regimens. Regular follow-up is essential.",
                    "source": "Clinical Practice Standards",
                    "doc_id": "practice_std_001",
                    "relevance": 0.75
                },
                {
                    "text": "Patient safety protocols require regular review of medication lists, allergy history, and contraindications before prescribing or administering drugs. Documentation is critical.",
                    "source": "Patient Safety Guidelines",
                    "doc_id": "safety_guide_002", 
                    "relevance": 0.70
                }
            ]
        }
        
        # Find most relevant knowledge
        query_lower = query.lower()
        knowledge_key = "default"
        
        # Check for specific medical terms
        for key in medical_knowledge.keys():
            if key in query_lower and key != "default":
                knowledge_key = key
                break
        
        # Get relevant documents
        docs = medical_knowledge.get(knowledge_key, medical_knowledge["default"])
        results = docs[:min(top_k, len(docs))]
        
        return MCPResponse(
            success=True,
            result={
                "documents": results,
                "total_found": len(results),
                "query": query,
                "source": "fallback_mock"
            }
        )
    
    def get_available_tools(self) -> List[str]:
        """Return list of available tools"""
        return ["vector.search"]
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            "milvus_connected": self.milvus_connected,
            "collection_name": self.collection_name,
            "embedding_model_available": self.embedding_model is not None,
            "collection_entities": self.collection.num_entities if self.collection else 0
        }


class MedicalContextBuilder:
    """Helper class for building medical context queries"""
    
    @staticmethod
    def build_drug_monitoring_query(drug_name: str) -> str:
        """Build query for drug monitoring guidelines"""
        return f"{drug_name} monitoring guidelines side effects contraindications dosing safety"
    
    @staticmethod
    def build_safety_query(drug_name: str) -> str:
        """Build query for drug safety information"""
        return f"{drug_name} safety warnings adverse reactions drug interactions precautions"
    
    @staticmethod
    def build_diagnostic_query(symptoms: str) -> str:
        """Build query for diagnostic information"""
        return f"{symptoms} diagnosis differential clinical assessment evaluation"
    
    @staticmethod
    def build_general_query(medical_condition: str) -> str:
        """Build general medical query"""
        return f"{medical_condition} clinical guidelines treatment protocols management"

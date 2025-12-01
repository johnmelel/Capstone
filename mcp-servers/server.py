#!/usr/bin/env python3
"""
Unified MCP Server for Medical Agent Data Access
Provides both EMR database and vector search capabilities through MCP protocol
"""
import os
import sys
import sqlite3
import json
import logging
from typing import Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# MCP SDK imports
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp import types

# Milvus imports
from pymilvus import connections, Collection, utility
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedMCPServer:
    """Unified MCP server providing both EMR database and vector search access"""
    
    def __init__(self):
        # Database configuration
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'database',
            'mimic_emr.db'
        )
        
        # Milvus configuration
        self.milvus_uri = os.getenv('MILVUS_URI')
        self.milvus_api_key = os.getenv('MILVUS_API_KEY')
        self.milvus_collection = os.getenv('MILVUS_COLLECTION_NAME', 'capstone_group_2')
        self.embedding_dimension = int(os.getenv('EMBEDDING_DIMENSION', '768'))
        
        # Gemini configuration
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        # Initialize connections
        self.collection = None
        self.milvus_connected = False
        self._initialize_milvus()
        self._initialize_gemini()
        
        logger.info(f"Initialized server with database: {self.db_path}")
        logger.info(f"Milvus connected: {self.milvus_connected}")
    
    def _initialize_gemini(self):
        """Initialize Gemini for embeddings"""
        try:
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                self.embedding_model = 'models/text-embedding-004'
                logger.info("Gemini embeddings initialized")
            else:
                logger.warning("Gemini API key not found")
                self.embedding_model = None
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.embedding_model = None
    
    def _initialize_milvus(self):
        """Initialize connection to Milvus Cloud"""
        try:
            if not self.milvus_uri or not self.milvus_api_key:
                logger.warning("Milvus credentials not found")
                return
            
            connections.connect(
                alias="default",
                uri=self.milvus_uri,
                token=self.milvus_api_key
            )
            
            if utility.has_collection(self.milvus_collection):
                self.collection = Collection(self.milvus_collection)
                self.collection.load()
                self.milvus_connected = True
                logger.info(f"Connected to Milvus collection: {self.milvus_collection}")
                logger.info(f"Collection contains {self.collection.num_entities} documents")
            else:
                logger.warning(f"Collection '{self.milvus_collection}' not found")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.milvus_connected = False
    
    def _get_db_connection(self):
        """Get a database connection with row factory"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    async def list_tables(self) -> dict:
        """Tool: List all tables in the EMR database"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return {
                "tables": tables,
                "count": len(tables)
            }
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            raise
    
    async def get_schema(self, table_name: str) -> dict:
        """Tool: Get schema information for a specific table"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "name": row[1],
                    "type": row[2],
                    "not_null": bool(row[3]),
                    "primary_key": bool(row[5])
                })
            
            # Get index information
            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = []
            for row in cursor.fetchall():
                index_name = row[1]
                cursor.execute(f"PRAGMA index_info({index_name})")
                index_columns = [col[2] for col in cursor.fetchall()]
                indexes.append({
                    "name": index_name,
                    "columns": index_columns
                })
            
            conn.close()
            
            return {
                "table_name": table_name,
                "columns": columns,
                "indexes": indexes
            }
        except Exception as e:
            logger.error(f"Error getting schema for {table_name}: {e}")
            raise
    
    async def run_sql(self, query: str) -> dict:
        """Tool: Execute a read-only SQL query"""
        try:
            # Safety check: only allow SELECT queries
            query_upper = query.strip().upper()
            if not query_upper.startswith('SELECT'):
                raise ValueError(
                    "Only SELECT queries are allowed. "
                    "INSERT, UPDATE, DELETE, and DROP operations are not permitted."
                )
            
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Fetch results
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
            
            conn.close()
            
            return {
                "query": query,
                "columns": columns,
                "rows": results,
                "row_count": len(results)
            }
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def _generate_embedding(self, text: str) -> Optional[list]:
        """Generate embedding using Gemini"""
        try:
            if self.embedding_model:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_query"
                )
                return result['embedding']
            return None
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def semantic_search(self, query: str, top_k: int = 5) -> dict:
        """Tool: Perform semantic search on medical literature vector store"""
        try:
            if not self.milvus_connected or not self.collection:
                raise Exception(
                    "Milvus is not connected. Cannot perform semantic search."
                )
            
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
                anns_field="vector",
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
            
            return {
                "query": query,
                "documents": documents,
                "total_found": len(documents)
            }
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            raise


async def main():
    """Main entry point for the MCP server"""
    # Create server instance
    server_instance = UnifiedMCPServer()
    
    # Create MCP server
    server = Server("unified-medical-mcp-server")
    
    # Register tool handlers
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List available tools"""
        return [
            types.Tool(
                name="list_tables",
                description="List all tables in the EMR database. Returns a list of table names.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            types.Tool(
                name="get_schema",
                description="Get schema information for a specific table including columns, types, and indexes.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to get schema for"
                        }
                    },
                    "required": ["table_name"]
                }
            ),
            types.Tool(
                name="run_sql",
                description=(
                    "Execute a read-only SQL query on the EMR database. "
                    "Only SELECT queries are allowed. "
                    "IMPORTANT: Always JOIN with dictionary tables (d_labitems, d_icd_diagnoses, d_icd_procedures) "
                    "to get human-readable names instead of codes."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL SELECT query to execute"
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="semantic_search",
                description=(
                    "Search medical literature using semantic/vector search. "
                    "Returns relevant documents with their text content, source, and relevance scores."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for medical literature"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            )
        ]
    
    @server.call_tool()
    async def handle_call_tool(
        name: str,
        arguments: dict
    ) -> list[types.TextContent]:
        """Handle tool execution"""
        try:
            if name == "list_tables":
                result = await server_instance.list_tables()
            elif name == "get_schema":
                result = await server_instance.get_schema(arguments["table_name"])
            elif name == "run_sql":
                result = await server_instance.run_sql(arguments["query"])
            elif name == "semantic_search":
                top_k = arguments.get("top_k", 5)
                result = await server_instance.semantic_search(arguments["query"], top_k)
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="unified-medical-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

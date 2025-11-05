"""Milvus vector store module"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
import uuid

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

from .config import Config


logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Class to interact with Milvus vector database"""
    
    def __init__(
        self,
        uri: str = None,
        api_key: str = None,
        collection_name: str = None,
        embedding_dim: int = 384  # Default for all-MiniLM-L6-v2
    ):
        """
        Initialize Milvus vector store
        
        Args:
            uri: Milvus server URI
            api_key: Milvus API key
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings
        """
        self.uri = uri or Config.MILVUS_URI
        self.api_key = api_key or Config.MILVUS_API_KEY
        self.collection_name = collection_name or Config.MILVUS_COLLECTION_NAME
        self.embedding_dim = embedding_dim
        
        # Connect to Milvus
        self._connect()
        
        # Create or load collection
        self.collection = self._get_or_create_collection()
    
    def _connect(self):
        """Connect to Milvus"""
        try:
            connections.connect(
                alias="default",
                uri=self.uri,
                token=self.api_key
            )
            logger.info(f"Connected to Milvus at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _create_schema(self) -> CollectionSchema:
        """Create collection schema"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="file_hash", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="total_chunks", dtype=DataType.INT64),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="PDF document embeddings"
        )
        
        return schema
    
    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one"""
        try:
            if utility.has_collection(self.collection_name):
                logger.info(f"Loading existing collection: {self.collection_name}")
                collection = Collection(self.collection_name)
            else:
                logger.info(f"Creating new collection: {self.collection_name}")
                schema = self._create_schema()
                collection = Collection(
                    name=self.collection_name,
                    schema=schema
                )
                
                # Create index for vector field
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
                logger.info("Created index on embedding field")
            
            # Load collection
            collection.load()
            logger.info(f"Collection loaded. Entity count: {collection.num_entities}")
            
            return collection
            
        except Exception as e:
            logger.error(f"Error getting/creating collection: {e}")
            raise
    
    def insert(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Insert embeddings with metadata into Milvus
        
        Args:
            embeddings: List of embedding vectors
            texts: List of text chunks
            metadatas: List of metadata dictionaries
            
        Returns:
            List of inserted IDs
        """
        if not (len(embeddings) == len(texts) == len(metadatas)):
            raise ValueError("embeddings, texts, and metadatas must have the same length")
        
        try:
            # Generate IDs
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
            
            # Extract metadata fields
            file_names = [m.get('file_name', '') for m in metadatas]
            file_hashes = [m.get('file_hash', '') for m in metadatas]
            chunk_indices = [m.get('chunk_index', 0) for m in metadatas]
            total_chunks_list = [m.get('total_chunks', 0) for m in metadatas]

            # Convert embeddings to a NumPy array for Milvus
            embeddings_np = np.array(embeddings, dtype=np.float32)
            
            # Prepare data
            data = [
                ids,
                embeddings_np,
                texts,
                file_names,
                file_hashes,
                chunk_indices,
                total_chunks_list,
            ]
            
            # Insert
            logger.info(f"Inserting {len(embeddings)} entities into Milvus")
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Successfully inserted {len(ids)} entities")
            return ids
            
        except Exception as e:
            logger.error(f"Error inserting data into Milvus: {e}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            output_fields: Fields to return in results
            
        Returns:
            List of search results with metadata
        """
        try:
            if output_fields is None:
                output_fields = ["text", "file_name", "chunk_index", "total_chunks"]
            
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "entity": hit.entity.to_dict()
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching Milvus: {e}")
            raise
    
    def delete_by_file(self, file_hash: str) -> int:
        """
        Delete all chunks from a specific file
        
        Args:
            file_hash: Hash of the file to delete
            
        Returns:
            Number of deleted entities
        """
        try:
            expr = f'file_hash == "{file_hash}"'
            result = self.collection.delete(expr)
            self.collection.flush()
            
            logger.info(f"Deleted entities for file_hash: {file_hash}")
            return result.delete_count
            
        except Exception as e:
            logger.error(f"Error deleting from Milvus: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            stats = {
                "collection_name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "schema": str(self.collection.schema),
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def close(self):
        """Close connection to Milvus"""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.warning(f"Error disconnecting from Milvus: {e}")

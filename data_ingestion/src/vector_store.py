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
            FieldSchema(name="primary_key", dtype=DataType.INT64, is_primary=True, auto_id=False, description="The Primary Key"),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_hash", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_index", dtype=DataType.INT16),
            FieldSchema(name="total_chunks", dtype=DataType.INT16),
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
                    field_name="vector",
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
            # Extract metadata fields
            file_names = [m.get('file_name', '') for m in metadatas]
            file_hashes = [m.get('file_hash', '') for m in metadatas]
            # Convert to numpy int16 for Milvus INT16 fields
            chunk_indices = np.array([m.get('chunk_index', 0) for m in metadatas], dtype=np.int16)
            total_chunks_list = np.array([m.get('total_chunks', 0) for m in metadatas], dtype=np.int16)

            # Convert embeddings to a NumPy array for Milvus
            embeddings_np = np.array(embeddings, dtype=np.float32)
            
            # DEBUG: Print schema information
            logger.info("=" * 80)
            logger.info("DEBUG: Schema Information")
            logger.info(f"Collection schema auto_id: {self.collection.schema.auto_id}")
            logger.info("Schema fields:")
            for field in self.collection.schema.fields:
                logger.info(f"  - {field.name}: {field.dtype}, auto_id={field.auto_id}, is_primary={field.is_primary}")
            
            # Check if primary key is auto_id
            is_auto_id = self.collection.schema.auto_id

            # DEBUG: Print data types before insertion
            logger.info("=" * 80)
            logger.info("DEBUG: Data Being Inserted")
            logger.info(f"Number of records: {len(embeddings)}")
            logger.info(f"embeddings_np type: {type(embeddings_np)}, dtype: {embeddings_np.dtype}, shape: {embeddings_np.shape}")
            logger.info(f"texts type: {type(texts)}, length: {len(texts)}")
            logger.info(f"  - Sample text types: {[type(t) for t in texts[:min(3, len(texts))]]}")
            logger.info(f"file_names type: {type(file_names)}, length: {len(file_names)}")
            logger.info(f"  - Sample file_names types: {[type(f) for f in file_names[:min(3, len(file_names))]]}")
            logger.info(f"file_hashes type: {type(file_hashes)}, length: {len(file_hashes)}")
            logger.info(f"  - Sample file_hashes types: {[type(h) for h in file_hashes[:min(3, len(file_hashes))]]}")
            logger.info(f"chunk_indices type: {type(chunk_indices)}, dtype: {chunk_indices.dtype}, length: {len(chunk_indices)}")
            logger.info(f"  - Sample chunk_indices: {chunk_indices[:min(3, len(chunk_indices))]}")
            logger.info(f"total_chunks_list type: {type(total_chunks_list)}, dtype: {total_chunks_list.dtype}, length: {len(total_chunks_list)}")
            logger.info(f"  - Sample total_chunks: {total_chunks_list[:min(3, len(total_chunks_list))]}")


            # Prepare data
            data = [
                embeddings_np,
                texts,
                file_names,
                file_hashes,
                chunk_indices,
                total_chunks_list,
            ]

            if not is_auto_id:
                # Generate primary keys if auto_id is false - must be numpy.int64
                # Mask to ensure values fit in signed int64 range
                primary_keys = np.array([uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF for _ in range(len(embeddings))], dtype=np.int64)
                logger.info(f"primary_keys type: {type(primary_keys)}, dtype: {primary_keys.dtype}, length: {len(primary_keys)}")
                logger.info(f"  - Sample primary_keys: {primary_keys[:min(3, len(primary_keys))]}")
                logger.info(f"  - Sample primary_keys types: {type(primary_keys[0]) if len(primary_keys) > 0 else 'N/A'}")
                data.insert(0, primary_keys)
            
            logger.info("=" * 80)
            logger.info(f"DEBUG: Final data list has {len(data)} fields")
            logger.info(f"Expected field order based on auto_id={is_auto_id}:")
            if not is_auto_id:
                logger.info("  0: primary_key (INT64)")
                logger.info("  1: vector (FLOAT_VECTOR)")
                logger.info("  2: text (VARCHAR)")
                logger.info("  3: file_name (VARCHAR)")
                logger.info("  4: file_hash (VARCHAR)")
                logger.info("  5: chunk_index (INT16)")
                logger.info("  6: total_chunks (INT16)")
            else:
                logger.info("  0: vector (FLOAT_VECTOR)")
                logger.info("  1: text (VARCHAR)")
                logger.info("  2: file_name (VARCHAR)")
                logger.info("  3: file_hash (VARCHAR)")
                logger.info("  4: chunk_index (INT16)")
                logger.info("  5: total_chunks (INT16)")
            logger.info("=" * 80)
            
            # Insert
            logger.info(f"Inserting {len(embeddings)} entities into Milvus")
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Successfully inserted {len(embeddings)} entities")
            return [] # Return empty list as IDs are auto-generated
            
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
                anns_field="vector",
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

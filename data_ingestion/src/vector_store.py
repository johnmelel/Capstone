"""Milvus vector store module"""

import json
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
from pymilvus.exceptions import MilvusException, ConnectionNotExistException

from .config import Config
from .exceptions import VectorStoreError
from .constants import MD5_HASH_PATTERN


logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Class to interact with Milvus vector database"""
    
    def __init__(
        self,
        uri: str = None,
        api_key: str = None,
        collection_name: str = None,
        embedding_dim: int = 384,  # Default for all-MiniLM-L6-v2
        metric_type: Optional[str] = None,
        recreate_collection: bool = False,
    ):
        """
        Initialize Milvus vector store
        
        Args:
            uri: Milvus server URI
            api_key: Milvus API key
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings
            metric_type: Distance metric type (COSINE, L2, IP)
            recreate_collection: If True, drop and recreate the collection
        """
        self.uri = uri or Config.MILVUS_URI
        self.api_key = api_key or Config.MILVUS_API_KEY
        self.collection_name = collection_name or Config.MILVUS_COLLECTION_NAME
        self.embedding_dim = embedding_dim
        self.metric_type = (metric_type or Config.MILVUS_METRIC_TYPE).upper()
        
        # Connect to Milvus
        self._connect()
        
        # Drop collection if requested
        if recreate_collection:
            self._drop_collection()
        
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
        except MilvusException as e:
            logger.error(f"Milvus connection error: {e}")
            raise VectorStoreError(f"Failed to connect to Milvus at {self.uri}") from e
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid Milvus configuration: {e}")
            raise VectorStoreError(f"Invalid Milvus configuration") from e
    
    def _drop_collection(self):
        """Drop the collection if it exists"""
        try:
            if utility.has_collection(self.collection_name):
                logger.warning(f"Dropping existing collection: {self.collection_name}")
                utility.drop_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' dropped successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' does not exist, nothing to drop")
        except MilvusException as e:
            logger.error(f"Milvus error dropping collection: {e}")
            raise VectorStoreError(f"Failed to drop collection {self.collection_name}") from e
    
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
            # Multimodal fields - Note: INT16 fields cannot have default values in Milvus
            FieldSchema(name="has_image", dtype=DataType.BOOL, default_value=False),
            FieldSchema(name="embedding_type", dtype=DataType.VARCHAR, max_length=32, default_value="text"),
            FieldSchema(name="image_count", dtype=DataType.INT16),  # No default value for INT16
            FieldSchema(name="image_gcs_paths", dtype=DataType.VARCHAR, max_length=10000, default_value="[]"),
            FieldSchema(name="image_metadata", dtype=DataType.VARCHAR, max_length=10000, default_value="{}"),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="PDF document embeddings with multimodal support"
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
                    "metric_type": self.metric_type,
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

            self._validate_collection_dimension(collection.schema)
            self._sync_index_metric(collection)
            
            return collection
            
        except MilvusException as e:
            logger.error(f"Milvus error getting/creating collection: {e}")
            raise VectorStoreError(f"Failed to get or create collection {self.collection_name}") from e
        except (ValueError, KeyError) as e:
            logger.error(f"Configuration error: {e}")
            raise VectorStoreError(f"Invalid collection configuration") from e

    def _extract_vector_dim(self, schema: CollectionSchema) -> Optional[int]:
        """Extract vector dimension from collection schema"""
        try:
            for field in schema.fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    params = getattr(field, "params", {}) or {}
                    dim = params.get("dim")
                    if dim:
                        return int(dim)
        except Exception as exc:
            logger.warning(f"Unable to read vector dimension from schema: {exc}")
        return None

    def _validate_collection_dimension(self, schema: CollectionSchema):
        """Ensure Milvus collection vector dimension matches configured embeddings"""
        dim = self._extract_vector_dim(schema)
        if dim is None:
            raise ValueError(
                f"Could not determine vector dimension for collection '{self.collection_name}'."
            )
        if dim != self.embedding_dim:
            raise ValueError(
                "Milvus collection '{name}' was created with vector dimension {col_dim}, "
                "but the configured embedding model outputs {embed_dim} dimensions. "
                "Update your Milvus collection/schema to use {embed_dim} dimensions or set EMBEDDING_DIMENSION "
                "to {col_dim} so both sides match.".format(
                    name=self.collection_name,
                    col_dim=dim,
                    embed_dim=self.embedding_dim,
                )
            )

    def _sync_index_metric(self, collection: Collection):
        """Make sure search metric matches Milvus index metric if defined"""
        metric = self._extract_index_metric(collection)
        if metric and metric != self.metric_type:
            logger.warning(
                "Milvus collection '%s' uses metric '%s' but config requested '%s'. "
                "Using collection metric for searches to avoid RPC errors.",
                self.collection_name,
                metric,
                self.metric_type,
            )
            self.metric_type = metric

    def _extract_index_metric(self, collection: Collection) -> Optional[str]:
        """Inspect collection indexes to determine vector field metric"""
        try:
            indexes = getattr(collection, "indexes", []) or []
            for index in indexes:
                if getattr(index, "field_name", None) != "vector":
                    continue
                params = getattr(index, "params", {}) or {}
                metric = self._parse_metric_type(params)
                if metric:
                    return metric
        except Exception as exc:
            logger.warning(f"Unable to inspect Milvus index metric: {exc}")
        return None

    @staticmethod
    def _parse_metric_type(params: Any) -> Optional[str]:
        """Extract metric type string from Milvus index params"""
        if isinstance(params, dict):
            metric = (
                params.get("metric_type")
                or params.get("metric")
                or params.get("metricType")
            )
            return metric.upper() if isinstance(metric, str) else None
        if isinstance(params, str):
            try:
                parsed = json.loads(params)
                return MilvusVectorStore._parse_metric_type(parsed)
            except json.JSONDecodeError:
                return None
        return None
    
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
            metadatas: List of metadata dictionaries (now includes multimodal fields)
            
        Returns:
            List of inserted IDs
        """
        if not (len(embeddings) == len(texts) == len(metadatas)):
            raise ValueError("embeddings, texts, and metadatas must have the same length")
        
        try:
            # Extract metadata fields (existing)
            file_names = [m.get('file_name', '') for m in metadatas]
            file_hashes = [m.get('file_hash', '') for m in metadatas]
            chunk_indices = np.array([m.get('chunk_index', 0) for m in metadatas], dtype=np.int16)
            total_chunks_list = np.array([m.get('total_chunks', 0) for m in metadatas], dtype=np.int16)
            
            # Extract multimodal metadata fields (new)
            has_image = [m.get('has_image', False) for m in metadatas]
            embedding_type = [m.get('embedding_type', 'text') for m in metadatas]
            image_count = np.array([m.get('image_count', 0) for m in metadatas], dtype=np.int16)
            image_gcs_paths = [m.get('image_gcs_paths', '[]') for m in metadatas]  # JSON string
            image_metadata = [m.get('image_metadata', '{}') for m in metadatas]  # JSON string

            # Convert embeddings to a NumPy array for Milvus
            embeddings_np = np.array(embeddings, dtype=np.float32)
            
            # Check if primary key is auto_id
            is_auto_id = self.collection.schema.auto_id
            
            logger.debug(f"Inserting {len(embeddings)} entities (auto_id={is_auto_id})")

            # Prepare data
            data = [
                embeddings_np,
                texts,
                file_names,
                file_hashes,
                chunk_indices,
                total_chunks_list,
                has_image,
                embedding_type,
                image_count,
                image_gcs_paths,
                image_metadata,
            ]

            if not is_auto_id:
                # Generate primary keys if auto_id is false - must be numpy.int64
                primary_keys = np.array([uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF for _ in range(len(embeddings))], dtype=np.int64)
                logger.debug(f"Generated {len(primary_keys)} primary keys")
                data.insert(0, primary_keys)
            
            # Insert
            logger.info(f"Inserting {len(embeddings)} entities into Milvus")
            insert_result = self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Successfully inserted {len(embeddings)} entities")
            
            # Return the primary keys from the insert result
            if hasattr(insert_result, 'primary_keys'):
                return [str(pk) for pk in insert_result.primary_keys]
            else:
                return []
            
        except MilvusException as e:
            logger.error(f"Milvus insertion error: {e}")
            raise VectorStoreError(f"Failed to insert {len(embeddings)} entities into Milvus") from e
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error during insertion: {e}")
            raise VectorStoreError(f"Invalid data format for Milvus insertion") from e
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            output_fields: Fields to return in results (includes multimodal fields)
            search_params: Optional dictionary of search parameters
            
        Returns:
            List of search results with metadata
        """
        try:
            if output_fields is None:
                output_fields = [
                    "text", "file_name", "chunk_index", "total_chunks",
                    "has_image", "embedding_type", "image_count",
                    "image_gcs_paths", "image_metadata"
                ]
            
            if search_params is None:
                search_params = {
                    "metric_type": self.metric_type,
                    "params": {"nprobe": 10}
                }
            else:
                if "metric_type" not in search_params:
                    search_params = dict(search_params)
                    search_params["metric_type"] = self.metric_type
            
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
                    # Extract entity fields manually to ensure they're included
                    entity_dict = {}
                    for field in output_fields:
                        try:
                            entity_dict[field] = hit.entity.get(field)
                        except (AttributeError, KeyError):
                            # Fallback: try accessing as attribute
                            entity_dict[field] = getattr(hit.entity, field, None)
                    
                    result = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "entity": entity_dict
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except MilvusException as e:
            logger.error(f"Milvus search error: {e}")
            raise VectorStoreError(f"Failed to search Milvus collection") from e
        except (AttributeError, KeyError) as e:
            logger.error(f"Error accessing search result fields: {e}")
            raise VectorStoreError(f"Invalid search result format") from e
    
    def delete_by_file_hash(self, file_hash: str) -> int:
        """
        Delete all chunks from a specific file
        
        Args:
            file_hash: Hash of the file to delete (must be valid MD5 hex string)
            
        Returns:
            Number of deleted entities
            
        Raises:
            ValueError: If file_hash is not a valid MD5 hash format
        """
        import re
        from pymilvus.exceptions import MilvusException
        
        # Validate file_hash is alphanumeric only (MD5 hash format)
        if not re.match(MD5_HASH_PATTERN, file_hash):
            raise ValueError(f"Invalid file hash format: {file_hash}. Must be a valid MD5 hash.")
        
        try:
            expr = f'file_hash == "{file_hash}"'
            result = self.collection.delete(expr)
            self.collection.flush()
            
            logger.info(f"Deleted entities for file_hash: {file_hash}")
            return result.delete_count
            
        except MilvusException as e:
            logger.error(f"Milvus error deleting file_hash {file_hash}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting from Milvus: {e}")
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
        except MilvusException as e:
            logger.error(f"Milvus error getting stats: {e}")
            return {"error": str(e)}
        except AttributeError as e:
            logger.error(f"Collection not initialized: {e}")
            return {"error": "Collection not available"}
    
    def close(self):
        """Close connection to Milvus"""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except (MilvusException, ConnectionNotExistException) as e:
            logger.warning(f"Error disconnecting from Milvus (may already be disconnected): {e}")

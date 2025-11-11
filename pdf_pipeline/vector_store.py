#!/usr/bin/env python
"""Milvus vector store module"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd
import hashlib

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

from config import Config


logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Class to interact with Milvus vector database"""

    def __init__(
        self,
        uri: str = None,
        api_key: str = None,
        collection_name: str = None,
        embedding_dim: int = None  # Will use Config.EMBEDDING_DIM if not provided
    ):
        """
        Initialize Milvus vector store

        Args:
            uri: Milvus server URI
            api_key: Milvus API key
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings (defaults to Config.EMBEDDING_DIM)
        """
        self.uri = uri or Config.MILVUS_URI
        self.api_key = api_key or Config.MILVUS_API_KEY
        self.collection_name = collection_name or Config.MILVUS_COLLECTION_NAME
        self.embedding_dim = embedding_dim or Config.EMBEDDING_DIM

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

    def insert_from_parquet(self, parquet_file: str):
        """
        Insert data from a parquet file into Milvus.
        """
        df = pd.read_parquet(parquet_file)

        embeddings = df['vector'].tolist()
        texts = df['text'].tolist()
        metadatas = df.drop(columns=['vector', 'text']).to_dict('records')

        self.insert(embeddings, texts, metadatas)

    def insert(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Insert embeddings with metadata into Milvus

        Args:
            embeddings: List of embedding vectors
            texts: List of text chunks
            metadatas: List of metadata dictionaries

        Returns:
            List of inserted primary keys
        """
        if not (len(embeddings) == len(texts) == len(metadatas)):
            raise ValueError("embeddings, texts, and metadatas must have the same length")
        
        # Validate embedding dimensions
        for idx, emb in enumerate(embeddings):
            if len(emb) != self.embedding_dim:
                logger.error(
                    f"Embedding dimension mismatch at index {idx}! "
                    f"Expected {self.embedding_dim}, got {len(emb)}. "
                    f"Text: '{texts[idx][:100]}...'"
                )
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                    f"got {len(emb)} at index {idx}"
                )

        try:
            # Extract metadata fields
            file_names = [m.get('file_name', '') for m in metadatas]
            file_hashes = [m.get('file_hash', '') for m in metadatas]
            # Convert to numpy int16 for Milvus INT16 fields
            chunk_indices = np.array([m.get('chunk_index', 0) for m in metadatas], dtype=np.int16)
            total_chunks_list = np.array([m.get('total_chunks', 0) for m in metadatas], dtype=np.int16)

            # Convert embeddings to a NumPy array for Milvus
            embeddings_np = np.array(embeddings, dtype=np.float32)

            # Generate deterministic primary keys
            primary_keys = []
            for i in range(len(texts)):
                key_string = f"{file_hashes[i]}-{chunk_indices[i]}-{texts[i][:100]}"
                hashed_key = hashlib.sha256(key_string.encode()).hexdigest()
                # Convert first 15 hex chars to int64 (15 hex chars = 60 bits, safely fits in 63-bit signed int64)
                primary_keys.append(int(hashed_key[:15], 16))

            primary_keys_np = np.array(primary_keys, dtype=np.int64)

            # Prepare data in the EXACT order of schema fields:
            # Schema order: primary_key, vector, text, file_name, file_hash, chunk_index, total_chunks
            data = [
                primary_keys_np,      # primary_key (INT64)
                embeddings_np,        # vector (FLOAT_VECTOR)
                texts,                # text (VARCHAR)
                file_names,           # file_name (VARCHAR)
                file_hashes,          # file_hash (VARCHAR)
                chunk_indices,        # chunk_index (INT16)
                total_chunks_list,    # total_chunks (INT16)
            ]

            # Insert
            logger.info(f"Inserting {len(embeddings)} entities into Milvus")
            self.collection.insert(data)
            self.collection.flush()

            logger.info(f"Successfully inserted {len(embeddings)} entities")
            return primary_keys

        except Exception as e:
            logger.error(f"Error inserting data into Milvus: {e}", exc_info=True)
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

if __name__ == '__main__':
    logging.basicConfig(level=Config.LOG_LEVEL)
    # Example usage:
    # This will be called by the main orchestrator in the final pipeline.
    # vector_store = MilvusVectorStore()
    # vector_store.insert_from_parquet("path/to/your/processed_data.parquet")
    # vector_store.close()

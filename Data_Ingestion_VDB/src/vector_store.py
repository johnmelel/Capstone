"""Milvus vector store interface."""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import List, Dict, Any
import numpy as np
import json
import os

class VectorStore:
    """Interface to Milvus vector database."""
    
    def __init__(self, config: Dict[str, Any], drop_if_exists: bool = False):
        print("\n[VectorStore] Initializing Milvus...")
        
        # Check if using cloud or embedded mode
        milvus_host = os.getenv("MILVUS_HOST")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        
        if milvus_host:
            # CLOUD MODE - Connect to remote Milvus server
            print(f"[VectorStore] Connecting to Cloud Milvus: {milvus_host}:{milvus_port}")
            try:
                connections.connect(
                    alias="default",
                    host=milvus_host,
                    port=milvus_port,
                    timeout=10
                )
                print(f"[VectorStore] Connected to Cloud Milvus successfully")
            except Exception as e:
                print(f"[Error] Failed to connect to Cloud Milvus: {e}")
                print(f"[Error] Make sure server is running and firewall allows port 19530")
                raise
        else:
            # EMBEDDED MODE - Use local file database
            print(f"[VectorStore] Using embedded Milvus (local database)")
            connections.connect(
                alias="default",
                uri="./milvus_data.db"
            )

        self.collection_name = config['vector_store']['collection_name']
        self.dimensions = config['embedding']['dimensions']
        
        # Create or get collection
        self._setup_collection(drop_if_exists=drop_if_exists)
    
    def _setup_collection(self, drop_if_exists: bool = False):
        """Create collection if it doesn't exist."""
        
        if drop_if_exists and utility.has_collection(self.collection_name):
            print(f"     Dropping existing collection: {self.collection_name}")
            Collection(self.collection_name).drop()

        if utility.has_collection(self.collection_name):
            print(f"   ✓ Collection '{self.collection_name}' already exists")
            self.collection = Collection(self.collection_name)
            self.collection.load()
        else:
            print(f"    Creating collection '{self.collection_name}'...")
            
            # Define schema - Milvus accepts JSON!
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimensions),
                FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=5000),  # Store as JSON string
                FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=5000)
            ]
            
            schema = CollectionSchema(fields=fields, description="Medical textbook embeddings")
            
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            self.collection.create_index(field_name="embedding", index_params=index_params)
            self.collection.load()
            
            print(f"   ✓ Collection created with COSINE index")
    
    def get_existing_chunk_ids(self) -> set:
        """Get all existing chunk_ids from the collection."""
        try:
            if not hasattr(self, 'collection'):
                return set()
            
            # Query Milvus database for ALL chunk_ids
            results = self.collection.query(
                expr="id >= 0",             # get all records
                output_fields=["chunk_id"], # only fetch chunk_id field
                limit=16384                 # max allowed by milvus
            )
            
            # run as set for fast lookup
            return set(r['chunk_id'] for r in results)
        except Exception as e:
            print(f"[Warning] Could not fetch existing chunks: {e}")
            return set()

    def count_entities(self) -> int:
        """Count total entities in collection."""
        try:
            if not hasattr(self, 'collection'):
                return 0
            return self.collection.num_entities
        except:
            return 0

    def add_items(self, items: List[Dict], embeddings: List[np.ndarray]):
        """Add items to vector store."""
        print(f"Uploading {len(items)} vectors to Milvus...")
        
        # Prepare data
        chunk_ids = [item["chunk_id"] for item in items]
        embedding_list = [emb.tolist() for emb in embeddings]
        content_types = [item["type"] for item in items]
        
        # Convert metadata to JSON strings (Milvus accepts this!)
        metadata_jsons = [json.dumps(item["metadata"]) for item in items]
        
        documents = [item.get("content", item.get("caption", ""))[:4900] for item in items]
        
        # Insert
        data = [
            chunk_ids,
            embedding_list,
            content_types,
            metadata_jsons,
            documents
        ]
        
        self.collection.insert(data)
        self.collection.flush()
        
        print(f"   ✓ Uploaded successfully")
        print(f"   Total vectors in collection: {self.collection.num_entities}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10):
        """Search for similar vectors."""
        print(f"\n Searching for top {top_k} results...")
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_id", "content_type", "metadata_json", "document"]
        )
        
        print(f"   ✓ Found {len(results[0])} results")
        return results[0]
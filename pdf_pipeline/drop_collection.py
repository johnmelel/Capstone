#!/usr/bin/env python
"""
Utility script to drop an existing Milvus collection.
Use this when you need to recreate a collection with different schema (e.g., new embedding dimension).
"""

import logging
from pymilvus import connections, utility
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def drop_collection(collection_name: str = None):
    """Drop a Milvus collection if it exists"""
    collection_name = collection_name or Config.MILVUS_COLLECTION_NAME
    
    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            uri=Config.MILVUS_URI,
            token=Config.MILVUS_API_KEY
        )
        logger.info(f"Connected to Milvus at {Config.MILVUS_URI}")
        
        # Check if collection exists
        if utility.has_collection(collection_name):
            logger.warning(f"Dropping collection: {collection_name}")
            utility.drop_collection(collection_name)
            logger.info(f"✓ Collection '{collection_name}' has been dropped successfully")
        else:
            logger.info(f"Collection '{collection_name}' does not exist. Nothing to drop.")
        
        # Disconnect
        connections.disconnect("default")
        
    except Exception as e:
        logger.error(f"Failed to drop collection: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    print("\n⚠️  WARNING: This will permanently delete the Milvus collection!")
    print(f"   Collection name: {Config.MILVUS_COLLECTION_NAME}")
    print(f"   Milvus URI: {Config.MILVUS_URI}\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        drop_collection()
    else:
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() in ["yes", "y"]:
            drop_collection()
        else:
            print("Operation cancelled.")

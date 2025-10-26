"""Quick test to check Milvus connection."""

import os
from dotenv import load_dotenv
from pymilvus import connections

load_dotenv()

milvus_host = os.getenv("MILVUS_HOST")
milvus_port = os.getenv("MILVUS_PORT", "19530")

print(f"Connecting to Milvus at {milvus_host}:{milvus_port}...")

try:
    connections.connect(
        alias="default",
        host=milvus_host,
        port=milvus_port,
        timeout=10
    )
    
    # Get server version
    from pymilvus import utility
    version = utility.get_server_version()
    print(f"Connected! Server version: {version}")
    
    connections.disconnect("default")
    
except Exception as e:
    print(f"Connection failed: {e}")

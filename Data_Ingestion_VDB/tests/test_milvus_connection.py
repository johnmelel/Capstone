"""Test connection to Cloud Milvus server."""

import os
from dotenv import load_dotenv
from pymilvus import connections, utility

load_dotenv()

milvus_host = os.getenv("MILVUS_HOST")
milvus_port = os.getenv("MILVUS_PORT", "19530")

print("\n" + "="*70)
print("TESTING MILVUS CONNECTION")
print("="*70)

if not milvus_host:
    print("\n[Error] MILVUS_HOST not found in .env file")
    print("[Error] Add this line to .env:")
    print("        MILVUS_HOST=34.10.71.21")
    exit(1)

print(f"\n[Info] Attempting to connect to Milvus...")
print(f"[Info] Host: {milvus_host}")
print(f"[Info] Port: {milvus_port}")

try:
    connections.connect(
        alias="test",
        host=milvus_host,
        port=milvus_port,
        timeout=10
    )
    print("\n[Success] Connection successful!")
    
    # List collections
    collections = utility.list_collections()
    print(f"[Info] Existing collections: {collections if collections else 'None'}")
    
    connections.disconnect("test")
    print("[Success] Disconnected successfully")
    
    print("\n" + "="*70)
    print("CONNECTION TEST PASSED")
    print("="*70)
    print("\nYou can now run: python run_pipeline.py --test")
    
except Exception as e:
    print(f"\n[Error] Connection failed: {e}")
    print("\n" + "="*70)
    print("TROUBLESHOOTING STEPS")
    print("="*70)
    print("\n1. Check if server is running:")
    print("   gcloud compute instances describe milvus-server --zone=us-central1-a")
    print("\n2. Check if Milvus service is running:")
    print("   gcloud compute ssh milvus-server --zone=us-central1-a")
    print("   sudo docker ps")
    print("\n3. Check firewall rules:")
    print("   gcloud compute firewall-rules list --filter=\"name~milvus\"")
    print("\n4. Verify .env file has:")
    print("   MILVUS_HOST=34.10.71.21")
    print("   MILVUS_PORT=19530")
    print("\n" + "="*70)
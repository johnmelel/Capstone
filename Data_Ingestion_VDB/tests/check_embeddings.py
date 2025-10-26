from pymilvus import connections, Collection
import os
from dotenv import load_dotenv

load_dotenv()

# Connect
milvus_host = os.getenv("MILVUS_HOST")
connections.connect("default", host=milvus_host, port="19530")

# Get collection
collection = Collection("medical_textbooks")
collection.load()

# Check count
print(f"\nTotal embeddings in database: {collection.num_entities}")

# Get a sample
results = collection.query(expr="id >= 0", limit=3, output_fields=["chunk_id", "content_type", "document"])

print(f"\nSample chunks:")
for i, r in enumerate(results, 1):
    print(f"\n[{i}] {r['chunk_id']}")
    print(f"    Type: {r['content_type']}")
    print(f"    Text: {r['document'][:100]}...")

connections.disconnect("default")

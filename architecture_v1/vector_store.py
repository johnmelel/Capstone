from pymilvus import connections, Collection
from typing import List, Dict
import config

class VectorStoreClient:
    def __init__(self, collection_name: str):
        connections.connect("default", uri=config.VECTOR_DB_URI)
        self.collection = Collection(name=collection_name)

    def upsert(self, embeddings: List[List[float]], ids: List[str], metadata: List[Dict]):
        self.collection.insert([ids, embeddings, metadata])

    def query(self, vector: List[float], top_k: int = 5) -> List[Dict]:
        results = self.collection.search(
            data=[vector], anns_field="embedding", param={"metric_type": "L2"}, limit=top_k
        )
        hits = [{"id": r.id, "score": r.distance, "metadata": r.entity.get_raw("metadata")} for r in results[0]]
        return hits
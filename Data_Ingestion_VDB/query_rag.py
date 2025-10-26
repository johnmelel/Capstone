"""Interactive RAG query tool with OpenAI."""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import connections, Collection
from vertexai.vision_models import MultiModalEmbeddingModel
from google.cloud import aiplatform
import numpy as np
import json

load_dotenv()

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Vertex AI for embeddings
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
aiplatform.init(project=project_id, location=location)
embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

# Connect to Milvus
connections.connect(alias="default", uri="./milvus_data.db")
collection = Collection("medical_textbooks")
collection.load()

def search_documents(query: str, top_k: int = 5):
    """Search for relevant documents."""
    print(f"\n🔍 Searching for: '{query}'")
    
    # Generate embedding for query
    emb = embedding_model.get_embeddings(contextual_text=query)
    query_vec = np.array(emb.text_embedding)
    query_vec = query_vec / np.linalg.norm(query_vec)  # Normalize
    
    # Search Milvus
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_vec.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_id", "content_type", "metadata_json", "document"]
    )
    
    # Format results
    contexts = []
    print(f"\n📚 Found {len(results[0])} relevant chunks:\n")
    
    for i, hit in enumerate(results[0]):
        score = hit.score
        doc = hit.entity.get("document")
        metadata = json.loads(hit.entity.get("metadata_json"))
        
        print(f"[{i+1}] Score: {score:.4f}")
        print(f"    Page: {metadata.get('page')}")
        print(f"    Text: {doc[:150]}...")
        print()
        
        contexts.append({
            'text': doc,
            'score': score,
            'page': metadata.get('page'),
            'pdf': metadata.get('pdf_name')
        })
    
    return contexts

def answer_question(query: str, contexts: list):
    """Use OpenAI to answer based on retrieved contexts."""
    
    # Build context string
    context_str = "\n\n".join([
        f"[Source {i+1}, Page {c['page']}, Score: {c['score']:.3f}]\n{c['text']}"
        for i, c in enumerate(contexts[:3])  # Use top 3
    ])
    
    # Create prompt
    prompt = f"""You are a medical imaging expert assistant. Answer the question based ONLY on the provided context.

Context from medical textbooks:
{context_str}

Question: {query}

Instructions:
- Answer concisely and accurately
- Cite which source number you're using (e.g., "According to Source 1...")
- If the context doesn't contain enough information, say so
- Do not make up information

Answer:"""
    
    print("Generating answer with OpenAI...\n")
    
    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-3.5-turbo" for cheaper
        messages=[
            {"role": "system", "content": "You are a helpful medical imaging expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    return answer

def interactive_query():
    """Interactive query loop."""
    print("\n" + "="*70)
    print("🏥 MEDICAL IMAGING RAG SYSTEM")
    print("="*70)
    print("\nAsk questions about prostate MRI, CT imaging, or medical imaging quality.")
    print("Type 'quit' to exit.\n")
    
    while True:
        # Get user query
        query = input("Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        try:
            # Search documents
            contexts = search_documents(query, top_k=5)
            
            # Generate answer
            answer = answer_question(query, contexts)
            
            # Display answer
            print("="*70)
            print("💡 ANSWER:")
            print("="*70)
            print(answer)
            print("\n" + "="*70 + "\n")
            
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    interactive_query()
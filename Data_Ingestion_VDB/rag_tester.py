# rag_tester.py
"""Test RAG with both chunk types."""

from openai import OpenAI
from vertexai.vision_models import MultiModalEmbeddingModel
from google.cloud import aiplatform
import numpy as np
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
aiplatform.init(project=project_id, location=location)
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")


def embed_text(text: str) -> np.ndarray:
    """Embed using Vertex AI."""
    if len(text) > 1000:
        text = text[:1000]
    emb = model.get_embeddings(contextual_text=text)
    vec = np.array(emb.text_embedding)
    return vec / np.linalg.norm(vec)


def search_chunks(query: str, chunks: list, top_k: int = 3):
    """Simple vector search."""
    query_emb = embed_text(query)
    
    # Embed all chunks
    chunk_embs = [embed_text(c['text']) for c in chunks]
    
    # Calculate similarities
    scores = [(i, np.dot(query_emb, emb)) for i, emb in enumerate(chunk_embs)]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    return [{'chunk': chunks[i], 'score': score} for i, score in scores[:top_k]]


def generate_answer(query: str, contexts: list, label: str):
    """Generate answer using OpenAI."""
    context_str = "\n\n".join([
        f"[Context {i+1}, Score: {c['score']:.3f}]\n{c['chunk']['text']}"
        for i, c in enumerate(contexts)
    ])
    
    prompt = f"""Answer the question based ONLY on the provided context.

Context:
{context_str}

Question: {query}

Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical imaging expert. Answer concisely based only on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300
    )
    
    return response.choices[0].message.content


# Load chunks
with open('raw_chunks.json') as f:
    raw_chunks = json.load(f)

with open('md_chunks.json') as f:
    md_chunks = json.load(f)

# Load questions
with open('test_questions.json') as f:
    test_data = json.load(f)

# Run test
results = []

for q in test_data['questions']:
    question = q['question']
    print(f"\n{'='*70}")
    print(f"Question: {question}")
    print('='*70)
    
    # RAW version
    print("\n🔵 RAW TEXT VERSION:")
    raw_contexts = search_chunks(question, raw_chunks, top_k=3)
    print(f"Top chunk score: {raw_contexts[0]['score']:.4f}")
    raw_answer = generate_answer(question, raw_contexts, "raw")
    print(f"Answer: {raw_answer}")
    
    # MARKDOWN version
    print("\n🟢 MARKDOWN VERSION:")
    md_contexts = search_chunks(question, md_chunks, top_k=3)
    print(f"Top chunk score: {md_contexts[0]['score']:.4f}")
    md_answer = generate_answer(question, md_contexts, "markdown")
    print(f"Answer: {md_answer}")
    
    # Save results
    results.append({
        'question': question,
        'raw': {
            'top_score': float(raw_contexts[0]['score']),
            'answer': raw_answer,
            'contexts': [c['chunk']['id'] for c in raw_contexts]
        },
        'markdown': {
            'top_score': float(md_contexts[0]['score']),
            'answer': md_answer,
            'contexts': [c['chunk']['id'] for c in md_contexts]
        }
    })

# Save results
with open('rag_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to rag_comparison.json")
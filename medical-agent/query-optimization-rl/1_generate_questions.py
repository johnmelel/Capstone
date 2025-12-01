"""
Generate 100 questions from Milvus chunks using Gemini
"""
import os
import json
import random
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import MilvusClient
import vertexai
from vertexai.generative_models import GenerativeModel
import config

# Load environment
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

def connect_to_milvus():
    """Connect to Milvus Cloud"""
    print("Connecting to Milvus...")
    client = MilvusClient(
        uri=config.MILVUS_URI,
        token=config.MILVUS_TOKEN
    )
    print(f"Connected to collection: {config.COLLECTION_NAME}")
    return client

def initialize_gemini():
    """Initialize Gemini"""
    print("Initializing Gemini...")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(config.SERVICE_ACCOUNT)
    vertexai.init(project=config.GCP_PROJECT, location=config.GCP_LOCATION)
    model = GenerativeModel("gemini-2.0-flash-exp")
    print("Gemini initialized")
    return model

def get_random_chunks(client, num_chunks=50):
    """Get random chunks from Milvus"""
    print(f"Sampling {num_chunks} random chunks...")
    
    # Query with random offset (simple sampling)
    results = client.query(
        collection_name=config.COLLECTION_NAME,
        filter="",  # No filter - get all
        output_fields=["text", "file_name"],
        limit=num_chunks
    )
    
    print(f"Retrieved {len(results)} chunks")
    return results

def generate_questions_for_chunk(model, chunk_text, file_name):
    """Generate 2-3 questions for a chunk using Gemini"""
    prompt = f"""Based on this medical text chunk, generate 2-3 natural user questions that could be answered by this text.

Make questions realistic - like what a medical student or clinician might ask.
Vary the style: some specific, some vague, some using layman's terms.

Text:
{chunk_text[:500]}...

Return ONLY the questions, one per line, no numbering or extra text."""

    try:
        response = model.generate_content(prompt)
        questions = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
        return questions
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

def main():
    """Generate 100 questions from chunks"""
    
    # Connect to services
    client = connect_to_milvus()
    model = initialize_gemini()
    
    # Get random chunks
    chunks = get_random_chunks(client, config.CHUNKS_TO_SAMPLE)
    
    # Generate questions
    all_questions = []
    questions_per_chunk = config.NUM_QUESTIONS // len(chunks) + 1
    
    print(f"\nGenerating {config.NUM_QUESTIONS} questions...")
    
    for i, chunk in enumerate(chunks):
        if len(all_questions) >= config.NUM_QUESTIONS:
            break
            
        print(f"Processing chunk {i+1}/{len(chunks)}...", end="\r")
        
        questions = generate_questions_for_chunk(
            model, 
            chunk.get("text", ""),
            chunk.get("file_name", "")
        )
        
        for question in questions:
            if len(all_questions) >= config.NUM_QUESTIONS:
                break
                
            all_questions.append({
                "question": question,
                "source_text": chunk.get("text", "")[:500],  # First 500 chars
                "source_file": chunk.get("file_name", ""),
                "chunk_id": i
            })
    
    print(f"\nGenerated {len(all_questions)} questions")
    
    # Save to file
    with open(config.QUESTIONS_FILE, 'w') as f:
        json.dump(all_questions, f, indent=2)
    
    print(f"Saved to: {config.QUESTIONS_FILE}")
    
    # Show samples
    print("\nSample questions:")
    for q in random.sample(all_questions, min(5, len(all_questions))):
        print(f"  - {q['question']}")

if __name__ == "__main__":
    main()

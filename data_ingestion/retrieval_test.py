"""
Standalone script to test retrieval from the Milvus vector database.

This script prompts the user for a query, generates an embedding for it,
and retrieves the top K most similar documents from the vector store.
"""

import logging
from src.config import Config
from src.embedder import TextEmbedder
from src.vector_store import MilvusVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the retrieval test.
    """
    try:
        # --- 1. Configuration and Initialization ---
        logger.info("Loading configuration and initializing components...")
        
        # Validate configuration
        Config.validate()
        
        # Initialize the text embedder
        embedder = TextEmbedder(
            model_name=Config.EMBEDDING_MODEL,
            embedding_dimension=Config.EMBEDDING_DIMENSION
        )
        
        # Initialize the vector store
        vector_store = MilvusVectorStore(
            uri=Config.MILVUS_URI,
            api_key=Config.MILVUS_API_KEY,
            collection_name=Config.MILVUS_COLLECTION_NAME,
            embedding_dim=Config.EMBEDDING_DIMENSION
        )
        
        logger.info("Initialization complete.")
        
        # --- 2. Get User Query ---
        prompt = input(f"Enter your query (or press Enter to use the default: '{Config.DEFAULT_RETRIEVAL_PROMPT}'): ")
        
        if not prompt:
            prompt = Config.DEFAULT_RETRIEVAL_PROMPT
            logger.info(f"Using default query: '{prompt}'")
        else:
            logger.info(f"Using user query: '{prompt}'")
            
        # --- 3. Generate Query Embedding ---
        logger.info("Generating embedding for the query...")
        query_embedding = embedder.embed_text(prompt)
        
        if query_embedding is None or query_embedding.size == 0:
            logger.error("Failed to generate query embedding. Exiting.")
            return
            
        # The embed_text method returns a numpy array of embeddings, 
        # for a single text it's the first element.
        query_vector = query_embedding[0].tolist()
        
        logger.info(f"Successfully generated query embedding of dimension {len(query_vector)}.")
        
        # --- 4. Perform Similarity Search ---
        top_k = Config.TOP_K_RESULTS
        logger.info(f"Performing similarity search to retrieve top {top_k} results...")
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        search_results = vector_store.search(
            query_embedding=query_vector,
            top_k=top_k,
            search_params=search_params,
            output_fields=["text", "file_name", "chunk_index"]
        )
        
        # --- 5. Display Results ---
        if not search_results:
            logger.warning("No results found for the given query.")
        else:
            logger.info(f"Found {len(search_results)} results:")
            print("\n" + "="*80)
            print(f"Top {len(search_results)} results for query: '{prompt}'")
            print("="*80 + "\n")
            
            for i, result in enumerate(search_results):
                entity = result.get('entity', {})
                text = entity.get('text', 'N/A')
                file_name = entity.get('file_name', 'N/A')
                chunk_index = entity.get('chunk_index', 'N/A')
                distance = result.get('distance', 'N/A')
                
                print(f"--- Result {i+1} ---")
                print(f"  Distance: {distance:.4f}")
                print(f"  File Name: {file_name}")
                print(f"  Chunk Index: {chunk_index}")
                print(f"  Text: \n\"{text}\"\n")
            
            print("="*80)

    except Exception as e:
        logger.error(f"An error occurred during the retrieval test: {e}", exc_info=True)
    finally:
        if 'vector_store' in locals() and vector_store:
            vector_store.close()
        logger.info("Retrieval test finished.")

if __name__ == "__main__":
    main()

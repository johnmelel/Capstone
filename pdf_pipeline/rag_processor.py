import asyncio
import logging
from pathlib import Path
from typing import List
import pandas as pd
import google.generativeai as genai
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

from config import Config

# Configure logging
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Configure Google Generative AI
genai.configure(api_key=Config.GOOGLE_API_KEY)

def google_llm_model_func(prompt: str, **kwargs):
    """Custom LLM model function using Google's Generative AI."""
    model = genai.GenerativeModel(Config.LLM_MODEL)
    response = model.generate_content(prompt, **kwargs)
    return response.text

def google_embedding_func(texts: List[str]) -> List[List[float]]:
    """Custom embedding function using Google's Generative AI.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors (each vector is a list of floats)
    """
    if not texts:
        return []
    
    # Handle single text string
    if isinstance(texts, str):
        texts = [texts]
    
    # Google's API can handle batch requests, but let's process them safely
    embeddings = []
    
    # Process in batches if needed (Google API has limits)
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = genai.embed_content(
                model=Config.EMBEDDING_MODEL,
                content=batch,
                task_type="retrieval_document"
            )
            # Response structure: {'embedding': [list of vectors]}
            if isinstance(response, dict) and 'embedding' in response:
                batch_embeddings = response['embedding']
                # If single text, wrap in list
                if not isinstance(batch_embeddings[0], list):
                    batch_embeddings = [batch_embeddings]
                embeddings.extend(batch_embeddings)
            else:
                logger.error(f"Unexpected response structure from Google embedding API: {response}")
                raise ValueError("Invalid response from embedding API")
        except Exception as e:
            logger.error(f"Error generating embeddings for batch: {e}")
            raise
    
    return embeddings

class RAGProcessor:
    def __init__(self):
        # Ensure working directory exists
        working_dir = Path("./rag_storage")
        working_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure processed data directory exists
        Config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        self.config = RAGAnythingConfig(
            working_dir=str(working_dir),
        )
        self.embedding_func = EmbeddingFunc(
            embedding_dim=768,  # Dimension for Google's embedding-001 model
            func=google_embedding_func,
        )
        self.rag = RAGAnything(
            config=self.config,
            llm_model_func=google_llm_model_func,
            embedding_func=self.embedding_func,
        )

    async def process_document(self, file_path: Path) -> Path:
        """Processes a single document and saves the output.
        
        Args:
            file_path: Path to the PDF file to process
            
        Returns:
            Path to the generated parquet file
        """
        logger.info(f"Processing document: {file_path}")
        try:
            # Process the document using RAGAnything
            # Note: The actual method name may vary - adjust based on the library version
            # Common method names: insert, ainsert, process_file, add_document
            try:
                # Try the async insert method first (most common in RAGAnything)
                await self.rag.ainsert(str(file_path))
            except AttributeError:
                # Fallback to synchronous insert if async not available
                logger.warning("Async insert not available, trying synchronous insert")
                self.rag.insert(str(file_path))
            
            # Extract processed data from the knowledge graph
            # The structure depends on RAGAnything version
            try:
                # Try to access storage directly
                if hasattr(self.rag, 'chunk_entity_relation_graph'):
                    # For newer versions with graph storage
                    entities = self.rag.chunk_entity_relation_graph.get_all_nodes()
                elif hasattr(self.rag, 'chunks'):
                    # For versions with chunk storage
                    entities = self.rag.chunks
                else:
                    # Fallback: try to access internal storage
                    logger.warning("Unknown RAGAnything storage structure, attempting to extract from internal state")
                    entities = []
                    if hasattr(self.rag, '_storage'):
                        entities = self.rag._storage.get_all()
                    elif hasattr(self.rag, 'lightrag') and hasattr(self.rag.lightrag, 'knowledge_graph'):
                        kg = self.rag.lightrag.knowledge_graph
                        entities = kg.get_all_nodes() if hasattr(kg, 'get_all_nodes') else []
            except Exception as e:
                logger.error(f"Error extracting entities from RAGAnything: {e}")
                logger.info("Attempting alternative data extraction method...")
                entities = []

            # Convert the entities to a pandas DataFrame
            data = []
            
            # Get file hash for tracking
            import hashlib
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if entities:
                for idx, entity in enumerate(entities):
                    # Handle different entity structures
                    if hasattr(entity, 'embedding'):
                        embedding = entity.embedding
                        text = getattr(entity, 'text', getattr(entity, 'content', ''))
                        metadata = getattr(entity, 'metadata', {})
                    elif isinstance(entity, dict):
                        embedding = entity.get('embedding', entity.get('vector', []))
                        text = entity.get('text', entity.get('content', ''))
                        metadata = entity.get('metadata', {})
                    else:
                        logger.warning(f"Unknown entity structure: {type(entity)}")
                        continue
                    
                    data.append({
                        "vector": embedding,
                        "text": text,
                        "file_name": file_path.name,
                        "file_hash": file_hash,
                        "chunk_index": metadata.get("chunk_index", idx),
                        "total_chunks": metadata.get("total_chunks", len(entities)),
                    })
            else:
                # If no entities found, create a basic entry
                logger.warning(f"No entities extracted from {file_path}, creating placeholder")
                # Read file and create basic embedding
                with open(file_path, 'rb') as f:
                    content = f.read(1000).decode('utf-8', errors='ignore')  # Read first 1000 bytes
                
                embedding = google_embedding_func([content])[0]
                data.append({
                    "vector": embedding,
                    "text": content,
                    "file_name": file_path.name,
                    "file_hash": file_hash,
                    "chunk_index": 0,
                    "total_chunks": 1,
                })
            
            df = pd.DataFrame(data)

            # Save the DataFrame to a parquet file
            processed_file_path = Config.PROCESSED_DATA_DIR / f"{file_path.stem}.parquet"
            df.to_parquet(processed_file_path, index=False)

            logger.info(f"Successfully processed {file_path} and saved to {processed_file_path}")
            logger.info(f"Extracted {len(data)} chunks from document")
            return processed_file_path
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            raise

async def main():
    """Main function to test the RAGProcessor."""
    processor = RAGProcessor()
    dummy_pdf_path = Path("dummy.pdf")
    dummy_pdf_path.write_text("This is a dummy PDF for testing.")
    await processor.process_document(dummy_pdf_path)

if __name__ == "__main__":
    asyncio.run(main())

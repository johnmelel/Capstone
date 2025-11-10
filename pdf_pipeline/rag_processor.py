import asyncio
import logging
from pathlib import Path
from typing import List
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
        
        # Configure RAGAnything with proper settings
        self.config = RAGAnythingConfig(
            working_dir=str(working_dir),
            # Use MinerU parser for better PDF extraction
            parser="mineru",
            # Enable multimodal features
            enable_multimodal=True,
        )
        
        # Set up embedding function
        self.embedding_func = EmbeddingFunc(
            embedding_dim=768,  # Dimension for Google's embedding-001 model
            func=google_embedding_func,
        )
        
        # Initialize RAGAnything
        logger.info("Initializing RAGAnything with MinerU parser...")
        self.rag = RAGAnything(
            config=self.config,
            llm_model_func=google_llm_model_func,
            embedding_func=self.embedding_func,
        )
        logger.info("RAGAnything initialized successfully")
        logger.info(f"Parser: {self.config.parser}")
        logger.info(f"Working directory: {self.config.working_dir}")

    async def process_document(self, file_path: Path) -> Path:
        """Processes a single document and saves the output.
        
        Args:
            file_path: Path to the PDF file to process
            
        Returns:
            Path to the generated parquet file
        """
        logger.info(f"Processing document: {file_path}")
        try:
            # Process the document using RAGAnything's correct API
            logger.info("Using process_document_complete method")
            await self.rag.process_document_complete(str(file_path))
            
            logger.info("Document processed, extracting data from LightRAG storage...")
            
            # Extract processed data from LightRAG's knowledge graph storage
            # RAGAnything uses LightRAG internally which stores data in its graph
            entities = []
            try:
                # Access LightRAG's internal storage
                if hasattr(self.rag, 'lightrag'):
                    lightrag = self.rag.lightrag
                    logger.info(f"LightRAG type: {type(lightrag)}")
                    
                    # Try to access the chunk storage
                    if hasattr(lightrag, 'chunk_entity_relation_graph'):
                        logger.info("Accessing chunk_entity_relation_graph...")
                        graph = lightrag.chunk_entity_relation_graph
                        
                        # Try different methods to get chunks
                        if hasattr(graph, 'get_all_chunks'):
                            entities = await graph.get_all_chunks()
                        elif hasattr(graph, 'get_node_by_key'):
                            # If we need to iterate through keys
                            logger.info("Graph uses key-based access")
                        elif hasattr(graph, '_graph'):
                            # Direct graph access
                            logger.info("Direct graph access available")
                            
                    # Alternative: check for document storage
                    if not entities and hasattr(lightrag, 'doc_chunks'):
                        logger.info("Accessing doc_chunks...")
                        entities = lightrag.doc_chunks
                        
                    # Alternative: check working directory for stored data
                    if not entities:
                        logger.info("Checking working directory for stored chunks...")
                        working_dir = Path(self.rag.config.working_dir)
                        
                        # Look for parquet files with chunks
                        chunk_files = list(working_dir.glob("**/chunks*.parquet"))
                        if chunk_files:
                            logger.info(f"Found chunk files: {chunk_files}")
                            import pandas as pd
                            for chunk_file in chunk_files:
                                df = pd.read_parquet(chunk_file)
                                logger.info(f"Loaded {len(df)} chunks from {chunk_file.name}")
                                # Convert dataframe to entity format
                                for _, row in df.iterrows():
                                    entities.append(row.to_dict())
                        
                        # Look for JSON files
                        json_files = list(working_dir.glob("**/*.json"))
                        if json_files and not entities:
                            logger.info(f"Found JSON files: {json_files}")
                            import json
                            for json_file in json_files:
                                try:
                                    with open(json_file, 'r') as f:
                                        data = json.load(f)
                                        if isinstance(data, list):
                                            entities.extend(data)
                                        elif isinstance(data, dict):
                                            entities.append(data)
                                except Exception as e:
                                    logger.debug(f"Could not load {json_file}: {e}")
                else:
                    logger.error("RAGAnything does not have lightrag attribute")
                    
            except Exception as e:
                logger.error(f"Error extracting entities from RAGAnything: {e}", exc_info=True)

            # Convert the entities to a pandas DataFrame
            data = []
            
            # Get file hash for tracking
            import hashlib
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if entities:
                logger.info(f"Processing {len(entities)} extracted entities...")
                for idx, entity in enumerate(entities):
                    # Handle different entity structures from RAGAnything/LightRAG
                    if hasattr(entity, 'embedding') or (isinstance(entity, dict) and 'embedding' in entity):
                        # Entity with embedding attribute
                        embedding = entity.embedding if hasattr(entity, 'embedding') else entity.get('embedding', entity.get('vector', []))
                        text = getattr(entity, 'text', None) or getattr(entity, 'content', None)
                        if text is None and isinstance(entity, dict):
                            text = entity.get('text', entity.get('content', entity.get('chunk', '')))
                        metadata = getattr(entity, 'metadata', {}) if hasattr(entity, 'metadata') else entity.get('metadata', {})
                    elif isinstance(entity, dict):
                        # Dictionary-based entity
                        embedding = entity.get('embedding', entity.get('vector', []))
                        text = entity.get('text', entity.get('content', entity.get('chunk', '')))
                        metadata = entity.get('metadata', {})
                    else:
                        logger.warning(f"Unknown entity structure: {type(entity)}")
                        continue
                    
                    # Generate embedding if not present
                    if not embedding or len(embedding) == 0:
                        logger.info(f"Generating embedding for chunk {idx}...")
                        embedding = google_embedding_func([text])[0]
                    
                    data.append({
                        "vector": embedding,
                        "text": text,
                        "file_name": file_path.name,
                        "file_hash": file_hash,
                        "chunk_index": metadata.get("chunk_index", idx),
                        "total_chunks": metadata.get("total_chunks", len(entities)),
                    })
            
            # If no entities extracted, fall back to direct processing
            if not data:
                logger.warning(f"No entities extracted from RAGAnything for {file_path}")
                logger.info("Using fallback: direct text extraction and chunking...")
                
                # Use PyMuPDF as fallback for text extraction
                try:
                    import fitz
                    doc = fitz.open(str(file_path))
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    doc.close()
                except ImportError:
                    # If PyMuPDF not available, read raw bytes
                    logger.warning("PyMuPDF not available, using basic extraction")
                    with open(file_path, 'rb') as f:
                        text = f.read(10000).decode('utf-8', errors='ignore')
                
                if text.strip():
                    # Simple chunking
                    chunk_size = 1000
                    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]
                    
                    logger.info(f"Created {len(chunks)} chunks from direct extraction")
                    
                    # Generate embeddings for chunks
                    embeddings = google_embedding_func(chunks)
                    
                    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        data.append({
                            "vector": embedding,
                            "text": chunk,
                            "file_name": file_path.name,
                            "file_hash": file_hash,
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                        })
            
            if not data:
                raise ValueError(f"Could not extract any data from {file_path}")
            
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

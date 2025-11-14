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

async def google_llm_model_func(prompt: str, **kwargs):
    """Custom async LLM model function using Google's Generative AI.
    
    Note: Only passes parameters that Google's API actually supports to avoid errors.
    """
    # List of parameters that Google's GenerativeModel.generate_content() actually supports
    supported_params = {
        'generation_config',  # GenerationConfig object or dict
        'safety_settings',    # List of SafetySetting objects
        'stream',            # Boolean for streaming
        'tools',             # List of Tool objects
        'tool_config',       # ToolConfig object
        'request_options',   # RequestOptions object
    }
    
    # Filter kwargs to only include supported parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
    
    # Log if we're filtering out parameters (for debugging)
    filtered_out = set(kwargs.keys()) - supported_params
    if filtered_out:
        logger.debug(f"Filtered out unsupported parameters: {filtered_out}")
    
    model = genai.GenerativeModel(Config.LLM_MODEL)
    # Run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, 
        lambda: model.generate_content(prompt, **filtered_kwargs)
    )
    return response.text

async def google_embedding_func(texts: List[str]) -> List[List[float]]:
    """Custom async embedding function using Google's Generative AI.
    
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
    
    # Filter out empty strings before processing
    texts = [text.strip() for text in texts if text and text.strip()]
    
    if not texts:
        logger.warning("All texts were empty after filtering")
        return []
    
    # Google's API can handle batch requests, but let's process them safely
    embeddings = []
    
    # Process in batches if needed (Google API has limits)
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda b=batch: genai.embed_content(
                    model=Config.EMBEDDING_MODEL,
                    content=b,
                    task_type="retrieval_document"
                )
            )
            # Response structure: {'embedding': [list of vectors]}
            if isinstance(response, dict) and 'embedding' in response:
                batch_embeddings = response['embedding']
                # If single text, wrap in list
                if not isinstance(batch_embeddings[0], list):
                    batch_embeddings = [batch_embeddings]
                
                # Validate embedding dimensions
                for idx, emb in enumerate(batch_embeddings):
                    if len(emb) != Config.EMBEDDING_DIM:
                        logger.error(
                            f"Dimension mismatch! Expected {Config.EMBEDDING_DIM}, got {len(emb)} "
                            f"for text: '{batch[idx][:100]}...'"
                        )
                        raise ValueError(
                            f"Embedding dimension mismatch: expected {Config.EMBEDDING_DIM}, got {len(emb)}"
                        )
                
                embeddings.extend(batch_embeddings)
            else:
                logger.error(f"Unexpected response structure from Google embedding API: {response}")
                raise ValueError("Invalid response from embedding API")
        except Exception as e:
            logger.error(f"Error generating embeddings for batch: {e}")
            raise
    
    return embeddings

class RAGProcessor:
    def __init__(self, clear_cache: bool = None):
        """Initialize RAGProcessor
        
        Args:
            clear_cache: If True, clears the working directory cache on initialization.
                        If None, uses Config.RAG_CLEAR_CACHE value.
        """
        # Ensure working directory exists
        working_dir = Path("./rag_storage")
        working_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine if cache should be cleared
        if clear_cache is None:
            clear_cache = Config.RAG_CLEAR_CACHE
        
        # Clear cache if requested
        if clear_cache:
            logger.info("Clearing RAGAnything cache to ensure fresh processing...")
            self._clear_cache(working_dir)
        else:
            logger.info("Cache clearing disabled - using existing cache if available")
        
        # Ensure processed data directory exists
        Config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configure RAGAnything with proper settings
        # RAGAnything should use MinerU for high-quality PDF parsing
        self.config = RAGAnythingConfig(
            working_dir=str(working_dir),
            # Add MinerU-specific configuration if supported
            # The exact parameter name may vary - check RAGAnything docs
        )
        
        # Validate MinerU availability
        self._validate_mineru()
        
        # Set up embedding function
        self.embedding_func = EmbeddingFunc(
            embedding_dim=Config.EMBEDDING_DIM,  # Use config value
            func=google_embedding_func,
        )
        
        # Initialize RAGAnything
        logger.info("Initializing RAGAnything...")
        self.rag = RAGAnything(
            config=self.config,
            llm_model_func=google_llm_model_func,
            embedding_func=self.embedding_func,
        )
        logger.info("RAGAnything initialized successfully")
        logger.info(f"Working directory: {self.config.working_dir}")
    
    def _clear_cache(self, working_dir: Path):
        """Clear cache files from working directory
        
        Args:
            working_dir: Path to the working directory to clean
        """
        try:
            import shutil
            
            cache_patterns = [
                "kv_store_*.json",      # Key-value store cache files
                "vdb_*.json",           # Vector database cache files
                "*_cache.json",         # General cache files
            ]
            
            cleared_count = 0
            for pattern in cache_patterns:
                for cache_file in working_dir.glob(f"**/{pattern}"):
                    try:
                        cache_file.unlink()
                        logger.debug(f"Deleted cache file: {cache_file.name}")
                        cleared_count += 1
                    except Exception as e:
                        logger.warning(f"Could not delete {cache_file}: {e}")
            
            if cleared_count > 0:
                logger.info(f"✓ Cleared {cleared_count} cache files from {working_dir}")
            else:
                logger.info(f"No cache files found in {working_dir}")
                
        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")
    
    def _validate_mineru(self):
        """Validate MinerU is available and properly configured"""
        try:
            # Try to import MinerU
            try:
                import mineru
                logger.info(f"✓ MinerU is installed: version {getattr(mineru, '__version__', 'unknown')}")
            except ImportError:
                logger.error("✗ MinerU is not installed!")
                logger.error("Install all dependencies with:")
                logger.error("  pip install uv")
                logger.error("  uv pip install -r requirements.txt")
                raise ImportError("MinerU is required but not installed")
            
            # Validate device configuration
            device = Config.MINERU_DEVICE.lower()
            logger.info(f"MinerU device configuration: {device}")
            
            if device == "cuda":
                # Check if CUDA is available
                try:
                    import torch
                    if torch.cuda.is_available():
                        logger.info(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
                    else:
                        logger.warning("⚠ CUDA device requested but not available. Will fall back to CPU.")
                        logger.warning("This may result in slower processing.")
                except ImportError:
                    logger.warning("⚠ PyTorch not installed, cannot verify CUDA availability")
            elif device == "mps":
                # Check if MPS (Apple Silicon) is available
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        logger.info("✓ MPS (Apple Silicon) is available")
                    else:
                        logger.warning("⚠ MPS device requested but not available. Will fall back to CPU.")
                except ImportError:
                    logger.warning("⚠ PyTorch not installed, cannot verify MPS availability")
            elif device == "cpu":
                logger.info("✓ Using CPU device (this may be slower)")
            else:
                logger.warning(f"⚠ Unknown device type: {device}. Defaulting to CPU.")
            
            logger.info("✓ MinerU validation completed")
            
        except ImportError as e:
            logger.error(f"MinerU validation failed: {e}")
            raise
        except Exception as e:
            logger.warning(f"MinerU validation warning: {e}")

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
            logger.info("Processing document with RAGAnything...")
            await self.rag.process_document_complete(str(file_path))
            logger.info("Document processing completed")
            
            # Extract chunks using RAGAnything's query/search functionality
            # RAGAnything uses LightRAG internally and should provide a way to retrieve chunks
            entities = []
            
            try:
                # Method 1: Try to query all content from the document
                # This is typically how you'd retrieve processed chunks
                logger.info("Attempting to extract chunks via query method...")
                
                if hasattr(self.rag, 'query'):
                    # Query for document content - this should return processed chunks
                    query_result = await self.rag.query(
                        query=f"What is in {file_path.name}?",
                        param={"mode": "local", "only_need_context": True}
                    )
                    logger.info(f"Query result type: {type(query_result)}")
                    
                    if isinstance(query_result, dict) and 'context' in query_result:
                        # Extract chunks from context
                        context = query_result['context']
                        if isinstance(context, list):
                            entities = context
                            logger.info(f"Extracted {len(entities)} chunks from query context")
                    elif isinstance(query_result, str):
                        # Single result string - need to split into chunks
                        logger.warning("Query returned single string, cannot extract individual chunks")
                
                # Method 2: Try to access internal storage directly
                if not entities and hasattr(self.rag, 'lightrag'):
                    logger.info("Attempting to access LightRAG internal storage...")
                    lightrag = self.rag.lightrag
                    
                    # Try accessing chunk storage via key-value store
                    if hasattr(lightrag, 'key_string_value_json_storage_cls'):
                        logger.info("Attempting to read from key-value storage...")
                        # The storage should have chunks indexed by document
                        
                    # Try to get chunks from full_docs storage
                    if hasattr(lightrag, 'chunks_vdb'):
                        logger.info("Found chunks_vdb attribute")
                        # This is the vector database storing chunks
                        # May need to query it to get all chunks
                
                # Method 3: Read from working directory parquet files (last resort, but valid)
                if not entities:
                    logger.info("Attempting to read chunks from working directory...")
                    working_dir = Path(self.rag.config.working_dir)
                    
                    # Look specifically for chunk data files (not cache files)
                    chunk_files = list(working_dir.glob("**/chunks*.parquet"))
                    if chunk_files:
                        logger.info(f"Found chunk parquet files: {chunk_files}")
                        for chunk_file in chunk_files:
                            df = pd.read_parquet(chunk_file)
                            logger.info(f"Loaded {len(df)} rows from {chunk_file.name}")
                            # Verify this is actually chunk data
                            if 'content' in df.columns or 'text' in df.columns or 'chunk' in df.columns:
                                for _, row in df.iterrows():
                                    entities.append(row.to_dict())
                                logger.info(f"Extracted {len(df)} chunks from {chunk_file.name}")
                            else:
                                logger.warning(f"File {chunk_file.name} doesn't contain expected chunk columns")
                
                if not entities:
                    logger.error("Failed to extract entities using any method")
                    logger.error("This indicates RAGAnything did not properly process the document")
                    raise ValueError("RAGAnything processing produced no extractable chunks")
                    
            except Exception as e:
                logger.error(f"Error extracting entities from RAGAnything: {e}", exc_info=True)
                raise ValueError(f"Failed to extract data from RAGAnything: {e}")

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
                    embedding = None
                    text = None
                    metadata = {}
                    
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
                        logger.warning(f"Unknown entity structure at index {idx}: {type(entity)}")
                        continue
                    
                    # Validate text content
                    if not text or not text.strip():
                        logger.warning(f"Skipping entity {idx}: empty text content")
                        continue
                    
                    # Validate text is actual content, not JSON/metadata
                    text_lower = text.strip().lower()
                    if (text_lower.startswith('{') and text_lower.endswith('}')) or \
                       (text_lower.startswith('[') and text_lower.endswith(']')):
                        logger.warning(f"Skipping entity {idx}: appears to be JSON data, not document content")
                        continue
                    
                    # Ensure text is substantial (minimum 20 characters)
                    if len(text.strip()) < 20:
                        logger.warning(f"Skipping entity {idx}: text too short ({len(text)} chars)")
                        continue
                    
                    # Generate embedding if not present
                    if not embedding or len(embedding) == 0:
                        logger.info(f"Generating embedding for chunk {idx}...")
                        try:
                            embedding_result = await google_embedding_func([text])
                            embedding = embedding_result[0]
                        except (ValueError, IndexError) as e:
                            logger.error(f"Failed to generate embedding for chunk {idx}: {e}")
                            continue
                    
                    # Validate embedding dimension before adding to data
                    if len(embedding) != Config.EMBEDDING_DIM:
                        logger.error(
                            f"Dimension mismatch in entity {idx}! Expected {Config.EMBEDDING_DIM}, "
                            f"got {len(embedding)}. Skipping this entity."
                        )
                        continue
                    
                    data.append({
                        "vector": embedding,
                        "text": text,
                        "file_name": file_path.name,
                        "file_hash": file_hash,
                        "chunk_index": metadata.get("chunk_index", idx),
                        "total_chunks": metadata.get("total_chunks", len(entities)),
                    })
                
                logger.info(f"Successfully validated and processed {len(data)} chunks out of {len(entities)} entities")
            
            # No fallback methods - if RAGAnything fails, the pipeline should fail
            if not data:
                error_msg = (
                    f"Failed to extract any valid data from {file_path}. "
                    f"RAGAnything processing did not produce usable chunks. "
                    f"This could be due to: "
                    f"1) RAGAnything not properly processing the document, "
                    f"2) Incorrect entity extraction logic, "
                    f"3) Document format issues. "
                    f"Check logs above for details."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
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

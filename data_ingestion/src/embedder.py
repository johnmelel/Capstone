"""Text embedding module using Google Gemini API"""

import logging
from typing import List, Union
import numpy as np
import time

from google import genai
from google.genai import types

from .config import Config


logger = logging.getLogger(__name__)


class TextEmbedder:
    """Class to generate embeddings from text using Gemini API"""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize text embedder with Gemini API
        
        Args:
            model_name: Name of the Gemini embedding model to use
            api_key: Gemini API key (optional, defaults to config)
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.max_tokens = Config.MAX_TOKENS_PER_CHUNK
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for text embeddings")
        
        logger.info(f"Initializing Gemini embedding model: {self.model_name}")
        
        # Initialize Genai client
        self.client = genai.Client(api_key=self.api_key)
        
        # Use configured embedding dimension
        self.embedding_dim = Config.EMBEDDING_DIMENSION
        logger.info(f"Model initialized. Embedding dimension: {self.embedding_dim}")
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Google's API
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            # Use Google's count_tokens API with the client
            result = self.client.models.count_tokens(
                model=self.model_name,
                contents=text
            )
            return result.total_tokens
        except Exception as e:
            logger.warning(f"Could not count tokens via API: {e}. Using estimation.")
            # Rough estimation: 1 token ≈ 4 characters for English text
            return len(text) // 4
    
    def _validate_text_length(self, text: str) -> bool:
        """
        Validate that text doesn't exceed token limit
        
        Args:
            text: Text to validate
            
        Returns:
            True if valid, False otherwise
        """
        token_count = self._count_tokens(text)
        if token_count > self.max_tokens:
            logger.warning(
                f"Text exceeds token limit: {token_count} tokens (max: {self.max_tokens}). "
                f"Text will be truncated."
            )
            return False
        return True
    
    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to fit within token limit using binary search
        
        Args:
            text: Text to truncate
            
        Returns:
            Truncated text
        """
        # Binary search to find the right length
        left, right = 0, len(text)
        result = text
        
        while left < right:
            mid = (left + right + 1) // 2
            candidate = text[:mid]
            
            if self._count_tokens(candidate) <= self.max_tokens:
                result = candidate
                left = mid
            else:
                right = mid - 1
        
        # Try to end at a sentence or word boundary
        if len(result) < len(text):
            last_period = result.rfind('.')
            last_space = result.rfind(' ')
            if last_period > len(result) * 0.9:
                result = result[:last_period + 1]
            elif last_space > len(result) * 0.9:
                result = result[:last_space]
        
        return result.strip()
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text using Gemini API
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Numpy array of embeddings
        """
        try:
            if isinstance(text, str):
                text = [text]
            
            # Validate and truncate texts if necessary
            processed_texts = []
            for t in text:
                if not self._validate_text_length(t):
                    t = self._truncate_text(t)
                processed_texts.append(t)
            
            # Generate embeddings using Gemini API
            embeddings = []
            for i, t in enumerate(processed_texts):
                try:
                    result = self.client.models.embed_content(
                        model=self.model_name,
                        contents=t,
                        config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=self.embedding_dim
                        )
                    )
                    # Extract embedding values from the response
                    embedding_obj = result.embeddings[0]
                    embeddings.append(embedding_obj.values)
                    
                    # Rate limiting - be nice to the API
                    if i < len(processed_texts) - 1:
                        time.sleep(0.1)  # 100ms between requests
                        
                except Exception as e:
                    logger.error(f"Error embedding text chunk {i}: {e}")
                    # Return zero vector on error
                    embeddings.append([0.0] * self.embedding_dim)
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            logger.debug(f"Generated embeddings for {len(text)} text chunks")
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = None) -> List[np.ndarray]:
        """
        Generate embeddings for a large list of texts in batches
        
        Args:
            texts: List of text strings
            batch_size: Size of batches to process
            
        Returns:
            List of embedding arrays
        """
        batch_size = batch_size or Config.BATCH_SIZE
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/"
                f"{(len(texts) + batch_size - 1) // batch_size}"
            )
            embeddings = self.embed_text(batch)
            all_embeddings.extend(embeddings)
            
            # Add delay between batches to respect rate limits
            if i + batch_size < len(texts):
                time.sleep(1)  # 1 second between batches
        
        return all_embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        return self.embedding_dim


def embed_texts(
    texts: Union[str, List[str]],
    model_name: str = None
) -> np.ndarray:
    """
    Convenience function to embed texts
    
    Args:
        texts: Text or list of texts to embed
        model_name: Name of the model to use
        
    Returns:
        Numpy array of embeddings
    """
    embedder = TextEmbedder(model_name=model_name)
    return embedder.embed_text(texts)


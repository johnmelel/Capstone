"""Text embedding module using Google Gemini API (New SDK)"""

import logging
from typing import List, Union
import numpy as np
import time

from google import genai
from google.genai import types

from .config import Config


logger = logging.getLogger(__name__)


class TextEmbedder:
    """Class to generate embeddings from text using Gemini API (New SDK)"""
    
    def __init__(self, model_name: str = None, api_key: str = None, embedding_dimension: int = None):
        """
        Initialize text embedder with Gemini API
        
        Args:
            model_name: Name of the Gemini embedding model to use
            api_key: Gemini API key (optional, defaults to config)
            embedding_dimension: Output dimension for embeddings (optional, defaults to config)
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.embedding_dim = embedding_dimension or Config.EMBEDDING_DIMENSION
        self.max_tokens = Config.MAX_TOKENS_PER_CHUNK
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for text embeddings")
        
        logger.info(f"Initializing Gemini embedding model: {self.model_name}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Initialize Gemini client with the new SDK
        self.client = genai.Client(api_key=self.api_key)
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = genai.LocalTokenizer(model_name='gemini-2.0-flash-exp')
        except Exception as e:
            logger.warning(f"Could not initialize tokenizer: {e}. Will use estimation.")
            self.tokenizer = None
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the tokenizer
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            if self.tokenizer:
                result = self.tokenizer.compute_tokens(text)
                return result.token_count
        except Exception as e:
            logger.warning(f"Could not count tokens via tokenizer: {e}. Using estimation.")
        
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
        Generate embeddings for text using Gemini API (New SDK)
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Numpy array of embeddings
        """
        try:
            if isinstance(text, str):
                text = [text]
            
            # Handle empty list
            if len(text) == 0:
                return np.empty((0, self.embedding_dim), dtype=np.float32)
            
            # Validate and truncate texts if necessary
            processed_texts = []
            for t in text:
                if not self._validate_text_length(t):
                    t = self._truncate_text(t)
                processed_texts.append(t)
            
            # Generate embeddings using Gemini API with new SDK
            embeddings = []
            for i, t in enumerate(processed_texts):
                try:
                    # Use the new SDK's embed_content method
                    config = types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=self.embedding_dim
                    )
                    
                    result = self.client.models.embed_content(
                        model=self.model_name,
                        contents=t,
                        config=config
                    )
                    
                    # Extract embedding values from the response
                    embedding_values = result.embeddings[0].values
                    embeddings.append(embedding_values)
                    
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
    model_name: str = None,
    embedding_dimension: int = None
) -> np.ndarray:
    """
    Convenience function to embed texts
    
    Args:
        texts: Text or list of texts to embed
        model_name: Name of the model to use
        embedding_dimension: Output dimension for embeddings
        
    Returns:
        Numpy array of embeddings
    """
    embedder = TextEmbedder(model_name=model_name, embedding_dimension=embedding_dimension)
    return embedder.embed_text(texts)

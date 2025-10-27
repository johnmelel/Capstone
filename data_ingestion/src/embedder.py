"""Text embedding module"""

import logging
from typing import List, Union
import numpy as np

from sentence_transformers import SentenceTransformer

from .config import Config


logger = logging.getLogger(__name__)


class TextEmbedder:
    """Class to generate embeddings from text"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize text embedder
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Numpy array of embeddings
        """
        try:
            if isinstance(text, str):
                text = [text]
            
            embeddings = self.model.encode(
                text,
                show_progress_bar=len(text) > 10,
                convert_to_numpy=True
            )
            
            logger.debug(f"Generated embeddings for {len(text)} text chunks")
            return embeddings
            
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
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
            embeddings = self.embed_text(batch)
            all_embeddings.extend(embeddings)
        
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

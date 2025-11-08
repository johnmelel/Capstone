"""Vertex AI multimodal embedding generation."""

import logging
from typing import List, Dict, Any
import numpy as np
from pathlib import Path

from google.cloud import aiplatform
from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage

from .config import Config


logger = logging.getLogger(__name__)


class VertexAIEmbedder:
    """Generate embeddings using Vertex AI multimodal model."""
    
    def __init__(self, project_id: str = None, location: str = None):
        """
        Initialize Vertex AI embedder.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud location (e.g., 'us-central1')
        """
        self.project_id = project_id or Config.GOOGLE_CLOUD_PROJECT
        self.location = location or Config.GOOGLE_CLOUD_LOCATION
        self.model_name = "multimodalembedding@001"
        self.dimensions = 1408  # Vertex AI multimodal embedding dimensions
        
        # Initialize Vertex AI
        logger.info("Initializing Vertex AI...")
        aiplatform.init(project=self.project_id, location=self.location)
        
        logger.info(f"Vertex AI Embedder initialized")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Dimensions: {self.dimensions}")
        logger.info(f"  Project: {self.project_id}")
        logger.info(f"  Location: {self.location}")
    
    def embed_batch(self, items: List[Dict]) -> List[np.ndarray]:
        """
        Embed a batch of items (text, images, tables).
        
        Args:
            items: List of dictionaries with 'type', 'content', 'image_path', etc.
            
        Returns:
            List of embedding arrays
        """
        logger.info(f"Embedding {len(items)} items...")
        
        embeddings = []
        
        for i, item in enumerate(items):
            if i % 10 == 0 and i > 0:
                logger.debug(f"  Progress: {i}/{len(items)}")
            
            try:
                if item["type"] == "text":
                    emb = self._embed_text(item["content"])
                elif item["type"] == "image":
                    emb = self._embed_image(item["image_path"], item.get("caption", ""))
                elif item["type"] == "table":
                    # Embed table as text
                    emb = self._embed_text(str(item["content"]))
                else:
                    logger.warning(f"Unknown item type: {item['type']}")
                    # Return zero vector
                    emb = np.zeros(self.dimensions)
                
                embeddings.append(emb)
                
            except Exception as e:
                logger.error(f"Failed to embed {item.get('chunk_id', 'unknown')}: {e}")
                # Return zero vector on error
                embeddings.append(np.zeros(self.dimensions))
        
        logger.info(f"Successfully embedded {len(embeddings)} items")
        return embeddings
    
    def _embed_text(self, text: str) -> np.ndarray:
        """
        Embed text content.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding array
        """
        model = MultiModalEmbeddingModel.from_pretrained(self.model_name)
        
        # Vertex AI has ~1000 char limit for text
        if len(text) > 1000:
            logger.debug(f"Text too long ({len(text)} chars), truncating to 1000")
            text = text[:1000]
            
            # Try to end at sentence boundary
            last_period = text.rfind('. ')
            if last_period > 700:  # Keep at least 70%
                text = text[:last_period + 1]
        
        embeddings = model.get_embeddings(contextual_text=text)
        embedding = np.array(embeddings.text_embedding, dtype=np.float32)
        
        # Normalize (L2 normalization)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _embed_image(self, image_path: str, caption: str = "") -> np.ndarray:
        """
        Embed image with optional caption.
        
        Args:
            image_path: Path to image file
            caption: Optional caption text
            
        Returns:
            Embedding array
        """
        model = MultiModalEmbeddingModel.from_pretrained(self.model_name)
        
        # Load image
        image = VertexImage.load_from_file(image_path)
        
        # Embed with context
        embeddings = model.get_embeddings(
            image=image,
            contextual_text=caption if caption else None
        )
        
        embedding = np.array(embeddings.image_embedding, dtype=np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.dimensions

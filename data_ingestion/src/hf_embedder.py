"""HuggingFace BiomedCLIP embedding client that communicates with the embedding service"""

import logging
from typing import List, Union, Optional
import numpy as np
import requests
import time
import base64
import io
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .config import Config
from .exceptions import EmbeddingError
from .constants import EMBEDDING_BATCH_DELAY_SECONDS


logger = logging.getLogger(__name__)


class HuggingFaceEmbedder:
    """Class to generate embeddings using HuggingFace BiomedCLIP via embedding service"""
    
    def __init__(
        self,
        service_url: str = None,
        embedding_dimension: int = None,
        timeout: int = 300,
        max_retries: int = 3
    ):
        """
        Initialize HuggingFace embedder client
        
        Args:
            service_url: URL of the embedding service
            embedding_dimension: Expected output dimension (default: 512 for BiomedCLIP)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.service_url = (service_url or Config.EMBEDDING_SERVICE_URL).rstrip('/')
        self.embedding_dim = embedding_dimension or Config.EMBEDDING_DIMENSION
        self.timeout = timeout
        self.max_retries = max_retries
        
        logger.info(f"Initializing HuggingFace embedder client")
        logger.info(f"Service URL: {self.service_url}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Verify service is available
        self._check_health()
    
    def _check_health(self):
        """Check if the embedding service is healthy"""
        try:
            response = requests.get(
                f"{self.service_url}/health",
                timeout=10
            )
            response.raise_for_status()
            
            health_data = response.json()
            logger.info(f"Embedding service is healthy")
            logger.info(f"Model: {health_data.get('model')}")
            logger.info(f"Device: {health_data.get('device')}")
            logger.info(f"Embedding dimension: {health_data.get('embedding_dimension')}")
            
            # Validate dimension matches
            service_dim = health_data.get('embedding_dimension')
            if service_dim and service_dim != self.embedding_dim:
                logger.warning(
                    f"Service embedding dimension ({service_dim}) differs from "
                    f"configured dimension ({self.embedding_dim}). "
                    f"Using service dimension."
                )
                self.embedding_dim = service_dim
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to embedding service at {self.service_url}: {e}")
            raise EmbeddingError(
                f"Embedding service is not available at {self.service_url}. "
                f"Please ensure the service is running."
            ) from e
    
    def _make_request(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        mode: str = "text",
        normalize: bool = True,
        fusion_method: str = "average"
    ) -> dict:
        """
        Make a request to the embedding service with retry logic
        
        Args:
            texts: List of texts to embed
            images: List of base64-encoded images
            mode: Embedding mode - "text", "image", or "multimodal"
            normalize: Whether to normalize embeddings
            fusion_method: Method to fuse text+image embeddings
            
        Returns:
            Response JSON data
        """
        payload = {
            "mode": mode,
            "normalize": normalize,
            "fusion_method": fusion_method
        }
        
        if texts:
            payload["texts"] = texts
        if images:
            payload["images"] = images
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.service_url}/embed",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        # All retries failed
        raise EmbeddingError(
            f"Failed to generate embeddings after {self.max_retries} attempts"
        ) from last_error
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text using the embedding service
        
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
            
            # Make request to service
            response_data = self._make_request(text, normalize=True)
            
            # Extract embeddings
            embeddings = response_data.get('embeddings', [])
            
            if not embeddings:
                logger.warning("No embeddings returned from service")
                # Return zero vectors
                return np.zeros((len(text), self.embedding_dim), dtype=np.float32)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            logger.debug(
                f"Generated embeddings for {len(text)} text chunks "
                f"in {response_data.get('processing_time', 0):.3f}s"
            )
            
            return embeddings_array
            
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings: {e}")
            raise EmbeddingError(f"Failed to generate embeddings") from e
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = None
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a large list of texts in batches
        
        Args:
            texts: List of text strings
            batch_size: Size of batches to process (default from config)
            
        Returns:
            List of embedding arrays
        """
        batch_size = batch_size or Config.BATCH_SIZE
        # Limit batch size for service requests
        batch_size = min(batch_size, 50)
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/"
                f"{(len(texts) + batch_size - 1) // batch_size}"
            )
            embeddings = self.embed_text(batch)
            all_embeddings.extend(embeddings)
            
            # Add delay between batches to respect service limits
            if i + batch_size < len(texts):
                time.sleep(EMBEDDING_BATCH_DELAY_SECONDS)
        
        return all_embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        return self.embedding_dim
    
    def _encode_image_to_base64(self, image_source: Union[Path, bytes, Image.Image]) -> str:
        """
        Encode image to base64 string
        
        Args:
            image_source: Path to image file, image bytes, or PIL Image
            
        Returns:
            Base64-encoded image string
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow is required for image processing")
        
        try:
            # Convert to PIL Image
            if isinstance(image_source, Path):
                img = Image.open(image_source)
            elif isinstance(image_source, bytes):
                img = Image.open(io.BytesIO(image_source))
            elif isinstance(image_source, Image.Image):
                img = image_source
            else:
                raise ValueError(f"Unsupported image source type: {type(image_source)}")
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Encode to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_bytes = buffer.getvalue()
            
            return base64.b64encode(img_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise EmbeddingError(f"Failed to encode image") from e
    
    def embed_image(self, image: Union[Path, bytes, Image.Image, List]) -> np.ndarray:
        """
        Generate embeddings for images using the embedding service
        
        Args:
            image: Single image (Path, bytes, or PIL Image) or list of images
            
        Returns:
            Numpy array of embeddings
        """
        try:
            # Handle single image or list
            if not isinstance(image, list):
                images = [image]
            else:
                images = image
            
            # Encode images to base64
            base64_images = [self._encode_image_to_base64(img) for img in images]
            
            # Make request
            response_data = self._make_request(
                images=base64_images,
                mode="image",
                normalize=True
            )
            
            # Extract embeddings
            embeddings = response_data.get('embeddings', [])
            
            if not embeddings:
                logger.warning("No embeddings returned from service")
                return np.zeros((len(images), self.embedding_dim), dtype=np.float32)
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            logger.debug(
                f"Generated embeddings for {len(images)} images "
                f"in {response_data.get('processing_time', 0):.3f}s"
            )
            
            return embeddings_array
            
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating image embeddings: {e}")
            raise EmbeddingError(f"Failed to generate image embeddings") from e
    
    def embed_multimodal(
        self,
        texts: List[str],
        images: List[Union[Path, bytes, Image.Image]],
        fusion_method: str = "average"
    ) -> np.ndarray:
        """
        Generate multimodal embeddings (text + image fusion)
        
        Args:
            texts: List of text strings
            images: List of images (same length as texts)
            fusion_method: How to fuse embeddings - "average", "max", or "concat"
            
        Returns:
            Numpy array of fused embeddings
        """
        try:
            if len(texts) != len(images):
                raise ValueError(
                    f"Number of texts ({len(texts)}) must match "
                    f"number of images ({len(images)})"
                )
            
            # Encode images to base64
            base64_images = [self._encode_image_to_base64(img) for img in images]
            
            # Make request
            response_data = self._make_request(
                texts=texts,
                images=base64_images,
                mode="multimodal",
                normalize=True,
                fusion_method=fusion_method
            )
            
            # Extract embeddings
            embeddings = response_data.get('embeddings', [])
            
            if not embeddings:
                logger.warning("No embeddings returned from service")
                # Dimension depends on fusion method
                dim = self.embedding_dim if fusion_method != "concat" else self.embedding_dim * 2
                return np.zeros((len(texts), dim), dtype=np.float32)
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            logger.debug(
                f"Generated {len(texts)} multimodal embeddings "
                f"in {response_data.get('processing_time', 0):.3f}s"
            )
            
            return embeddings_array
            
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating multimodal embeddings: {e}")
            raise EmbeddingError(f"Failed to generate multimodal embeddings") from e


def embed_texts(
    texts: Union[str, List[str]],
    service_url: str = None,
    embedding_dimension: int = None
) -> np.ndarray:
    """
    Convenience function to embed texts
    
    Args:
        texts: Text or list of texts to embed
        service_url: URL of the embedding service
        embedding_dimension: Expected output dimension
        
    Returns:
        Numpy array of embeddings
    """
    embedder = HuggingFaceEmbedder(
        service_url=service_url,
        embedding_dimension=embedding_dimension
    )
    return embedder.embed_text(texts)

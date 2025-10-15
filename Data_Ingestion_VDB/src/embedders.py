"""Vertex AI multimodal embedding generation."""

from google.cloud import aiplatform
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import base64
import io

class VertexAIEmbedder:
    """Generate embeddings using Vertex AI multimodal model."""
    
    def __init__(self, config: Dict[str, Any], project_id: str, location: str):
        print("\n🔧 Initializing Vertex AI embedder...")
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        self.project_id = project_id
        self.location = location
        self.model_name = "multimodalembedding@001"
        self.dimensions = 1408  # Vertex AI multimodal embedding dimensions
        self.normalize = config['embedding']['normalize']
        
        print(f"   ✓ Model: {self.model_name}")
        print(f"   ✓ Dimensions: {self.dimensions}")
        print(f"   ✓ Project: {project_id}")
        print(f"   ✓ Location: {location}")
    
    def embed_batch(self, items: List[Dict]) -> List[np.ndarray]:
        """Embed a batch of items."""
        print(f"\n Embedding {len(items)} items...")
        
        embeddings = []
        
        for i, item in enumerate(items):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(items)}")
            
            try:
                if item["type"] == "text":
                    emb = self._embed_text(item["content"])
                elif item["type"] == "image":
                    emb = self._embed_image(item["image_path"], item.get("caption", ""))
                elif item["type"] == "table":
                    emb = self._embed_text(str(item["content"]))
                else:
                    print(f"     Unknown type: {item['type']}")
                    continue
                
                embeddings.append(emb)
                
            except Exception as e:
                print(f"     Failed to embed {item['chunk_id']}: {e}")
                embeddings.append(np.zeros(self.dimensions))
        
        print(f"   ✓ Embedded {len(embeddings)} items successfully")
        return embeddings
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text content."""
        from vertexai.vision_models import MultiModalEmbeddingModel
        
        model = MultiModalEmbeddingModel.from_pretrained(self.model_name)

        # SAFTEY: vertex AI has 1024 char limit, so we'll impose on that (because we are doing semantic chunking)
        if len(text) >1000:
            print(f"       Text too long ({len(text)} chars), truncating to 1000...")
            text = text[:1000]

            #try to end at sentence boundary
            last_period = text.rfind('. ')
            if last_period > 700: # keep at least 70%
                text = text[:last_period + 1]

        embeddings = model.get_embeddings(
            contextual_text=text
        )
        
        embedding = np.array(embeddings.text_embedding)
        
        if self.normalize:
            embedding = self._normalize(embedding)
        
        return embedding
    


    def _embed_image(self, image_path: str, caption: str = "") -> np.ndarray:
        """Embed image with optional caption."""
        from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage
        
        model = MultiModalEmbeddingModel.from_pretrained(self.model_name)
        
        # Load image
        image = VertexImage.load_from_file(image_path)
        
        # Embed with context
        embeddings = model.get_embeddings(
            image=image,
            contextual_text=caption if caption else None
        )
        
        embedding = np.array(embeddings.image_embedding)
        
        if self.normalize:
            embedding = self._normalize(embedding)
        
        return embedding
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalization."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
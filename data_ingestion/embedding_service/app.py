"""
FastAPI service for BiomedCLIP text embeddings.

This service loads the microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 model
using the open_clip library and provides REST endpoints for generating embeddings.

The model outputs 512-dimensional embeddings optimized for biomedical text.
"""

import logging
import time
import base64
import io
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from open_clip import create_model_from_pretrained, get_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
preprocess = None  # Image preprocessing function
device = None

# Model configuration
MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EMBEDDING_DIMENSION = 512
MAX_LENGTH = 256  # Model's maximum sequence length
IMAGE_SIZE = 224  # Expected image size for ViT


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation"""
    texts: Optional[List[str]] = Field(None, description="List of texts to embed", max_items=100)
    images: Optional[List[str]] = Field(None, description="List of base64-encoded images", max_items=50)
    mode: str = Field("text", description="Embedding mode: 'text', 'image', or 'multimodal'")
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")
    fusion_method: str = Field("average", description="Method to fuse text+image: 'average', 'concat', 'max'")


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation"""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimension: int = Field(..., description="Dimension of each embedding")
    model: str = Field(..., description="Model used for embeddings")
    processing_time: float = Field(..., description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model: str
    device: str
    embedding_dimension: int


def load_model():
    """Load the BiomedCLIP model and tokenizer using open_clip"""
    global model, tokenizer, preprocess, device
    
    logger.info(f"Loading model: {MODEL_NAME}")
    start_time = time.time()
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    try:
        # Load model and tokenizer from open_clip
        # create_model_from_pretrained returns (model, preprocess)
        model, preprocess = create_model_from_pretrained(MODEL_NAME)
        tokenizer = get_tokenizer(MODEL_NAME)
        
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Image preprocessing available: {preprocess is not None}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading/unloading model"""
    # Startup
    logger.info("Starting embedding service...")
    load_model()
    logger.info("Embedding service ready")
    yield
    # Shutdown
    logger.info("Shutting down embedding service...")
    global model, tokenizer, preprocess
    model = None
    tokenizer = None
    preprocess = None


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        # Remove data URI prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',', 1)[1]
        
        # Decode base64
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise ValueError(f"Invalid base64 image data: {e}")


def fuse_embeddings(
    text_embeddings: Optional[torch.Tensor],
    image_embeddings: Optional[torch.Tensor],
    method: str = "average"
) -> torch.Tensor:
    """
    Fuse text and image embeddings
    
    Args:
        text_embeddings: Text embeddings tensor (N, D)
        image_embeddings: Image embeddings tensor (M, D)
        method: Fusion method - 'average', 'concat', or 'max'
    
    Returns:
        Fused embeddings tensor
    """
    if text_embeddings is None and image_embeddings is None:
        raise ValueError("At least one of text or image embeddings must be provided")
    
    if text_embeddings is None:
        return image_embeddings
    
    if image_embeddings is None:
        return text_embeddings
    
    if method == "average":
        # Average text and image embeddings (proven effective for CLIP)
        return (text_embeddings + image_embeddings) / 2
    
    elif method == "max":
        # Element-wise maximum
        return torch.maximum(text_embeddings, image_embeddings)
    
    elif method == "concat":
        # Concatenate (doubles dimensionality)
        return torch.cat([text_embeddings, image_embeddings], dim=-1)
    
    else:
        logger.warning(f"Unknown fusion method '{method}', using 'average'")
        return (text_embeddings + image_embeddings) / 2


# Initialize FastAPI app
app = FastAPI(
    title="BiomedCLIP Embedding Service",
    description="REST API for generating text embeddings using BiomedCLIP model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return HealthResponse(
        status="healthy",
        model=MODEL_NAME,
        device=str(device),
        embedding_dimension=EMBEDDING_DIMENSION
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "BiomedCLIP Embedding Service",
        "model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "endpoints": {
            "health": "/health",
            "embed": "/embed (POST)",
            "docs": "/docs"
        }
    }


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: EmbeddingRequest):
    """
    Generate embeddings for input texts and/or images.
    
    Args:
        request: EmbeddingRequest with texts, images, and mode
        
    Returns:
        EmbeddingResponse with embeddings and metadata
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # Validate request
    if request.mode == "text" and not request.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="texts is required for mode='text'"
        )
    if request.mode == "image" and not request.images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="images is required for mode='image'"
        )
    if request.mode == "multimodal" and not (request.texts and request.images):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both texts and images are required for mode='multimodal'"
        )
    
    start_time = time.time()
    
    try:
        embeddings_list = []
        
        if request.mode == "text":
            # Text-only embeddings
            text_tokens = tokenizer(request.texts).to(device)
            
            with torch.no_grad():
                embeddings = model.encode_text(text_tokens, normalize=request.normalize)
                embeddings_list = embeddings.cpu().numpy().tolist()
        
        elif request.mode == "image":
            # Image-only embeddings
            if preprocess is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Image preprocessing not available"
                )
            
            # Decode and preprocess images
            pil_images = [decode_base64_image(img_b64) for img_b64 in request.images]
            image_tensors = torch.stack([preprocess(img) for img in pil_images]).to(device)
            
            with torch.no_grad():
                embeddings = model.encode_image(image_tensors, normalize=request.normalize)
                embeddings_list = embeddings.cpu().numpy().tolist()
        
        elif request.mode == "multimodal":
            # Multimodal fusion
            if preprocess is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Image preprocessing not available"
                )
            
            # Ensure equal lengths or broadcast
            num_texts = len(request.texts)
            num_images = len(request.images)
            
            if num_texts != num_images:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Number of texts ({num_texts}) must match number of images ({num_images}) for multimodal mode"
                )
            
            # Encode texts
            text_tokens = tokenizer(request.texts).to(device)
            
            # Decode and preprocess images
            pil_images = [decode_base64_image(img_b64) for img_b64 in request.images]
            image_tensors = torch.stack([preprocess(img) for img in pil_images]).to(device)
            
            with torch.no_grad():
                text_embeddings = model.encode_text(text_tokens, normalize=False)
                image_embeddings = model.encode_image(image_tensors, normalize=False)
                
                # Fuse embeddings
                fused_embeddings = fuse_embeddings(
                    text_embeddings,
                    image_embeddings,
                    method=request.fusion_method
                )
                
                # Normalize if requested
                if request.normalize:
                    fused_embeddings = fused_embeddings / fused_embeddings.norm(dim=-1, keepdim=True)
                
                embeddings_list = fused_embeddings.cpu().numpy().tolist()
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid mode: {request.mode}. Must be 'text', 'image', or 'multimodal'"
            )
        
        processing_time = time.time() - start_time
        
        logger.info(f"Generated {len(embeddings_list)} {request.mode} embeddings in {processing_time:.3f}s")
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimension=EMBEDDING_DIMENSION if request.mode != "multimodal" or request.fusion_method != "concat" else EMBEDDING_DIMENSION * 2,
            model=MODEL_NAME,
            processing_time=processing_time
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run the service
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )
"""
FastAPI service for BiomedCLIP text embeddings.

This service loads the microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 model
using the open_clip library and provides REST endpoints for generating embeddings.

The model outputs 512-dimensional embeddings optimized for biomedical text.
"""

import logging
import time
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
import numpy as np
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
device = None

# Model configuration
MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EMBEDDING_DIMENSION = 512
MAX_LENGTH = 256  # Model's maximum sequence length


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation"""
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1, max_items=100)
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")


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
    global model, tokenizer, device
    
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
        model, _, _ = create_model_from_pretrained(MODEL_NAME)
        tokenizer = get_tokenizer(MODEL_NAME)
        
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
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
    global model, tokenizer
    model = None
    tokenizer = None


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
    Generate embeddings for input texts.
    
    Args:
        request: EmbeddingRequest containing texts to embed
        
    Returns:
        EmbeddingResponse with embeddings and metadata
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    start_time = time.time()
    
    try:
        # Tokenize texts
        text_tokens = tokenizer(request.texts).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = model.encode_text(text_tokens, normalize=request.normalize)
            
            # Convert to numpy and then to list
            embeddings_np = embeddings.cpu().numpy()
            embeddings_list = embeddings_np.tolist()
        
        processing_time = time.time() - start_time
        
        logger.info(f"Generated {len(embeddings_list)} embeddings in {processing_time:.3f}s")
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimension=EMBEDDING_DIMENSION,
            model=MODEL_NAME,
            processing_time=processing_time
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
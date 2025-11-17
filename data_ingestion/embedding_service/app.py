"""
FastAPI service for BiomedCLIP text embeddings.

This service loads the microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 model
and provides REST endpoints for generating embeddings.

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
from transformers import AutoTokenizer, AutoModel

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
MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
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
    """Load the BiomedCLIP model and tokenizer"""
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
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info("Tokenizer loaded successfully")
        
        # Load model
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings.
    
    Args:
        model_output: Model output containing last_hidden_state
        attention_mask: Attention mask to ignore padding tokens
        
    Returns:
        Pooled embeddings
    """
    token_embeddings = model_output[0]  # First element is last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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
        encoded_input = tokenizer(
            request.texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Move to device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Generate embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            
            # Perform mean pooling
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings if requested
            if request.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
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

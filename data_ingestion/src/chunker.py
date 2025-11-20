"""Text chunking module with exact tokenization support"""

import logging
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .config import Config
from .utils import create_metadata, get_file_hash
from .types import ImageData


logger = logging.getLogger(__name__)


class TextChunker:
    """Base class for text chunking"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
    def chunk_text(self, text: str) -> List[str]:
        """Override in subclasses"""
        raise NotImplementedError


class SimpleTokenChunker(TextChunker):
    """Legacy chunker using character heuristic (1 token ≈ 4 chars)"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        super().__init__(chunk_size, chunk_overlap)
        logger.info(f"SimpleTokenChunker initialized: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def _chars_from_tokens(self, num_tokens: int) -> int:
        return num_tokens * 4
    
    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        chunk_size_chars = self._chars_from_tokens(self.chunk_size)
        overlap_chars = self._chars_from_tokens(self.chunk_overlap)
        step_size = chunk_size_chars - overlap_chars
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size_chars, text_length)
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            start += step_size
            if end >= text_length:
                break
        
        return chunks


class ExactTokenChunker(TextChunker):
    """Chunker using HuggingFace tokenizer for exact counts"""
    
    def __init__(self, model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", chunk_size: int = None, chunk_overlap: int = None):
        super().__init__(chunk_size, chunk_overlap)
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, falling back to SimpleTokenChunker")
            self._fallback = SimpleTokenChunker(chunk_size, chunk_overlap)
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"ExactTokenChunker initialized with model: {model_name}")
            self._fallback = None
        except Exception as e:
            logger.error(f"Failed to load tokenizer {model_name}: {e}")
            logger.warning("Falling back to SimpleTokenChunker")
            self._fallback = SimpleTokenChunker(chunk_size, chunk_overlap)

    def chunk_text(self, text: str) -> List[str]:
        if self._fallback:
            return self._fallback.chunk_text(text)
            
        if not text:
            return []

        # Tokenize entire text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(tokens)
        
        chunks = []
        start = 0
        step_size = self.chunk_size - self.chunk_overlap
        
        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            start += step_size
            if end >= total_tokens:
                break
                
        logger.debug(f"Split text into {len(chunks)} chunks using exact tokenization")
        return chunks


class ImageCaptionChunker:
    """Chunker for handling Image+Caption pairs"""
    
    def __init__(self, chunker: TextChunker):
        self.text_chunker = chunker
        
    def chunk_image(self, image: ImageData) -> List[Tuple[ImageData, Optional[str]]]:
        """
        Chunk an image based on its caption.
        Returns list of (image, caption_chunk) pairs.
        If caption is empty, returns [(image, None)].
        """
        caption = image.get('caption')
        
        # Case 1: No caption
        if not caption:
            return [(image, None)]
            
        # Case 2: Caption fits in one chunk
        # We use the text chunker to check/split
        caption_chunks = self.text_chunker.chunk_text(caption)
        
        if not caption_chunks:
            # Should not happen if caption is not empty, but safety check
            return [(image, None)]
            
        # Case 3: Caption split into multiple chunks
        # Duplicate image for each caption chunk
        return [(image, chunk) for chunk in caption_chunks]


def chunk_with_metadata(
    text: str,
    file_path: Any,
    chunker: TextChunker,
    **additional_metadata
) -> List[Dict[str, Any]]:
    """
    Chunk text and create metadata for each chunk
    """
    chunks = chunker.chunk_text(text)
    
    # Generate file hash
    if isinstance(file_path, Path):
        file_hash = get_file_hash(file_path)
    else:
        # GCS blob handling
        if hasattr(file_path, 'etag'):
            file_hash = file_path.etag.strip('"')
        elif hasattr(file_path, 'size') and hasattr(file_path, 'name'):
            import hashlib
            hash_input = f"{file_path.name}:{file_path.size}"
            file_hash = hashlib.md5(hash_input.encode()).hexdigest()
        else:
            import hashlib
            file_hash = hashlib.md5(file_path.name.encode()).hexdigest()
    
    chunks_with_metadata = []
    for idx, chunk in enumerate(chunks):
        metadata = create_metadata(
            file_name=file_path.name,
            file_hash=file_hash,
            chunk_index=idx,
            total_chunks=len(chunks),
            token_count=len(chunk.split()), # Rough estimate for metadata
            **additional_metadata
        )
        
        chunks_with_metadata.append({
            'text': chunk,
            'metadata': metadata
        })
    
    return chunks_with_metadata




"""Text chunking module with simple token-based chunking"""

import logging
from typing import List, Dict, Any
from pathlib import Path

from .config import Config
from .utils import create_metadata, get_file_hash


logger = logging.getLogger(__name__)


class TextChunker:
    """Class to split text into chunks with simple token-based sliding window"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize text chunker with token-based limits
        
        Args:
            chunk_size: Target size of each chunk in TOKENS (not characters)
            chunk_overlap: Number of TOKENS to overlap between chunks
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        logger.info(f"TextChunker initialized: chunk_size={self.chunk_size} tokens, overlap={self.chunk_overlap} tokens")
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using simple heuristic: 1 token ≈ 4 characters
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        return len(text) // 4
    
    def _chars_from_tokens(self, num_tokens: int) -> int:
        """
        Convert token count to approximate character count
        
        Args:
            num_tokens: Number of tokens
            
        Returns:
            Estimated character count
        """
        return num_tokens * 4
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using simple sliding window
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Convert token sizes to character estimates
        chunk_size_chars = self._chars_from_tokens(self.chunk_size)
        overlap_chars = self._chars_from_tokens(self.chunk_overlap)
        step_size = chunk_size_chars - overlap_chars
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position for this chunk
            end = min(start + chunk_size_chars, text_length)
            
            # Extract chunk
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
                token_count = self._estimate_tokens(chunk)
                logger.debug(f"Chunk {len(chunks)}: ~{token_count} tokens, {len(chunk)} chars")
            
            # Move to next chunk with overlap
            start += step_size
            
            # If we're at the end, stop
            if end >= text_length:
                break
        
        avg_tokens = self._estimate_tokens(text) / len(chunks) if chunks else 0
        logger.info(f"Split text into {len(chunks)} chunks (avg ~{avg_tokens:.0f} tokens)")
        return chunks
    
    def chunk_with_metadata(
        self,
        text: str,
        file_path: Any,
        **additional_metadata
    ) -> List[Dict[str, Any]]:
        """
        Chunk text and create metadata for each chunk
        
        Args:
            text: Text to chunk
            file_path: Path to source file or blob object
            **additional_metadata: Additional metadata to include
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        chunks = self.chunk_text(text)
        
        # Generate file hash based on input type
        if isinstance(file_path, Path):
            file_hash = get_file_hash(file_path)
        else:
            # For GCS blobs, use etag if available, otherwise generate hash from name and size
            if hasattr(file_path, 'etag'):
                file_hash = file_path.etag.strip('"')  # Remove quotes from etag
            elif hasattr(file_path, 'size') and hasattr(file_path, 'name'):
                # Generate hash from name and size
                import hashlib
                hash_input = f"{file_path.name}:{file_path.size}"
                file_hash = hashlib.md5(hash_input.encode()).hexdigest()
            else:
                # Fallback to name-based hash
                import hashlib
                file_hash = hashlib.md5(file_path.name.encode()).hexdigest()
        
        chunks_with_metadata = []
        for idx, chunk in enumerate(chunks):
            # Add estimated token count to metadata
            token_count = self._estimate_tokens(chunk)
            
            metadata = create_metadata(
                file_name=file_path.name,
                file_hash=file_hash,
                chunk_index=idx,
                total_chunks=len(chunks),
                token_count=token_count,
                **additional_metadata
            )
            
            chunks_with_metadata.append({
                'text': chunk,
                'metadata': metadata
            })
        
        return chunks_with_metadata


def chunk_text_simple(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[str]:
    """
    Convenience function to chunk text using token-based limits
    
    Args:
        text: Text to chunk
        chunk_size: Target number of tokens per chunk (default: 1800)
        chunk_overlap: Number of tokens to overlap between chunks (default: 100)
        
    Returns:
        List of text chunks
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_text(text)


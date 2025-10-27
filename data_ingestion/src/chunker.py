"""Text chunking module"""

import logging
from typing import List, Dict, Any
from pathlib import Path

from .config import Config
from .utils import create_metadata, get_file_hash


logger = logging.getLogger(__name__)


class TextChunker:
    """Class to split text into chunks for embedding"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk and we're in the middle of a word,
            # try to find a better breaking point
            if end < text_length:
                # Look for sentence ending
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    space = text.rfind(' ', start, end)
                    if space != -1 and space > start + self.chunk_size // 2:
                        end = space + 1
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        logger.info(f"Split text into {len(chunks)} chunks")
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
            metadata = create_metadata(
                file_name=file_path.name,
                file_hash=file_hash,
                chunk_index=idx,
                total_chunks=len(chunks),
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
    Convenience function to chunk text
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_text(text)

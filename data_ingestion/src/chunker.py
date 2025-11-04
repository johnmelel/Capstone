"""Text chunking module with token-aware chunking for Gemini embeddings"""

import logging
from typing import List, Dict, Any
from pathlib import Path

import google.generativeai as genai

from .config import Config
from .utils import create_metadata, get_file_hash


logger = logging.getLogger(__name__)


class TextChunker:
    """Class to split text into chunks for embedding with token-based limits"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, max_tokens: int = None):
        """
        Initialize text chunker with token-based limits
        
        Args:
            chunk_size: Target size of each chunk in TOKENS (not characters)
            chunk_overlap: Number of TOKENS to overlap between chunks
            max_tokens: Hard maximum number of tokens per chunk (for Gemini: 2048)
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.max_tokens = max_tokens or Config.MAX_TOKENS_PER_CHUNK
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        if self.chunk_size > self.max_tokens:
            raise ValueError(f"chunk_size ({self.chunk_size}) cannot exceed max_tokens ({self.max_tokens})")
        
        # Configure Gemini for token counting
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Using Google Gemini API for token-based chunking")
        except Exception as e:
            logger.warning(f"Could not initialize Gemini for token counting: {e}. Using estimation.")
            self.model = None
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Google's API
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.model:
            try:
                result = self.model.count_tokens(text)
                return result.total_tokens
            except Exception as e:
                logger.debug(f"Token counting failed: {e}. Using estimation.")
        
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def _find_chunk_end_by_tokens(self, text: str, start: int, target_tokens: int) -> tuple[int, int]:
        """
        Find the end position in text that fits within target token count
        
        Args:
            text: Full text to chunk
            start: Starting position in text
            target_tokens: Target number of tokens for this chunk
            
        Returns:
            Tuple of (end_position, actual_token_count)
        """
        # Estimate initial character length (average 4 chars per token)
        estimated_chars = target_tokens * 4
        end = min(start + estimated_chars, len(text))
        
        # Extract initial chunk
        chunk = text[start:end]
        token_count = self._count_tokens(chunk)
        
        # Binary search approach for efficiency
        if token_count > target_tokens:
            # Chunk too big, reduce it
            while token_count > target_tokens and end > start + 100:
                # Reduce by estimated overage
                overage_ratio = token_count / target_tokens
                end = start + int((end - start) / overage_ratio)
                chunk = text[start:end]
                token_count = self._count_tokens(chunk)
        else:
            # Chunk might be too small, try to expand it
            while token_count < target_tokens and end < len(text):
                # Estimate how much more we can add
                remaining_tokens = target_tokens - token_count
                additional_chars = remaining_tokens * 4
                new_end = min(end + additional_chars, len(text))
                
                new_chunk = text[start:new_end]
                new_token_count = self._count_tokens(new_chunk)
                
                if new_token_count <= target_tokens:
                    end = new_end
                    chunk = new_chunk
                    token_count = new_token_count
                    
                    # If we didn't grow much, stop trying
                    if new_end == end or new_end >= len(text):
                        break
                else:
                    # Would exceed limit, stop here
                    break
        
        # Try to end at a natural boundary (sentence or word)
        if end < len(text):
            # Look for sentence boundary in the last 20% of chunk
            search_start = max(start, end - len(chunk) // 5)
            sentence_end = text.rfind('.', search_start, end)
            if sentence_end != -1 and sentence_end > start:
                test_chunk = text[start:sentence_end + 1]
                test_tokens = self._count_tokens(test_chunk)
                if test_tokens <= target_tokens:
                    end = sentence_end + 1
                    token_count = test_tokens
            else:
                # Look for word boundary
                space = text.rfind(' ', search_start, end)
                if space != -1 and space > start:
                    test_chunk = text[start:space]
                    test_tokens = self._count_tokens(test_chunk)
                    if test_tokens <= target_tokens:
                        end = space
                        token_count = test_tokens
        
        return end, token_count
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using token-based limits
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks (all within token limit)
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        total_tokens_processed = 0
        
        while start < text_length:
            # Find chunk end based on target token count
            end, token_count = self._find_chunk_end_by_tokens(
                text, start, self.chunk_size
            )
            
            # Extract and clean chunk
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
                total_tokens_processed += token_count
                logger.debug(f"Chunk {len(chunks)}: {token_count} tokens, {len(chunk)} chars")
                
                # Calculate overlap in characters for next chunk
                # We need to go back by approximately chunk_overlap tokens
                if end < text_length:
                    # Estimate character position for overlap
                    overlap_chars = self.chunk_overlap * 4  # Rough estimate
                    
                    # Find actual position that gives us the desired token overlap
                    overlap_start = max(start, end - overlap_chars)
                    test_chunk = text[overlap_start:end]
                    overlap_tokens = self._count_tokens(test_chunk)
                    
                    # Adjust if needed
                    while overlap_tokens < self.chunk_overlap and overlap_start > start:
                        overlap_start = max(start, overlap_start - 100)
                        test_chunk = text[overlap_start:end]
                        overlap_tokens = self._count_tokens(test_chunk)
                    
                    while overlap_tokens > self.chunk_overlap and overlap_start < end - 100:
                        overlap_start = min(end - 100, overlap_start + 100)
                        test_chunk = text[overlap_start:end]
                        overlap_tokens = self._count_tokens(test_chunk)
                    
                    start = overlap_start
                else:
                    start = end
            else:
                # Empty chunk, move forward to avoid infinite loop
                start = end
            
            # Safety check: ensure forward progress
            if start >= text_length or (len(chunks) > 0 and start <= end - text_length // 1000):
                break
        
        avg_tokens = total_tokens_processed / len(chunks) if chunks else 0
        logger.info(f"Split text into {len(chunks)} chunks (avg {avg_tokens:.0f} tokens, max {self.max_tokens})")
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
            # Add token count to metadata
            token_count = self._count_tokens(chunk)
            
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


"""Text chunking module with simple token-based chunking"""

import logging
import json
from typing import List, Dict, Any
from pathlib import Path

from .config import Config
from .constants import CHARS_PER_TOKEN_ESTIMATE
from .utils import create_metadata, get_file_hash
from .types import ImageData


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


class MultimodalChunker(TextChunker):
    """Class to chunk text and associate images with chunks"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize multimodal chunker
        
        Args:
            chunk_size: Target size of each chunk in TOKENS
            chunk_overlap: Number of TOKENS to overlap between chunks
        """
        super().__init__(chunk_size, chunk_overlap)
        logger.info("MultimodalChunker initialized for text + image processing")
    
    def _associate_images_with_chunks(
        self,
        chunks: List[str],
        images: List[ImageData]
    ) -> List[List[ImageData]]:
        """
        Associate images with text chunks based on page proximity
        
        Strategy: Distribute images evenly across chunks to avoid memory issues
        - Prefer page-based distribution if page info is available
        - Otherwise, distribute images evenly across all chunks
        - Never put all images in a single chunk
        
        Args:
            chunks: List of text chunks
            images: List of ImageData objects with page_num info
            
        Returns:
            List of image lists, one per chunk
        """
        if not images:
            return [[] for _ in chunks]
        
        num_chunks = len(chunks)
        chunk_images = [[] for _ in chunks]
        
        # Check if we have page number information
        has_page_info = any(img.get('page_num', 0) > 0 for img in images)
        
        if not has_page_info or num_chunks == 1:
            # No page info: distribute images evenly across chunks
            # This prevents putting all images in one chunk
            if num_chunks == 1:
                chunk_images[0] = images
                logger.debug(f"Single chunk: assigned all {len(images)} images")
            else:
                # Round-robin distribution
                for idx, img in enumerate(images):
                    chunk_idx = idx % num_chunks
                    chunk_images[chunk_idx].append(img)
                logger.info(
                    f"Distributed {len(images)} images evenly across {num_chunks} chunks "
                    f"(no page info available)"
                )
        else:
            # Distribute images based on page numbers
            # Simple approach: divide page range into chunk ranges
            max_page = max(img.get('page_num', 1) for img in images)
            pages_per_chunk = max(1, max_page / num_chunks)
            
            for img in images:
                page_num = img.get('page_num', 1)
                # Calculate which chunk this page belongs to
                chunk_idx = min(int((page_num - 1) / pages_per_chunk), num_chunks - 1)
                chunk_images[chunk_idx].append(img)
                logger.debug(f"Assigned image from page {page_num} to chunk {chunk_idx}")
        
        return chunk_images
    
    def chunk_with_images(
        self,
        text: str,
        images: List[ImageData],
        file_path: Any,
        **additional_metadata
    ) -> List[Dict[str, Any]]:
        """
        Chunk text and associate images with chunks
        
        Args:
            text: Text to chunk
            images: List of ImageData objects
            file_path: Path to source file or blob object
            **additional_metadata: Additional metadata to include
            
        Returns:
            List of dictionaries containing chunk text, images, and metadata
        """
        # Get text chunks
        text_chunks = self.chunk_text(text)
        
        # Associate images with chunks
        chunk_images_lists = self._associate_images_with_chunks(text_chunks, images)
        
        # Generate file hash
        if isinstance(file_path, Path):
            file_hash = get_file_hash(file_path)
        else:
            if hasattr(file_path, 'etag'):
                file_hash = file_path.etag.strip('"')
            elif hasattr(file_path, 'size') and hasattr(file_path, 'name'):
                import hashlib
                hash_input = f"{file_path.name}:{file_path.size}"
                file_hash = hashlib.md5(hash_input.encode()).hexdigest()
            else:
                import hashlib
                file_hash = hashlib.md5(file_path.name.encode()).hexdigest()
        
        # Create chunks with metadata
        chunks_with_metadata = []
        for idx, (chunk_text, chunk_images) in enumerate(zip(text_chunks, chunk_images_lists)):
            token_count = self._estimate_tokens(chunk_text)
            has_image = len(chunk_images) > 0
            embedding_type = "multimodal" if has_image else "text"
            
            # Basic metadata
            metadata = create_metadata(
                file_name=file_path.name,
                file_hash=file_hash,
                chunk_index=idx,
                total_chunks=len(text_chunks),
                token_count=token_count,
                **additional_metadata
            )
            
            # Add multimodal fields
            metadata['has_image'] = has_image
            metadata['image_count'] = len(chunk_images)
            metadata['embedding_type'] = embedding_type
            metadata['image_gcs_paths'] = '[]'  # Will be set after GCS upload
            
            # Create image metadata JSON
            image_metadata_list = []
            for img in chunk_images:
                img_meta = {
                    'page_num': img.get('page_num', 0),
                    'image_index': img.get('image_index', 0),
                    'size': img.get('size', (0, 0)),
                    'bbox': img.get('bbox')
                }
                image_metadata_list.append(img_meta)
            
            metadata['image_metadata'] = json.dumps(image_metadata_list)
            
            chunk_data = {
                'text': chunk_text,
                'images': chunk_images,  # Keep ImageData objects for processing
                'metadata': metadata
            }
            
            chunks_with_metadata.append(chunk_data)
            
            logger.debug(
                f"Chunk {idx}: {token_count} tokens, {len(chunk_images)} images, "
                f"type={embedding_type}"
            )
        
        logger.info(
            f"Created {len(chunks_with_metadata)} chunks: "
            f"{sum(1 for c in chunks_with_metadata if c['metadata']['has_image'])} with images, "
            f"{sum(1 for c in chunks_with_metadata if not c['metadata']['has_image'])} text-only"
        )
        
        return chunks_with_metadata



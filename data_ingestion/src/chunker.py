"""Text chunking module with recursive splitting and page awareness"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .config import Config
from .utils import create_metadata, get_file_hash
from .types import ImageData, PageData


logger = logging.getLogger(__name__)


class TextChunker:
    """Base class for text chunking"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
    def chunk_text(self, text_or_pages: Union[str, List[PageData]]) -> List[Dict[str, Any]]:
        """
        Chunk text or pages.
        Returns list of dicts: {'text': str, 'page_num': int}
        """
        raise NotImplementedError


class RecursiveTokenChunker(TextChunker):
    """
    Chunker that splits text recursively by separators (paragraphs, sentences, etc.)
    while respecting token limits.
    """
    
    def __init__(self, model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", chunk_size: int = None, chunk_overlap: int = None):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = ["\n\n", "\n", ". ", " ", ""]
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info(f"RecursiveTokenChunker initialized with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load tokenizer {model_name}: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
            logger.warning("Transformers not available, using character-based approximation")

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            # verbose=False to suppress token length warnings
            # truncation=False to ensure we count ALL tokens, not just up to model max length
            return len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False))
        else:
            return len(text) // 4  # Rough approximation

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text into chunks that fit within chunk_size.
        """
        final_chunks = []
        
        # If text fits, return it
        if self._count_tokens(text) <= self.chunk_size:
            return [text]
        
        # If no separators left, force split
        if not separators:
            # Hard split by tokens
            if self.tokenizer:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                chunks = []
                for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                    chunk_tokens = tokens[i:i + self.chunk_size]
                    chunks.append(self.tokenizer.decode(chunk_tokens, skip_special_tokens=True))
                return chunks
            else:
                # Hard split by chars
                char_limit = self.chunk_size * 4
                return [text[i:i+char_limit] for i in range(0, len(text), char_limit)]

        # Use current separator
        separator = separators[0]
        next_separators = separators[1:]
        
        if separator == "":
            splits = list(text) # Split by character
        else:
            splits = text.split(separator)
            
        # Merge splits into chunks
        current_chunk = []
        current_len = 0
        
        for split in splits:
            split_len = self._count_tokens(split)
            
            if current_len + split_len + (1 if separator else 0) > self.chunk_size:
                # Current chunk is full, save it
                if current_chunk:
                    doc = separator.join(current_chunk)
                    if self._count_tokens(doc) > self.chunk_size:
                        # If even one split is too big, recurse on it
                        if len(current_chunk) == 1:
                             final_chunks.extend(self._split_text(doc, next_separators))
                        else:
                             # This shouldn't happen often if logic is right, but safety net
                             final_chunks.append(doc) 
                    else:
                        final_chunks.append(doc)
                    
                    # Apply overlap (simplified: just keep last item if it fits)
                    # Implementing proper overlap in recursive split is tricky.
                    # For now, we reset.
                    current_chunk = []
                    current_len = 0
            
            current_chunk.append(split)
            current_len += split_len
            
        if current_chunk:
            final_chunks.append(separator.join(current_chunk))
            
        return final_chunks

    def chunk_text(self, text_or_pages: Union[str, List[PageData]]) -> List[Dict[str, Any]]:
        chunks_with_page = []
        
        if isinstance(text_or_pages, str):
            # Legacy mode: single string, unknown page (assume 1)
            raw_chunks = self._split_text(text_or_pages, self.separators)
            for chunk in raw_chunks:
                if chunk.strip():
                    chunks_with_page.append({'text': chunk.strip(), 'page_num': 1})
        else:
            # Page-aware mode
            for page in text_or_pages:
                page_text = page['text']
                page_num = page['page_num']
                
                raw_chunks = self._split_text(page_text, self.separators)
                for chunk in raw_chunks:
                    if chunk.strip():
                        chunks_with_page.append({'text': chunk.strip(), 'page_num': page_num})
                        
        return chunks_with_page


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
            
        # Case 2: Chunk caption
        # We pass caption as string, so page_num defaults to 1 (irrelevant for image chunks usually)
        caption_chunks_dicts = self.text_chunker.chunk_text(caption)
        
        if not caption_chunks_dicts:
            return [(image, None)]
            
        # Case 3: Caption split into multiple chunks
        # Duplicate image for each caption chunk
        return [(image, chunk_dict['text']) for chunk_dict in caption_chunks_dicts]


def chunk_with_metadata(
    text_or_pages: Union[str, List[PageData]],
    file_path: Any,
    chunker: TextChunker,
    **additional_metadata
) -> List[Dict[str, Any]]:
    """
    Chunk text and create metadata for each chunk
    """
    chunks_data = chunker.chunk_text(text_or_pages)
    
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
    for idx, item in enumerate(chunks_data):
        chunk_text = item['text']
        page_num = item['page_num']
        
        metadata = create_metadata(
            file_name=file_path.name,
            file_hash=file_hash,
            chunk_index=idx,
            page_num=page_num,
            **additional_metadata
        )
        
        chunks_with_metadata.append({
            'text': chunk_text,
            'metadata': metadata
        })
    
    return chunks_with_metadata

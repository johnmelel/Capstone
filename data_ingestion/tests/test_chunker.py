"""Tests for text chunker"""

import pytest
from pathlib import Path
from src.chunker import TextChunker, chunk_text_simple


class TestTextChunker:
    """Test TextChunker class"""
    
    def test_initialization(self):
        """Test chunker initialization"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 10
    
    def test_initialization_invalid_overlap(self):
        """Test that overlap must be less than chunk size"""
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)
    
    def test_chunk_empty_text(self):
        """Test chunking empty text"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_text("")
        assert chunks == []
    
    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk size"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a short text."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_long_text(self):
        """Test chunking long text with token-based chunking"""
        text = "This is a long text. " * 50  # Make it long enough for multiple chunks
        chunker = TextChunker(chunk_size=30, chunk_overlap=5)  # 30 tokens per chunk, 5 tokens overlap
        chunks = chunker.chunk_text(text)
        # Should create multiple chunks if total tokens > chunk_size
        assert len(chunks) > 1
        # Each chunk should be roughly the chunk size in tokens
        for chunk in chunks[:-1]:  # All but last
            token_count = chunker._count_tokens(chunk)
            assert token_count <= chunker.chunk_size + 5  # Some flexibility
    
    def test_chunk_with_sentence_boundaries(self):
        """Test that chunker respects sentence boundaries"""
        chunker = TextChunker(chunk_size=50, chunk_overlap=5)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk_text(text)
        
        # Should break at sentence boundaries when possible
        assert all(chunk.strip() for chunk in chunks)
    
    def test_chunk_with_metadata(self, tmp_path):
        """Test chunking with metadata"""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "This is test text. " * 10
        
        # Create a temporary file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy content")
        
        chunks_with_metadata = chunker.chunk_with_metadata(
            text=text,
            file_path=pdf_path,
            extra_field="extra_value"
        )
        
        assert len(chunks_with_metadata) > 0
        
        # Check first chunk
        first_chunk = chunks_with_metadata[0]
        assert 'text' in first_chunk
        assert 'metadata' in first_chunk
        
        # Check metadata fields
        metadata = first_chunk['metadata']
        assert metadata['file_name'] == 'test.pdf'
        assert 'file_hash' in metadata
        assert metadata['chunk_index'] == 0
        assert 'total_chunks' in metadata
        assert metadata['extra_field'] == 'extra_value'
    
    def test_chunk_indices(self, tmp_path):
        """Test that chunk indices are correct"""
        chunker = TextChunker(chunk_size=30, chunk_overlap=5)
        text = "Word " * 50  # Create text that will be chunked
        
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")
        
        chunks_with_metadata = chunker.chunk_with_metadata(
            text=text,
            file_path=pdf_path
        )
        
        total_chunks = len(chunks_with_metadata)
        
        for i, chunk_data in enumerate(chunks_with_metadata):
            assert chunk_data['metadata']['chunk_index'] == i
            assert chunk_data['metadata']['total_chunks'] == total_chunks
    
    def test_convenience_function(self):
        """Test convenience function"""
        text = "This is a test text. " * 5
        chunks = chunk_text_simple(text, chunk_size=50, chunk_overlap=10)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

"""Tests for text embedder using the new google-genai SDK"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.embedder import TextEmbedder, embed_texts


@pytest.fixture
def mock_genai_client():
    """Mock Gemini API client for the new SDK"""
    with patch('src.embedder.genai.Client') as mock_client_class:
        # Mock client instance
        mock_client = MagicMock()
        
        # Mock embed_content response
        mock_embedding_response = Mock()
        mock_embedding = Mock()
        mock_embedding.values = list(np.random.rand(768))
        mock_embedding_response.embeddings = [mock_embedding]
        
        mock_client.models.embed_content.return_value = mock_embedding_response
        
        mock_client_class.return_value = mock_client
        
        yield mock_client_class


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for the new SDK"""
    with patch('src.embedder.genai.LocalTokenizer', create=True) as mock_tokenizer_class:
        mock_tokenizer_instance = Mock()
        mock_token_result = Mock()
        mock_token_result.token_count = 10
        mock_tokenizer_instance.compute_tokens.return_value = mock_token_result
        mock_tokenizer_class.return_value = mock_tokenizer_instance
        
        yield mock_tokenizer_class


class TestTextEmbedder:
    """Test TextEmbedder class with new SDK"""
    
    def test_initialization(self, mock_genai_client, mock_tokenizer):
        """Test embedder initialization with new SDK"""
        embedder = TextEmbedder(model_name="text-embedding-004", api_key="test-key")
        
        assert embedder.model_name == "text-embedding-004"
        assert embedder.embedding_dim == 768
        mock_genai_client.assert_called_once_with(api_key="test-key")
    
    def test_embed_single_text(self, mock_genai_client, mock_tokenizer):
        """Test embedding single text"""
        embedder = TextEmbedder()
        
        text = "This is a test sentence."
        embedding = embedder.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 1
        assert embedding.shape[1] == 768
    
    def test_embed_multiple_texts(self, mock_genai_client, mock_tokenizer):
        """Test embedding multiple texts"""
        embedder = TextEmbedder()
        
        texts = ["First text", "Second text", "Third text"]
        embeddings = embedder.embed_text(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 768
        
        # With new SDK, we call embed_content for each text
        assert embedder.client.models.embed_content.call_count == 3
    
    def test_embed_batch(self, mock_genai_client, mock_tokenizer):
        """Test batch embedding"""
        embedder = TextEmbedder()
        
        # Create list of texts
        texts = [f"Text {i}" for i in range(25)]
        
        embeddings = embedder.embed_batch(texts, batch_size=10)
        
        assert len(embeddings) == 25
        # Should have called embed_content 25 times
        assert embedder.client.models.embed_content.call_count == 25
    
    def test_get_embedding_dimension(self, mock_genai_client, mock_tokenizer):
        """Test getting embedding dimension"""
        embedder = TextEmbedder()
        
        dim = embedder.get_embedding_dimension()
        assert dim == 768
    
    def test_convenience_function(self, mock_genai_client, mock_tokenizer):
        """Test convenience function"""
        texts = ["Text 1", "Text 2"]
        embeddings = embed_texts(texts, model_name="test-model")
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 768)
    
    def test_embed_empty_list(self, mock_genai_client, mock_tokenizer):
        """Test embedding empty list"""
        embedder = TextEmbedder()
        
        embeddings = embedder.embed_text([])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, 768)
    
    def test_token_counting(self, mock_genai_client, mock_tokenizer):
        """Test token counting functionality"""
        embedder = TextEmbedder()
        
        text = "This is a test sentence."
        token_count = embedder._count_tokens(text)
        
        assert token_count == 10  # From our mock
        mock_tokenizer.return_value.compute_tokens.assert_called_once_with(text)
    
    def test_text_truncation(self, mock_genai_client, mock_tokenizer):
        """Test text truncation when exceeding token limit"""
        embedder = TextEmbedder()
        embedder.max_tokens = 5  # Set a low limit for testing
        
        long_text = "This is a very long text that exceeds the token limit."
        truncated = embedder._truncate_text(long_text)
        
        assert len(truncated) < len(long_text)
        assert embedder._count_tokens(truncated) <= embedder.max_tokens

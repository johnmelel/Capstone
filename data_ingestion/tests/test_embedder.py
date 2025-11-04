"""Tests for text embedder"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.embedder import TextEmbedder, embed_texts


@pytest.fixture
def mock_gemini_api():
    """Mock Gemini API calls"""
    with patch('src.embedder.genai') as mock_genai:
        # Mock configure
        mock_genai.configure = Mock()
        
        # Mock embed_content
        mock_genai.embed_content.return_value = {'embedding': list(np.random.rand(768))}
        
        # Mock GenerativeModel and count_tokens
        mock_model_instance = Mock()
        mock_token_result = Mock()
        mock_token_result.total_tokens = 10
        mock_model_instance.count_tokens.return_value = mock_token_result
        mock_genai.GenerativeModel.return_value = mock_model_instance
        
        yield mock_genai


class TestTextEmbedder:
    """Test TextEmbedder class"""
    
    def test_initialization(self, mock_gemini_api):
        """Test embedder initialization"""
        embedder = TextEmbedder(model_name="text-embedding-004", api_key="test-key")
        
        assert embedder.model_name == "text-embedding-004"
        assert embedder.embedding_dim == 768
        mock_gemini_api.configure.assert_called_with(api_key="test-key")
    
    def test_embed_single_text(self, mock_gemini_api):
        """Test embedding single text"""
        embedder = TextEmbedder()
        
        text = "This is a test sentence."
        embedding = embedder.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 1
        assert embedding.shape[1] == 768
    
    def test_embed_multiple_texts(self, mock_gemini_api):
        """Test embedding multiple texts"""
        embedder = TextEmbedder()
        
        texts = ["First text", "Second text", "Third text"]
        embeddings = embedder.embed_text(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 768
        assert mock_gemini_api.embed_content.call_count == 3
    
    def test_embed_batch(self, mock_gemini_api):
        """Test batch embedding"""
        embedder = TextEmbedder()
        
        # Create list of texts
        texts = [f"Text {i}" for i in range(25)]
        
        embeddings = embedder.embed_batch(texts, batch_size=10)
        
        assert len(embeddings) == 25
        # Should have called embed_content 25 times
        assert mock_gemini_api.embed_content.call_count == 25
    
    def test_get_embedding_dimension(self, mock_gemini_api):
        """Test getting embedding dimension"""
        embedder = TextEmbedder()
        
        dim = embedder.get_embedding_dimension()
        assert dim == 768
    
    def test_convenience_function(self, mock_gemini_api):
        """Test convenience function"""
        texts = ["Text 1", "Text 2"]
        embeddings = embed_texts(texts, model_name="test-model")
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 768)
    
    def test_embed_empty_list(self, mock_gemini_api):
        """Test embedding empty list"""
        embedder = TextEmbedder()
        
        embeddings = embedder.embed_text([])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, 768)

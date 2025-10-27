"""Tests for text embedder"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.embedder import TextEmbedder, embed_texts


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model"""
    with patch('src.embedder.SentenceTransformer') as mock_st:
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(2, 384)
        mock_st.return_value = mock_model
        yield mock_model


class TestTextEmbedder:
    """Test TextEmbedder class"""
    
    def test_initialization(self, mock_sentence_transformer):
        """Test embedder initialization"""
        embedder = TextEmbedder(model_name="test-model")
        
        assert embedder.model_name == "test-model"
        assert embedder.embedding_dim == 384
    
    def test_embed_single_text(self, mock_sentence_transformer):
        """Test embedding single text"""
        embedder = TextEmbedder()
        
        # Mock encode to return single embedding
        mock_sentence_transformer.encode.return_value = np.random.rand(1, 384)
        
        text = "This is a test sentence."
        embedding = embedder.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 1
        assert embedding.shape[1] == 384
    
    def test_embed_multiple_texts(self, mock_sentence_transformer):
        """Test embedding multiple texts"""
        embedder = TextEmbedder()
        
        # Mock encode to return multiple embeddings
        texts = ["First text", "Second text", "Third text"]
        mock_sentence_transformer.encode.return_value = np.random.rand(3, 384)
        
        embeddings = embedder.embed_text(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 384
    
    def test_embed_batch(self, mock_sentence_transformer):
        """Test batch embedding"""
        embedder = TextEmbedder()
        
        # Create list of texts
        texts = [f"Text {i}" for i in range(25)]
        
        # Mock encode to return appropriate sized arrays
        def mock_encode(batch, **kwargs):
            return np.random.rand(len(batch), 384)
        
        mock_sentence_transformer.encode.side_effect = mock_encode
        
        embeddings = embedder.embed_batch(texts, batch_size=10)
        
        assert len(embeddings) == 25
        # Should have called encode 3 times (10, 10, 5)
        assert mock_sentence_transformer.encode.call_count >= 3
    
    def test_get_embedding_dimension(self, mock_sentence_transformer):
        """Test getting embedding dimension"""
        embedder = TextEmbedder()
        
        dim = embedder.get_embedding_dimension()
        assert dim == 384
    
    def test_convenience_function(self, mock_sentence_transformer):
        """Test convenience function"""
        mock_sentence_transformer.encode.return_value = np.random.rand(2, 384)
        
        texts = ["Text 1", "Text 2"]
        embeddings = embed_texts(texts, model_name="test-model")
        
        assert isinstance(embeddings, np.ndarray)
    
    def test_embed_empty_list(self, mock_sentence_transformer):
        """Test embedding empty list"""
        embedder = TextEmbedder()
        mock_sentence_transformer.encode.return_value = np.array([]).reshape(0, 384)
        
        embeddings = embedder.embed_text([])
        assert isinstance(embeddings, np.ndarray)

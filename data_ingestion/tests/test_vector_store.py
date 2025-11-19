"""Tests for Milvus vector store"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pymilvus import DataType
from src.vector_store import MilvusVectorStore


@pytest.fixture
def mock_milvus():
    """Mock Milvus connections and collections"""
    with patch('src.vector_store.connections') as mock_conn, \
         patch('src.vector_store.Collection') as mock_coll, \
         patch('src.vector_store.utility') as mock_util:
        
        # Mock schema with proper structure
        mock_schema = MagicMock()
        mock_schema.auto_id = True
        mock_vector_field = MagicMock()
        mock_vector_field.name = "vector"
        mock_vector_field.dtype = DataType.FLOAT_VECTOR
        mock_vector_field.params = {"dim": 384}

        mock_schema.fields = [mock_vector_field]
        
        # Mock insert result
        mock_insert_result = MagicMock()
        mock_insert_result.primary_keys = ['id1', 'id2']  # Mock IDs
        
        # Mock collection
        mock_collection = MagicMock()
        mock_collection.num_entities = 100
        mock_collection.schema = mock_schema
        mock_collection.indexes = []
        mock_collection.insert.return_value = mock_insert_result
        mock_coll.return_value = mock_collection
        
        # Mock utility
        mock_util.has_collection.return_value = False
        
        yield {
            'connections': mock_conn,
            'Collection': mock_coll,
            'utility': mock_util,
            'collection': mock_collection
        }


class TestMilvusVectorStore:
    """Test MilvusVectorStore class"""
    
    def test_initialization(self, mock_milvus):
        """Test vector store initialization"""
        store = MilvusVectorStore(
            uri="http://localhost:19530",
            api_key="test_key",
            collection_name="test_collection",
            embedding_dim=384
        )
        
        assert store.collection_name == "test_collection"
        assert store.embedding_dim == 384
        mock_milvus['connections'].connect.assert_called_once()
    
    def test_create_new_collection(self, mock_milvus):
        """Test creating a new collection"""
        mock_milvus['utility'].has_collection.return_value = False
        
        store = MilvusVectorStore(
            uri="http://localhost:19530",
            api_key="test_key",
            collection_name="new_collection",
            embedding_dim=384
        )
        
        # Should create new collection and index
        mock_milvus['collection'].create_index.assert_called_once()
    
    def test_load_existing_collection(self, mock_milvus):
        """Test loading existing collection"""
        mock_milvus['utility'].has_collection.return_value = True
        
        store = MilvusVectorStore(
            uri="http://localhost:19530",
            api_key="test_key",
            collection_name="existing_collection",
            embedding_dim=384
        )
        
        # Should load existing collection
        mock_milvus['collection'].load.assert_called()

    def test_existing_collection_dimension_mismatch(self, mock_milvus):
        """Ensure mismatch between schema and embedder raises error"""
        mock_milvus['utility'].has_collection.return_value = True
        mock_milvus['collection'].schema.fields[0].params = {"dim": 1024}

        with pytest.raises(ValueError, match="vector dimension"):
            MilvusVectorStore(
                uri="http://localhost:19530",
                api_key="test_key",
                collection_name="mismatch_collection",
                embedding_dim=384
            )

    def test_existing_collection_metric_override(self, mock_milvus):
        """Ensure store adopts collection metric when different from config"""
        mock_milvus['utility'].has_collection.return_value = True
        mock_index = MagicMock()
        mock_index.field_name = "vector"
        mock_index.params = {"metric_type": "L2"}
        mock_milvus['collection'].indexes = [mock_index]

        store = MilvusVectorStore(
            uri="http://localhost:19530",
            api_key="test_key",
            collection_name="metric_collection",
            embedding_dim=384,
            metric_type="COSINE",
        )

        assert store.metric_type == "L2"
    
    def test_insert_data(self, mock_milvus):
        """Test inserting data"""
        store = MilvusVectorStore(
            uri="http://localhost:19530",
            api_key="test_key",
            collection_name="test_collection",
            embedding_dim=384
        )
        
        embeddings = [[0.1] * 384, [0.2] * 384]
        texts = ["Text 1", "Text 2"]
        metadatas = [
            {'file_name': 'test1.pdf', 'file_hash': 'hash1', 'chunk_index': 0, 'total_chunks': 2},
            {'file_name': 'test1.pdf', 'file_hash': 'hash1', 'chunk_index': 1, 'total_chunks': 2}
        ]
        
        ids = store.insert(embeddings, texts, metadatas)
        
        assert len(ids) == 2
        mock_milvus['collection'].insert.assert_called_once()
        mock_milvus['collection'].flush.assert_called()
    
    def test_insert_mismatched_lengths(self, mock_milvus):
        """Test insert with mismatched lengths raises error"""
        store = MilvusVectorStore(
            uri="http://localhost:19530",
            api_key="test_key",
            collection_name="test_collection",
            embedding_dim=384
        )
        
        embeddings = [[0.1] * 384]
        texts = ["Text 1", "Text 2"]  # Different length
        metadatas = [{'file_name': 'test.pdf', 'file_hash': 'hash', 'chunk_index': 0, 'total_chunks': 1}]
        
        with pytest.raises(ValueError):
            store.insert(embeddings, texts, metadatas)
    
    def test_search(self, mock_milvus):
        """Test searching"""
        # Mock search results
        mock_hit = Mock()
        mock_hit.id = "test_id"
        mock_hit.distance = 0.5
        mock_hit.entity.to_dict.return_value = {'text': 'Test text'}
        
        mock_hits = [mock_hit]
        mock_milvus['collection'].search.return_value = [mock_hits]
        
        store = MilvusVectorStore(
            uri="http://localhost:19530",
            api_key="test_key",
            collection_name="test_collection",
            embedding_dim=384
        )
        
        query_embedding = [0.1] * 384
        results = store.search(query_embedding, top_k=5)
        
        assert len(results) == 1
        assert results[0]['id'] == 'test_id'
        assert results[0]['distance'] == 0.5
    
    def test_delete_by_file(self, mock_milvus):
        """Test deleting by file hash"""
        mock_result = Mock()
        mock_result.delete_count = 10
        mock_milvus['collection'].delete.return_value = mock_result
        
        store = MilvusVectorStore(
            uri="http://localhost:19530",
            api_key="test_key",
            collection_name="test_collection",
            embedding_dim=384
        )
        
        count = store.delete_by_file("test_hash")
        
        assert count == 10
        mock_milvus['collection'].delete.assert_called_once()
    
    def test_get_stats(self, mock_milvus):
        """Test getting collection stats"""
        store = MilvusVectorStore(
            uri="http://localhost:19530",
            api_key="test_key",
            collection_name="test_collection",
            embedding_dim=384
        )
        
        stats = store.get_stats()
        
        assert 'collection_name' in stats
        assert stats['collection_name'] == 'test_collection'
        assert 'num_entities' in stats
    
    def test_close(self, mock_milvus):
        """Test closing connection"""
        store = MilvusVectorStore(
            uri="http://localhost:19530",
            api_key="test_key",
            collection_name="test_collection",
            embedding_dim=384
        )
        
        store.close()
        
        mock_milvus['connections'].disconnect.assert_called_once()

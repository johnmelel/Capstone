"""Tests for Google Cloud Storage downloader"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.gcs_downloader import GCSDownloader, download_pdfs_from_gcs


@pytest.fixture
def mock_service_account():
    """Mock service account credentials"""
    with patch('src.gcs_downloader.service_account') as mock_sa:
        mock_creds = Mock()
        mock_sa.Credentials.from_service_account_file.return_value = mock_creds
        yield mock_sa


@pytest.fixture
def mock_storage_client():
    """Mock Google Cloud Storage client"""
    with patch('src.gcs_downloader.storage.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = True
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        yield {
            'client_class': mock_client_class,
            'client': mock_client,
            'bucket': mock_bucket
        }


@pytest.fixture
def temp_download_dir(tmp_path):
    """Create temporary download directory"""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    return download_dir


class TestGCSDownloader:
    """Test GCSDownloader class"""
    
    def test_initialization(self, mock_service_account, mock_storage_client, temp_download_dir):
        """Test downloader initialization"""
        downloader = GCSDownloader(
            service_account_file="test_account.json",
            bucket_name="test-bucket",
            bucket_prefix="documents",
            download_dir=temp_download_dir,
            recursive=True
        )
        
        assert downloader.bucket_name == "test-bucket"
        assert downloader.bucket_prefix == "documents"
        assert downloader.download_dir == temp_download_dir
        assert downloader.recursive is True
    
    def test_list_pdf_blobs(self, mock_service_account, mock_storage_client, temp_download_dir):
        """Test listing PDF blobs"""
        # Mock blob objects
        mock_blob1 = Mock()
        mock_blob1.name = "document1.pdf"
        
        mock_blob2 = Mock()
        mock_blob2.name = "folder/document2.pdf"
        
        mock_blob3 = Mock()
        mock_blob3.name = "document.txt"  # Not a PDF
        
        mock_storage_client['bucket'].list_blobs.return_value = [
            mock_blob1, mock_blob2, mock_blob3
        ]
        
        downloader = GCSDownloader(
            service_account_file="test_account.json",
            bucket_name="test-bucket",
            bucket_prefix="",
            download_dir=temp_download_dir
        )
        
        pdf_blobs = downloader.list_pdf_blobs()
        
        assert len(pdf_blobs) == 2
        assert pdf_blobs[0].name == "document1.pdf"
        assert pdf_blobs[1].name == "folder/document2.pdf"
    
    def test_download_blob(self, mock_service_account, mock_storage_client, temp_download_dir):
        """Test downloading a single blob"""
        mock_blob = Mock()
        mock_blob.name = "test/document.pdf"
        mock_blob.download_to_filename = Mock()
        
        downloader = GCSDownloader(
            service_account_file="test_account.json",
            bucket_name="test-bucket",
            bucket_prefix="",
            download_dir=temp_download_dir
        )
        
        result = downloader.download_blob(mock_blob)
        
        assert result == temp_download_dir / "test" / "document.pdf"
        mock_blob.download_to_filename.assert_called_once()
    
    def test_download_blob_with_prefix(self, mock_service_account, mock_storage_client, temp_download_dir):
        """Test downloading blob with bucket prefix"""
        mock_blob = Mock()
        mock_blob.name = "documents/reports/test.pdf"
        mock_blob.download_to_filename = Mock()
        
        downloader = GCSDownloader(
            service_account_file="test_account.json",
            bucket_name="test-bucket",
            bucket_prefix="documents",
            download_dir=temp_download_dir
        )
        
        result = downloader.download_blob(mock_blob)
        
        # Should strip the prefix
        assert result == temp_download_dir / "reports" / "test.pdf"
    
    def test_download_all_pdfs(self, mock_service_account, mock_storage_client, temp_download_dir):
        """Test downloading all PDFs"""
        mock_blob1 = Mock()
        mock_blob1.name = "doc1.pdf"
        mock_blob1.download_to_filename = Mock()
        
        mock_blob2 = Mock()
        mock_blob2.name = "doc2.pdf"
        mock_blob2.download_to_filename = Mock()
        
        mock_storage_client['bucket'].list_blobs.return_value = [mock_blob1, mock_blob2]
        
        downloader = GCSDownloader(
            service_account_file="test_account.json",
            bucket_name="test-bucket",
            bucket_prefix="",
            download_dir=temp_download_dir
        )
        
        results = downloader.download_all_pdfs()
        
        assert len(results) == 2
        assert all(isinstance(path, Path) for path in results)
    
    def test_bucket_not_exists(self, mock_service_account, mock_storage_client, temp_download_dir):
        """Test error when bucket doesn't exist"""
        mock_storage_client['bucket'].exists.return_value = False
        
        with pytest.raises(ValueError, match="does not exist"):
            GCSDownloader(
                service_account_file="test_account.json",
                bucket_name="nonexistent-bucket",
                bucket_prefix="",
                download_dir=temp_download_dir
            )


def test_convenience_function(mock_service_account, mock_storage_client, tmp_path):
    """Test the convenience function"""
    with patch('src.gcs_downloader.Config') as mock_config:
        mock_config.GOOGLE_SERVICE_ACCOUNT_JSON = "test.json"
        mock_config.GCS_BUCKET_NAME = "test-bucket"
        mock_config.GCS_BUCKET_PREFIX = ""
        mock_config.DOWNLOAD_DIR = tmp_path
        mock_config.GCS_RECURSIVE = True
        
        mock_storage_client['bucket'].list_blobs.return_value = []
        
        result = download_pdfs_from_gcs()
        
        assert isinstance(result, list)

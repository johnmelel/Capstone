"""Tests for PDF text extractor with MinerU"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import shutil
from src.pdf_extractor import PDFExtractor, extract_text_from_pdf


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a mock PDF path"""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()
    return pdf_path


@pytest.fixture
def mock_mineru_available():
    """Mock MinerU availability"""
    with patch('src.pdf_extractor.MINERU_AVAILABLE', True):
        with patch('src.pdf_extractor.TORCH_AVAILABLE', True):
            yield


class TestPDFExtractor:
    """Test PDFExtractor class"""
    
    @patch('src.pdf_extractor.torch.cuda.is_available')
    @patch('src.pdf_extractor.torch.cuda.get_device_name')
    def test_initialization_with_gpu(self, mock_gpu_name, mock_cuda_available, mock_mineru_available):
        """Test extractor initialization with GPU"""
        mock_cuda_available.return_value = True
        mock_gpu_name.return_value = "Tesla T4"
        
        extractor = PDFExtractor(extract_images=True)
        assert extractor.extract_images is True
        assert extractor.use_gpu is True
    
    @patch('src.pdf_extractor.torch.cuda.is_available')
    def test_initialization_without_gpu(self, mock_cuda_available, mock_mineru_available):
        """Test extractor initialization without GPU"""
        mock_cuda_available.return_value = False
        
        extractor = PDFExtractor(extract_images=False)
        assert extractor.extract_images is False
        assert extractor.use_gpu is False
    
    @patch('src.pdf_extractor.Config')
    @patch('src.pdf_extractor.torch.cuda.is_available')
    def test_initialization_config_loading(self, mock_cuda_available, mock_config, mock_mineru_available):
        """Test configuration loading during initialization"""
        mock_cuda_available.return_value = False
        mock_config.MINERU_BACKEND = 'pipeline'
        mock_config.MINERU_MODEL_SOURCE = 'huggingface'
        mock_config.MINERU_LANG = 'en'
        mock_config.PDF_EXTRACTION_TIMEOUT = 900
        mock_config.MINERU_DEBUG_MODE = False
        mock_config.MINERU_ENABLE_TABLES = False
        mock_config.MINERU_ENABLE_FORMULAS = False
        
        extractor = PDFExtractor()
        assert extractor.backend == 'pipeline'
        assert extractor.model_source == 'huggingface'
        assert extractor.timeout == 900
    
    @patch('src.pdf_extractor.UNIPipe')
    @patch('src.pdf_extractor.DiskReaderWriter')
    @patch('src.pdf_extractor.torch.cuda.is_available')
    @patch('builtins.open', new_callable=mock_open, read_data=b'PDF_BYTES')
    @patch('src.pdf_extractor.tempfile.mkdtemp')
    def test_extract_text_success(self, mock_mkdtemp, mock_file_open, mock_cuda, 
                                  mock_disk_writer, mock_unipipe, sample_pdf_path, mock_mineru_available):
        """Test successful text extraction using MinerU"""
        mock_cuda.return_value = False
        
        # Mock temp directory
        temp_dir = Path(tempfile.gettempdir()) / "mineru_output_test"
        mock_mkdtemp.return_value = str(temp_dir)
        
        # Mock UNIPipe
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.jso_useful_key = {"_pdf_type": "text"}
        mock_pipe_instance.pipe_mk_markdown.return_value = "# Test Document\n\nThis is test content."
        mock_unipipe.return_value = mock_pipe_instance
        
        # Mock markdown file write
        with patch('builtins.open', mock_open()) as mock_md_file:
            extractor = PDFExtractor()
            text = extractor.extract_text(sample_pdf_path)
        
        assert text is not None
        assert "Test Document" in text or "test content" in text.lower()
    
    @patch('src.pdf_extractor.torch.cuda.is_available')
    def test_extract_text_file_not_found(self, mock_cuda, mock_mineru_available):
        """Test extraction with non-existent file"""
        mock_cuda.return_value = False
        
        extractor = PDFExtractor()
        text = extractor.extract_text(Path("nonexistent.pdf"))
        
        assert text is None
    
    @patch('src.pdf_extractor.torch.cuda.is_available')
    def test_extract_text_not_pdf(self, mock_cuda, tmp_path, mock_mineru_available):
        """Test extraction with non-PDF file"""
        mock_cuda.return_value = False
        
        not_pdf = tmp_path / "test.txt"
        not_pdf.touch()
        
        extractor = PDFExtractor()
        text = extractor.extract_text(not_pdf)
        
        assert text is None
    
    @patch('src.pdf_extractor.UNIPipe')
    @patch('src.pdf_extractor.DiskReaderWriter')
    @patch('src.pdf_extractor.torch.cuda.is_available')
    @patch('builtins.open', new_callable=mock_open, read_data=b'PDF_BYTES')
    @patch('src.pdf_extractor.tempfile.mkdtemp')
    def test_extract_with_metadata(self, mock_mkdtemp, mock_file_open, mock_cuda,
                                   mock_disk_writer, mock_unipipe, sample_pdf_path, mock_mineru_available):
        """Test extraction with metadata"""
        mock_cuda.return_value = False
        
        # Mock temp directory
        temp_dir = Path(tempfile.gettempdir()) / "mineru_output_test"
        mock_mkdtemp.return_value = str(temp_dir)
        
        # Mock UNIPipe
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.jso_useful_key = {"_pdf_type": "text"}
        mock_pipe_instance.pipe_mk_markdown.return_value = "Test content"
        mock_unipipe.return_value = mock_pipe_instance
        
        with patch('builtins.open', mock_open()):
            extractor = PDFExtractor()
            result = extractor.extract_with_metadata(sample_pdf_path)
        
        assert result is not None
        assert 'text' in result
        assert 'metadata' in result
        assert result['metadata']['file_name'] == 'test.pdf'
        assert result['metadata']['creator'] == 'MinerU'
    
    @patch('src.pdf_extractor.torch.cuda.is_available')
    def test_get_page_count(self, mock_cuda, sample_pdf_path, mock_mineru_available):
        """Test getting page count (returns 0 in Phase 1)"""
        mock_cuda.return_value = False
        
        extractor = PDFExtractor()
        count = extractor.get_page_count(sample_pdf_path)
        
        # Phase 1: page count not implemented
        assert count == 0
    
    @patch('src.pdf_extractor.UNIPipe')
    @patch('src.pdf_extractor.torch.cuda.is_available')
    @patch('builtins.open', new_callable=mock_open, read_data=b'PDF_BYTES')
    @patch('src.pdf_extractor.tempfile.mkdtemp')
    def test_extract_text_error_handling(self, mock_mkdtemp, mock_file_open, mock_cuda,
                                        mock_unipipe, sample_pdf_path, mock_mineru_available):
        """Test error handling during extraction"""
        mock_cuda.return_value = False
        mock_unipipe.side_effect = Exception("Processing failed")
        
        temp_dir = Path(tempfile.gettempdir()) / "mineru_output_test"
        mock_mkdtemp.return_value = str(temp_dir)
        
        extractor = PDFExtractor()
        text = extractor.extract_text(sample_pdf_path)
        
        assert text is None
    
    @patch('src.pdf_extractor.UNIPipe')
    @patch('src.pdf_extractor.DiskReaderWriter')
    @patch('src.pdf_extractor.torch.cuda.is_available')
    @patch('builtins.open', new_callable=mock_open, read_data=b'PDF_BYTES')
    @patch('src.pdf_extractor.tempfile.mkdtemp')
    @patch('src.pdf_extractor.shutil.rmtree')
    def test_temp_file_cleanup(self, mock_rmtree, mock_mkdtemp, mock_file_open, mock_cuda,
                               mock_disk_writer, mock_unipipe, sample_pdf_path, mock_mineru_available):
        """Test temporary file cleanup"""
        mock_cuda.return_value = False
        
        temp_dir = Path(tempfile.gettempdir()) / "mineru_output_test"
        mock_mkdtemp.return_value = str(temp_dir)
        
        # Mock UNIPipe
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.jso_useful_key = {"_pdf_type": "text"}
        mock_pipe_instance.pipe_mk_markdown.return_value = "Test content"
        mock_unipipe.return_value = mock_pipe_instance
        
        with patch('builtins.open', mock_open()):
            extractor = PDFExtractor()
            extractor.extract_text(sample_pdf_path)
        
        # Verify cleanup was called (debug mode is False by default)
        assert mock_rmtree.called
    
    @patch('src.pdf_extractor.Config')
    @patch('src.pdf_extractor.UNIPipe')
    @patch('src.pdf_extractor.DiskReaderWriter')
    @patch('src.pdf_extractor.torch.cuda.is_available')
    @patch('builtins.open', new_callable=mock_open, read_data=b'PDF_BYTES')
    @patch('src.pdf_extractor.tempfile.mkdtemp')
    @patch('src.pdf_extractor.shutil.rmtree')
    def test_debug_mode_no_cleanup(self, mock_rmtree, mock_mkdtemp, mock_file_open, mock_cuda,
                                   mock_disk_writer, mock_unipipe, mock_config, 
                                   sample_pdf_path, mock_mineru_available):
        """Test that debug mode preserves temporary files"""
        mock_cuda.return_value = False
        mock_config.MINERU_DEBUG_MODE = True
        mock_config.MINERU_BACKEND = 'pipeline'
        mock_config.MINERU_MODEL_SOURCE = 'huggingface'
        mock_config.MINERU_LANG = 'en'
        mock_config.PDF_EXTRACTION_TIMEOUT = 900
        mock_config.MINERU_ENABLE_TABLES = False
        mock_config.MINERU_ENABLE_FORMULAS = False
        
        temp_dir = Path(tempfile.gettempdir()) / "mineru_output_test"
        mock_mkdtemp.return_value = str(temp_dir)
        
        # Mock UNIPipe
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.jso_useful_key = {"_pdf_type": "text"}
        mock_pipe_instance.pipe_mk_markdown.return_value = "Test content"
        mock_unipipe.return_value = mock_pipe_instance
        
        with patch('builtins.open', mock_open()):
            extractor = PDFExtractor()
            extractor.debug_mode = True  # Override for test
            extractor.extract_text(sample_pdf_path)
        
        # Verify cleanup was NOT called in debug mode
        assert not mock_rmtree.called
    
    @patch('src.pdf_extractor.UNIPipe')
    @patch('src.pdf_extractor.DiskReaderWriter')
    @patch('src.pdf_extractor.torch.cuda.is_available')
    @patch('builtins.open', new_callable=mock_open, read_data=b'PDF_BYTES')
    @patch('src.pdf_extractor.tempfile.mkdtemp')
    def test_convenience_function(self, mock_mkdtemp, mock_file_open, mock_cuda,
                                  mock_disk_writer, mock_unipipe, sample_pdf_path, mock_mineru_available):
        """Test convenience function"""
        mock_cuda.return_value = False
        
        temp_dir = Path(tempfile.gettempdir()) / "mineru_output_test"
        mock_mkdtemp.return_value = str(temp_dir)
        
        # Mock UNIPipe
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.jso_useful_key = {"_pdf_type": "text"}
        mock_pipe_instance.pipe_mk_markdown.return_value = "Test content"
        mock_unipipe.return_value = mock_pipe_instance
        
        with patch('builtins.open', mock_open()):
            text = extract_text_from_pdf(sample_pdf_path)
        
        assert text is not None
        assert "Test content" in text or "test" in text.lower()


"""Tests for PDF text extractor"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.pdf_extractor import PDFExtractor, extract_text_from_pdf


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a mock PDF path"""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()
    return pdf_path


class TestPDFExtractor:
    """Test PDFExtractor class"""
    
    def test_initialization(self):
        """Test extractor initialization"""
        extractor = PDFExtractor(extract_images=True)
        assert extractor.extract_images is True
        
        extractor = PDFExtractor(extract_images=False)
        assert extractor.extract_images is False
    
    @patch('src.pdf_extractor.pdfplumber.open')
    def test_extract_text_success(self, mock_pdfplumber_open, sample_pdf_path):
        """Test successful text extraction using pdfplumber"""
        # Mock page objects
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "This is page 1 content"
        
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "This is page 2 content"
        
        # Mock document
        mock_doc = MagicMock()
        mock_doc.pages = [mock_page1, mock_page2]
        mock_doc.close = Mock()
        
        mock_pdfplumber_open.return_value = mock_doc
        
        extractor = PDFExtractor()
        text = extractor.extract_text(sample_pdf_path)
        
        assert text is not None
        assert "page 1 content" in text
        assert "page 2 content" in text
    
    @patch('src.pdf_extractor.pdfplumber.open')
    def test_extract_text_empty_pages(self, mock_pdfplumber_open, sample_pdf_path):
        """Test extraction with empty pages"""
        mock_page = Mock()
        mock_page.extract_text.return_value = ""
        
        mock_doc = MagicMock()
        mock_doc.pages = [mock_page]
        mock_doc.close = Mock()
        
        mock_pdfplumber_open.return_value = mock_doc
        
        extractor = PDFExtractor()
        text = extractor.extract_text(sample_pdf_path)
        
        # Should return None for empty text
        assert text is None
    
    def test_extract_text_file_not_found(self):
        """Test extraction with non-existent file"""
        extractor = PDFExtractor()
        text = extractor.extract_text(Path("nonexistent.pdf"))
        
        assert text is None
    
    def test_extract_text_not_pdf(self, tmp_path):
        """Test extraction with non-PDF file"""
        not_pdf = tmp_path / "test.txt"
        not_pdf.touch()
        
        extractor = PDFExtractor()
        text = extractor.extract_text(not_pdf)
        
        assert text is None
    
    @patch('src.pdf_extractor.pdfplumber.open')
    def test_extract_with_metadata(self, mock_pdfplumber_open, sample_pdf_path):
        """Test extraction with metadata"""
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test content"
        
        mock_metadata = {
            'Title': 'Test Document',
            'Author': 'Test Author',
            'Subject': 'Test Subject',
            'Creator': 'Test Creator',
            'Producer': 'Test Producer',
            'CreationDate': '2024-01-01',
            'ModDate': '2024-01-02'
        }
        
        mock_doc = MagicMock()
        mock_doc.pages = [mock_page]
        mock_doc.metadata = mock_metadata
        mock_doc.close = Mock()
        
        mock_pdfplumber_open.return_value = mock_doc
        
        extractor = PDFExtractor()
        result = extractor.extract_with_metadata(sample_pdf_path)
        
        assert result is not None
        assert 'text' in result
        assert 'metadata' in result
        assert result['metadata']['title'] == 'Test Document'
        assert result['metadata']['author'] == 'Test Author'
        assert result['metadata']['num_pages'] == 1
    
    @patch('src.pdf_extractor.pdfplumber.open')
    def test_get_page_count(self, mock_pdfplumber_open, sample_pdf_path):
        """Test getting page count"""
        mock_doc = MagicMock()
        mock_doc.pages = [Mock(), Mock(), Mock(), Mock(), Mock()]  # 5 pages
        mock_doc.close = Mock()
        
        mock_pdfplumber_open.return_value = mock_doc
        
        extractor = PDFExtractor()
        count = extractor.get_page_count(sample_pdf_path)
        
        assert count == 5
    
    @patch('src.pdf_extractor.pdfplumber.open')
    def test_extract_text_error_handling(self, mock_pdfplumber_open, sample_pdf_path):
        """Test error handling during extraction"""
        mock_pdfplumber_open.side_effect = Exception("PDF is corrupted")
        
        extractor = PDFExtractor()
        text = extractor.extract_text(sample_pdf_path)
        
        assert text is None
    
    @patch('src.pdf_extractor.pdfplumber.open')
    def test_convenience_function(self, mock_pdfplumber_open, sample_pdf_path):
        """Test convenience function"""
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test content"
        
        mock_doc = MagicMock()
        mock_doc.pages = [mock_page]
        mock_doc.close = Mock()
        
        mock_pdfplumber_open.return_value = mock_doc
        
        text = extract_text_from_pdf(sample_pdf_path)
        
        assert text is not None
        assert "Test content" in text


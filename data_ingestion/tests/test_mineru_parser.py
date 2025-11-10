"""Tests for MinerU PDF parser with Gemini Vision integration"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.mineru_parser import MinerUParser


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        'chunk_size': 800,
        'overlap': 150
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a mock PDF file"""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock content")
    return pdf_path


class TestMinerUParser:
    """Test MinerU parser class"""
    
    def test_initialization(self, mock_config, temp_output_dir):
        """Test parser initialization"""
        parser = MinerUParser(
            config=mock_config,
            output_dir=temp_output_dir,
            gemini_api_key="test-key",
            vision_model="gemini-2.0-flash-exp",
            image_prompt="Test prompt"
        )
        
        assert parser.output_dir == temp_output_dir
        assert parser.vision_model == "gemini-2.0-flash-exp"
        assert parser.image_prompt == "Test prompt"
        assert parser.chunk_size == 800
        assert parser.overlap == 150
        assert (temp_output_dir / "images").exists()
        assert (temp_output_dir / "temp").exists()
    
    def test_initialization_default_vision_model(self, mock_config, temp_output_dir):
        """Test parser initialization with default vision model"""
        parser = MinerUParser(
            config=mock_config,
            output_dir=temp_output_dir,
            gemini_api_key="test-key"
        )
        
        assert parser.vision_model == "gemini-2.0-flash-exp"
        assert "Describe this image in detail" in parser.image_prompt
    
    @patch('src.mineru_parser.subprocess.run')
    @patch('src.mineru_parser.Path.glob')
    def test_run_mineru_cli_success(self, mock_glob, mock_run, mock_config, temp_output_dir, sample_pdf_path):
        """Test successful MinerU CLI execution"""
        # Mock subprocess success
        mock_run.return_value = Mock(returncode=0, stdout="Success", stderr="")
        
        # Mock markdown file
        mock_md_file = Mock(spec=Path)
        mock_md_file.exists.return_value = True
        mock_md_file.read_text.return_value = "# Test Content\n\nThis is test markdown."
        mock_glob.return_value = [mock_md_file]
        
        parser = MinerUParser(mock_config, temp_output_dir, "test-key")
        result = parser._run_mineru_cli(sample_pdf_path)
        
        assert result == "# Test Content\n\nThis is test markdown."
        mock_run.assert_called_once()
    
    @patch('src.mineru_parser.subprocess.run')
    def test_run_mineru_cli_failure(self, mock_run, mock_config, temp_output_dir, sample_pdf_path):
        """Test MinerU CLI failure handling"""
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Error occurred")
        
        parser = MinerUParser(mock_config, temp_output_dir, "test-key")
        result = parser._run_mineru_cli(sample_pdf_path)
        
        assert result is None
    
    @patch('src.mineru_parser.subprocess.run')
    def test_run_mineru_cli_timeout(self, mock_run, mock_config, temp_output_dir, sample_pdf_path):
        """Test MinerU CLI timeout handling"""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=300)
        
        parser = MinerUParser(mock_config, temp_output_dir, "test-key")
        result = parser._run_mineru_cli(sample_pdf_path)
        
        assert result is None
    
    def test_process_text_chunking(self, mock_config, temp_output_dir):
        """Test text processing and chunking"""
        parser = MinerUParser(mock_config, temp_output_dir, "test-key")
        
        # Create text longer than chunk size
        test_text = "This is a test sentence. " * 100
        
        chunks = parser._process_text(test_text, "test.pdf")
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert 'text' in chunk
            assert 'metadata' in chunk
            assert chunk['metadata']['file_name'] == "test.pdf"
            assert chunk['metadata']['source'] == 'text'
            assert 'chunk_index' in chunk['metadata']
    
    def test_process_text_empty(self, mock_config, temp_output_dir):
        """Test processing empty text"""
        parser = MinerUParser(mock_config, temp_output_dir, "test-key")
        
        chunks = parser._process_text("", "test.pdf")
        
        assert len(chunks) == 0
    
    @patch('src.mineru_parser.MinerUParser._describe_image_with_gemini')
    def test_process_images(self, mock_describe, mock_config, temp_output_dir, tmp_path):
        """Test image processing"""
        # Create mock image files
        images_dir = temp_output_dir / "temp" / "test" / "images"
        images_dir.mkdir(parents=True)
        
        img1 = images_dir / "image_001.png"
        img2 = images_dir / "image_002.jpg"
        img1.write_bytes(b"fake image data")
        img2.write_bytes(b"fake image data")
        
        # Mock Gemini Vision descriptions
        mock_describe.side_effect = [
            "This is a medical chart showing patient data.",
            "This is a diagram of the human heart."
        ]
        
        parser = MinerUParser(mock_config, temp_output_dir, "test-key")
        chunks = parser._process_images(images_dir, "test.pdf")
        
        assert len(chunks) == 2
        assert chunks[0]['text'] == "This is a medical chart showing patient data."
        assert chunks[0]['metadata']['source'] == 'image'
        assert chunks[0]['metadata']['image_file'] == 'image_001.png'
        assert chunks[1]['text'] == "This is a diagram of the human heart."
        assert mock_describe.call_count == 2
    
    @patch('src.mineru_parser.MinerUParser._describe_image_with_gemini')
    def test_process_images_no_images(self, mock_describe, mock_config, temp_output_dir):
        """Test processing when no images exist"""
        images_dir = temp_output_dir / "temp" / "test" / "images"
        
        parser = MinerUParser(mock_config, temp_output_dir, "test-key")
        chunks = parser._process_images(images_dir, "test.pdf")
        
        assert len(chunks) == 0
        mock_describe.assert_not_called()
    
    @patch('src.mineru_parser.genai.Client')
    def test_describe_image_with_gemini_success(self, mock_client_class, mock_config, temp_output_dir, tmp_path):
        """Test successful image description with Gemini Vision"""
        # Create mock image
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"fake image data")
        
        # Mock Gemini client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock file upload
        mock_uploaded_file = Mock()
        mock_uploaded_file.uri = "https://generativelanguage.googleapis.com/v1beta/files/test"
        mock_uploaded_file.mime_type = "image/png"
        mock_client.files.upload.return_value = mock_uploaded_file
        
        # Mock generate_content response
        mock_response = Mock()
        mock_response.text = "This image shows a medical chart with patient data."
        mock_client.models.generate_content.return_value = mock_response
        
        parser = MinerUParser(mock_config, temp_output_dir, "test-key", vision_model="gemini-2.0-flash-exp")
        description = parser._describe_image_with_gemini(img_path)
        
        assert description == "This image shows a medical chart with patient data."
        mock_client.files.upload.assert_called_once()
        mock_client.models.generate_content.assert_called_once()
    
    @patch('src.mineru_parser.genai.Client')
    def test_describe_image_with_gemini_failure(self, mock_client_class, mock_config, temp_output_dir, tmp_path):
        """Test image description failure handling"""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"fake image data")
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.files.upload.side_effect = Exception("API Error")
        
        parser = MinerUParser(mock_config, temp_output_dir, "test-key")
        description = parser._describe_image_with_gemini(img_path)
        
        assert description == ""
    
    @patch('src.mineru_parser.MinerUParser._run_mineru_cli')
    @patch('src.mineru_parser.MinerUParser._process_images')
    @patch('src.mineru_parser.MinerUParser._process_text')
    def test_parse_pdf_success(self, mock_process_text, mock_process_images, 
                               mock_run_cli, mock_config, temp_output_dir, sample_pdf_path):
        """Test successful PDF parsing"""
        # Mock MinerU CLI output
        mock_run_cli.return_value = "# Test Document\n\nThis is test content."
        
        # Mock text chunks
        mock_process_text.return_value = [
            {'text': 'Text chunk 1', 'metadata': {'source': 'text', 'chunk_index': 0}}
        ]
        
        # Mock image chunks
        mock_process_images.return_value = [
            {'text': 'Image description 1', 'metadata': {'source': 'image', 'image_file': 'img1.png'}}
        ]
        
        parser = MinerUParser(mock_config, temp_output_dir, "test-key")
        chunks = parser.parse_pdf(sample_pdf_path)
        
        assert len(chunks) == 2
        assert chunks[0]['text'] == 'Image description 1'  # Images come first
        assert chunks[1]['text'] == 'Text chunk 1'
        mock_run_cli.assert_called_once_with(sample_pdf_path)
        mock_process_text.assert_called_once()
        mock_process_images.assert_called_once()
    
    @patch('src.mineru_parser.MinerUParser._run_mineru_cli')
    def test_parse_pdf_mineru_failure(self, mock_run_cli, mock_config, temp_output_dir, sample_pdf_path):
        """Test PDF parsing when MinerU fails"""
        mock_run_cli.return_value = None
        
        parser = MinerUParser(mock_config, temp_output_dir, "test-key")
        chunks = parser.parse_pdf(sample_pdf_path)
        
        assert len(chunks) == 0
    
    @patch('src.mineru_parser.MinerUParser._run_mineru_cli')
    @patch('src.mineru_parser.MinerUParser._process_images')
    @patch('src.mineru_parser.MinerUParser._process_text')
    def test_parse_pdf_combines_all_content(self, mock_process_text, mock_process_images,
                                           mock_run_cli, mock_config, temp_output_dir, sample_pdf_path):
        """Test that parse_pdf combines both text and image content"""
        mock_run_cli.return_value = "Test content"
        
        mock_process_text.return_value = [
            {'text': 'Text 1', 'metadata': {'source': 'text'}},
            {'text': 'Text 2', 'metadata': {'source': 'text'}}
        ]
        
        mock_process_images.return_value = [
            {'text': 'Image 1', 'metadata': {'source': 'image'}},
            {'text': 'Image 2', 'metadata': {'source': 'image'}}
        ]
        
        parser = MinerUParser(mock_config, temp_output_dir, "test-key")
        chunks = parser.parse_pdf(sample_pdf_path)
        
        assert len(chunks) == 4
        # Images should come first
        assert chunks[0]['metadata']['source'] == 'image'
        assert chunks[1]['metadata']['source'] == 'image'
        assert chunks[2]['metadata']['source'] == 'text'
        assert chunks[3]['metadata']['source'] == 'text'
    
    def test_custom_image_prompt(self, mock_config, temp_output_dir):
        """Test custom image prompt is used"""
        custom_prompt = "Describe this medical image in detail, including anatomical structures."
        
        parser = MinerUParser(
            mock_config,
            temp_output_dir,
            "test-key",
            image_prompt=custom_prompt
        )
        
        assert parser.image_prompt == custom_prompt
    
    def test_custom_vision_model(self, mock_config, temp_output_dir):
        """Test custom vision model is used"""
        custom_model = "gemini-1.5-pro"
        
        parser = MinerUParser(
            mock_config,
            temp_output_dir,
            "test-key",
            vision_model=custom_model
        )
        
        assert parser.vision_model == custom_model

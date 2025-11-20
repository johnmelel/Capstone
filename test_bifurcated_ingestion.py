
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "data_ingestion" / "src"))

from chunker import ExactTokenChunker, ImageCaptionChunker, SimpleTokenChunker
from pdf_extractor import PDFExtractor

class TestBifurcatedIngestion(unittest.TestCase):
    
    def test_exact_token_chunker(self):
        """Test that ExactTokenChunker works or falls back gracefully"""
        chunker = ExactTokenChunker(chunk_size=10, chunk_overlap=2)
        text = "This is a test sentence that should be split into multiple chunks because it is long enough."
        chunks = chunker.chunk_text(text)
        print(f"Chunks: {chunks}")
        self.assertTrue(len(chunks) > 0)
        
    def test_image_caption_chunker(self):
        """Test image caption chunking logic"""
        mock_text_chunker = MagicMock()
        mock_text_chunker.chunk_text.return_value = ["Part 1", "Part 2"]
        
        chunker = ImageCaptionChunker(mock_text_chunker)
        
        # Case 1: Long caption
        image_data = {'caption': 'Long caption', 'bytes': b'123'}
        chunks = chunker.chunk_image(image_data)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0][1], "Part 1")
        self.assertEqual(chunks[1][1], "Part 2")
        
        # Case 2: No caption
        image_data_empty = {'caption': None, 'bytes': b'456'}
        chunks_empty = chunker.chunk_image(image_data_empty)
        self.assertEqual(len(chunks_empty), 1)
        self.assertIsNone(chunks_empty[0][1])
        
    def test_markdown_replacement(self):
        """Test markdown image tag replacement logic (simulated)"""
        # This logic is inside PDFExtractor.extract_with_images, but we can simulate the regex part
        import re
        
        text = "Here is an image: ![](images/fig1.jpg) and another ![](images/fig2.jpg)"
        image_caption_map = {
            'fig1.jpg': 'Figure 1: A nice chart',
            'fig2.jpg': None # No caption
        }
        
        def replace_image_tag(match):
            full_path = match.group(1)
            filename = Path(full_path).name
            caption = image_caption_map.get(filename)
            if caption:
                return f"\n[Image: {caption}]\n"
            else:
                return ""
                
        pattern = r'!\[\]\((.*?)\)'
        processed_text = re.sub(pattern, replace_image_tag, text)
        
        expected = "Here is an image: \n[Image: Figure 1: A nice chart]\n and another "
        self.assertEqual(processed_text.strip(), expected.strip())

if __name__ == '__main__':
    unittest.main()

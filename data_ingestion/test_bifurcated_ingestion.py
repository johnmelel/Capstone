
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add data_ingestion to path to allow src imports
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from src.chunker import RecursiveTokenChunker, ImageCaptionChunker, SimpleTokenChunker
from src.pdf_extractor import PDFExtractor

class TestBifurcatedIngestion(unittest.TestCase):
    
    def test_recursive_token_chunker(self):
        """Test that RecursiveTokenChunker works with text and pages"""
        chunker = RecursiveTokenChunker(chunk_size=10, chunk_overlap=2)
        
        # Test string input (legacy)
        text = "This is a test sentence that should be split into multiple chunks because it is long enough."
        chunks = chunker.chunk_text(text)
        print(f"String chunks: {chunks}")
        self.assertTrue(len(chunks) > 0)
        self.assertIsInstance(chunks[0], dict)
        self.assertIn('text', chunks[0])
        self.assertIn('page_num', chunks[0])
        
        # Test page input
        pages = [
            {'text': "Page 1 content.", 'page_num': 1},
            {'text': "Page 2 content is longer and might split.", 'page_num': 2}
        ]
        page_chunks = chunker.chunk_text(pages)
        print(f"Page chunks: {page_chunks}")
        self.assertTrue(len(page_chunks) >= 2)
        self.assertEqual(page_chunks[0]['page_num'], 1)
        self.assertEqual(page_chunks[-1]['page_num'], 2)

    def test_image_caption_chunker(self):
        """Test image caption chunking logic"""
        mock_text_chunker = MagicMock()
        # Mock return value to match new format: list of dicts
        mock_text_chunker.chunk_text.return_value = [{'text': "Part 1", 'page_num': 1}, {'text': "Part 2", 'page_num': 1}]
        
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

    def test_caption_tagger(self):
        """Test CaptionTagger logic for tables vs images"""
        from src.caption_tagger import CaptionTagger
        tagger = CaptionTagger()
        
        # Mock content list
        content_list = [
            {'type': 'text', 'text': 'Some text', 'page_idx': 0, 'bbox': [0, 0, 100, 10]},
            {'type': 'text', 'text': 'Table 1: Data', 'page_idx': 0, 'bbox': [0, 20, 100, 30]},
            {'type': 'table', 'img_path': 'table1.png', 'page_idx': 0, 'bbox': [0, 40, 100, 140]}, # Table image
            {'type': 'text', 'text': 'Figure 1: Graph', 'page_idx': 0, 'bbox': [0, 150, 100, 160]},
            {'type': 'image', 'img_path': 'fig1.png', 'page_idx': 0, 'bbox': [0, 170, 100, 270]}, # Image
            {'type': 'text', 'text': 'Far away text', 'page_idx': 0, 'bbox': [0, 500, 100, 510]}
        ]
        
        tagged = tagger.tag_images(content_list)
        
        # Check Table (index 2) - should match "Table 1: Data" (index 1)
        table_block = next(b for b in tagged if b.get('img_path') == 'table1.png')
        self.assertEqual(table_block['caption'], 'Table 1: Data')
        
        # Check Image (index 4) - should match "Figure 1: Graph" (index 3)
        image_block = next(b for b in tagged if b.get('img_path') == 'fig1.png')
        self.assertEqual(image_block['caption'], 'Figure 1: Graph')
        
        # Test Image max distance
        content_list_far = [
            {'type': 'image', 'img_path': 'fig2.png', 'page_idx': 0, 'bbox': [0, 0, 100, 100]},
            {'type': 'text', 'text': 'Figure 2: Far', 'page_idx': 0, 'bbox': [0, 400, 100, 410]} # > 200 units away
        ]
        tagged_far = tagger.tag_images(content_list_far)
        image_far = next(b for b in tagged_far if b.get('img_path') == 'fig2.png')
        self.assertIsNone(image_far.get('caption'))

if __name__ == '__main__':
    unittest.main()

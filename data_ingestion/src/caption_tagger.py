import re
from typing import List, Dict, Optional, Tuple

class CaptionTagger:
    """
    Tags images with captions based on proximity and text patterns.
    """

    def __init__(self):
        # Regex to identify potential caption text (e.g., "Figure 1:", "Fig. 2", "Table 1")
        # We can expand this list based on observation
        self.caption_patterns = [
            re.compile(r'^(Figure|Fig\.?)\s*\d+', re.IGNORECASE),
            re.compile(r'^(Table|Tab\.?)\s*\d+', re.IGNORECASE),
            re.compile(r'^(Graph|Chart)\s*\d+', re.IGNORECASE),
            re.compile(r'^(Source|Note):', re.IGNORECASE)
        ]

    def is_caption(self, text: str) -> bool:
        """Check if text matches common caption patterns."""
        if not text:
            return False
        text = text.strip()
        for pattern in self.caption_patterns:
            if pattern.match(text):
                return True
        return False

    def tag_images(self, content_list: List[Dict]) -> List[Dict]:
        """
        Iterates through the content list and associates images with captions.
        
        Args:
            content_list: List of content blocks from Mineru's content_list.json
            
        Returns:
            List of image blocks with added 'caption' and 'caption_id' fields.
        """
        tagged_images = []
        
        for i, block in enumerate(content_list):
            if block.get('type') == 'image':
                caption, caption_id = self._find_caption_for_image(content_list, i)
                
                image_block = block.copy()
                image_block['caption'] = caption
                image_block['caption_id'] = caption_id
                tagged_images.append(image_block)
                
        return tagged_images

    def _find_caption_for_image(self, content_list: List[Dict], image_index: int) -> Tuple[Optional[str], Optional[int]]:
        """
        Look for a caption near the image.
        
        Returns:
            Tuple of (caption_text, caption_block_index)
        """
        
        # Check next block (common for Figures)
        if image_index + 1 < len(content_list):
            next_block = content_list[image_index + 1]
            if next_block.get('type') == 'text' and self.is_caption(next_block.get('text', '')):
                return next_block.get('text', '').strip(), image_index + 1

        # Check previous block (common for Tables, sometimes Figures)
        if image_index - 1 >= 0:
            prev_block = content_list[image_index - 1]
            if prev_block.get('type') == 'text' and self.is_caption(prev_block.get('text', '')):
                return prev_block.get('text', '').strip(), image_index - 1
                
        return None, None

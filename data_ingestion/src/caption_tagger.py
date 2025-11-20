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

    def _calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Euclidean distance between two bounding boxes.
        BBox format: [x0, y0, x1, y1] (left, top, right, bottom)
        We use the closest points between the two rectangles.
        """
        if not bbox1 or not bbox2:
            return float('inf')
            
        l1, t1, r1, b1 = bbox1
        l2, t2, r2, b2 = bbox2
        
        # Horizontal distance
        if l1 > r2: x_dist = l1 - r2
        elif l2 > r1: x_dist = l2 - r1
        else: x_dist = 0
        
        # Vertical distance
        if t1 > b2: y_dist = t1 - b2
        elif t2 > b1: y_dist = t2 - b1
        else: y_dist = 0
        
        return (x_dist**2 + y_dist**2)**0.5

    def _get_relative_position(self, img_bbox: List[float], text_bbox: List[float]) -> str:
        """
        Determine relative position of text to image.
        Returns: 'right', 'bottom', 'left', 'top', or 'overlap'
        """
        if not img_bbox or not text_bbox:
            return 'unknown'
            
        il, it, ir, ib = img_bbox
        tl, tt, tr, tb = text_bbox
        
        # Center points
        icx, icy = (il + ir) / 2, (it + ib) / 2
        tcx, tcy = (tl + tr) / 2, (tt + tb) / 2
        
        dx = tcx - icx
        dy = tcy - icy
        
        # Determine primary direction based on larger offset
        # Note: y increases downwards in PDF coordinates usually
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'bottom' if dy > 0 else 'top'

    def _find_closest_text_geometric(self, content_list: List[Dict], image_index: int, window: int = 2, grace: float = 10.0) -> Tuple[Optional[str], Optional[int]]:
        """
        Find closest text block geometrically within a window.
        Priority: Shortest Distance (with grace), then Position (Bottom > Right > Left > Top)
        
        Args:
            grace: Distance difference to consider "equal" (in PDF points)
        """
        img_block = content_list[image_index]
        img_bbox = img_block.get('bbox')
        img_page = img_block.get('page_idx')
        
        if not img_bbox or img_page is None:
            return None, None
            
        candidates = []
        
        start_idx = max(0, image_index - window)
        end_idx = min(len(content_list), image_index + window + 1)
        
        for i in range(start_idx, end_idx):
            if i == image_index:
                continue
                
            block = content_list[i]
            
            # Must be text and on same page
            if block.get('type') != 'text' or block.get('page_idx') != img_page:
                continue
                
            text = block.get('text', '').strip()
            if not text:
                continue
                
            text_bbox = block.get('bbox')
            if not text_bbox:
                continue
                
            dist = self._calculate_distance(img_bbox, text_bbox)
            pos = self._get_relative_position(img_bbox, text_bbox)
            
            # Priority score for position (lower is better)
            # Bottom (0) > Right (1) > Left (2) > Top (3)
            pos_score = {'bottom': 0, 'right': 1, 'left': 2, 'top': 3}.get(pos, 4)
            
            candidates.append({
                'text': text,
                'index': i,
                'dist': dist,
                'pos_score': pos_score
            })
            
        if not candidates:
            return None, None
            
        # Find minimum distance
        min_dist = min(c['dist'] for c in candidates)
        
        # Filter candidates within grace of minimum distance
        # This allows a slightly further text to win if it has better position priority
        valid_candidates = [c for c in candidates if c['dist'] <= min_dist + grace]
        
        # Sort by position score (primary for valid candidates) and then distance (secondary)
        valid_candidates.sort(key=lambda x: (x['pos_score'], x['dist']))
        
        best = valid_candidates[0]
        return best['text'], best['index']

    def _find_caption_for_image(self, content_list: List[Dict], image_index: int) -> Tuple[Optional[str], Optional[int]]:
        """
        Look for a caption near the image.
        Strategy:
        1. Regex pattern check on immediate neighbors (+/- 1)
        2. Geometric proximity check within window (+/- 2)
        
        Returns:
            Tuple of (caption_text, caption_block_index)
        """
        
        # 1. Regex Check (Immediate neighbors)
        # Check next block
        if image_index + 1 < len(content_list):
            next_block = content_list[image_index + 1]
            if next_block.get('type') == 'text' and self.is_caption(next_block.get('text', '')):
                return next_block.get('text', '').strip(), image_index + 1

        # Check previous block
        if image_index - 1 >= 0:
            prev_block = content_list[image_index - 1]
            if prev_block.get('type') == 'text' and self.is_caption(prev_block.get('text', '')):
                return prev_block.get('text', '').strip(), image_index - 1
                
        # 2. Geometric Fallback
        return self._find_closest_text_geometric(content_list, image_index, window=2)

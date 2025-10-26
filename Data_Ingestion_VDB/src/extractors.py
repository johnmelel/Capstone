"""PDF content extraction with HYBRID RAW/MARKDOWN support."""

import fitz  # PyMuPDF
import pymupdf4llm  # For markdown extraction
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import re
import io

class PDFExtractor:
    """Extract text, images, and tables with smart metadata - HYBRID VERSION."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config['chunking']['chunk_size']
        self.overlap = config['chunking']['overlap']
        self.min_img_width = config['images']['min_width']
        self.min_img_height = config['images']['min_height']
    
    def extract_from_pdf(self, pdf_path: Path, max_pages: int = None) -> List[Dict]:
        """Extract all content from a PDF using HYBRID method."""
        print(f"\n{'='*70}")
        print(f" EXTRACTING: {pdf_path.name}")
        
        # DETECT EXTRACTION METHOD based on folder
        extraction_method = self._detect_extraction_method(pdf_path)
        print(f"   Method: {extraction_method.upper()}")
        print(f"{'='*70}")
        
        doc = fitz.open(pdf_path)
        total_pages = min(max_pages, doc.page_count) if max_pages else doc.page_count
        
        print(f" Total pages to process: {total_pages}")
        
        all_items = []
        
        for page_num in range(total_pages):
            page = doc[page_num]
            print(f"\n   Page {page_num + 1}/{total_pages}")
            
            # Extract text with appropriate method
            print(f"      Extracting text ({extraction_method})...")
            if extraction_method == "markdown":
                text_items = self._extract_text_markdown(pdf_path, page_num, doc)
            else:
                text_items = self._extract_text_raw(page, pdf_path.stem, page_num)
            
            all_items.extend(text_items)
            print(f"        {len(text_items)} text chunks")
            
            # Extract images with smart metadata
            print("       Extracting images...")
            image_items = self._extract_images_smart(page, doc, pdf_path.stem, page_num)
            all_items.extend(image_items)
            print(f"        {len(image_items)} images")
            
            # Detect tables
            print("      Detecting tables...")
            table_items = self._detect_tables(page, pdf_path.stem, page_num)
            all_items.extend(table_items)
            print(f"        {len(table_items)} tables")
        
        doc.close()
        
        print(f"\n{'='*70}")
        print(f" EXTRACTION COMPLETE: {len(all_items)} total items")
        print(f"{'='*70}")
        
        return all_items
    
    def _detect_extraction_method(self, pdf_path: Path) -> str:
        """Detect if PDF should use raw or markdown extraction based on folder."""
        # Check if in raw_extraction folder
        if "raw_extraction" in str(pdf_path.parent):
            return "raw"
        # Check if in markdown_extraction folder
        elif "markdown_extraction" in str(pdf_path.parent):
            return "markdown"
        # Default to markdown if not in a specific folder
        else:
            print(f"      PDF not in raw_extraction or markdown_extraction folder")
            print(f"      Defaulting to MARKDOWN extraction")
            return "markdown"
    
    def _extract_text_raw(self, page, pdf_name: str, page_num: int) -> List[Dict]:
        """Extract text using RAW PyMuPDF method."""
        
        #  Let PyMuPDF handle columns automatically
        text = page.get_text("text", sort=True)
        
        if not text.strip():
            return []
        
        #  Simple chunking with hard 950-char limit
        chunks = []
        chunk_idx = 0
        MAX_SIZE = 950
        OVERLAP_SIZE = self.overlap
        
        # Split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 10]
        
        current_chunk = ""
        overlap_text = ""
        
        for para in paragraphs:
            # If single para is too long, split by sentences
            if len(para) > MAX_SIZE:
                if current_chunk.strip():
                    chunks.append(self._create_chunk(current_chunk.strip(), pdf_name, page_num, chunk_idx, "raw"))
                    chunk_idx += 1
                    overlap_text = self._get_overlap(current_chunk, OVERLAP_SIZE)
                    current_chunk = overlap_text
                
                # Split by sentences
                sentences = [s.strip() + '.' for s in para.split('.') if s.strip()]
                for sent in sentences:
                    if len(current_chunk) + len(sent) > MAX_SIZE:
                        if current_chunk.strip():
                            chunks.append(self._create_chunk(current_chunk.strip(), pdf_name, page_num, chunk_idx, "raw"))
                            chunk_idx += 1
                            overlap_text = self._get_overlap(current_chunk, OVERLAP_SIZE)
                        current_chunk = overlap_text + sent + " "
                    else:
                        current_chunk += sent + " "
            # Check if adding para would exceed
            elif len(current_chunk) + len(para) > MAX_SIZE:
                if current_chunk.strip():
                    chunks.append(self._create_chunk(current_chunk.strip(), pdf_name, page_num, chunk_idx, "raw"))
                    chunk_idx += 1
                    overlap_text = self._get_overlap(current_chunk, OVERLAP_SIZE)
                current_chunk = overlap_text + para + "\n\n"
            else:
                current_chunk += para + "\n\n"
        
        # Save last chunk
        if current_chunk.strip():
            if len(current_chunk) > 1000:  # Safety truncate
                current_chunk = current_chunk[:1000]
            chunks.append(self._create_chunk(current_chunk.strip(), pdf_name, page_num, chunk_idx, "raw"))
        
        return chunks
    
    def _extract_text_markdown(self, pdf_path: Path, page_num: int, doc) -> List[Dict]:
        """Extract text using MARKDOWN pymupdf4llm method."""
        
        try:
            # Extract markdown for this specific page
            md_dict = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_num])
            
            if isinstance(md_dict, dict) and 'text' in md_dict:
                text = md_dict['text']
            else:
                text = md_dict
            
            if not text.strip():
                return []
            
            # Same chunking logic as raw
            chunks = []
            chunk_idx = 0
            MAX_SIZE = 950
            OVERLAP_SIZE = self.overlap
            
            # Split by paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 10]
            
            current_chunk = ""
            overlap_text = ""
            
            for para in paragraphs:
                # If single para is too long, split by sentences
                if len(para) > MAX_SIZE:
                    if current_chunk.strip():
                        chunks.append(self._create_chunk(current_chunk.strip(), pdf_path.stem, page_num, chunk_idx, "markdown"))
                        chunk_idx += 1
                        overlap_text = self._get_overlap(current_chunk, OVERLAP_SIZE)
                        current_chunk = overlap_text
                    
                    # Split by sentences
                    sentences = [s.strip() + '.' for s in para.split('.') if s.strip()]
                    for sent in sentences:
                        if len(current_chunk) + len(sent) > MAX_SIZE:
                            if current_chunk.strip():
                                chunks.append(self._create_chunk(current_chunk.strip(), pdf_path.stem, page_num, chunk_idx, "markdown"))
                                chunk_idx += 1
                                overlap_text = self._get_overlap(current_chunk, OVERLAP_SIZE)
                            current_chunk = sent + " "
                        else:
                            current_chunk += sent + " "
                # Check if adding para would exceed
                elif len(current_chunk) + len(para) > MAX_SIZE:
                    if current_chunk.strip():
                        chunks.append(self._create_chunk(current_chunk.strip(), pdf_path.stem, page_num, chunk_idx, "markdown"))
                        chunk_idx += 1
                        overlap_text = self._get_overlap(current_chunk, OVERLAP_SIZE)
                    current_chunk = para + "\n\n"
                else:
                    current_chunk += para + "\n\n"
            
            # Save last chunk
            if current_chunk.strip():
                if len(current_chunk) > 1000:  # Safety truncate
                    current_chunk = current_chunk[:1000]
                chunks.append(self._create_chunk(current_chunk.strip(), pdf_path.stem, page_num, chunk_idx, "markdown"))
            
            return chunks
            
        except Exception as e:
            print(f"       Markdown extraction failed: {e}")
            print(f"       Falling back to RAW extraction...")
            # Fallback to raw if markdown fails
            page = doc[page_num]
            return self._extract_text_raw(page, pdf_path.stem, page_num)

    def _create_chunk(self, text: str, pdf_name: str, page_num: int, 
                    chunk_idx: int, extraction_method: str) -> Dict:
        """Create a chunk with metadata including extraction method."""
        fig_refs = self._find_figure_refs(text)
        table_refs = self._find_table_refs(text)
        
        return {
            "chunk_id": f"{pdf_name}_p{page_num+1}_text_{chunk_idx}",
            "type": "text",
            "content": text,
            "metadata": {
                "pdf_name": pdf_name,
                "page": page_num + 1,
                "chunk_index": chunk_idx,
                "extraction_method": extraction_method,  # Track which method used
                "figure_refs": ", ".join(fig_refs) if fig_refs else None,
                "table_refs": ", ".join(table_refs) if table_refs else None
            }
        }

    def _get_overlap(self, text: str, overlap_size: int) -> str:
        """Get the last N characters from text, trying to break at sentence boundary."""
        if len(text) <= overlap_size:
            return text
        
        # Get last overlap_size characters
        overlap = text[-overlap_size:]
        
        # Try to start at a sentence boundary (look for '. ')
        sentence_start = overlap.find('. ')
        if sentence_start != -1 and sentence_start < overlap_size * 0.5:
            # If we found a sentence start in the first half, use everything after it
            overlap = overlap[sentence_start + 2:]
        
        return overlap

    def _extract_images_smart(self, page, doc, pdf_name: str, page_num: int) -> List[Dict]:
        """Extract images with smart figure numbering and panel detection."""
        image_list = page.get_images(full=True)
        
        if not image_list:
            return []
        
        page_text = page.get_text()
        
        image_items = []
        
        for img_idx, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                
                if (base_image["width"] < self.min_img_width or 
                    base_image["height"] < self.min_img_height):
                    continue
                
                rects = page.get_image_rects(xref)
                bbox = list(rects[0]) if rects else None
                
                #  SIMPLE CAPTION EXTRACTION (no complex strategies yet)
                caption = self._extract_caption_simple(page, bbox)
                
                figure_id = self._detect_figure_number(caption)
                if not figure_id:
                    figure_id = f"AUTO_P{page_num+1}_IMG_{img_idx}"
                
                panel_info = self._detect_panel_side(caption)
                
                image_filename = f"{pdf_name}_p{page_num+1}_img_{img_idx}.png"
                image_path = Path("data/extracted/images") / image_filename
                
                with open(image_path, "wb") as f:
                    f.write(base_image["image"])
                
                image_items.append({
                    "chunk_id": f"{pdf_name}_p{page_num+1}_img_{img_idx}",
                    "type": "image",
                    "image_path": str(image_path),
                    "caption": caption,
                    "metadata": {
                        "pdf_name": pdf_name,
                        "page": page_num + 1,
                        "width": base_image["width"],
                        "height": base_image["height"],
                        "figure_id": figure_id,
                        "panel_side": panel_info.get("side"),
                        "panel_index": panel_info.get("index"),
                        "bbox": str(bbox) if bbox else None
                    }
                })
                
                print(f"         → {figure_id} {panel_info.get('side', '')}")
                
            except Exception as e:
                print(f"          Failed to extract image {img_idx}: {e}")
        
        return image_items
    
    def _extract_caption_simple(self, page, bbox) -> str:
        """Simple caption extraction - look near image."""
        if not bbox:
            return ""
        
        blocks = page.get_text("dict")["blocks"]
        
        img_bottom = bbox[3]
        img_top = bbox[1]
        img_center_y = (img_top + img_bottom) / 2
        
        candidates = []
        
        for block in blocks:
            if block["type"] != 0:
                continue
            
            block_text = self._extract_block_text(block)
            
            # Check if looks like caption
            if any(block_text.lower().startswith(w) for w in ['fig', 'figure', 'table', 'panel']):
                block_center_y = (block["bbox"][1] + block["bbox"][3]) / 2
                distance = abs(block_center_y - img_center_y)
                
                if distance < 150:
                    candidates.append((distance, block_text))
        
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1][:300]
        
        return ""
    
    def _detect_figure_number(self, text: str) -> str:
        """Detect figure numbers like 'Figure 3.2', 'Fig 5-1', 'FIG 10A'."""
        if not text:
            return None
        
        patterns = [
            r'Fig(?:ure)?\s*(\d+[-–.]?\d*[A-Z]?)',
            r'FIG\s*(\d+[-–.]?\d*[A-Z]?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fig_num = match.group(1).strip()
                return f"FIG_{fig_num.replace('.', '-').replace('–', '-')}"
        
        return None
    
    def _detect_panel_side(self, text: str) -> Dict[str, Any]:
        """Detect panel indicators like (Left), (Right), (A), (B)."""
        if not text:
            return {}
        
        patterns = {
            r'\(Left\)': {'side': 'left', 'index': 0},
            r'\(Right\)': {'side': 'right', 'index': 1},
            r'\(Upper\)': {'side': 'upper', 'index': 0},
            r'\(Lower\)': {'side': 'lower', 'index': 1},
            r'\(A\)': {'side': 'panel_A', 'index': 0},
            r'\(B\)': {'side': 'panel_B', 'index': 1},
            r'\(C\)': {'side': 'panel_C', 'index': 2},
        }
        
        for pattern, info in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return info
        
        return {}
    
    def _find_figure_refs(self, text: str) -> List[str]:
        """Find references to figures in text."""
        matches = re.findall(r'Fig(?:ure)?\s*\d+[-–.]?\d*[A-Z]?', text, re.IGNORECASE)
        return [m.strip() for m in matches]
    
    def _find_table_refs(self, text: str) -> List[str]:
        """Find references to tables in text."""
        matches = re.findall(r'Table\s*\d+[-–.]?\d*', text, re.IGNORECASE)
        return [m.strip() for m in matches]
    
    def _detect_tables(self, page, pdf_name: str, page_num: int) -> List[Dict]:
        """Basic table detection."""
        return []
        """Basic table detection."""
        # For now, return empty - can enhance later
        return []

    def _extract_block_text(self, block) -> str:
        """Extract text from a block."""
        text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text += span.get("text", "") + " "
        return text.strip()

    def _detect_tables(self, page, pdf_name: str, page_num: int) -> List[Dict]:
        """Basic table detection."""
        return []
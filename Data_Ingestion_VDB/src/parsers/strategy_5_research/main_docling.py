"""
Strategy 5 - main.py


CHANGEABLE'S
    - self.images_root = Path("strategy_5_research/images")
    - self.save_images = True



"""

from pathlib import Path
import io
from typing import List, Dict, Any, List, Optional
import re
from ..base_parser import BaseParser


try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError:
    print("[Warning] Docling not installed.")
    print("[Info] Install with: pip install docling")
    DOCLING_AVAILABLE = False
    import fitz  # PyMuPDF fallback





pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True  

class ResearchParser(BaseParser):
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config, output_dir) # calls the parent class (BaseParser)
        self.strategy_name = "research_paper"
        
        if DOCLING_AVAILABLE:
            print(f"        [ResearchParser] Initialized with docling parser")
            #self.converter = DocumentConverter(pipeline_options=pipeline_options)
            self.converter = DocumentConverter()
        else:
            print(f"[ResearchParser] WARNING: Using fallback PyMuPDF parser")
            print(f"[ResearchParser] Install docling for better results")
            self.converter = None
        
        # Section patterns for research papers
        self.section_keywords = {
            'abstract': ['abstract'],
            'introduction': ['introduction'],
            'methods': ['methods', 'methodology', 'materials and methods'],
            'results': ['results'],
            'discussion': ['discussion'],
            'conclusion': ['conclusion', 'conclusions'],
            'references': ['references', 'bibliography']
        }

        self.save_images = getattr(self, "save_images", True)
        self.images_root = Path("strategy_5_research/images")
    
    def parse_pdf(self, pdf_path: Path, max_pages: int = None) -> List[Dict]: # returns list of dictionaries, each representing a chunk
        # print(f"\n{'='*70}")
        # print(f"[ResearchParser] Starting extraction: {pdf_path.name}")
        # print(f"{'='*70}")
        print("-"*30,f"[ResearchParser] Starting extraction: {pdf_path.name}","-"*30)
        
        if DOCLING_AVAILABLE and self.converter:
            return self._parse_with_docling(pdf_path, max_pages)
        else:
            return self._parse_with_pymupdf_fallback(pdf_path, max_pages)
    
    def _parse_with_docling(self, pdf_path: Path, max_pages: int = None) -> List[Dict]:
        """
        Parse PDF using docling for high-quality extraction.
        
        docling Process:
        1. Initialize reader/writer for disk I/O
        2. Parse PDF with layout analysis
        3. Extract structured content
        4. Process each content type
        5. Create chunks
        
        Args:
            pdf_path: Path to PDF
            max_pages: Page limit
            
        Returns:
            List of chunks
        """
        print(f"[Docling] Parsing...")
        
        pdf_name = pdf_path.stem
        all_chunks = []
        
        try:
            # Convert PDF with docling
            print(f"\n[Docling] Converting PDF...\n")
            result = self.converter.convert(str(pdf_path))

            # get document
            doc = result.document

            # Extract markdown
            print(f"\n[Docling] Extractig text content...")
            md_content = doc.export_to_markdown()

            text_chunks = self._process_docling_text(
                md_content,
                pdf_name,
                max_pages
            )
            
            all_chunks.extend(text_chunks)
            #print(f"[Docling] Extracted {len(text_chunks)} text chunks")

            # Extract tables
            print(f"\n[Docling] Extracting tables...")
            table_chunks = self._process_docling_tables(
                doc,
                pdf_name,
                max_pages
            )
            all_chunks.extend(table_chunks)
            #print(f"[Docling] Extracted {len(table_chunks)} tables")
            
            # Extract images/figures
            print(f"\n[Docling] Extracting images...")
            image_chunks = self._process_docling_images(
                doc,
                pdf_name,
                max_pages
            )
            all_chunks.extend(image_chunks)
            #print(f"[Docling] Extracted {len(image_chunks)} images")

        except Exception as e:
            print(f"[ERROR] Docling parsing failed: {e}")
            print(f"[INFO] Falling back to PyMuPDF...")
            import traceback
            traceback.print_exc()
            return self._parse_with_pymupdf_fallback(pdf_path, max_pages)
        
        # Count chunks by type
        text_count = len([c for c in all_chunks if c['type'] == 'text'])
        image_count = len([c for c in all_chunks if c['type'] == 'image'])
        table_count = len([c for c in all_chunks if c['type'] == 'table'])

        # Print clean summary
        print(f"\n{'─' * 70}")
        print(f"|  [Summary] Extraction complete for {pdf_name}")
        print(f"{'─' * 70}")
        print(f"|  Text chunks: {text_count}")
        print(f"|  Images:      {image_count}")
        print(f"|  Tables:      {table_count}")
        print(f"|  Total:       {len(all_chunks)}")
        print(f"{'─' * 70}")
        
        return all_chunks   

    def _process_docling_text(self, md_content: str, pdf_name: str, 
                         max_pages: int = None) -> List[Dict]:
        """
        Process text content from Docling markdown output with cleaning.
        
        NEW: Uses post_processor to clean and chunk text.
        
        Args:
            md_content: Markdown text from Docling
            pdf_name: PDF name
            max_pages: Page limit
            
        Returns:
            List of clean text chunks
        """
        if not md_content or not md_content.strip():
            return []
        
        # Import post_processor
        from .post_processor import clean_and_chunk
        
        # Clean and chunk the text (this does EVERYTHING)
        chunks = clean_and_chunk(md_content, pdf_name, page_num=1)
        
        print(f"    [Text] Created {len(chunks)} clean chunks")
        
        return chunks

    def _process_docling_tables(self, doc, pdf_name: str,
                                max_pages: int = None) -> List[Dict]:
        """
        Process tables extracted by Docling.
        
        Args:
            doc: Docling document object
            pdf_name: PDF name
            max_pages: Page limit
            
        Returns:
            List of table chunks
        """
        table_chunks = []
        
        # Iterate through document elements
        table_index = 0
        for item in doc.tables:
            try:
                table_content = item.export_to_markdown()
                    
                # Try to get caption
                caption = getattr(item, 'caption', '')
                if not caption and hasattr(item, 'text'):
                    caption = item.text[:200]  # Use first 200 chars as caption
                    
                # Get page number if available
                if hasattr(item, 'prov') and len(item.prov) > 0 and hasattr(item.prov[0], 'page_no'):
                    page_num = item.prov[0].page_no
                else:
                    page_num = 1
                    
                if max_pages and page_num > max_pages:
                    continue
                    
                if table_content and table_content.strip():
                    # Detect table ID
                    table_id = self._detect_table_number(caption)
                        
                    # Create chunk
                    chunk_id = self.create_chunk_id(pdf_name, page_num, 'table', table_index)
                        
                    metadata = self.create_metadata(
                        pdf_name=pdf_name,
                        page_num=page_num,
                        strategy_name=self.strategy_name,
                        table_id=table_id
                    )
                        
                    table_chunks.append({
                        'chunk_id': chunk_id,
                        'type': 'table',
                        'content': table_content,
                        'caption': caption,
                        'metadata': metadata
                    })
                        
                    table_index += 1
                
            except Exception as e:
                print(f"      [Error] Failed to process table {table_index}: {e}")
        
        return table_chunks
    



    def _as_image_bytes(img_obj) -> bytes:
        """
        Accepts bytes, PIL.Image, or numpy array and returns PNG bytes.
        """
        try:
            # Case 1: already bytes
            if isinstance(img_obj, (bytes, bytearray, memoryview)):
                return bytes(img_obj)

            # Case 2: PIL.Image
            try:
                from PIL import Image
                if isinstance(img_obj, Image.Image):
                    buf = io.BytesIO()
                    img_obj.save(buf, format="PNG")
                    return buf.getvalue()
            except Exception:
                pass

            # Case 3: numpy array (HWC / HW)
            try:
                import numpy as np
                from PIL import Image
                if isinstance(img_obj, np.ndarray):
                    if img_obj.dtype != np.uint8:
                        arr = img_obj
                        # Normalize to 0–255
                        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                        img = Image.fromarray((arr * 255).astype('uint8'))
                    else:
                        img = Image.fromarray(img_obj)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    return buf.getvalue()
            except Exception:
                pass

            # Case 4: common attribute fallbacks (Docling variants)
            for attr in ("data", "bytes", "image_bytes", "content"):
                if hasattr(img_obj, attr):
                    maybe = getattr(img_obj, attr)
                    if isinstance(maybe, (bytes, bytearray, memoryview)):
                        return bytes(maybe)

        except Exception:
            pass

        return b""
        
    def _safe_write_image(image_bytes: bytes, out_path: Path) -> Path: 
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(image_bytes)
        return out_path






    def _process_docling_images(self, doc, pdf_name: str, max_pages: int = None):
        """
        Find images in the Docling doc, convert them to bytes, (optionally) save to disk,
        and build a list of 'image chunks' for your summary / downstream steps.
        """
        image_chunks = []

        # 1) Figure out where images will be saved (if saving is enabled)
        images_root = getattr(self, "images_root", Path("outputs/images"))
        strategy = getattr(self, "strategy_name", "strategy")
        save_images = getattr(self, "save_images", True)

        # 2) Get picture items from the doc
        pictures = []
        if hasattr(doc, "pictures") and doc.pictures:
            pictures = list(doc.pictures)
            print(f"[DEBUG] doc.pictures has {len(pictures)} images")
        elif hasattr(doc, "elements"):
            # Fallback: some doc versions keep figures under elements with type "picture"/"figure"/"image"
            pictures = [el for el in getattr(doc, "elements", [])
                        if getattr(el, "type", "").lower() in {"picture", "figure", "image"}]
            print(f"[DEBUG] doc.elements fallback found {len(pictures)} images")
        else:
            print("[DEBUG] No doc.pictures or elements with images found")

        image_index = 0

        # 3) Loop through each picture
        for item in pictures:
            try:
                # Page number (if available)
                page_num = getattr(item, "page_no", None) or getattr(item, "page", None) or 1
                if max_pages and page_num and page_num > max_pages:
                    # You *are* skipping pages if max_pages is set and exceeded
                    # (Your old log text was confusing)
                    print(f"[DEBUG] skipping page {page_num} due to max_pages={max_pages}")
                    continue

                # Caption (if available)
                caption = getattr(item, "caption", "") or getattr(item, "text", "") or ""

                # The raw image object (Docling may use different attribute names)
                img_obj = getattr(item, "image", None) or getattr(item, "bitmap", None) or getattr(item, "content", None)

                # Convert to bytes we can save or embed
                image_bytes = _as_image_bytes(img_obj)
                if not image_bytes:
                    print(f"[DEBUG] skipping image {image_index}: could not get bytes")
                    continue

                print("[DEBUG] got image data!! uhul")

                # Build a stable id and filename
                figure_id = f"fig-{page_num}-{image_index}"
                file_name = f"{pdf_name}__{strategy}__{figure_id}.png"

                # 4) (Optional) Save to disk
                if save_images:
                    out_path = images_root / pdf_name / file_name
                    _safe_write_image(image_bytes, out_path)
                    image_path_str = str(out_path)
                else:
                    image_path_str = None  # not saved to disk

                # 5) Build the metadata chunk (this is what your summary counts)
                chunk_id = f"{pdf_name}__img__{image_index}"
                metadata = {
                    "pdf_name": pdf_name,
                    "page_num": page_num,
                    "caption": caption,
                    "file_name": file_name,
                    "image_relpath": str(Path(pdf_name) / file_name) if image_path_str else None,
                    "strategy_name": strategy,
                    "figure_id": figure_id,
                    "type": "image"
                }

                image_chunks.append({
                    "chunk_id": chunk_id,
                    "type": "image",
                    "image_path": image_path_str,  # full path or None
                    "caption": caption,
                    "metadata": metadata
                })

                image_index += 1

            except Exception as e:
                print(f"[Error] Failed to process image {image_index}: {e}")

        return image_chunks



    # def _process_docling_images(self, doc, pdf_name: str,
    #                             max_pages: int = None) -> List[Dict]:
    #     image_chunks = []

    #     # DEBUG: Check if doc.pictures exists
    #     if not hasattr(doc, 'pictures'):
    #         print(f"[DEBUG]  ❌  doc has NO pictures attribute")
    #         return image_chunks

    #     # DEBUG: count pictures
    #     pics_list = list(doc.pictures)
    #     print(f"[DEBUG] doc.pictures has {len(pics_list)} images")
        
    #     # Iterate through document elements
    #     image_index = 0
    #     for item in doc.pictures:
    #         try:



    #             caption = getattr(item, 'caption', '')
    
    #             if not caption and hasattr(item, 'text'):
    #                 caption = item.text
                    
    #             # Get page number if available
    #             page_num = getattr(item, 'page_no', 1)
                    
    #             if max_pages and page_num > max_pages:
    #                 print(f"[DEBUG] skipping (page {page_num} > max {max_pages})")
    #                 continue
    #             print(f"[DEBUG] pages not skipped, continuing to save.")
                    
    #             # Try to get image data
    #             image_data = None
    #             if hasattr(item, 'image'):
    #                 image_data = item.image
    #                 print("[DEBUG] got image data!! uhul")
    #             elif hasattr(item, 'data'):
    #                 image_data = item.data
    #                 print("[DEBUG] didn't get image data")
                    
    #             if image_data:
    #                 # Save image
    #                 image_path = self.save_image(
    #                     image_data,
    #                     pdf_name,
    #                     page_num,
    #                     image_index
    #                 )

                        
    #                 # Detect figure ID
    #                 figure_id = self._detect_figure_number(caption)
                        
    #                 # Create chunk
    #                 chunk_id = self.create_chunk_id(pdf_name, page_num, 'image', image_index)
                        
    #                 metadata = self.create_metadata(
    #                     pdf_name=pdf_name,
    #                     page_num=page_num,
    #                     strategy_name=self.strategy_name,
    #                     figure_id=figure_id
    #                 )
                        
    #                 image_chunks.append({
    #                     'chunk_id': chunk_id,
    #                     'type': 'image',
    #                     'image_path': str(image_path),
    #                     'caption': caption,
    #                     'metadata': metadata
    #                 })
                        
    #                 image_index += 1
                
    #         except Exception as e:
    #             print(f"      [Error] Failed to process image {image_index}: {e}")
        
    #     return image_chunks
    
    def _detect_section(self, text: str) -> Optional[str]:
        """
        Detect if text is a section header.
        
        Args:
            text: Text to check
            
        Returns:
            Section name or None
        """
        text_lower = text.lower().strip()
        
        # Check if this looks like a header (short, capitalized, etc.)
        if len(text) > 100:
            return None
        
        for section_name, keywords in self.section_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return section_name
        
        return None
    
    def _create_text_chunk(self, text: str, pdf_name: str, page_num: int,
                           chunk_index: int, section: str = None) -> Dict:
        """
        Create text chunk with metadata.
        
        Args:
            text: Chunk text
            pdf_name: PDF name
            page_num: Page number
            chunk_index: Chunk index
            section: Section name if detected
            
        Returns:
            Chunk dictionary
        """
        # Detect references
        fig_refs = self._find_figure_references(text)
        table_refs = self._find_table_references(text)
        
        chunk_id = self.create_chunk_id(pdf_name, page_num, 'text', chunk_index)
        
        metadata = self.create_metadata(
            pdf_name=pdf_name,
            page_num=page_num,
            strategy_name=self.strategy_name,
            chunk_index=chunk_index,
            char_count=len(text),
            word_count=len(text.split()),
            section=section,
            figure_refs=fig_refs if fig_refs else None,
            table_refs=table_refs if table_refs else None
        )
        
        return {
            'chunk_id': chunk_id,
            'type': 'text',
            'content': text,
            'metadata': metadata
        }

    def _parse_with_pymupdf_fallback(self, pdf_path: Path, max_pages: int = None) -> List[Dict]:
        """
        Fallback parser using PyMuPDF if Docling fails or unavailable.
        
        This is a simplified version that just extracts basic content.
        For production, you should install Docling for better results.
        
        Args:
            pdf_path: Path to PDF
            max_pages: Page limit
            
        Returns:
            List of chunks
        """
        print(f"[Fallback] Using basic PyMuPDF extraction...")
        print(f"[Warning] Limited functionality without Docling")
        
        import fitz
        
        doc = fitz.open(pdf_path)
        pdf_name = pdf_path.stem
        total_pages = min(max_pages, doc.page_count) if max_pages else doc.page_count
        
        all_chunks = []
        
        for page_num in range(total_pages):
            page = doc[page_num]
            page_number = page_num + 1
            
            # Extract text
            text = page.get_text("text", sort=True)
            
            if text.strip():
                # Simple chunking
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                chunk_index = 0
                current_chunk = ""
                
                for para in paragraphs:
                    if len(current_chunk) + len(para) > self.chunk_size:
                        if current_chunk.strip():
                            chunk_dict = self._create_text_chunk(
                                current_chunk.strip(),
                                pdf_name,
                                page_number,
                                chunk_index
                            )
                            all_chunks.append(chunk_dict)
                            chunk_index += 1
                        current_chunk = para + "\n\n"
                    else:
                        current_chunk += para + "\n\n"
                
                if current_chunk.strip():
                    chunk_dict = self._create_text_chunk(
                        current_chunk.strip(),
                        pdf_name,
                        page_number,
                        chunk_index
                    )
                    all_chunks.append(chunk_dict)
        
        doc.close()
        
        print(f"\n[Fallback] Extracted {len(all_chunks)} chunks")
        return all_chunks
    
    # Helper methods
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap from end of chunk."""
        if len(text) <= self.overlap:
            return text
        
        overlap = text[-self.overlap:]
        sentence_start = overlap.find('. ')
        if sentence_start != -1 and sentence_start < self.overlap * 0.5:
            overlap = overlap[sentence_start + 2:]
        
        return overlap
    
    def _find_figure_references(self, text: str) -> List[str]:
        """Find figure references in text."""
        patterns = [
            r'Figure\s+\d+[A-Z]?',
            r'Fig\.\s*\d+[A-Z]?',
            r'FIG\.\s*\d+[A-Z]?',
        ]
        
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            matches.extend(found)
        
        return list(set(matches))
    
    def _find_table_references(self, text: str) -> List[str]:
        """Find table references in text."""
        pattern = r'Table\s+\d+'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return list(set(matches))
    
    def _detect_figure_number(self, text: str) -> Optional[str]:
        """Extract figure number from caption."""
        if not text:
            return None
        
        patterns = [
            r'Fig(?:ure)?\s*(\d+[A-Z]?)',
            r'FIG\s*(\d+[A-Z]?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"Figure_{match.group(1)}"
        
        return None
    
    def _detect_table_number(self, text: str) -> Optional[str]:
        """Extract table number from caption."""
        if not text:
            return None
        
        pattern = r'Table\s+(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"Table_{match.group(1)}"
        
        return None
        """Extract table number from caption."""
        if not text:
            return None
        
        pattern = r'Table\s+(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"Table_{match.group(1)}"
        
        return None
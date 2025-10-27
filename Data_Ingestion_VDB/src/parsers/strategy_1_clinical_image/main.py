

from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from ..base_parser import BaseParser

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.document import PdfFormatOption
    DOCLING_AVAILABLE = True
except ImportError:
    print("[Warning] Docling not installed.")
    print("[Info] Install with: pip install docling")
    DOCLING_AVAILABLE = False
    import fitz  # PyMuPDF fallback

pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True  

class ClinicalImageParser(BaseParser):

    def __init__(self, config, output_dir):          # ← ADD PARAMETERS
        super().__init__(config, output_dir)
        self.strategy_name = "strategy_1_clinical_image"
        
        # Setup output directory for images
        self.output_dir = Path("outputs") / self.strategy_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chunking config (text chunking done by post_processor)
        self.chunk_config = {
            'max_chunk_size': 950,
            'overlap_size': 150
        }
    
    def parse_pdf(self, pdf_path: Path, max_pages: int = None) -> List[Dict]:

        if not DOCLING_AVAILABLE:
            return self._fallback_parse(pdf_path, max_pages)
        
        print(f"\n[Strategy 1] Parsing {pdf_path.name}...")
        
        # Step 1: Convert PDF with docling
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(str(pdf_path))
        
        # Step 2: Get markdown content
        md_content = result.document.export_to_markdown()
        
        # Step 3: Process text (cleaning + chunking)
        text_chunks = self._process_docling_text(md_content, pdf_path.stem, max_pages)
        
        # Step 4: Extract images (if any)
        image_chunks = self._extract_images(result, pdf_path.stem)
        
        # Step 5: Extract tables (if any)
        table_chunks = self._extract_tables(result, pdf_path.stem)
        
        # Combine all chunks
        all_chunks = text_chunks + image_chunks + table_chunks
        
        print(f"    [Total] {len(all_chunks)} chunks created")
        print(f"      - {len(text_chunks)} text chunks")
        print(f"      - {len(image_chunks)} image chunks")
        print(f"      - {len(table_chunks)} table chunks")
        
        return all_chunks
    
    def _process_docling_text(self, md_content: str, pdf_name: str, 
                             max_pages: int = None) -> List[Dict]:
        """
        Process text content from Docling markdown output.
        
        NEW: Uses post_processor to clean and chunk text.
        
        PROCESS:
        1. Import post_processor
        2. Call clean_and_chunk() which handles:
           - Loading cleaning rules from cleaning_rules.yaml
           - Removing noise patterns (headers, keywords, etc.)
           - Normalizing text (fixing line breaks, spacing)
           - Creating semantic chunks with overlap
           - Detecting figure/table references
        3. Return clean chunks
        
        Args:
            md_content: Markdown text from Docling
            pdf_name: PDF name
            max_pages: Page limit (optional)
            
        Returns:
            List of clean text chunks with metadata
        """
        if not md_content or not md_content.strip():
            return []
        
        # Import post_processor
        from .post_processor import clean_and_chunk
        
        # Clean and chunk the text (this does EVERYTHING)
        chunks = clean_and_chunk(md_content, pdf_name, page_num=1)
        
        print(f"    [Text] Created {len(chunks)} clean chunks")
        
        return chunks
    
    def _extract_images(self, result, pdf_name: str) -> List[Dict]:
        """
        Extract images from docling result.
        
        PROCESS:
        1. Iterate through docling figures
        2. Save each image to disk
        3. Extract caption
        4. Create chunk with image path and metadata
        
        Args:
            result: Docling conversion result
            pdf_name: PDF name
            
        Returns:
            List of image chunks
        """
        image_chunks = []
        
        # Extract images from docling
        for idx, figure in enumerate(result.document.pictures):
            image_path = self.output_dir / f"{pdf_name}_figure_{idx}.png"
            
            # Save image
            figure.image.save(image_path)
            
            # Get caption if available
            caption = figure.caption.text if figure.caption else None
            
            # Create chunk
            chunk = {
                'chunk_id': f"{pdf_name}_image_{idx}",
                'type': 'image',
                'image_path': str(image_path),
                'caption': caption,
                'metadata': {
                    'pdf_name': pdf_name,
                    'page': figure.page,
                    'strategy_name': self.strategy_name,
                    'figure_id': f"Figure_{idx}"
                }
            }
            
            image_chunks.append(chunk)
        
        return image_chunks
    
    def _extract_tables(self, result, pdf_name: str) -> List[Dict]:
        """
        Extract tables from docling result.
        
        PROCESS:
        1. Iterate through docling tables
        2. Extract table data as text
        3. Extract caption
        4. Create chunk with table content and metadata
        
        Args:
            result: Docling conversion result
            pdf_name: PDF name
            
        Returns:
            List of table chunks
        """
        table_chunks = []
        
        # Extract tables from docling
        for idx, table in enumerate(result.document.tables):
            # Get table as markdown or text
            table_text = table.export_to_markdown()
            
            # Get caption if available
            caption = table.caption.text if table.caption else None
            
            # Create chunk
            chunk = {
                'chunk_id': f"{pdf_name}_table_{idx}",
                'type': 'table',
                'content': table_text,
                'caption': caption,
                'metadata': {
                    'pdf_name': pdf_name,
                    'page': table.page,
                    'strategy_name': self.strategy_name,
                    'table_id': f"Table_{idx}"
                }
            }
            
            table_chunks.append(chunk)
        
        return table_chunks
    
    def _fallback_parse(self, pdf_path: Path, max_pages: int = None) -> List[Dict]:
        """
        Fallback parser if docling is not available.
        Uses PyMuPDF for basic text extraction.
        """
        print("[Warning] Using fallback parser (PyMuPDF)")
        import fitz
        
        doc = fitz.open(pdf_path)
        chunks = []
        
        for page_num in range(len(doc)):
            if max_pages and page_num >= max_pages:
                break
            
            page = doc[page_num]
            text = page.get_text()
            
            chunk = {
                'chunk_id': f"{pdf_path.stem}_p{page_num}_text_0",
                'type': 'text',
                'content': text,
                'metadata': {
                    'pdf_name': pdf_path.stem,
                    'page': page_num,
                    'strategy_name': self.strategy_name
                }
            }
            
            chunks.append(chunk)
        
        return chunks
"""
================================================================================
Strategy 5 Research Parser - MinerU Implementation
================================================================================

ARCHITECTURE OVERVIEW:
---------------------
This parser is part of a multi-strategy PDF parsing system designed to handle
different document types with specialized processing. This specific parser
(Strategy 5) is optimized for research papers using the MinerU library.

PIPELINE FLOW:
1. PDF Input → MinerU Parser → Text/Image/Table Extraction
2. Raw Content → Post-Processing → Clean Chunks
3. Clean Chunks → Embeddings → Milvus Vector Database

KEY COMPONENTS IN THIS FILE:
- ResearchParser: Main parser class using MinerU
- PDF parsing with layout analysis
- Text extraction and chunking  
- Image extraction and saving
- Table extraction and formatting
- Metadata generation

INTEGRATION POINTS:
- Inherits from BaseParser for common functionality
- Uses post_processor.py for text cleaning
- Outputs chunks compatible with embedder.py
- Images saved to disk for multimodal processing

WHY MINERU OVER DOCLING:
- Better handling of complex research paper layouts
- More accurate figure/table extraction
- Superior handling of mathematical formulas
- Better preservation of document structure

================================================================================
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import base64
import io
import json
from PIL import Image
from ..base_parser import BaseParser

# MinerU imports with fallback
try:
    from magic_pdf.pipe.UNIPipe import UNIPipe
    from magic_pdf.pipe.OCRPipe import OCRPipe
    from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
    import magic_pdf.model as model_config
    MINERU_AVAILABLE = True
    print("[✓] MinerU successfully imported")
except ImportError:
    print("[Warning] MinerU not installed.")
    print("[Info] Install with: pip install magic-pdf[full]")
    print("[Info] Also need: pip install paddlepaddle paddleocr")
    MINERU_AVAILABLE = False
    
    # Fallback to PyMuPDF
    try:
        import fitz
        print("[Info] Using PyMuPDF as fallback")
    except ImportError:
        print("[Error] Neither MinerU nor PyMuPDF available!")


class ResearchParser(BaseParser):
    """
    Research Paper Parser using MinerU
    
    This parser is specifically designed for academic research papers with:
    - Complex multi-column layouts
    - Figures with captions
    - Tables with structured data
    - Mathematical formulas
    - References and citations
    
    Processing Pipeline:
    1. Load PDF with MinerU
    2. Perform layout analysis
    3. Extract structured content
    4. Process each content type separately
    5. Apply post-processing and cleaning
    6. Generate metadata for each chunk
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        Initialize the Research Parser
        
        Args:
            config: Configuration dictionary containing:
                - chunk_size: Maximum size of text chunks
                - overlap_size: Overlap between chunks
                - extract_images: Whether to extract images
                - extract_tables: Whether to extract tables
            output_dir: Directory to save extracted images
        """
        super().__init__(config, output_dir)
        self.strategy_name = "research_paper"
        
        # Setup output directory for images
        self.image_output_dir = Path("outputs") / self.strategy_name / "images"
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chunking configuration
        self.chunk_size = config.get('chunk_size', 1000)
        self.overlap_size = config.get('overlap_size', 200)
        self.extract_images = config.get('extract_images', True)
        self.extract_tables = config.get('extract_tables', True)
        
        # MinerU configuration
        if MINERU_AVAILABLE:
            print(f"[ResearchParser] Initialized with MinerU parser")
            print(f"[ResearchParser] Image output: {self.image_output_dir}")
            
            # Setup MinerU model path if needed
            self.setup_mineru_models()
        else:
            print(f"[ResearchParser] WARNING: Using fallback PyMuPDF parser")
        
        # Section patterns for research papers
        self.section_keywords = {
            'abstract': ['abstract', 'summary'],
            'introduction': ['introduction', 'background'],
            'methods': ['methods', 'methodology', 'materials and methods', 'experimental'],
            'results': ['results', 'findings'],
            'discussion': ['discussion', 'analysis'],
            'conclusion': ['conclusion', 'conclusions', 'summary and conclusions'],
            'references': ['references', 'bibliography', 'works cited']
        }
    
    def setup_mineru_models(self):
        """
        Setup MinerU model paths and configuration
        
        MinerU requires pretrained models for:
        - Layout analysis
        - Table structure recognition
        - Formula detection
        - OCR (if needed)
        """
        try:
            # Set model paths if not already configured
            import os
            
            # Check if models are downloaded
            model_dir = Path.home() / ".cache" / "magic_pdf"
            if not model_dir.exists():
                print("[Info] MinerU models not found. They will be downloaded on first use.")
                print("[Info] This may take a few minutes...")
            else:
                print(f"[Info] MinerU models found at: {model_dir}")
                
        except Exception as e:
            print(f"[Warning] Could not setup MinerU models: {e}")
    
    def parse_pdf(self, pdf_path: Path, max_pages: int = None) -> List[Dict]:
        """
        Main entry point for PDF parsing
        
        This method orchestrates the entire parsing process:
        1. Validates the PDF file
        2. Chooses parsing method (MinerU or fallback)
        3. Extracts all content types
        4. Returns structured chunks
        
        Args:
            pdf_path: Path to the PDF file
            max_pages: Maximum number of pages to process (None = all)
            
        Returns:
            List of chunk dictionaries containing:
                - chunk_id: Unique identifier
                - type: 'text', 'image', or 'table'
                - content: The actual content
                - metadata: Additional information
        """
        print(f"\n{'='*70}")
        print(f"[ResearchParser] Starting extraction: {pdf_path.name}")
        print(f"{'='*70}")
        
        if not pdf_path.exists():
            print(f"[ERROR] PDF file not found: {pdf_path}")
            return []
        
        if MINERU_AVAILABLE:
            return self._parse_with_mineru(pdf_path, max_pages)
        else:
            return self._parse_with_pymupdf_fallback(pdf_path, max_pages)
    
    def _parse_with_mineru(self, pdf_path: Path, max_pages: int = None) -> List[Dict]:
        """
        Parse PDF using MinerU for high-quality extraction
        
        MinerU Processing Pipeline:
        1. Initialize DiskReaderWriter for file I/O
        2. Create parsing pipeline (UNIPipe or OCRPipe)
        3. Run layout analysis and content extraction
        4. Process extracted JSON structure
        5. Extract text, images, and tables separately
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Page limit
            
        Returns:
            List of all extracted chunks
        """
        print(f"[MinerU] Starting advanced parsing...")
        
        pdf_name = pdf_path.stem
        all_chunks = []
        
        try:
            # Step 1: Setup MinerU pipeline
            print(f"[MinerU] Initializing pipeline...")
            
            # Create temporary directory for MinerU output
            temp_dir = Path("temp") / "mineru_output" / pdf_name
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize reader/writer
            reader_writer = DiskReaderWriter(str(temp_dir))
            
            # Read PDF bytes
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            # Step 2: Create parsing pipeline
            # Use UNIPipe for general PDFs, OCRPipe for scanned documents
            print(f"[MinerU] Running layout analysis...")
            
            # Determine if OCR is needed (simple check)
            need_ocr = self._check_if_ocr_needed(pdf_bytes)
            
            if need_ocr:
                print(f"[MinerU] Using OCR pipeline (scanned document detected)")
                pipe = OCRPipe(pdf_bytes, str(pdf_path), reader_writer)
            else:
                print(f"[MinerU] Using standard pipeline (digital document)")
                pipe = UNIPipe(pdf_bytes, str(pdf_path), reader_writer)
            
            # Step 3: Run the pipeline
            print(f"[MinerU] Extracting content...")
            pipe_result = pipe.pipe_classify()
            
            if not pipe_result:
                print(f"[ERROR] MinerU pipeline failed")
                return self._parse_with_pymupdf_fallback(pdf_path, max_pages)
            
            # Parse the result
            pipe.pipe_analyze()
            pipe.pipe_parse()
            
            # Get the parsed content
            parsed_content = pipe.get_result()
            
            # Step 4: Process extracted content
            # Extract text chunks
            print(f"\n[MinerU] Processing text content...")
            text_chunks = self._process_mineru_text(
                parsed_content,
                pdf_name,
                max_pages
            )
            all_chunks.extend(text_chunks)
            
            # Extract tables
            if self.extract_tables:
                print(f"\n[MinerU] Processing tables...")
                table_chunks = self._process_mineru_tables(
                    parsed_content,
                    pdf_name,
                    max_pages
                )
                all_chunks.extend(table_chunks)
            
            # Extract images
            if self.extract_images:
                print(f"\n[MinerU] Processing images...")
                image_chunks = self._process_mineru_images(
                    parsed_content,
                    pdf_name,
                    max_pages,
                    reader_writer
                )
                all_chunks.extend(image_chunks)
            
            # Cleanup temporary files
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"[ERROR] MinerU parsing failed: {e}")
            print(f"[INFO] Falling back to PyMuPDF...")
            import traceback
            traceback.print_exc()
            return self._parse_with_pymupdf_fallback(pdf_path, max_pages)
        
        # Generate summary
        self._print_extraction_summary(all_chunks, pdf_name)
        
        return all_chunks
    
    def _check_if_ocr_needed(self, pdf_bytes: bytes) -> bool:
        """
        Check if PDF needs OCR (is scanned/image-based)
        
        Simple heuristic: Check if PDF has extractable text
        """
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Check first few pages for text
            pages_to_check = min(3, len(doc))
            for i in range(pages_to_check):
                page = doc[i]
                text = page.get_text()
                if len(text.strip()) > 100:  # Has substantial text
                    doc.close()
                    return False
            
            doc.close()
            return True  # No text found, likely scanned
            
        except:
            return False  # Default to non-OCR
    
    def _process_mineru_text(self, parsed_content: Dict, pdf_name: str, 
                            max_pages: int = None) -> List[Dict]:
        """
        Process text content extracted by MinerU
        
        MinerU returns structured JSON with:
        - Pages array containing blocks
        - Each block has type (text, figure, table, etc.)
        - Text blocks contain the actual text content
        
        Processing steps:
        1. Extract text from each page
        2. Identify section headers
        3. Group text by sections
        4. Apply cleaning via post_processor
        5. Create chunks with overlap
        
        Args:
            parsed_content: MinerU parsed JSON structure
            pdf_name: Name of the PDF
            max_pages: Maximum pages to process
            
        Returns:
            List of text chunks with metadata
        """
        text_chunks = []
        
        # Import post-processor for cleaning
        from .post_processor import clean_and_chunk
        
        # Extract pages from MinerU result
        pages = parsed_content.get('pages', [])
        
        if max_pages:
            pages = pages[:max_pages]
        
        print(f"[Text] Processing {len(pages)} pages...")
        
        for page_idx, page in enumerate(pages, 1):
            # Extract text blocks from page
            page_text = []
            
            for block in page.get('blocks', []):
                if block.get('type') == 'text':
                    text = block.get('text', '').strip()
                    if text:
                        page_text.append(text)
            
            if page_text:
                # Combine page text
                full_text = '\n\n'.join(page_text)
                
                # Clean and chunk using post-processor
                chunks = clean_and_chunk(
                    full_text, 
                    pdf_name, 
                    page_num=page_idx,
                    chunk_size=self.chunk_size,
                    overlap_size=self.overlap_size
                )
                
                text_chunks.extend(chunks)
        
        print(f"[Text] Created {len(text_chunks)} text chunks")
        return text_chunks
    
    def _process_mineru_tables(self, parsed_content: Dict, pdf_name: str,
                              max_pages: int = None) -> List[Dict]:
        """
        Process tables extracted by MinerU
        
        MinerU provides structured table data with:
        - Cell positions and content
        - Table structure (rows, columns)
        - Table captions if detected
        
        Args:
            parsed_content: MinerU parsed content
            pdf_name: PDF name
            max_pages: Page limit
            
        Returns:
            List of table chunks
        """
        table_chunks = []
        table_index = 0
        
        pages = parsed_content.get('pages', [])
        if max_pages:
            pages = pages[:max_pages]
        
        for page_idx, page in enumerate(pages, 1):
            for block in page.get('blocks', []):
                if block.get('type') == 'table':
                    try:
                        # Extract table content
                        table_data = block.get('table', {})
                        
                        # Convert table to markdown format
                        table_md = self._table_to_markdown(table_data)
                        
                        if table_md:
                            # Get caption if available
                            caption = block.get('caption', '')
                            
                            # Detect table ID from caption
                            table_id = self._detect_table_number(caption)
                            
                            # Create chunk
                            chunk_id = self.create_chunk_id(
                                pdf_name, page_idx, 'table', table_index
                            )
                            
                            metadata = self.create_metadata(
                                pdf_name=pdf_name,
                                page_num=page_idx,
                                strategy_name=self.strategy_name,
                                table_id=table_id
                            )
                            
                            table_chunks.append({
                                'chunk_id': chunk_id,
                                'type': 'table',
                                'content': table_md,
                                'caption': caption,
                                'metadata': metadata
                            })
                            
                            table_index += 1
                            
                    except Exception as e:
                        print(f"[Error] Failed to process table {table_index}: {e}")
        
        print(f"[Tables] Extracted {len(table_chunks)} tables")
        return table_chunks
    
    def _process_mineru_images(self, parsed_content: Dict, pdf_name: str,
                              max_pages: int, reader_writer) -> List[Dict]:
        """
        Process images extracted by MinerU
        
        IMPORTANT: Images MUST be saved to disk for:
        1. Displaying in test.py (markdown references)
        2. Multimodal embedding later
        3. User inspection and validation
        
        MinerU provides:
        - Image data (base64 or file path)
        - Bounding boxes
        - Associated captions
        
        Args:
            parsed_content: MinerU parsed content
            pdf_name: PDF name
            max_pages: Page limit
            reader_writer: MinerU disk reader/writer
            
        Returns:
            List of image chunks with file paths
        """
        image_chunks = []
        image_index = 0
        
        pages = parsed_content.get('pages', [])
        if max_pages:
            pages = pages[:max_pages]
        
        for page_idx, page in enumerate(pages, 1):
            for block in page.get('blocks', []):
                if block.get('type') in ['figure', 'image']:
                    try:
                        # Get image data
                        image_path_in_result = block.get('img_path')
                        caption = block.get('caption', '')
                        
                        if image_path_in_result:
                            # Read image from MinerU output
                            image_data = reader_writer.read(
                                image_path_in_result, 
                                mode='rb'
                            )
                            
                            # Save image to our output directory
                            image_filename = f"{pdf_name}_page{page_idx}_fig{image_index}.png"
                            image_path = self.image_output_dir / image_filename
                            
                            # Convert and save image
                            image = Image.open(io.BytesIO(image_data))
                            image.save(image_path, 'PNG')
                            
                            print(f"[Image] Saved: {image_filename}")
                            
                            # Detect figure ID
                            figure_id = self._detect_figure_number(caption)
                            
                            # Create chunk
                            chunk_id = self.create_chunk_id(
                                pdf_name, page_idx, 'image', image_index
                            )
                            
                            metadata = self.create_metadata(
                                pdf_name=pdf_name,
                                page_num=page_idx,
                                strategy_name=self.strategy_name,
                                figure_id=figure_id
                            )
                            
                            image_chunks.append({
                                'chunk_id': chunk_id,
                                'type': 'image',
                                'image_path': str(image_path),
                                'caption': caption,
                                'metadata': metadata
                            })
                            
                            image_index += 1
                            
                    except Exception as e:
                        print(f"[Error] Failed to process image {image_index}: {e}")
        
        print(f"[Images] Extracted and saved {len(image_chunks)} images")
        return image_chunks
    
    def _table_to_markdown(self, table_data: Dict) -> str:
        """
        Convert MinerU table structure to markdown format
        
        Args:
            table_data: Table structure from MinerU
            
        Returns:
            Markdown formatted table string
        """
        try:
            # Extract cells
            cells = table_data.get('cells', [])
            if not cells:
                return ""
            
            # Determine table dimensions
            max_row = max(cell.get('row', 0) for cell in cells)
            max_col = max(cell.get('col', 0) for cell in cells)
            
            # Create 2D array
            table = [['' for _ in range(max_col + 1)] for _ in range(max_row + 1)]
            
            # Fill table
            for cell in cells:
                row = cell.get('row', 0)
                col = cell.get('col', 0)
                text = cell.get('text', '').strip()
                table[row][col] = text
            
            # Convert to markdown
            md_lines = []
            for i, row in enumerate(table):
                md_lines.append('| ' + ' | '.join(row) + ' |')
                if i == 0:  # Add header separator
                    md_lines.append('|' + '---|' * len(row))
            
            return '\n'.join(md_lines)
            
        except Exception as e:
            print(f"[Warning] Failed to convert table to markdown: {e}")
            return ""
    
    def _detect_figure_number(self, caption: str) -> Optional[str]:
        """
        Extract figure number from caption
        
        Args:
            caption: Figure caption text
            
        Returns:
            Figure ID (e.g., "Figure_1") or None
        """
        if not caption:
            return None
        
        # Pattern to match "Figure 1", "Fig. 1", "Figure 1.", etc.
        patterns = [
            r'Figure\s+(\d+)',
            r'Fig\.\s*(\d+)',
            r'FIG\s+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, caption, re.IGNORECASE)
            if match:
                return f"Figure_{match.group(1)}"
        
        return None
    
    def _detect_table_number(self, caption: str) -> Optional[str]:
        """
        Extract table number from caption
        
        Args:
            caption: Table caption text
            
        Returns:
            Table ID (e.g., "Table_1") or None
        """
        if not caption:
            return None
        
        # Pattern to match "Table 1", "Tab. 1", etc.
        patterns = [
            r'Table\s+(\d+)',
            r'Tab\.\s*(\d+)',
            r'TABLE\s+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, caption, re.IGNORECASE)
            if match:
                return f"Table_{match.group(1)}"
        
        return None
    
    def _parse_with_pymupdf_fallback(self, pdf_path: Path, max_pages: int = None) -> List[Dict]:
        """
        Fallback parser using PyMuPDF if MinerU fails or is unavailable
        
        This provides basic extraction but won't have:
        - Advanced layout analysis
        - Accurate figure/table detection
        - Formula extraction
        
        Args:
            pdf_path: Path to PDF
            max_pages: Page limit
            
        Returns:
            List of basic text chunks
        """
        print(f"[PyMuPDF] Using fallback parser...")
        
        try:
            import fitz
            
            doc = fitz.open(pdf_path)
            pdf_name = pdf_path.stem
            all_chunks = []
            
            pages_to_process = min(len(doc), max_pages) if max_pages else len(doc)
            
            for page_idx in range(pages_to_process):
                page = doc[page_idx]
                text = page.get_text()
                
                if text.strip():
                    # Use post-processor for cleaning
                    from .post_processor import clean_and_chunk
                    
                    chunks = clean_and_chunk(
                        text,
                        pdf_name,
                        page_num=page_idx + 1,
                        chunk_size=self.chunk_size,
                        overlap_size=self.overlap_size
                    )
                    
                    all_chunks.extend(chunks)
            
            doc.close()
            
            print(f"[PyMuPDF] Extracted {len(all_chunks)} chunks")
            return all_chunks
            
        except Exception as e:
            print(f"[ERROR] Fallback parsing also failed: {e}")
            return []
    
    def _print_extraction_summary(self, all_chunks: List[Dict], pdf_name: str):
        """
        Print a clean summary of extraction results
        
        Args:
            all_chunks: All extracted chunks
            pdf_name: Name of the PDF
        """
        # Count chunks by type
        text_count = len([c for c in all_chunks if c['type'] == 'text'])
        image_count = len([c for c in all_chunks if c['type'] == 'image'])
        table_count = len([c for c in all_chunks if c['type'] == 'table'])
        
        # Print summary
        print(f"\n{'─' * 70}")
        print(f"|  [Summary] Extraction complete for {pdf_name}")
        print(f"{'─' * 70}")
        print(f"|  Text chunks: {text_count}")
        print(f"|  Images:      {image_count}")
        print(f"|  Tables:      {table_count}")
        print(f"|  Total:       {len(all_chunks)}")
        print(f"{'─' * 70}")
        
        if image_count > 0:
            print(f"|  Images saved to: {self.image_output_dir}")
            print(f"{'─' * 70}")


# ============================================================================
# ANSWERS TO YOUR QUESTIONS
# ============================================================================

"""
Q1: Do we need to save images to disk to print them in test.py?
A: YES! When test.py creates markdown output, it needs actual file paths to 
   reference images. Markdown can't display base64 encoded images directly.
   The test.py will generate markdown like: ![Figure 1](path/to/image.png)

Q2: Do we need to save images for embedding?
A: YES! Most multimodal embedding models (like CLIP) need to load images 
   from disk. Some can work with in-memory data, but saving to disk is:
   - More reliable
   - Allows for inspection
   - Enables caching
   - Makes debugging easier

Q3: Does the current code save images?
A: The current Docling code attempts to save images BUT there's an issue:
   - It tries to save using `figure.image.save()` directly
   - This might not work with all Docling versions
   - MinerU has a more reliable image extraction pipeline

STRATEGY EXPLANATION:
--------------------
1. TESTING PHASE (what you're doing now):
   - Keep both main.py (Docling) and main_mineru.py
   - Test MinerU with test.py to verify image extraction
   - Compare outputs between both parsers
   
2. INTEGRATION PHASE:
   - Once MinerU works well, rename main.py → main_docling.py
   - Rename main_mineru.py → main.py
   - Update imports in __init__.py
   
3. IMAGE HANDLING:
   - Images MUST be saved to disk for both display and embedding
   - Use a consistent naming convention: {pdf_name}_page{N}_fig{M}.png
   - Store in organized directories: outputs/strategy_name/images/

4. WHY THIS APPROACH:
   - MinerU handles complex layouts better than Docling
   - Better mathematical formula extraction
   - More accurate table structure preservation
   - Superior figure/caption association
"""
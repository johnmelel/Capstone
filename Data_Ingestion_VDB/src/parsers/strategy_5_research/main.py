
"""
Research Paper Parser (Strategy 5) - USING docling

STEPS: 
1. Opens PDF using docling parser
2. Extracts text with structure awareness (sections, paragraphs)
3. Detects and extracts tables with structure preservation
4. Identifies figures and associates them with captions
5. Creates chunks with rich metadata
6. Saves extracted content to disk

docling FEATURES USED:
- PDF parsing with layout analysis
- Table detection and structure extraction
- Figure extraction with bounding boxes
- Caption detection and association
- Section hierarchy detection

OUTPUT STRUCTURE:
Each chunk dictionary contains:
{
    'chunk_id': str,              # Format: {pdf_name}_p{page}_{type}_{index}
    'type': 'text'|'image'|'table',
    'content': str,               # Text content (for text/table chunks)
    'image_path': str,            # Path to saved image (for image chunks)
    'caption': str,               # Caption text (for images/tables)
    'metadata': {
        'pdf_name': str,
        'page': int,
        'strategy_name': str,
        'figure_id': str,         # e.g., "Figure_1A"
        'table_id': str,          # e.g., "Table_2"
        'section': str,           # Document section if detected
        ... additional fields ...
    }
}

CHUNKING STRATEGY:
- Text: Semantic chunking with overlap
- Images: One chunk per image with caption
- Tables: One chunk per table with caption and data
- Maintains references between text and figures/tables

DEPENDENCIES:
Requires: pip install docling
"""

from pathlib import Path
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

class ResearchParser(BaseParser):
    """
    Parser for research papers using docling.
    
    INITIALIZATION:
    - Checks if docling is available
    - Sets up output directories
    - Configures chunking parameters
    
    PARSING WORKFLOW:
    1. Load PDF with docling
    2. Extract structured content (text blocks, images, tables)
    3. Process and chunk text content
    4. Extract images with captions
    5. Extract tables with structure
    6. Create metadata for all chunks
    7. Save extracted content
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config, output_dir)
        self.strategy_name = "research_paper"
        
        if DOCLING_AVAILABLE:
            print(f"        [ResearchParser] Initialized with docling parser")
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
    
    def parse_pdf(self, pdf_path: Path, max_pages: int = None) -> List[Dict]:
        """
        Parse research paper PDF and extract all content.
        
        Steps:
        1. Check if docling is available
        2. Parse PDF (docling or fallback)
        3. Extract text, images, tables
        4. Create chunks with metadata
        5. Log extraction summary
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Optional page limit for testing
            
        Returns:
            List of chunk dictionaries
        """
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
        
        # Summary
        print(f"\n{'-'*70}")
        print(f"|   [Summary] Extraction complete for {pdf_name}                             |")
        print(f"|     - Text chunks: {len([c for c in all_chunks if c['type']=='text'])}     |")
        print(f"|     - Images: {len([c for c in all_chunks if c['type']=='image'])}         |")
        print(f"|     - Tables: {len([c for c in all_chunks if c['type']=='table'])}         |")
        print(f"|     - Total: {len(all_chunks)}                                             |")
        print(f"{'-'*70}")
        
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

    # def _process_docling_text(self, md_content: str, pdf_name: str, 
    #                          max_pages: int = None) -> List[Dict]:
    #     """
    #     Process text content from Docling markdown output.
        
    #     Process:
    #     1. Split markdown into sections
    #     2. Detect section types (abstract, methods, etc.)
    #     3. Chunk text with overlap
    #     4. Detect figure/table references
    #     5. Create metadata
        
    #     Args:
    #         md_content: Markdown text from Docling
    #         pdf_name: PDF name
    #         max_pages: Page limit
            
    #     Returns:
    #         List of text chunks
    #     """
    #     if not md_content or not md_content.strip():
    #         return []
        
    #     chunks = []
        
    #     # Split by paragraphs (double newline)
    #     paragraphs = [p.strip() for p in md_content.split('\n\n') if p.strip()]
        
    #     print(f"    [Text] Processing {len(paragraphs)} paragraphs...")
        
    #     chunk_index = 0
    #     current_chunk = ""
    #     current_section = None
        
    #     for para in paragraphs:
    #         # Detect section headers
    #         section = self._detect_section(para)
    #         if section:
    #             current_section = section
            
    #         # Check if we need to start new chunk
    #         if len(current_chunk) + len(para) > self.chunk_size:
    #             if current_chunk.strip():
    #                 # Save current chunk
    #                 chunk_dict = self._create_text_chunk(
    #                     current_chunk.strip(),
    #                     pdf_name,
    #                     page_num=1,  # Docling doesn't give us page numbers easily
    #                     chunk_index=chunk_index,
    #                     section=current_section
    #                 )
    #                 chunks.append(chunk_dict)
    #                 chunk_index += 1
                    
    #                 # Start new chunk with overlap
    #                 overlap = self._get_overlap_text(current_chunk)
    #                 current_chunk = overlap + para + "\n\n"
    #             else:
    #                 current_chunk = para + "\n\n"
    #         else:
    #             current_chunk += para + "\n\n"
        
    #     # Save last chunk
    #     if current_chunk.strip():
    #         chunk_dict = self._create_text_chunk(
    #             current_chunk.strip(),
    #             pdf_name,
    #             page_num=1,
    #             chunk_index=chunk_index,
    #             section=current_section
    #         )
    #         chunks.append(chunk_dict)
        
    #     return chunks 
 
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
                page_num = getattr(item, 'page_no', 1)
                    
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
    
    def _process_docling_images(self, doc, pdf_name: str,
                                max_pages: int = None) -> List[Dict]:
        """
        Process images/figures extracted by Docling.
        
        Args:
            doc: Docling document object
            pdf_name: PDF name
            max_pages: Page limit
            
        Returns:
            List of image chunks
        """
        image_chunks = []
        
        # Iterate through document elements
        image_index = 0
        for item in doc.pictures:
            try:
                caption = getattr(item, 'caption', '')
    
                if not caption and hasattr(item, 'text'):
                    caption = item.text
                    
                # Get page number if available
                page_num = getattr(item, 'page_no', 1)
                    
                if max_pages and page_num > max_pages:
                    continue
                    
                # Try to get image data
                image_data = None
                if hasattr(item, 'image'):
                    image_data = item.image
                elif hasattr(item, 'data'):
                    image_data = item.data
                    
                if image_data:
                    # Save image
                    image_path = self.save_image(
                        image_data,
                        pdf_name,
                        page_num,
                        image_index
                    )
                        
                    # Detect figure ID
                    figure_id = self._detect_figure_number(caption)
                        
                    # Create chunk
                    chunk_id = self.create_chunk_id(pdf_name, page_num, 'image', image_index)
                        
                    metadata = self.create_metadata(
                        pdf_name=pdf_name,
                        page_num=page_num,
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
                print(f"      [Error] Failed to process image {image_index}: {e}")
        
        return image_chunks
    
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
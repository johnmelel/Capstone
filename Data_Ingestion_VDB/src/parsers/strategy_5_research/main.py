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


import subprocess
import json
import shutil

MINERU_AVAILABLE = True

class ResearchParser(BaseParser):
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config, output_dir) # calls the parent class (BaseParser)
        self.strategy_name = "research_paper"
        
        # MinerU output directories
        self.mineru_temp_dir = Path("temp/mineru_output")
        self.images_root = Path("strategy_5_research/images")
        self.images_root.mkdir(parents=True, exist_ok=True)

        print(f"[ResearchParser] Initialized with MinerU CLI parser")
        print(f"[ResearchParser] Images will be saved to: {self.images_root}")

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
        print(f"\n{'='*70}")
        print(f"[ResearchParser] Starting MinerU extraction: {pdf_path.name}")
        print(f"{'='*70}")
        
        if not pdf_path.exists():
            print(f"[ERROR] PDF not found: {pdf_path}")
            return []
        
        # Call MinerU and process output
        return self._parse_with_mineru_cli(pdf_path, max_pages)
    
    def _parse_with_mineru_cli(self, pdf_path: Path, max_pages: int = None) -> List[Dict]:
        """
        Parse PDF using MinerU CLI command.
        
        Steps:
        1. Create temp output directory
        2. Run mineru command
        3. Read generated markdown
        4. Copy images to our folder
        5. Process into chunks
        """
        pdf_name = pdf_path.stem
        
        # Setup output directory
        output_dir = self.mineru_temp_dir / pdf_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Run MinerU command
            print(f"\n[MinerU] Running CLI command...")
            
            cmd = [
                'mineru',
                '-p', str(pdf_path),  
                '-o', str(output_dir) 
            ]
            
            print(f"[MinerU DEBUG] Command: {' '.join(cmd)}")
                        
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"[ERROR] MinerU command failed:")
                print(f"  Return code: {result.returncode}")
                print(f"  STDERR: {result.stderr}")
                print(f"  STDOUT: {result.stdout}")
                return []
            
            print(f"[MinerU] ✓ Extraction complete")
            
            # Step 2: Find the markdown file MinerU created
            # MinerU creates: output_dir/{pdf_name}/auto/
            auto_dir = output_dir / "auto"
            if not auto_dir.exists():
                # Try without auto subdirectory
                auto_dir = output_dir
        

            md_files = list(output_dir.glob("**/*.md"))
            if not md_files:
                print(f"[ERROR] No markdown file found in {output_dir}")
                return []
            
            md_file = md_files[0]  # MinerU creates one .md file per PDF
            print(f"\n[MinerU] Reading markdown: {md_file.name}")
            
            # Step 3: Read markdown content
            markdown = md_file.read_text(encoding='utf-8')
            
            # Step 4: Copy images to our folder
            image_chunks = self._process_mineru_images(output_dir, pdf_name)
            
            # Step 5: Process markdown into text chunks
            text_chunks = self._process_mineru_text(markdown, pdf_name)
            
            # Combine all chunks
            all_chunks = text_chunks + image_chunks
            
            # Print summary
            print(f"\n{'─' * 70}")
            print(f"|  [Summary] Extraction complete for {pdf_name}")
            print(f"{'─' * 70}")
            print(f"|  Text chunks: {len(text_chunks)}")
            print(f"|  Images:      {len(image_chunks)}")
            print(f"|  Total:       {len(all_chunks)}")
            print(f"{'─' * 70}")
            
            return all_chunks
            
        except subprocess.TimeoutExpired:
            print(f"[ERROR] MinerU command timed out after 5 minutes")
            return []
        except Exception as e:
            print(f"[ERROR] MinerU parsing failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _process_mineru_images(self, output_dir: Path, pdf_name: str) -> List[Dict]:
        """
        Find images extracted by MinerU and copy to our images folder.
        
        MinerU saves images in: output_dir/images/*.png
        We copy them to: self.images_root/{pdf_name}_*.png
        """
        image_chunks = []
        
        # MinerU puts images in an 'images' subfolder
        mineru_images_dir = output_dir / "images"
        
        if not mineru_images_dir.exists():
            print(f"[Images] No images folder found")
            return []
        
        image_files = list(mineru_images_dir.glob("*.png")) + \
                    list(mineru_images_dir.glob("*.jpg"))
        
        print(f"[Images] Found {len(image_files)} images")
        
        for idx, img_file in enumerate(image_files):
            try:
                # Copy to our images folder with better naming
                new_name = f"{pdf_name}_fig{idx}.png"
                dest_path = self.images_root / new_name
                shutil.copy2(img_file, dest_path)
                
                # Create chunk
                chunk_id = self.create_chunk_id(pdf_name, 1, 'image', idx)
                
                metadata = self.create_metadata(
                    pdf_name=pdf_name,
                    page_num=1,  # MinerU doesn't provide page numbers easily
                    strategy_name=self.strategy_name
                )
                
                image_chunks.append({
                    'chunk_id': chunk_id,
                    'type': 'image',
                    'image_path': str(dest_path),
                    'caption': '',  # Extract from markdown if needed
                    'metadata': metadata
                })
                
            except Exception as e:
                print(f"[Images] Failed to process {img_file.name}: {e}")
        
        return image_chunks

    def _process_mineru_text(self, markdown: str, pdf_name: str, max_pages: int = None) -> List[Dict]:
        """
        Process text content from MinerU markdown output with cleaning.
        
        NEW: Uses post_processor to clean and chunk text.
        
        Args:
            markdown: Markdown text from MinerU
            pdf_name: PDF name
            max_pages: Page limit (ignored)
            
        Returns:
            List of clean text chunks
        """
        if not markdown or not markdown.strip():
            return []
        
        # Import post_processor
        from .post_processor import clean_and_chunk
        
        # Clean and chunk the text (this does EVERYTHING)
        chunks = clean_and_chunk(markdown, pdf_name, page_num=1)
        
        print(f"    [Text] Created {len(chunks)} clean chunks")
        
        return chunks
  
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
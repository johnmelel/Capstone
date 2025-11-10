"""
Simplified MinerU parser for PDF extraction.
Extracts text and images using MinerU CLI, then converts images to text descriptions using Gemini Vision API.
"""

from pathlib import Path
from typing import List, Dict, Any
import subprocess
import shutil
import re
import base64
from google import genai
from google.genai import types


class MinerUParser:
    """Parser using MinerU for advanced PDF extraction with Gemini Vision for image descriptions."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path, gemini_api_key: str, vision_model: str = None, image_prompt: str = None):
        """
        Initialize MinerU parser.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory for extracted content
            gemini_api_key: Gemini API key for image descriptions
            vision_model: Gemini model to use for image descriptions (optional)
            image_prompt: Custom prompt for image descriptions (optional)
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.gemini_api_key = gemini_api_key
        self.vision_model = vision_model or "gemini-2.0-flash-exp"
        self.image_prompt = image_prompt or (
            "Describe this image in detail. Focus on medical/scientific content, "
            "figures, charts, diagrams, or any text visible in the image. "
            "Be concise but comprehensive."
        )
        
        # Initialize Gemini client for image descriptions
        self.gemini_client = genai.Client(api_key=gemini_api_key)
        
        # Create output subdirectories
        self.images_dir = self.output_dir / "images"
        self.temp_dir = self.output_dir / "temp"
        
        for dir_path in [self.images_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Chunking parameters
        self.chunk_size = config.get('chunk_size', 800)
        self.overlap = config.get('overlap', 150)
        
        print("[MinerUParser] Initialized")
        print(f"  Output: {self.output_dir}")
        print(f"  Chunk size: {self.chunk_size}")
        print("  Image-to-text: Gemini Vision API")
    
    def parse_pdf(self, pdf_path: Path, max_pages: int = None) -> List[Dict]:
        """
        Parse PDF using MinerU.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Optional page limit (not used by MinerU)
            
        Returns:
            List of chunks with embeddings and metadata
        """
        print(f"\n[MinerUParser] Processing: {pdf_path.name}")
        
        if not pdf_path.exists():
            print(f"[ERROR] PDF not found: {pdf_path}")
            return []
        
        pdf_name = pdf_path.stem
        
        # Setup temp output directory for this PDF
        pdf_output_dir = self.temp_dir / pdf_name
        pdf_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Run MinerU CLI
            print(f"  [1/3] Running MinerU extraction...")
            success = self._run_mineru(pdf_path, pdf_output_dir)
            
            if not success:
                return []
            
            # Step 2: Find and read markdown output
            markdown = self._read_markdown_output(pdf_output_dir)
            
            if not markdown:
                print(f"  [ERROR] No markdown content extracted")
                return []
            
            # Step 3: Process images
            print(f"  [2/3] Processing images...")
            image_chunks = self._process_images(pdf_output_dir, pdf_name)
            
            # Step 4: Process text
            print(f"  [3/3] Processing text...")
            text_chunks = self._process_text(markdown, pdf_name)
            
            # Combine all chunks
            all_chunks = text_chunks + image_chunks
            
            print(f"  ✓ Complete: {len(text_chunks)} text chunks, {len(image_chunks)} images")
            
            return all_chunks
            
        except Exception as e:
            print(f"  [ERROR] Failed to parse PDF: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _run_mineru(self, pdf_path: Path, output_dir: Path) -> bool:
        """Run MinerU CLI command."""
        try:
            cmd = [
                'mineru',
                '-p', str(pdf_path),
                '-o', str(output_dir)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"    [ERROR] MinerU failed:")
                print(f"    STDERR: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            print(f"    [ERROR] MinerU timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"    [ERROR] Failed to run MinerU: {e}")
            return False
    
    def _read_markdown_output(self, output_dir: Path) -> str:
        """Find and read markdown file created by MinerU."""
        md_files = list(output_dir.glob("**/*.md"))
        
        if not md_files:
            print(f"    [ERROR] No markdown file found in {output_dir}")
            return ""
        
        md_file = md_files[0]
        return md_file.read_text(encoding='utf-8')
    
    def _process_images(self, output_dir: Path, pdf_name: str) -> List[Dict]:
        """Extract images and convert to text descriptions using Gemini Vision."""
        image_chunks = []
        
        # MinerU saves images in 'images' subfolder
        mineru_images_dir = output_dir / "images"
        
        if not mineru_images_dir.exists():
            return []
        
        image_files = list(mineru_images_dir.glob("*.png")) + \
                     list(mineru_images_dir.glob("*.jpg"))
        
        print(f"    Found {len(image_files)} images, generating descriptions...")
        
        for idx, img_file in enumerate(image_files):
            try:
                # Copy to our images folder
                new_name = f"{pdf_name}_img{idx}.png"
                dest_path = self.images_dir / new_name
                shutil.copy2(img_file, dest_path)
                
                # Generate text description using Gemini Vision
                description = self._describe_image_with_gemini(img_file)
                
                if not description:
                    description = f"[Image {idx} from {pdf_name}]"
                
                # Create text chunk with image description
                chunk_id = f"{pdf_name}_image_{idx}"
                
                image_chunks.append({
                    'chunk_id': chunk_id,
                    'type': 'text',  # Now it's text!
                    'content': f"[IMAGE DESCRIPTION]: {description}",
                    'metadata': {
                        'pdf_name': pdf_name,
                        'chunk_index': idx,
                        'content_type': 'image_description',
                        'original_image_path': str(dest_path)
                    }
                })
                
                if (idx + 1) % 5 == 0:
                    print(f"      Processed {idx + 1}/{len(image_files)} images")
                
            except Exception as e:
                print(f"    [WARNING] Failed to process image {img_file.name}: {e}")
        
        return image_chunks
    
    def _describe_image_with_gemini(self, image_path: Path) -> str:
        """
        Use Gemini Vision API to generate a text description of an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Text description of the image
        """
        try:
            # Upload the image
            uploaded_file = self.gemini_client.files.upload(path=str(image_path))
            
            # Generate description using Gemini
            response = self.gemini_client.models.generate_content(
                model=self.vision_model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(
                                file_uri=uploaded_file.uri,
                                mime_type=uploaded_file.mime_type
                            ),
                            types.Part.from_text(self.image_prompt)
                        ]
                    )
                ]
            )
            
            # Extract text from response
            if response.text:
                return response.text.strip()
            
            return ""
            
        except Exception as e:
            print(f"      [WARNING] Failed to describe image: {e}")
            return ""
    
    def _process_text(self, markdown: str, pdf_name: str) -> List[Dict]:
        """Process markdown text into chunks."""
        if not markdown or not markdown.strip():
            return []
        
        # Clean text
        text = self._clean_text(markdown)
        
        # Create chunks
        chunks = self._chunk_text(text, pdf_name)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove markdown formatting
        text = re.sub(r'#+\s*', '', text)  # Headers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        
        # Fix hyphenation: "diagnos-\ntic" → "diagnostic"
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Clean whitespace
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _chunk_text(self, text: str, pdf_name: str) -> List[Dict]:
        """Split text into chunks with overlap."""
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current = ""
        chunk_idx = 0
        
        for para in paragraphs:
            # Check if adding paragraph would exceed limit
            if len(current) + len(para) > self.chunk_size:
                # Save current chunk if substantial
                if len(current) >= 50:
                    chunks.append(self._create_chunk(current.strip(), pdf_name, chunk_idx))
                    chunk_idx += 1
                    
                    # Get overlap for next chunk
                    overlap_text = self._get_overlap(current)
                else:
                    overlap_text = ""
                
                # Start new chunk
                current = overlap_text + para + "\n\n"
            else:
                current += para + "\n\n"
        
        # Save final chunk
        if len(current) >= 50:
            chunks.append(self._create_chunk(current.strip(), pdf_name, chunk_idx))
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from end of chunk."""
        if len(text) <= self.overlap:
            return text
        
        overlap = text[-self.overlap:]
        
        # Try to start at sentence boundary
        sentence_start = overlap.find('. ')
        if sentence_start != -1:
            overlap = overlap[sentence_start + 2:]
        
        return overlap + "\n\n"
    
    def _create_chunk(self, text: str, pdf_name: str, chunk_idx: int) -> Dict:
        """Create chunk dictionary with metadata."""
        chunk_id = f"{pdf_name}_text_{chunk_idx}"
        
        # Find references to figures/tables
        fig_refs = re.findall(r'Fig\.?\s*\d+[A-Z]?', text, re.IGNORECASE)
        table_refs = re.findall(r'Table\s*\d+', text, re.IGNORECASE)
        
        return {
            'chunk_id': chunk_id,
            'type': 'text',
            'content': text,
            'metadata': {
                'pdf_name': pdf_name,
                'chunk_index': chunk_idx,
                'char_count': len(text),
                'word_count': len(text.split()),
                'content_type': 'text',
                'figure_refs': ', '.join(fig_refs) if fig_refs else None,
                'table_refs': ', '.join(table_refs) if table_refs else None
            }
        }

"""
Base parser class - All strategy parsers inherit from this.

Provides common functionality:
- Chunk ID generation
- Metadata creation
- Image/table path management
- Logging utilities
"""

from pathlib import Path
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import json
from PIL import Image
import io



class BaseParser(ABC):
    """Abstract base class for all PDF parsers."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        Initialize parser.
        
        Args:
            config: Configuration dictionary from config.yaml
            output_dir: Base directory for extracted content (e.g., extracted_content/)
        """
        self.config = config
        self.output_dir = output_dir
        
        # Create output subdirectories
        self.images_dir = output_dir / "images"
        self.tables_dir = output_dir / "tables"
        self.metadata_dir = output_dir / "metadata"
        
        for dir_path in [self.images_dir, self.tables_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get chunking parameters from config
        self.chunk_size = config.get('chunking', {}).get('chunk_size', 800)
        self.overlap = config.get('chunking', {}).get('overlap', 200)
        
        # Get image/table filters
        self.min_img_width = config.get('images', {}).get('min_width', 100)
        self.min_img_height = config.get('images', {}).get('min_height', 100)
        
        print(f"        [BaseParser] Initialized with chunk_size={self.chunk_size}, overlap={self.overlap}")
    
    @abstractmethod
    def parse_pdf(self, pdf_path: Path, max_pages: int = None) -> List[Dict]:
        """
        Parse a PDF and extract structured content.
        
        Must be implemented by each strategy parser.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Optional limit on number of pages to process
            
        Returns:
            List of dictionaries, each representing a chunk with:
            {
                'chunk_id': str,
                'type': 'text' | 'image' | 'table',
                'content': str (for text/table) or None (for image),
                'image_path': str (for images) or None,
                'metadata': dict
            }
        """
        pass
    
    def create_chunk_id(self, pdf_name: str, page_num: int, 
                       chunk_type: str, index: int) -> str:
        """
        Generate a unique chunk ID.
        
        Format: {pdf_name}_p{page}_{type}_{index}
        Example: research_paper_p5_text_2
        """
        return f"{pdf_name}_p{page_num}_{chunk_type}_{index}"
    
    def create_metadata(self, pdf_name: str, page_num: int, 
                       strategy_name: str, **kwargs) -> Dict[str, Any]:
        """
        Create standardized metadata dictionary.
        
        Args:
            pdf_name: Name of the PDF (without extension)
            page_num: Page number (1-indexed)
            strategy_name: Name of parsing strategy used
            **kwargs: Additional metadata fields
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'pdf_name': pdf_name,
            'page': page_num,
            'parsing_strategy': strategy_name,
        }
        metadata.update(kwargs)
        return metadata
    
    def save_image(self, image_data, pdf_name: str, 
                page_num: int, img_index: int, ext: str = 'png') -> Path:
        """Save both PIL Images and raw bytes"""
        
        filename = f"{pdf_name}_p{page_num}_img_{img_index}.{ext}"
        filepath = self.images_dir / filename
        
        # Check type and convert if needed
        if isinstance(image_data, Image.Image):
            # Convert PIL Image to bytes
            buffer = io.BytesIO()
            image_data.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
        elif isinstance(image_data, bytes):
            # Already bytes
            image_bytes = image_data
        else:
            raise TypeError(f"Cannot handle image type: {type(image_data)}")
        
        # Write to disk
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        return filepath

    def save_table_csv(self, table_data: str, pdf_name: str, 
                       page_num: int, table_index: int) -> Path:
        """
        Save an extracted table as CSV.
        
        Args:
            table_data: CSV-formatted string
            pdf_name: PDF name for filename
            page_num: Page number
            table_index: Table index on that page
            
        Returns:
            Path to saved CSV file
        """
        filename = f"{pdf_name}_p{page_num}_table_{table_index}.csv"
        filepath = self.tables_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(table_data)
        
        return filepath
    
    def log_extraction_summary(self, pdf_name: str, summary: Dict[str, Any]):
        """
        Save extraction summary as JSON for debugging.
        
        Args:
            pdf_name: PDF name
            summary: Dictionary with extraction statistics
        """
        log_file = self.metadata_dir / f"{pdf_name}_extraction_log.json"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[BaseParser] Saved extraction log to {log_file}")
    
    def chunk_text(self, text: str, max_size: int = None) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            max_size: Maximum chunk size (uses self.chunk_size if None)
            
        Returns:
            List of text chunks
        """
        if max_size is None:
            max_size = self.chunk_size
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + max_size
            
            # If this is not the last chunk, try to break at sentence boundary
            if end < text_len:
                # Look for sentence end in the last 20% of chunk
                search_start = end - int(max_size * 0.2)
                sentence_end = text.rfind('. ', search_start, end)
                
                if sentence_end != -1:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap if end < text_len else text_len
        
        return chunks
    
    def print_progress(self, current: int, total: int, prefix: str = "Progress"):
        """Simple progress printer."""
        percentage = (current / total) * 100 if total > 0 else 0
        print(f"[{prefix}] {current}/{total} ({percentage:.1f}%)")
"""PDF text extraction module"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import io

import pdfplumber

from .utils import clean_text


logger = logging.getLogger(__name__)
    
class PDFExtractor:
    def __init__(self, extract_images: bool = False):
        """
        Initialize PDF extractor
        
        Args:
            extract_images: Whether to extract text from images using OCR (not currently implemented)
        """
        self.extract_images = extract_images
        logger.info(f"PDFExtractor initialized (extract_images={extract_images})")
    
    def extract_text(self, pdf_source: Union[Path, Any]) -> Optional[str]:
        """
        Extract text from PDF using PyMuPDF
        
        Args:
            pdf_source: Path to PDF file or GCS blob object
            
        Returns:
            Extracted text or None if failed
        """
        import time
        start_time = time.time()
        
        try:
            # Handle different input types
            if isinstance(pdf_source, Path):
                # Local file
                if not pdf_source.exists():
                    logger.error(f"PDF file not found: {pdf_source}")
                    return None
                
                if not pdf_source.suffix.lower() == '.pdf':
                    logger.error(f"File is not a PDF: {pdf_source}")
                    return None
                
                source_name = pdf_source.name
                pdf = pdfplumber.open(pdf_source)
                
            else:
                # GCS blob (assume it has download_as_bytes method)
                source_name = getattr(pdf_source, 'name', 'unknown.pdf')
                
                # Check file size before downloading (avoid memory issues with very large files)
                if hasattr(pdf_source, 'size') and pdf_source.size > 100 * 1024 * 1024:  # 100MB limit
                    logger.warning(f"Skipping large file {source_name} ({pdf_source.size} bytes) - too big for memory processing")
                    return None
                
                logger.info(f"Extracting text from GCS: {source_name}")
                
                # Open blob as stream with timeout consideration
                blob_data = pdf_source.download_as_bytes()
                pdf = pdfplumber.open(io.BytesIO(blob_data))
            
            text_parts = []
            num_pages = len(pdf.pages)
            logger.debug(f"Processing {num_pages} pages")
            
            # Limit processing to reasonable number of pages to prevent timeouts
            max_pages = min(num_pages, 500)  # Process max 500 pages
            
            for page_num in range(max_pages):
                # Check for timeout (5 minutes max per PDF)
                if time.time() - start_time > 300:
                    logger.warning(f"Timeout reached processing {source_name}, stopping at page {page_num}")
                    break
                    
                page = pdf.pages[page_num]
                
                # Extract text from page
                text = page.extract_text()
                
                if text and text.strip():
                    text_parts.append(text)
            
            pdf.close()
            
            # Combine all text
            full_text = "\n".join(text_parts)
            cleaned_text = clean_text(full_text) if full_text else None
            
            processing_time = time.time() - start_time
            if cleaned_text:
                logger.info(f"Successfully extracted {len(cleaned_text)} characters from {source_name} in {processing_time:.2f}s")
            else:
                logger.warning(f"No text extracted from {source_name}")
            
            return cleaned_text
            
        except Exception as e:
            processing_time = time.time() - start_time
            source_name = getattr(pdf_source, 'name', str(pdf_source)) if hasattr(pdf_source, 'name') else str(pdf_source)
            logger.error(f"Error extracting text from {source_name} after {processing_time:.2f}s: {e}")
            return None
    
    def extract_with_metadata(self, pdf_source: Union[Path, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract text and metadata from PDF
        
        Args:
            pdf_source: Path to PDF file or GCS blob object
            
        Returns:
            Dictionary containing text and metadata, or None if failed
        """
        try:
            # Handle different input types
            if isinstance(pdf_source, Path):
                # Local file
                if not pdf_source.exists():
                    logger.error(f"PDF file not found: {pdf_source}")
                    return None
                
                source_name = pdf_source.name
                pdf = pdfplumber.open(pdf_source)
                
            else:
                # GCS blob
                source_name = getattr(pdf_source, 'name', 'unknown.pdf')
                logger.info(f"Extracting metadata from GCS: {source_name}")
                
                # Open blob as stream
                blob_data = pdf_source.download_as_bytes()
                pdf = pdfplumber.open(io.BytesIO(blob_data))
            
            # Extract text
            text_parts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(text)
            
            full_text = "\n".join(text_parts)
            cleaned_text = clean_text(full_text) if full_text else ""
            
            # Extract metadata (pdfplumber's metadata is more limited)
            metadata = {
                'title': pdf.metadata.get('Title', '') if pdf.metadata else '',
                'author': pdf.metadata.get('Author', '') if pdf.metadata else '',
                'subject': pdf.metadata.get('Subject', '') if pdf.metadata else '',
                'creator': pdf.metadata.get('Creator', '') if pdf.metadata else '',
                'producer': pdf.metadata.get('Producer', '') if pdf.metadata else '',
                'creation_date': pdf.metadata.get('CreationDate', '') if pdf.metadata else '',
                'modification_date': pdf.metadata.get('ModDate', '') if pdf.metadata else '',
                'num_pages': len(pdf.pages),
                'file_name': source_name
            }
            
            pdf.close()
            
            return {
                'text': cleaned_text,
                'metadata': metadata
            }
            
        except Exception as e:
            source_name = getattr(pdf_source, 'name', str(pdf_source)) if hasattr(pdf_source, 'name') else str(pdf_source)
            logger.error(f"Error extracting from {source_name}: {e}")
            return None
    
    def get_page_count(self, pdf_source: Union[Path, Any]) -> int:
        """
        Get the number of pages in a PDF
        
        Args:
            pdf_source: Path to PDF file or GCS blob object
            
        Returns:
            Number of pages, or 0 if error
        """
        try:
            if isinstance(pdf_source, Path):
                # Local file
                if not pdf_source.exists():
                    return 0
                pdf = pdfplumber.open(pdf_source)
            else:
                # GCS blob
                blob_data = pdf_source.download_as_bytes()
                pdf = pdfplumber.open(io.BytesIO(blob_data))
            
            page_count = len(pdf.pages)
            pdf.close()
            return page_count
            
        except Exception as e:
            source_name = getattr(pdf_source, 'name', str(pdf_source)) if hasattr(pdf_source, 'name') else str(pdf_source)
            logger.error(f"Error getting page count from {source_name}: {e}")
            return 0


def extract_text_from_pdf(pdf_source: Union[Path, Any], extract_images: bool = False) -> Optional[str]:
    """
    Convenience function to extract text from a PDF
    
    Args:
        pdf_source: Path to PDF file or GCS blob object
        extract_images: Whether to extract text from images using OCR
        
    Returns:
        Extracted text or None if failed
    """
    extractor = PDFExtractor(extract_images=extract_images)
    return extractor.extract_text(pdf_source)

"""PDF text extraction module using MinerU"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import tempfile
import time
import shutil
import json

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    from mineru.cli.common import do_parse, read_fn
    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False

from .utils import clean_text
from .config import Config
from .exceptions import PDFExtractionError
from .constants import MAX_PDF_SIZE_BYTES
from .types import PDFExtractionResult, PDFMetadata, MultimodalPDFExtractionResult, ImageData


logger = logging.getLogger(__name__)
    
class PDFExtractor:
    def __init__(self, extract_images: bool = False):
        """
        Initialize PDF extractor with MinerU
        
        Args:
            extract_images: Whether to extract images (Phase 1: not used)
        """
        self.extract_images = extract_images
        
        # Check MinerU availability
        if not MINERU_AVAILABLE:
            logger.error("MinerU is not installed. Please install with: pip install mineru[core]")
            raise ImportError("MinerU not available")
        
        # Detect GPU availability
        self.use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
        if self.use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"PDFExtractor initialized with GPU acceleration: {gpu_name}")
        else:
            logger.info("PDFExtractor initialized with CPU (GPU not available)")
        
        # Configure MinerU backend
        self.backend = Config.MINERU_BACKEND
        self.model_source = Config.MINERU_MODEL_SOURCE
        self.lang = Config.MINERU_LANG
        self.timeout = Config.PDF_EXTRACTION_TIMEOUT
        self.debug_mode = Config.MINERU_DEBUG_MODE
        self.enable_tables = Config.MINERU_ENABLE_TABLES
        self.enable_formulas = Config.MINERU_ENABLE_FORMULAS
        
        logger.info(f"MinerU config: backend={self.backend}, model_source={self.model_source}, "
                   f"lang={self.lang}, timeout={self.timeout}s, debug={self.debug_mode}")
    
    def _extract_text_from_markdown(self, md_file_path: Path) -> Optional[str]:
        """
        Extract plain text from MinerU's markdown output
        
        Args:
            md_file_path: Path to markdown file
            
        Returns:
            Extracted text or None
        """
        try:
            if not md_file_path.exists():
                logger.error(f"Markdown file not found: {md_file_path}")
                return None
            
            with open(md_file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # For Phase 1, we use the markdown as-is (it's already clean text)
            # Phase 2 could parse markdown for structured content
            return markdown_content.strip() if markdown_content else None
            
        except (IOError, OSError) as e:
            logger.error(f"File I/O error reading markdown {md_file_path}: {e}")
            return None
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading markdown {md_file_path}: {e}")
            return None
    
    def _extract_images_from_output(self, output_dir: Path, pdf_name: str) -> List[ImageData]:
        """
        Extract images from MinerU output directory
        
        Args:
            output_dir: Base output directory
            pdf_name: Name of the PDF (without extension)
            
        Returns:
            List of ImageData objects
        """
        images = []
        
        try:
            # Images are in: output_dir/pdf_name/auto/images/
            images_dir = output_dir / pdf_name / 'auto' / 'images'
            
            if not images_dir.exists():
                logger.debug(f"No images directory found at {images_dir}")
                return images
            
            # Find all image files
            image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
            
            if not image_files:
                logger.debug(f"No images found in {images_dir}")
                return images
            
            logger.info(f"Found {len(image_files)} images in {images_dir}")
            
            # Load content_list.json for metadata if available
            content_list_path = output_dir / pdf_name / 'auto' / f"{pdf_name}_content_list.json"
            image_metadata_map = {}
            
            if content_list_path.exists():
                try:
                    with open(content_list_path, 'r', encoding='utf-8') as f:
                        content_data = json.load(f)
                        # Extract image metadata from content list
                        # Format varies, but typically has image info with page numbers
                        if isinstance(content_data, list):
                            for item in content_data:
                                if isinstance(item, dict) and item.get('type') == 'image':
                                    img_path = item.get('img_path', '')
                                    if img_path:
                                        image_metadata_map[Path(img_path).name] = item
                except Exception as e:
                    logger.warning(f"Failed to parse content_list.json: {e}")
            
            # Process each image
            for img_path in image_files:
                try:
                    # Read image bytes
                    with open(img_path, 'rb') as f:
                        img_bytes = f.read()
                    
                    # Get image dimensions
                    size = (0, 0)
                    if PIL_AVAILABLE:
                        try:
                            with Image.open(img_path) as img:
                                size = img.size
                        except Exception as e:
                            logger.warning(f"Failed to get image size for {img_path}: {e}")
                    
                    # Parse filename for page/index info
                    # MinerU format: image_1_2.png (page 1, image 2)
                    filename = img_path.stem
                    page_num = 0
                    image_index = 0
                    
                    try:
                        parts = filename.replace('image_', '').split('_')
                        if len(parts) >= 2:
                            page_num = int(parts[0])
                            image_index = int(parts[1])
                    except (ValueError, IndexError):
                        logger.debug(f"Could not parse page/index from filename: {filename}")
                    
                    # Get metadata from content list if available
                    bbox = None
                    metadata = image_metadata_map.get(img_path.name, {})
                    if metadata.get('bbox'):
                        bbox = metadata['bbox']
                    
                    # Create ImageData object
                    image_data: ImageData = {
                        'path': img_path,
                        'bytes': img_bytes,
                        'page_num': page_num,
                        'image_index': image_index,
                        'bbox': bbox,
                        'size': size,
                        'gcs_path': None  # Will be set after GCS upload
                    }
                    
                    images.append(image_data)
                    logger.debug(f"Extracted image: {img_path.name}, page={page_num}, size={size}")
                    
                except Exception as e:
                    logger.error(f"Failed to process image {img_path}: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(images)} images")
            return images
            
        except Exception as e:
            logger.error(f"Error extracting images from {output_dir}: {e}")
            return []
    
    def _process_pdf_with_mineru(self, pdf_path: Path, output_dir: Path) -> Optional[str]:
        """
        Process PDF using MinerU official API
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory for MinerU outputs
            
        Returns:
            Extracted text or None
        """
        try:
            # Read PDF bytes using official read_fn
            pdf_bytes = read_fn(pdf_path)
            
            # Use official do_parse function
            pdf_name = pdf_path.stem
            do_parse(
                output_dir=str(output_dir),
                pdf_file_names=[pdf_name],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[self.lang],
                backend=self.backend,
                parse_method='auto',
                formula_enable=self.enable_formulas,
                table_enable=self.enable_tables,
                f_dump_md=True,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=True,  # Always enable for image extraction
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False
            )
            
            # Read the generated markdown file
            md_file_path = output_dir / pdf_name / 'auto' / f"{pdf_name}.md"
            
            if not md_file_path.exists():
                logger.error(f"Markdown file not found at expected path: {md_file_path}")
                return None
            
            with open(md_file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            logger.info(f"MinerU processing complete: {md_file_path}")
            
            # Extract text from markdown
            return md_content.strip() if md_content else None
            
        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"File I/O error processing PDF with MinerU: {e}")
            return None
        except ImportError as e:
            logger.error(f"Missing MinerU dependency: {e}")
            raise PDFExtractionError("MinerU dependencies not properly installed") from e
        except Exception as e:
            logger.error(f"Unexpected error processing PDF with MinerU: {e}")
            return None
    
    def extract_text(self, pdf_source: Union[Path, Any]) -> Optional[str]:
        """
        Extract text from PDF using MinerU
        
        Args:
            pdf_source: Path to PDF file or GCS blob object
            
        Returns:
            Extracted text or None if failed
        """
        start_time = time.time()
        temp_pdf_path = None
        temp_output_dir = None
        
        try:
            # Determine source name for logging
            if isinstance(pdf_source, Path):
                source_name = pdf_source.name
                pdf_path = pdf_source
                
                # Validate file
                if not pdf_path.exists():
                    logger.error(f"PDF file not found: {pdf_path}")
                    return None
                
                if not pdf_path.suffix.lower() == '.pdf':
                    logger.error(f"File is not a PDF: {pdf_path}")
                    return None
                
                # Create temp output directory
                temp_output_dir = Path(tempfile.mkdtemp(prefix="mineru_output_"))
                
            else:
                # GCS blob - need to download to temp file
                source_name = getattr(pdf_source, 'name', 'unknown.pdf')
                
                # Check file size before downloading (avoid memory issues)
                if hasattr(pdf_source, 'size') and pdf_source.size > MAX_PDF_SIZE_BYTES:
                    logger.warning(f"Skipping large file {source_name} ({pdf_source.size} bytes) - exceeds {MAX_PDF_SIZE_BYTES} byte limit")
                    return None
                
                logger.info(f"Extracting text from GCS: {source_name}")
                
                # Download blob to temporary file
                temp_pdf_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                temp_pdf_path = Path(temp_pdf_file.name)
                
                try:
                    blob_data = pdf_source.download_as_bytes()
                    temp_pdf_file.write(blob_data)
                    temp_pdf_file.flush()
                    temp_pdf_file.close()
                    pdf_path = temp_pdf_path
                    
                    # Create temp output directory
                    temp_output_dir = Path(tempfile.mkdtemp(prefix="mineru_output_"))
                    
                except (IOError, OSError) as e:
                    logger.error(f"I/O error downloading GCS blob {source_name}: {e}")
                    temp_pdf_file.close()
                    if temp_pdf_path and temp_pdf_path.exists():
                        temp_pdf_path.unlink()
                    return None
                except Exception as e:
                    logger.error(f"Unexpected error downloading GCS blob {source_name}: {e}")
                    temp_pdf_file.close()
                    if temp_pdf_path and temp_pdf_path.exists():
                        temp_pdf_path.unlink()
                    return None
            
            # Check for timeout before processing
            if time.time() - start_time > self.timeout:
                logger.warning(f"Timeout reached before processing {source_name}")
                return None
            
            # Process PDF with MinerU
            logger.debug(f"Processing {source_name} with MinerU")
            extracted_text = self._process_pdf_with_mineru(pdf_path, temp_output_dir)
            
            # Clean text
            cleaned_text = clean_text(extracted_text) if extracted_text else None
            
            processing_time = time.time() - start_time
            if cleaned_text:
                logger.info(f"Successfully extracted {len(cleaned_text)} characters from {source_name} "
                           f"in {processing_time:.2f}s")
            else:
                logger.warning(f"No text extracted from {source_name}")
            
            return cleaned_text
            
        except (IOError, OSError, FileNotFoundError) as e:
            processing_time = time.time() - start_time
            source_name = getattr(pdf_source, 'name', str(pdf_source)) if hasattr(pdf_source, 'name') else str(pdf_source)
            logger.error(f"File I/O error extracting text from {source_name} after {processing_time:.2f}s: {e}")
            return None
        except TimeoutError as e:
            processing_time = time.time() - start_time
            source_name = getattr(pdf_source, 'name', str(pdf_source)) if hasattr(pdf_source, 'name') else str(pdf_source)
            logger.error(f"Timeout extracting text from {source_name} after {processing_time:.2f}s: {e}")
            return None
        except Exception as e:
            processing_time = time.time() - start_time
            source_name = getattr(pdf_source, 'name', str(pdf_source)) if hasattr(pdf_source, 'name') else str(pdf_source)
            logger.error(f"Unexpected error extracting text from {source_name} after {processing_time:.2f}s: {e}")
            return None
            
        finally:
            # Cleanup temporary files
            if not self.debug_mode:
                # Delete temp PDF (from GCS download)
                if temp_pdf_path and temp_pdf_path.exists():
                    try:
                        temp_pdf_path.unlink()
                        logger.debug(f"Deleted temp PDF: {temp_pdf_path}")
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Failed to delete temp PDF {temp_pdf_path}: {e}")
                
                # Delete temp output directory
                if temp_output_dir and temp_output_dir.exists():
                    try:
                        shutil.rmtree(temp_output_dir)
                        logger.debug(f"Deleted temp output dir: {temp_output_dir}")
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Failed to delete temp output dir {temp_output_dir}: {e}")
            else:
                # Debug mode - keep files
                if temp_pdf_path:
                    logger.info(f"Debug mode: Kept temp PDF at {temp_pdf_path}")
                if temp_output_dir:
                    logger.info(f"Debug mode: Kept output dir at {temp_output_dir}")
    
    def extract_with_metadata(self, pdf_source: Union[Path, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract text and metadata from PDF using MinerU
        
        Args:
            pdf_source: Path to PDF file or GCS blob object
            
        Returns:
            Dictionary containing text and metadata, or None if failed
        """
        temp_pdf_path = None
        temp_output_dir = None
        
        try:
            # Determine source name
            if isinstance(pdf_source, Path):
                source_name = pdf_source.name
                pdf_path = pdf_source
                
                if not pdf_path.exists():
                    logger.error(f"PDF file not found: {pdf_path}")
                    return None
                
                temp_output_dir = Path(tempfile.mkdtemp(prefix="mineru_output_"))
                
            else:
                # GCS blob
                source_name = getattr(pdf_source, 'name', 'unknown.pdf')
                logger.info(f"Extracting metadata from GCS: {source_name}")
                
                # Download to temp file
                temp_pdf_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                temp_pdf_path = Path(temp_pdf_file.name)
                
                blob_data = pdf_source.download_as_bytes()
                temp_pdf_file.write(blob_data)
                temp_pdf_file.flush()
                temp_pdf_file.close()
                pdf_path = temp_pdf_path
                
                temp_output_dir = Path(tempfile.mkdtemp(prefix="mineru_output_"))
            
            # Extract text using MinerU
            extracted_text = self._process_pdf_with_mineru(pdf_path, temp_output_dir)
            cleaned_text = clean_text(extracted_text) if extracted_text else ""
            
            # Basic metadata (MinerU doesn't extract PDF metadata like title/author)
            # We can extend this in Phase 2 if needed
            metadata = {
                'title': '',
                'author': '',
                'subject': '',
                'creator': 'MinerU',
                'producer': 'MinerU',
                'creation_date': '',
                'modification_date': '',
                'num_pages': 0,  # Could parse from content_list.json if needed
                'file_name': source_name
            }
            
            return {
                'text': cleaned_text,
                'metadata': metadata
            }
            
        except (IOError, OSError, FileNotFoundError) as e:
            source_name = getattr(pdf_source, 'name', str(pdf_source)) if hasattr(pdf_source, 'name') else str(pdf_source)
            logger.error(f"File I/O error extracting from {source_name}: {e}")
            return None
        except Exception as e:
            source_name = getattr(pdf_source, 'name', str(pdf_source)) if hasattr(pdf_source, 'name') else str(pdf_source)
            logger.error(f"Unexpected error extracting from {source_name}: {e}")
            return None
            
        finally:
            # Cleanup
            if not self.debug_mode:
                if temp_pdf_path and temp_pdf_path.exists():
                    try:
                        temp_pdf_path.unlink()
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Failed to delete temp PDF: {e}")
                
                if temp_output_dir and temp_output_dir.exists():
                    try:
                        shutil.rmtree(temp_output_dir)
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Failed to delete temp output dir: {e}")
    
    def get_page_count(self, pdf_source: Union[Path, Any]) -> int:
        """
        Get the number of pages in a PDF
        Note: MinerU doesn't provide direct page count, returning 0 for now
        Phase 2 could parse content_list.json for accurate page count
        
        Args:
            pdf_source: Path to PDF file or GCS blob object
            
        Returns:
            Number of pages, or 0 if not available
        """
        try:
            # For Phase 1, we don't extract page count (would require full processing)
            # Phase 2 could add lightweight page count extraction
            logger.debug("Page count not available in Phase 1 (MinerU)")
            return 0
            
        except Exception as e:
            source_name = getattr(pdf_source, 'name', str(pdf_source)) if hasattr(pdf_source, 'name') else str(pdf_source)
            logger.error(f"Error getting page count from {source_name}: {e}")
            return 0
    
    def extract_with_images(self, pdf_source: Union[Path, Any]) -> Optional[MultimodalPDFExtractionResult]:
        """
        Extract text, images, and metadata from PDF using MinerU
        
        Args:
            pdf_source: Path to PDF file or GCS blob object
            
        Returns:
            MultimodalPDFExtractionResult with text, images, and metadata, or None if failed
        """
        start_time = time.time()
        temp_pdf_path = None
        temp_output_dir = None
        
        try:
            # Determine source name
            if isinstance(pdf_source, Path):
                source_name = pdf_source.name
                pdf_path = pdf_source
                
                if not pdf_path.exists():
                    logger.error(f"PDF file not found: {pdf_path}")
                    return None
                
                temp_output_dir = Path(tempfile.mkdtemp(prefix="mineru_output_"))
                
            else:
                # GCS blob
                source_name = getattr(pdf_source, 'name', 'unknown.pdf')
                logger.info(f"Extracting text and images from GCS: {source_name}")
                
                # Check file size
                if hasattr(pdf_source, 'size') and pdf_source.size > MAX_PDF_SIZE_BYTES:
                    logger.warning(f"Skipping large file {source_name} ({pdf_source.size} bytes)")
                    return None
                
                # Download to temp file
                temp_pdf_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                temp_pdf_path = Path(temp_pdf_file.name)
                
                blob_data = pdf_source.download_as_bytes()
                temp_pdf_file.write(blob_data)
                temp_pdf_file.flush()
                temp_pdf_file.close()
                pdf_path = temp_pdf_path
                
                temp_output_dir = Path(tempfile.mkdtemp(prefix="mineru_output_"))
            
            # Check timeout
            if time.time() - start_time > self.timeout:
                logger.warning(f"Timeout reached before processing {source_name}")
                return None
            
            # Process PDF with MinerU
            logger.debug(f"Processing {source_name} with MinerU (images enabled)")
            extracted_text = self._process_pdf_with_mineru(pdf_path, temp_output_dir)
            cleaned_text = clean_text(extracted_text) if extracted_text else ""
            
            # Extract images - always extract when extract_with_images is called
            images = []
            pdf_name = pdf_path.stem
            images = self._extract_images_from_output(temp_output_dir, pdf_name)
            
            # Basic metadata
            metadata: PDFMetadata = {
                'title': '',
                'author': '',
                'subject': '',
                'creator': 'MinerU',
                'producer': 'MinerU',
                'creation_date': '',
                'modification_date': '',
                'num_pages': 0,
                'file_name': source_name
            }
            
            processing_time = time.time() - start_time
            logger.info(
                f"Successfully extracted {len(cleaned_text)} chars and {len(images)} images "
                f"from {source_name} in {processing_time:.2f}s"
            )
            
            result: MultimodalPDFExtractionResult = {
                'text': cleaned_text,
                'images': images,
                'metadata': metadata
            }
            
            return result
            
        except (IOError, OSError, FileNotFoundError) as e:
            processing_time = time.time() - start_time
            source_name = getattr(pdf_source, 'name', str(pdf_source)) if hasattr(pdf_source, 'name') else str(pdf_source)
            logger.error(f"File I/O error extracting from {source_name} after {processing_time:.2f}s: {e}")
            return None
        except TimeoutError as e:
            processing_time = time.time() - start_time
            source_name = getattr(pdf_source, 'name', str(pdf_source)) if hasattr(pdf_source, 'name') else str(pdf_source)
            logger.error(f"Timeout extracting from {source_name} after {processing_time:.2f}s: {e}")
            return None
        except Exception as e:
            processing_time = time.time() - start_time
            source_name = getattr(pdf_source, 'name', str(pdf_source)) if hasattr(pdf_source, 'name') else str(pdf_source)
            logger.error(f"Unexpected error extracting from {source_name} after {processing_time:.2f}s: {e}")
            return None
            
        finally:
            # Cleanup temporary files
            if not self.debug_mode:
                if temp_pdf_path and temp_pdf_path.exists():
                    try:
                        temp_pdf_path.unlink()
                        logger.debug(f"Deleted temp PDF: {temp_pdf_path}")
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Failed to delete temp PDF {temp_pdf_path}: {e}")
                
                if temp_output_dir and temp_output_dir.exists():
                    try:
                        shutil.rmtree(temp_output_dir)
                        logger.debug(f"Deleted temp output dir: {temp_output_dir}")
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Failed to delete temp output dir {temp_output_dir}: {e}")
            else:
                if temp_pdf_path:
                    logger.info(f"Debug mode: Kept temp PDF at {temp_pdf_path}")
                if temp_output_dir:
                    logger.info(f"Debug mode: Kept output dir at {temp_output_dir}")


def extract_text_from_pdf(pdf_source: Union[Path, Any], extract_images: bool = False) -> Optional[str]:
    """
    Convenience function to extract text from a PDF using MinerU
    
    Args:
        pdf_source: Path to PDF file or GCS blob object
        extract_images: Whether to extract images (Phase 1: not used)
        
    Returns:
        Extracted text or None if failed
    """
    extractor = PDFExtractor(extract_images=extract_images)
    return extractor.extract_text(pdf_source)

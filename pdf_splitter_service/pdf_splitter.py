"""
Standalone PDF Splitter Service for GCS

This service checks all PDFs in a GCS bucket (recursively), identifies large PDFs,
and splits them into smaller parts with an average target size of 25MB.
Split PDFs are uploaded to a new folder with a modified name.
"""

import logging
import sys
import os
import math
from pathlib import Path
from typing import List, Tuple

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
from google.oauth2 import service_account
import fitz  # PyMuPDF

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pdf_splitter.log')
    ]
)
logger = logging.getLogger(__name__)


class PDFSplitterService:
    """Service to split large PDFs in GCS bucket"""
    
    # Target average size for split PDFs (25MB)
    TARGET_SIZE_MB = 25
    TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024
    
    # Minimum size to consider splitting (to avoid splitting small files)
    MIN_SIZE_FOR_SPLIT_MB = 50
    MIN_SIZE_FOR_SPLIT_BYTES = MIN_SIZE_FOR_SPLIT_MB * 1024 * 1024
    
    def __init__(
        self,
        service_account_file: str,
        bucket_name: str,
        bucket_prefix: str = "",
        split_folder_suffix: str = "_split"
    ):
        """
        Initialize PDF Splitter Service
        
        Args:
            service_account_file: Path to GCS service account JSON
            bucket_name: GCS bucket name
            bucket_prefix: Prefix/folder in bucket to check
            split_folder_suffix: Suffix to add to folder names for split PDFs
        """
        self.bucket_name = bucket_name
        self.bucket_prefix = bucket_prefix.rstrip('/') if bucket_prefix else ''
        self.split_folder_suffix = split_folder_suffix
        
        # Authenticate with GCS
        self.client, self.bucket = self._authenticate(service_account_file)
        
        logger.info("PDFSplitterService initialized")
        logger.info(f"Bucket: {bucket_name}, Prefix: {bucket_prefix or '(root)'}")
        logger.info(f"Target size: {self.TARGET_SIZE_MB}MB, Min split size: {self.MIN_SIZE_FOR_SPLIT_MB}MB")
    
    def _authenticate(self, service_account_file: str) -> Tuple[storage.Client, storage.Bucket]:
        """Authenticate with GCS"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file
            )
            client = storage.Client(credentials=credentials)
            bucket = client.bucket(self.bucket_name)
            
            if not bucket.exists():
                raise ValueError(f"Bucket '{self.bucket_name}' does not exist")
            
            logger.info(f"Successfully authenticated with GCS bucket: {self.bucket_name}")
            return client, bucket
            
        except Exception as e:
            logger.error(f"Failed to authenticate with GCS: {e}")
            raise
    
    def list_all_pdfs(self) -> List[storage.Blob]:
        """
        List all PDF blobs in bucket (recursively)
        
        Returns:
            List of PDF blob objects
        """
        try:
            prefix = f"{self.bucket_prefix}/" if self.bucket_prefix else None
            
            # List all blobs recursively
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            # Filter for PDFs only
            pdf_blobs = [
                blob for blob in blobs
                if blob.name.lower().endswith('.pdf') and not blob.name.endswith('/')
            ]
            
            logger.info(f"Found {len(pdf_blobs)} PDF files in bucket")
            return pdf_blobs
            
        except GoogleCloudError as e:
            logger.error(f"Failed to list PDFs from bucket: {e}")
            raise
    
    def identify_large_pdfs(self, blobs: List[storage.Blob]) -> List[storage.Blob]:
        """
        Identify PDFs that exceed the minimum size for splitting
        
        Args:
            blobs: List of PDF blobs
            
        Returns:
            List of large PDF blobs that should be split
        """
        large_blobs = []
        
        for blob in blobs:
            # Skip if already in a split folder
            if self.split_folder_suffix in blob.name:
                logger.debug(f"Skipping already-split PDF: {blob.name}")
                continue
            
            # Check size
            size_mb = blob.size / (1024 * 1024)
            if blob.size >= self.MIN_SIZE_FOR_SPLIT_BYTES:
                logger.info(f"Large PDF found: {blob.name} ({size_mb:.2f}MB)")
                large_blobs.append(blob)
            else:
                logger.debug(f"Skipping small PDF: {blob.name} ({size_mb:.2f}MB)")
        
        logger.info(f"Identified {len(large_blobs)} large PDFs for splitting")
        return large_blobs
    
    def calculate_split_strategy(self, total_size_bytes: int, num_pages: int) -> Tuple[int, int]:
        """
        Calculate how to split a PDF based on size and pages
        
        Args:
            total_size_bytes: Total size of PDF in bytes
            num_pages: Total number of pages
            
        Returns:
            Tuple of (num_parts, pages_per_part)
        """
        # Calculate number of parts needed to get close to target size
        num_parts = max(1, math.ceil(total_size_bytes / self.TARGET_SIZE_BYTES))
        
        # Calculate pages per part (rounded up to avoid empty parts)
        pages_per_part = math.ceil(num_pages / num_parts)
        
        # Recalculate num_parts based on pages (may be less than original)
        actual_num_parts = math.ceil(num_pages / pages_per_part)
        
        logger.info(
            f"Split strategy: {num_pages} pages into {actual_num_parts} parts "
            f"({pages_per_part} pages/part, ~{total_size_bytes/(actual_num_parts*1024*1024):.1f}MB/part)"
        )
        
        return actual_num_parts, pages_per_part
    
    def split_pdf(self, pdf_bytes: bytes, blob_name: str) -> List[Tuple[bytes, str]]:
        """
        Split a PDF into multiple smaller PDFs using PyMuPDF
        
        Args:
            pdf_bytes: PDF file bytes
            blob_name: Original blob name
            
        Returns:
            List of tuples (pdf_bytes, part_name) for each split part
        """
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            num_pages = len(doc)
            
            logger.info(f"Splitting PDF: {blob_name} ({num_pages} pages)")
            
            # Calculate split strategy
            num_parts, pages_per_part = self.calculate_split_strategy(
                len(pdf_bytes), num_pages
            )
            
            # If only 1 part needed, don't split
            if num_parts <= 1:
                logger.info(f"PDF doesn't need splitting: {blob_name}")
                doc.close()
                return []
            
            # Create split PDFs
            split_pdfs = []
            
            for part_num in range(num_parts):
                # Calculate page range for this part
                start_page = part_num * pages_per_part
                end_page = min(start_page + pages_per_part, num_pages)
                
                # Create new PDF with selected pages
                part_doc = fitz.open()
                part_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)
                
                # Write to bytes
                part_bytes = part_doc.tobytes()
                part_doc.close()
                
                # Generate part name
                part_name = self._generate_part_name(blob_name, part_num + 1, num_parts)
                
                part_size_mb = len(part_bytes) / (1024 * 1024)
                logger.info(
                    f"Created part {part_num + 1}/{num_parts}: "
                    f"pages {start_page + 1}-{end_page}, {part_size_mb:.2f}MB"
                )
                
                split_pdfs.append((part_bytes, part_name))
            
            doc.close()
            return split_pdfs
            
        except Exception as e:
            logger.error(f"Failed to split PDF {blob_name}: {e}")
            raise
    
    def _generate_part_name(self, original_blob_name: str, part_num: int, total_parts: int) -> str:
        """
        Generate new blob name for a split part
        
        Example: 
            Input: "documents/medical_paper.pdf"
            Output: "documents/medical_paper_split/medical_paper_part_1_of_3.pdf"
        
        Args:
            original_blob_name: Original blob path in bucket
            part_num: Part number (1-indexed)
            total_parts: Total number of parts
            
        Returns:
            New blob name for the part
        """
        # Parse original name
        blob_path = Path(original_blob_name)
        folder = blob_path.parent
        filename = blob_path.stem
        extension = blob_path.suffix
        
        # Create new folder name with suffix
        new_folder = folder / f"{filename}{self.split_folder_suffix}"
        
        # Create new filename with part number
        new_filename = f"{filename}_part_{part_num}_of_{total_parts}{extension}"
        
        # Combine into full path
        new_blob_name = str(new_folder / new_filename).replace('\\', '/')
        
        return new_blob_name
    
    def upload_split_pdfs(self, split_pdfs: List[Tuple[bytes, str]]) -> List[str]:
        """
        Upload split PDFs to GCS bucket
        
        Args:
            split_pdfs: List of (pdf_bytes, blob_name) tuples
            
        Returns:
            List of uploaded blob names
        """
        uploaded_names = []
        
        for pdf_bytes, blob_name in split_pdfs:
            try:
                # Create blob
                blob = self.bucket.blob(blob_name)
                
                # Upload
                blob.upload_from_string(
                    pdf_bytes,
                    content_type='application/pdf'
                )
                
                size_mb = len(pdf_bytes) / (1024 * 1024)
                logger.info(f"Uploaded: {blob_name} ({size_mb:.2f}MB)")
                uploaded_names.append(blob_name)
                
            except GoogleCloudError as e:
                logger.error(f"Failed to upload {blob_name}: {e}")
                continue
        
        return uploaded_names
    
    def process_large_pdf(self, blob: storage.Blob) -> List[str]:
        """
        Process a single large PDF: download, split, upload
        
        Args:
            blob: Large PDF blob to process
            
        Returns:
            List of uploaded split PDF names
        """
        try:
            logger.info(f"Processing large PDF: {blob.name}")
            
            # Download PDF bytes
            pdf_bytes = blob.download_as_bytes()
            
            # Split PDF
            split_pdfs = self.split_pdf(pdf_bytes, blob.name)
            
            # If no split needed, return empty list
            if not split_pdfs:
                return []
            
            # Upload split PDFs
            uploaded_names = self.upload_split_pdfs(split_pdfs)
            
            logger.info(
                f"Successfully processed {blob.name}: "
                f"created {len(uploaded_names)} split parts"
            )
            
            return uploaded_names
            
        except Exception as e:
            logger.error(f"Failed to process {blob.name}: {e}")
            return []
    
    def run(self, dry_run: bool = False) -> dict:
        """
        Run the PDF splitting service
        
        Args:
            dry_run: If True, only identify large PDFs without splitting
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info("=" * 80)
        logger.info("Starting PDF Splitter Service")
        logger.info("=" * 80)
        
        # List all PDFs
        all_pdfs = self.list_all_pdfs()
        
        # Identify large PDFs
        large_pdfs = self.identify_large_pdfs(all_pdfs)
        
        if not large_pdfs:
            logger.info("No large PDFs found that need splitting")
            return {
                'total_pdfs': len(all_pdfs),
                'large_pdfs': 0,
                'processed': 0,
                'split_parts_created': 0,
                'failed': 0
            }
        
        if dry_run:
            logger.info("DRY RUN MODE - No PDFs will be split")
            logger.info(f"Would split {len(large_pdfs)} PDFs:")
            for blob in large_pdfs:
                size_mb = blob.size / (1024 * 1024)
                logger.info(f"  - {blob.name} ({size_mb:.2f}MB)")
            
            return {
                'total_pdfs': len(all_pdfs),
                'large_pdfs': len(large_pdfs),
                'processed': 0,
                'split_parts_created': 0,
                'failed': 0,
                'dry_run': True
            }
        
        # Process each large PDF
        processed = 0
        split_parts_created = 0
        failed = 0
        
        for blob in large_pdfs:
            uploaded_parts = self.process_large_pdf(blob)
            
            if uploaded_parts:
                processed += 1
                split_parts_created += len(uploaded_parts)
            else:
                failed += 1
        
        # Summary
        logger.info("=" * 80)
        logger.info("PDF Splitter Service Complete")
        logger.info(f"Total PDFs scanned: {len(all_pdfs)}")
        logger.info(f"Large PDFs identified: {len(large_pdfs)}")
        logger.info(f"Successfully processed: {processed}")
        logger.info(f"Split parts created: {split_parts_created}")
        logger.info(f"Failed: {failed}")
        logger.info("=" * 80)
        
        return {
            'total_pdfs': len(all_pdfs),
            'large_pdfs': len(large_pdfs),
            'processed': processed,
            'split_parts_created': split_parts_created,
            'failed': failed
        }


def main():
    """Main entry point for PDF splitter service"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Split large PDFs in GCS bucket into smaller parts'
    )
    parser.add_argument(
        '--service-account',
        default=os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON', 'service_account.json'),
        help='Path to GCS service account JSON'
    )
    parser.add_argument(
        '--bucket',
        default=os.getenv('GCS_BUCKET_NAME'),
        help='GCS bucket name'
    )
    parser.add_argument(
        '--prefix',
        default=os.getenv('GCS_BUCKET_PREFIX', ''),
        help='Bucket prefix/folder to check'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=25,
        help='Target size in MB for split PDFs (default: 25)'
    )
    parser.add_argument(
        '--min-size',
        type=int,
        default=50,
        help='Minimum size in MB to consider splitting (default: 50)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only identify large PDFs without splitting'
    )
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.bucket:
        logger.error("GCS_BUCKET_NAME is required (set via --bucket or environment variable)")
        sys.exit(1)
    
    # Update target sizes if provided
    if args.target_size:
        PDFSplitterService.TARGET_SIZE_MB = args.target_size
        PDFSplitterService.TARGET_SIZE_BYTES = args.target_size * 1024 * 1024
    
    if args.min_size:
        PDFSplitterService.MIN_SIZE_FOR_SPLIT_MB = args.min_size
        PDFSplitterService.MIN_SIZE_FOR_SPLIT_BYTES = args.min_size * 1024 * 1024
    
    # Create and run service
    try:
        service = PDFSplitterService(
            service_account_file=args.service_account,
            bucket_name=args.bucket,
            bucket_prefix=args.prefix
        )
        
        results = service.run(dry_run=args.dry_run)
        
        # Exit with appropriate code
        if results.get('failed', 0) > 0:
            logger.warning("Some PDFs failed to process")
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

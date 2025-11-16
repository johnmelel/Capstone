"""Google Cloud Storage downloader module"""

import logging
from pathlib import Path
from typing import List, Optional

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError, NotFound
from google.oauth2 import service_account

from .config import Config
from .exceptions import GCSError
from .utils import sanitize_filename


logger = logging.getLogger(__name__)


class GCSDownloader:
    """Class to handle Google Cloud Storage PDF downloads"""
    
    def __init__(
        self,
        service_account_file: str,
        bucket_name: str,
        bucket_prefix: str,
        download_dir: Path,
        recursive: bool = True
    ):
        """
        Initialize Google Cloud Storage downloader
        
        Args:
            service_account_file: Path to service account JSON file
            bucket_name: GCS bucket name
            bucket_prefix: Prefix/folder path in bucket (e.g., 'documents/' or '')
            download_dir: Local directory to save downloaded files
            recursive: If True, download from all subfolders recursively
        """
        self.bucket_name = bucket_name
        self.bucket_prefix = bucket_prefix.rstrip('/') if bucket_prefix else ''
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.recursive = recursive
        
        # Authenticate and create client
        self.client, self.bucket = self._authenticate(service_account_file)
        
    def _authenticate(self, service_account_file: str):
        """Authenticate with Google Cloud Storage using service account"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file
            )
            client = storage.Client(credentials=credentials)
            bucket = client.bucket(self.bucket_name)
            
            # Verify bucket exists
            if not bucket.exists():
                raise ValueError(f"Bucket '{self.bucket_name}' does not exist or is not accessible")
            
            logger.info(f"Successfully authenticated with Google Cloud Storage")
            logger.info(f"Connected to bucket: {self.bucket_name}")
            return client, bucket
            
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Invalid service account file or bucket configuration: {e}")
            raise GCSError(f"Failed to authenticate with GCS: {e}") from e
        except GoogleCloudError as e:
            logger.error(f"Google Cloud Storage error: {e}")
            raise GCSError(f"Failed to connect to GCS bucket {self.bucket_name}") from e
    
    def list_pdf_blobs(self) -> List[storage.Blob]:
        """
        List all PDF blobs in the bucket with the specified prefix
        
        Returns:
            List of blob objects
        """
        try:
            # Set delimiter for non-recursive listing
            delimiter = None if self.recursive else '/'
            
            # List blobs with prefix
            if self.bucket_prefix:
                prefix = f"{self.bucket_prefix}/"
            else:
                prefix = None
            
            blobs = list(self.bucket.list_blobs(prefix=prefix, delimiter=delimiter))
            
            # Filter for PDF files only
            pdf_blobs = [
                blob for blob in blobs 
                if blob.name.lower().endswith('.pdf') and not blob.name.endswith('/')
            ]
            
            logger.info(f"Found {len(pdf_blobs)} PDF files in bucket")
            
            if self.bucket_prefix:
                logger.info(f"Prefix: {self.bucket_prefix}")
            
            return pdf_blobs
            
        except GoogleCloudError as error:
            logger.error(f"GCS error listing blobs: {error}")
            raise GCSError(f"Failed to list PDFs from bucket {self.bucket_name}") from error
    
    def download_blob(self, blob: storage.Blob) -> Optional[Path]:
        """
        Download a single blob from GCS
        
        Args:
            blob: GCS blob object
            
        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            # Get the relative path within the bucket
            blob_name = blob.name
            
            # Remove prefix to get relative path
            if self.bucket_prefix and blob_name.startswith(f"{self.bucket_prefix}/"):
                relative_path = blob_name[len(self.bucket_prefix) + 1:]
            else:
                relative_path = blob_name
            
            # Sanitize the path components
            path_parts = relative_path.split('/')
            safe_parts = [sanitize_filename(part) for part in path_parts]
            safe_relative_path = '/'.join(safe_parts)
            
            # Create full local path
            file_path = self.download_dir / safe_relative_path
            
            # Create directory structure if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if file already exists
            if file_path.exists():
                logger.info(f"File already exists: {safe_relative_path}")
                return file_path
            
            # Download blob
            blob.download_to_filename(str(file_path))
            
            logger.info(f"Successfully downloaded: {safe_relative_path}")
            return file_path
            
        except (IOError, OSError) as e:
            logger.error(f"File system error downloading {blob.name}: {e}")
            return None
        except GoogleCloudError as e:
            logger.error(f"GCS download error for {blob.name}: {e}")
            return None
    
    def download_all_pdfs(self) -> List[Path]:
        """
        Download all PDF files from the bucket
        
        Returns:
            List of paths to downloaded files
        """
        blobs = self.list_pdf_blobs()
        downloaded_files = []
        
        logger.info(f"Starting download of {len(blobs)} PDF files")
        
        for blob in blobs:
            file_path = self.download_blob(blob)
            
            if file_path:
                downloaded_files.append(file_path)
        
        logger.info(f"Successfully downloaded {len(downloaded_files)} files")
        return downloaded_files


def download_pdfs_from_gcs(
    service_account_file: Optional[str] = None,
    bucket_name: Optional[str] = None,
    bucket_prefix: Optional[str] = None,
    download_dir: Optional[Path] = None,
    recursive: Optional[bool] = None
) -> List[Path]:
    """
    Convenience function to download PDFs from Google Cloud Storage
    
    Args:
        service_account_file: Path to service account JSON (defaults to Config)
        bucket_name: GCS bucket name (defaults to Config)
        bucket_prefix: Bucket prefix/folder (defaults to Config)
        download_dir: Download directory (required - no default in Config)
        recursive: Whether to download from subfolders recursively (defaults to Config)
        
    Returns:
        List of paths to downloaded files
        
    Raises:
        ValueError: If download_dir is not provided
    """
    service_account_file = service_account_file or Config.GOOGLE_SERVICE_ACCOUNT_JSON
    bucket_name = bucket_name or Config.GCS_BUCKET_NAME
    bucket_prefix = bucket_prefix or getattr(Config, 'GCS_BUCKET_PREFIX', '')
    
    # download_dir is required since Config.DOWNLOAD_DIR was removed for stream processing
    if download_dir is None:
        raise ValueError("download_dir must be provided (Config.DOWNLOAD_DIR has been removed)")
    
    # Default to True if not specified
    if recursive is None:
        recursive = getattr(Config, 'GCS_RECURSIVE', True)
    
    downloader = GCSDownloader(
        service_account_file,
        bucket_name,
        bucket_prefix,
        download_dir,
        recursive
    )
    return downloader.download_all_pdfs()

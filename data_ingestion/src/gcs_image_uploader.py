"""Google Cloud Storage image uploader module"""

import logging
from pathlib import Path
from typing import List, Optional
import io

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
from google.oauth2 import service_account
from PIL import Image

from .config import Config
from .exceptions import GCSError
from .types import ImageData


logger = logging.getLogger(__name__)


class GCSImageUploader:
    """Class to handle uploading images to Google Cloud Storage"""
    
    def __init__(
        self,
        service_account_file: str = None,
        bucket_name: str = None,
        images_prefix: str = "images"
    ):
        """
        Initialize Google Cloud Storage image uploader
        
        Args:
            service_account_file: Path to service account JSON file
            bucket_name: GCS bucket name
            images_prefix: Prefix/folder for images in bucket (default: 'images')
        """
        self.bucket_name = bucket_name or Config.GCS_BUCKET_NAME
        self.images_prefix = images_prefix.rstrip('/')
        self.service_account_file = service_account_file or Config.GOOGLE_SERVICE_ACCOUNT_JSON
        
        if not self.bucket_name:
            raise ValueError("GCS_BUCKET_NAME is required")
        
        # Authenticate and create client
        self.client, self.bucket = self._authenticate(self.service_account_file)
        logger.info(f"GCS Image Uploader initialized for bucket: {self.bucket_name}")
    
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
                raise GCSError(f"Bucket {self.bucket_name} does not exist")
            
            logger.info(f"Successfully authenticated with GCS bucket: {self.bucket_name}")
            return client, bucket
            
        except FileNotFoundError as e:
            logger.error(f"Service account file not found: {service_account_file}")
            raise GCSError(f"Service account file not found: {service_account_file}") from e
        except GoogleCloudError as e:
            logger.error(f"GCS authentication error: {e}")
            raise GCSError(f"Failed to authenticate with GCS") from e
    
    def upload_image(
        self,
        image_data: ImageData,
        file_hash: str,
        chunk_index: int,
        optimize: bool = True,
        max_size_kb: int = 1000,
        max_retries: int = 5,
        base_delay: float = 1.0
    ) -> str:
        """
        Upload a single image to GCS with retry logic for rate limiting
        
        Args:
            image_data: ImageData object containing image bytes and metadata
            file_hash: Hash of the source PDF file
            chunk_index: Index of the chunk this image belongs to
            optimize: Whether to optimize/compress the image
            max_size_kb: Maximum size in KB before compression (default: 1000)
            max_retries: Maximum number of retries for rate limit errors
            base_delay: Base delay in seconds for exponential backoff
            
        Returns:
            GCS URI (gs://bucket/path/to/image.png)
        """
        import time
        
        # Prepare blob name
        # Format: images/{file_hash}/{chunk_index}_{image_index}.png
        image_index = image_data['image_index']
        blob_name = f"{self.images_prefix}/{file_hash}/{chunk_index}_{image_index}.png"
        
        # Get image bytes
        img_bytes = image_data['bytes']
        
        # Optimize image if needed
        if optimize and len(img_bytes) > (max_size_kb * 1024):
            logger.debug(f"Optimizing image {blob_name} (original size: {len(img_bytes)} bytes)")
            img_bytes = self._optimize_image(img_bytes, max_size_kb)
            logger.debug(f"Optimized to {len(img_bytes)} bytes")
        
        # Upload with retry logic for rate limiting
        for attempt in range(max_retries):
            try:
                blob = self.bucket.blob(blob_name)
                blob.upload_from_string(img_bytes, content_type='image/png')
                
                gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
                logger.debug(f"Uploaded image to {gcs_uri}")
                
                return gcs_uri
                
            except GoogleCloudError as e:
                # Check if it's a rate limit error (429)
                if '429' in str(e) or 'rateLimitExceeded' in str(e):
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + (time.time() % 1)
                        logger.warning(
                            f"Rate limit hit for {blob_name}, retrying in {delay:.2f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Failed to upload image after {max_retries} retries due to rate limiting")
                        raise GCSError(f"Failed to upload image due to rate limiting") from e
                else:
                    logger.error(f"Failed to upload image to GCS: {e}")
                    raise GCSError(f"Failed to upload image") from e
            except Exception as e:
                logger.error(f"Unexpected error uploading image: {e}")
                raise GCSError(f"Unexpected error uploading image") from e
        
        raise GCSError(f"Failed to upload image after {max_retries} attempts")
    
    def _optimize_image(self, img_bytes: bytes, max_size_kb: int = 1000) -> bytes:
        """
        Optimize image by resizing and compressing
        
        Args:
            img_bytes: Original image bytes
            max_size_kb: Target maximum size in KB
            
        Returns:
            Optimized image bytes
        """
        try:
            # Open image
            img = Image.open(io.BytesIO(img_bytes))
            
            # Convert to RGB if necessary (for JPEG compatibility)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            
            # Calculate target size
            # Start with 80% of original dimensions
            original_size = img.size
            scale_factor = 0.8
            quality = 85
            
            # Try progressively smaller sizes until under target
            for attempt in range(5):
                # Resize
                new_size = (
                    int(original_size[0] * scale_factor),
                    int(original_size[1] * scale_factor)
                )
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Compress to PNG
                buffer = io.BytesIO()
                resized_img.save(buffer, format='PNG', optimize=True)
                compressed_bytes = buffer.getvalue()
                
                # Check size
                if len(compressed_bytes) <= (max_size_kb * 1024):
                    logger.debug(f"Optimized: {original_size} -> {new_size}, {len(img_bytes)} -> {len(compressed_bytes)} bytes")
                    return compressed_bytes
                
                # Reduce scale factor for next attempt
                scale_factor *= 0.8
            
            # If still too large, return the smallest we got
            logger.warning(f"Could not optimize image below {max_size_kb}KB, using best effort")
            return compressed_bytes
            
        except Exception as e:
            logger.error(f"Error optimizing image: {e}")
            # Return original if optimization fails
            return img_bytes
    
    def upload_images_batch(
        self,
        images: List[ImageData],
        file_hash: str,
        chunk_index: int = 0,
        optimize: bool = True
    ) -> List[dict]:
        """
        Upload multiple images for a chunk
        
        Args:
            images: List of ImageData objects
            file_hash: Hash of the source PDF file
            chunk_index: Index of the chunk these images belong to
            optimize: Whether to optimize/compress images
            
        Returns:
            List of dicts with image metadata including gcs_path
        """
        uploaded_images = []
        
        for image_data in images:
            try:
                gcs_uri = self.upload_image(image_data, file_hash, chunk_index, optimize)
                # Update image_data with GCS path
                image_data['gcs_path'] = gcs_uri
                uploaded_images.append({
                    'gcs_path': gcs_uri,
                    'page_num': image_data.get('page_num', 0),
                    'image_index': image_data.get('image_index', 0),
                    'size': image_data.get('size', (0, 0)),
                    'bbox': image_data.get('bbox')
                })
            except Exception as e:
                logger.error(f"Failed to upload image {image_data.get('path', 'unknown')}: {e}")
                # Continue with other images
                continue
        
        logger.info(f"Uploaded {len(uploaded_images)}/{len(images)} images for chunk {chunk_index}")
        return uploaded_images
    
    def delete_images_by_file_hash(self, file_hash: str) -> int:
        """
        Delete all images for a specific PDF file
        
        Args:
            file_hash: Hash of the PDF file
            
        Returns:
            Number of images deleted
        """
        try:
            prefix = f"{self.images_prefix}/{file_hash}/"
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            count = 0
            for blob in blobs:
                try:
                    blob.delete()
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to delete blob {blob.name}: {e}")
            
            logger.info(f"Deleted {count} images for file_hash: {file_hash}")
            return count
            
        except GoogleCloudError as e:
            logger.error(f"Error deleting images from GCS: {e}")
            raise GCSError(f"Failed to delete images for file_hash: {file_hash}") from e
    
    def get_signed_url(self, gcs_uri: str, expiration_minutes: int = 60) -> str:
        """
        Generate a signed URL for accessing an image
        
        Args:
            gcs_uri: GCS URI (gs://bucket/path/to/image.png)
            expiration_minutes: URL expiration time in minutes
            
        Returns:
            Signed URL for accessing the image
        """
        try:
            # Parse GCS URI
            if not gcs_uri.startswith('gs://'):
                raise ValueError(f"Invalid GCS URI: {gcs_uri}")
            
            # Extract blob name
            uri_parts = gcs_uri.replace('gs://', '').split('/', 1)
            if len(uri_parts) != 2:
                raise ValueError(f"Invalid GCS URI format: {gcs_uri}")
            
            bucket_name, blob_name = uri_parts
            
            if bucket_name != self.bucket_name:
                logger.warning(f"URI bucket ({bucket_name}) doesn't match configured bucket ({self.bucket_name})")
            
            # Get blob and generate signed URL
            blob = self.bucket.blob(blob_name)
            
            from datetime import timedelta
            signed_url = blob.generate_signed_url(
                version='v4',
                expiration=timedelta(minutes=expiration_minutes),
                method='GET'
            )
            
            return signed_url
            
        except Exception as e:
            logger.error(f"Failed to generate signed URL for {gcs_uri}: {e}")
            raise GCSError(f"Failed to generate signed URL") from e
    
    def download_image(self, gcs_uri: str) -> bytes:
        """
        Download image bytes from GCS
        
        Args:
            gcs_uri: GCS URI (gs://bucket/path/to/image.png)
            
        Returns:
            Image bytes
        """
        try:
            # Parse GCS URI
            if not gcs_uri.startswith('gs://'):
                raise ValueError(f"Invalid GCS URI: {gcs_uri}")
            
            uri_parts = gcs_uri.replace('gs://', '').split('/', 1)
            if len(uri_parts) != 2:
                raise ValueError(f"Invalid GCS URI format: {gcs_uri}")
            
            bucket_name, blob_name = uri_parts
            
            # Download blob
            blob = self.bucket.blob(blob_name)
            img_bytes = blob.download_as_bytes()
            
            return img_bytes
            
        except GoogleCloudError as e:
            logger.error(f"Failed to download image from {gcs_uri}: {e}")
            raise GCSError(f"Failed to download image") from e

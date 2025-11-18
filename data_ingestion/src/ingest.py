"""Main ingestion orchestrator"""

import logging
from typing import List, Dict, Any
from tqdm import tqdm

from google.cloud import storage
from google.oauth2 import service_account

from .config import Config
from .pdf_extractor import PDFExtractor
from .chunker import TextChunker
from .vector_store import MilvusVectorStore
from .utils import setup_logging


logger = setup_logging()


def _get_embedder():
    """Get the appropriate embedder based on configuration"""
    backend = Config.EMBEDDING_BACKEND
    
    if backend == "huggingface":
        from .hf_embedder import HuggingFaceEmbedder
        logger.info(f"Using HuggingFace embedder (service URL: {Config.EMBEDDING_SERVICE_URL})")
        return HuggingFaceEmbedder(
            service_url=Config.EMBEDDING_SERVICE_URL,
            embedding_dimension=Config.EMBEDDING_DIMENSION
        )
    elif backend == "gemini":
        from .embedder import TextEmbedder
        logger.info(f"Using Gemini embedder (model: {Config.EMBEDDING_MODEL})")
        return TextEmbedder(
            model_name=Config.EMBEDDING_MODEL,
            embedding_dimension=Config.EMBEDDING_DIMENSION
        )
    else:
        raise ValueError(f"Invalid EMBEDDING_BACKEND: {backend}")


class IngestionPipeline:
    """Main pipeline for ingesting PDFs into Milvus"""
    
    def __init__(self, recreate_collection: bool = False):
        """Initialize the ingestion pipeline
        
        Args:
            recreate_collection: If True, drop and recreate the Milvus collection
        """
        # Validate configuration
        Config.validate()
        
        # Initialize GCS client
        self._init_gcs_client()
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        # Initialize PDF extractor with multimodal support if enabled
        self.pdf_extractor = PDFExtractor()
        
        # Initialize appropriate chunker based on multimodal mode
        if Config.ENABLE_MULTIMODAL:
            logger.info("🎨 Multimodal mode enabled - will extract and process images")
            self.chunker = MultimodalChunker(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            # Initialize image uploader for GCS
            self.image_uploader = GCSImageUploader(
                bucket_name=Config.GCS_BUCKET_NAME,
                service_account_json=Config.GOOGLE_SERVICE_ACCOUNT_JSON,
                images_prefix=Config.GCS_IMAGES_PREFIX
            )
        else:
            logger.info("📝 Text-only mode - images will be skipped")
            self.chunker = TextChunker(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            self.image_uploader = None
        
        self.embedder = _get_embedder()
        self.vector_store = MilvusVectorStore(
            uri=Config.MILVUS_URI,
            api_key=Config.MILVUS_API_KEY,
            collection_name=Config.MILVUS_COLLECTION_NAME,
            embedding_dim=self.embedder.get_embedding_dimension(),
            recreate_collection=recreate_collection
        )
        
        logger.info("Pipeline initialized successfully")
    
    def _close_gcs_client(self):
        """Close GCS client connection"""
        try:
            if hasattr(self, 'gcs_client'):
                # GCS client doesn't have an explicit close method, but we can disconnect
                pass
        except Exception as e:
            logger.warning(f"Error closing GCS client: {e}")
    
    def _init_gcs_client(self):
        """Initialize Google Cloud Storage client"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                Config.GOOGLE_SERVICE_ACCOUNT_JSON
            )
            self.gcs_client = storage.Client(credentials=credentials)
            self.bucket = self.gcs_client.bucket(Config.GCS_BUCKET_NAME)
            
            # Verify bucket exists
            if not self.bucket.exists():
                raise ValueError(f"Bucket '{Config.GCS_BUCKET_NAME}' does not exist or is not accessible")
            
            logger.info(f"Connected to GCS bucket: {Config.GCS_BUCKET_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise
    
    def list_pdf_blobs(self) -> List[storage.Blob]:
        """
        List all PDF blobs in the GCS bucket
        
        Returns:
            List of PDF blob objects
        """
        try:
            # Set delimiter for non-recursive listing
            delimiter = None if Config.GCS_RECURSIVE else '/'
            
            # List blobs with prefix
            if Config.GCS_BUCKET_PREFIX:
                prefix = f"{Config.GCS_BUCKET_PREFIX}/"
            else:
                prefix = None
            
            blobs = list(self.bucket.list_blobs(prefix=prefix, delimiter=delimiter))
            
            # Filter for PDF files only
            pdf_blobs = [
                blob for blob in blobs 
                if blob.name.lower().endswith('.pdf') and not blob.name.endswith('/')
            ]
            
            logger.info(f"Found {len(pdf_blobs)} PDF files in bucket")
            
            if Config.GCS_BUCKET_PREFIX:
                logger.info(f"Prefix: {Config.GCS_BUCKET_PREFIX}")
            
            return pdf_blobs
            
        except Exception as e:
            logger.error(f"Error listing PDF blobs: {e}")
            raise
    
    def process_pdf_blob(self, pdf_blob: storage.Blob) -> List[Dict[str, Any]]:
        """
        Process a single PDF blob from GCS
        
        Args:
            pdf_blob: GCS blob object
            
        Returns:
            List of chunks with embeddings and metadata
        """
        logger.info(f"Processing: {pdf_blob.name}")
        
        # Create a mock file path object for chunker compatibility
        class MockFilePath:
            def __init__(self, name):
                self.name = name
                self.etag = getattr(pdf_blob, 'etag', None)
                self.size = getattr(pdf_blob, 'size', None)
        
        mock_file_path = MockFilePath(pdf_blob.name)
        
        # Extract and chunk based on multimodal mode
        if Config.ENABLE_MULTIMODAL:
            # Multimodal extraction: text + images
            result = self.pdf_extractor.extract_with_images(pdf_blob)
            if not result or not result.text:
                logger.warning(f"No text extracted from {pdf_blob.name}, skipping")
                return []
            
            logger.info(f"📸 Extracted {len(result.images)} images from {pdf_blob.name}")
            
            # Chunk with image association
            chunks_with_metadata = self.chunker.chunk_with_images(
                text=result.text,
                images=result.images,
                file_path=mock_file_path
            )
        else:
            # Text-only extraction
            text = self.pdf_extractor.extract_text(pdf_blob)
            if not text:
                logger.warning(f"No text extracted from {pdf_blob.name}, skipping")
                return []
            
            # Chunk text
            chunks_with_metadata = self.chunker.chunk_with_metadata(
                text=text,
                file_path=mock_file_path
            )
        
        if not chunks_with_metadata:
            logger.warning(f"No chunks created from {pdf_blob.name}, skipping")
            return []
        
        logger.info(f"Created {len(chunks_with_metadata)} chunks from {pdf_blob.name}")
        
        # Process chunks in smaller batches to manage memory for HUGE files
        # For very large files (>1000 chunks), use smaller batch size
        num_chunks = len(chunks_with_metadata)
        if num_chunks > 1000:
            chunk_batch_size = 20  # Very conservative for huge files
            logger.info(f"Large file detected ({num_chunks} chunks), using batch size of {chunk_batch_size}")
        else:
            chunk_batch_size = min(Config.BATCH_SIZE, 50)
        
        all_chunks_with_embeddings = []
        
        for i in range(0, len(chunks_with_metadata), chunk_batch_size):
            batch_chunks = chunks_with_metadata[i:i + chunk_batch_size]
            
            batch_num = i // chunk_batch_size + 1
            total_batches = (len(chunks_with_metadata) + chunk_batch_size - 1) // chunk_batch_size
            logger.info(f"📊 Processing batch {batch_num}/{total_batches} for {pdf_blob.name}")
            
            try:
                # Process each chunk in the batch
                for chunk_data in batch_chunks:
                    # Handle multimodal chunks with images
                    if Config.ENABLE_MULTIMODAL and chunk_data.get('images'):
                        images = chunk_data['images']
                        logger.info(f"🎨 Chunk has {len(images)} associated images")
                        
                        # Upload images to GCS first
                        uploaded_images = self.image_uploader.upload_images_batch(
                            images=images,
                            file_hash=chunk_data['file_name']  # Use filename as identifier
                        )
                        
                        # Store GCS paths and metadata in chunk
                        chunk_data['image_gcs_paths'] = [img['gcs_path'] for img in uploaded_images]
                        chunk_data['image_metadata'] = uploaded_images
                        chunk_data['image_count'] = len(uploaded_images)
                        chunk_data['has_image'] = True
                        chunk_data['embedding_type'] = 'multimodal'
                        
                        # Generate multimodal embedding (text + first image)
                        # For multiple images, we use the first one as representative
                        primary_image = images[0]
                        embedding = self.embedder.embed_multimodal(
                            text=chunk_data['text'],
                            image_bytes=primary_image.image_bytes
                        )
                        
                        # Clean up temporary image data (not needed in Milvus)
                        del chunk_data['images']
                    else:
                        # Text-only chunk
                        chunk_data['has_image'] = False
                        chunk_data['embedding_type'] = 'text'
                        chunk_data['image_count'] = 0
                        chunk_data['image_gcs_paths'] = []
                        chunk_data['image_metadata'] = []
                        
                        # Generate text-only embedding
                        embedding = self.embedder.embed_text([chunk_data['text']])[0]
                    
                    # Add embedding to chunk
                    chunk_data['embedding'] = embedding.tolist()
                    all_chunks_with_embeddings.append(chunk_data)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {batch_num}: {e}")
                raise  # Re-raise to handle at file level
        
        return all_chunks_with_embeddings
    
    def ingest_chunks(self, chunks_with_embeddings: List[Dict[str, Any]], file_name: str = "unknown"):
        """
        Insert chunks into Milvus with better error handling
        
        Args:
            chunks_with_embeddings: List of chunks with embeddings and metadata
            file_name: Name of source file for logging
        """
        if not chunks_with_embeddings:
            return
        
        embeddings = [chunk['embedding'] for chunk in chunks_with_embeddings]
        texts = [chunk['text'] for chunk in chunks_with_embeddings]
        metadatas = [chunk['metadata'] for chunk in chunks_with_embeddings]
        
        # Insert in batches with smaller size for huge files
        num_chunks = len(embeddings)
        batch_size = min(Config.BATCH_SIZE, 100) if num_chunks > 500 else Config.BATCH_SIZE
        
        total_batches = (num_chunks + batch_size - 1) // batch_size
        logger.info(f"💾 Inserting {num_chunks} chunks in {total_batches} batches to Milvus")
        
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            batch_num = i // batch_size + 1
            logger.info(f"💾 Milvus batch {batch_num}/{total_batches} for {file_name}")
            
            try:
                self.vector_store.insert(
                    embeddings=batch_embeddings,
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
            except Exception as e:
                logger.error(f"Failed to insert batch {batch_num} for {file_name}: {e}")
                raise  # Re-raise to handle at file level
    
    def run(self, force: bool = False):
        """
        Run the complete ingestion pipeline
        
        Args:
            force: If True, re-process files that have already been ingested
        """
        try:
            # List PDF blobs from GCS
            logger.info("Listing PDF files from Google Cloud Storage...")
            pdf_blobs = self.list_pdf_blobs()
            
            if not pdf_blobs:
                logger.warning("No PDF files found in GCS bucket")
                return
            
            logger.info(f"Starting ingestion of {len(pdf_blobs)} PDF files")
            
            # Track processing results
            successful_uploads = 0
            failed_uploads = 0
            total_chunks = 0
            failed_files = []
            
            # Process each PDF blob
            for pdf_blob in tqdm(pdf_blobs, desc="Processing PDFs"):
                try:
                    # Check if file has already been processed (basic deduplication)
                    # Use COUNT to avoid loading data into memory
                    if not force:
                        try:
                            # Escape filename for Milvus query
                            safe_filename = pdf_blob.name.replace('"', '\\"')
                            
                            # Use count instead of loading data
                            existing_results = self.vector_store.collection.query(
                                expr=f'file_name == "{safe_filename}"',
                                output_fields=["primary_key"],
                                limit=1
                            )
                            
                            if existing_results and len(existing_results) > 0:
                                logger.info(f"✓ Skipping {pdf_blob.name} - already processed")
                                successful_uploads += 1  # Count as success since it's already done
                                continue
                        except Exception as e:
                            logger.warning(f"Error checking duplicates for {pdf_blob.name}: {e}. Proceeding with processing.")
                    else:
                        # In force mode, delete existing entries for this file
                        try:
                            safe_filename = pdf_blob.name.replace('"', '\\"')
                            existing_results = self.vector_store.collection.query(
                                expr=f'file_name == "{safe_filename}"',
                                output_fields=["file_hash"],
                                limit=1
                            )
                            if existing_results and len(existing_results) > 0:
                                # Get file_hash and delete all chunks for this file
                                file_hash = existing_results[0].get('file_hash')
                                if file_hash:
                                    deleted = self.vector_store.delete_by_file_hash(file_hash)
                                    logger.info(f"🗑️  Deleted {deleted} existing chunks for {pdf_blob.name}")
                        except Exception as e:
                            logger.warning(f"Error deleting duplicates for {pdf_blob.name}: {e}. Proceeding with processing.")
                    
                    # Log checkpoint before processing
                    logger.info(f"🔄 [{successful_uploads + 1}/{len(pdf_blobs)}] Processing: {pdf_blob.name} ({pdf_blob.size / 1024 / 1024:.1f} MB)")
                    
                    chunks_with_embeddings = self.process_pdf_blob(pdf_blob)
                    
                    if chunks_with_embeddings:
                        self.ingest_chunks(chunks_with_embeddings, file_name=pdf_blob.name)
                        successful_uploads += 1
                        total_chunks += len(chunks_with_embeddings)
                        logger.info(f"✅ [{successful_uploads}/{len(pdf_blobs)}] Successfully ingested {len(chunks_with_embeddings)} chunks from {pdf_blob.name}")
                    else:
                        failed_uploads += 1
                        failed_files.append(pdf_blob.name)
                        logger.warning(f"No chunks generated from {pdf_blob.name}")
                    
                except Exception as e:
                    failed_uploads += 1
                    failed_files.append(pdf_blob.name)
                    logger.error(f"Error processing {pdf_blob.name}: {e}", exc_info=True)
                    continue
            
            # Print comprehensive statistics
            logger.info("=" * 60)
            logger.info("Ingestion completed!")
            logger.info(f"Total PDFs found: {len(pdf_blobs)}")
            logger.info(f"Successfully processed: {successful_uploads}")
            logger.info(f"Failed to process: {failed_uploads}")
            logger.info(f"Total chunks ingested: {total_chunks}")
            
            if failed_files:
                logger.warning(f"Failed files: {', '.join(failed_files[:5])}{'...' if len(failed_files) > 5 else ''}")
            
            stats = self.vector_store.get_stats()
            logger.info(f"Collection stats: {stats}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error in ingestion pipeline: {e}", exc_info=True)
            raise
        
        finally:
            # Clean up
            self.vector_store.close()
            self._close_gcs_client()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest PDF files from GCS into Milvus")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of files that have already been ingested"
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop and recreate the Milvus collection (WARNING: deletes all existing data)"
    )
    args = parser.parse_args()
    
    logger.info("Starting data ingestion pipeline")
    if args.force:
        logger.info("Force mode enabled: Will re-process already ingested files")
    if args.recreate_collection:
        logger.warning("Recreate collection mode enabled: Will drop and recreate the collection")
    
    try:
        pipeline = IngestionPipeline(recreate_collection=args.recreate_collection)
        pipeline.run(force=args.force)
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

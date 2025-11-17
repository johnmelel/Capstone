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
        self.pdf_extractor = PDFExtractor()
        self.chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
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
        
        # Extract text
        text = self.pdf_extractor.extract_text(pdf_blob)
        if not text:
            logger.warning(f"No text extracted from {pdf_blob.name}, skipping")
            return []
        
        # Create a mock file path object for chunker compatibility
        class MockFilePath:
            def __init__(self, name):
                self.name = name
                self.etag = getattr(pdf_blob, 'etag', None)
                self.size = getattr(pdf_blob, 'size', None)
        
        mock_file_path = MockFilePath(pdf_blob.name)
        
        # Chunk text
        chunks_with_metadata = self.chunker.chunk_with_metadata(
            text=text,
            file_path=mock_file_path
        )
        
        if not chunks_with_metadata:
            logger.warning(f"No chunks created from {pdf_blob.name}, skipping")
            return []
        
        logger.info(f"Created {len(chunks_with_metadata)} chunks from {pdf_blob.name}")
        
        # Process chunks in smaller batches to manage memory
        chunk_batch_size = min(Config.BATCH_SIZE, 50)  # Limit embedding batch size
        all_chunks_with_embeddings = []
        
        for i in range(0, len(chunks_with_metadata), chunk_batch_size):
            batch_chunks = chunks_with_metadata[i:i + chunk_batch_size]
            
            # Extract texts for embedding
            batch_texts = [chunk['text'] for chunk in batch_chunks]
            
            # Generate embeddings for this batch
            logger.debug(f"Generating embeddings for batch {i // chunk_batch_size + 1}")
            batch_embeddings = self.embedder.embed_text(batch_texts)
            
            # Combine embeddings with chunks
            for j, chunk_data in enumerate(batch_chunks):
                chunk_data['embedding'] = batch_embeddings[j].tolist()
                all_chunks_with_embeddings.append(chunk_data)
        
        return all_chunks_with_embeddings
    
    def ingest_chunks(self, chunks_with_embeddings: List[Dict[str, Any]]):
        """
        Insert chunks into Milvus
        
        Args:
            chunks_with_embeddings: List of chunks with embeddings and metadata
        """
        if not chunks_with_embeddings:
            return
        
        embeddings = [chunk['embedding'] for chunk in chunks_with_embeddings]
        texts = [chunk['text'] for chunk in chunks_with_embeddings]
        metadatas = [chunk['metadata'] for chunk in chunks_with_embeddings]
        
        # Insert in batches
        batch_size = Config.BATCH_SIZE
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            logger.info(f"Inserting batch {i // batch_size + 1}/{(len(embeddings) + batch_size - 1) // batch_size}")
            self.vector_store.insert(
                embeddings=batch_embeddings,
                texts=batch_texts,
                metadatas=batch_metadatas
            )
    
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
                    if not force:
                        existing_count = self.vector_store.collection.query(
                            expr=f'file_name == "{pdf_blob.name}"',
                            output_fields=["primary_key"],
                            limit=1
                        )
                        
                        if existing_count:
                            logger.info(f"Skipping {pdf_blob.name} - already processed ({len(existing_count)} chunks found)")
                            continue
                    else:
                        # In force mode, delete existing entries for this file
                        existing_count = self.vector_store.collection.query(
                            expr=f'file_name == "{pdf_blob.name}"',
                            output_fields=["file_hash"],
                            limit=1
                        )
                        if existing_count:
                            # Get file_hash and delete all chunks for this file
                            file_hash = existing_count[0].get('file_hash')
                            if file_hash:
                                deleted = self.vector_store.delete_by_file_hash(file_hash)
                                logger.info(f"Deleted {deleted} existing chunks for {pdf_blob.name}")
                    
                    chunks_with_embeddings = self.process_pdf_blob(pdf_blob)
                    
                    if chunks_with_embeddings:
                        self.ingest_chunks(chunks_with_embeddings)
                        successful_uploads += 1
                        total_chunks += len(chunks_with_embeddings)
                        logger.info(f"Successfully ingested {len(chunks_with_embeddings)} chunks from {pdf_blob.name}")
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

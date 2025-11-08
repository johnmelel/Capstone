"""Main ingestion orchestrator"""

import logging
from typing import List, Dict, Any
from tqdm import tqdm

from google.cloud import storage
from google.oauth2 import service_account

from .config import Config
from .mineru_parser import MinerUParser
from .vertex_embedder import VertexAIEmbedder
from .vector_store import MilvusVectorStore
from .utils import setup_logging


logger = setup_logging()


class IngestionPipeline:
    """Main pipeline for ingesting PDFs into Milvus"""
    
    def __init__(self):
        """Initialize the ingestion pipeline"""
        # Validate configuration
        Config.validate()
        
        # Initialize GCS client
        self._init_gcs_client()
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        # MinerU parser for PDF extraction
        parser_config = {
            'chunk_size': Config.CHUNK_SIZE,
            'overlap': Config.CHUNK_OVERLAP
        }
        self.parser = MinerUParser(parser_config, Config.OUTPUT_DIR)
        
        # Vertex AI embedder for multimodal embeddings
        self.embedder = VertexAIEmbedder(
            project_id=Config.GOOGLE_CLOUD_PROJECT,
            location=Config.GOOGLE_CLOUD_LOCATION
        )
        
        # Milvus vector store
        self.vector_store = MilvusVectorStore(
            uri=Config.MILVUS_URI,
            api_key=Config.MILVUS_API_KEY,
            collection_name=Config.MILVUS_COLLECTION_NAME,
            embedding_dim=self.embedder.get_embedding_dimension()
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
        Process a single PDF blob from GCS using MinerU
        
        Args:
            pdf_blob: GCS blob object
            
        Returns:
            List of chunks with embeddings and metadata
        """
        logger.info(f"Processing: {pdf_blob.name}")
        
        # Download PDF to temp location
        import tempfile
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
            pdf_blob.download_to_filename(str(temp_path))
        
        try:
            # Parse PDF with MinerU (extracts text, images, tables)
            chunks = self.parser.parse_pdf(temp_path)
            
            if not chunks:
                logger.warning(f"No chunks extracted from {pdf_blob.name}, skipping")
                return []
            
            logger.info(f"Extracted {len(chunks)} chunks from {pdf_blob.name}")
            
            # Generate embeddings for all chunks (text + images)
            logger.info(f"Generating embeddings...")
            embeddings = self.embedder.embed_batch(chunks)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding.tolist()
            
            return chunks
            
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
    
    def ingest_chunks(self, chunks_with_embeddings: List[Dict[str, Any]]):
        """
        Insert chunks into Milvus
        
        Args:
            chunks_with_embeddings: List of chunks with embeddings and metadata
        """
        if not chunks_with_embeddings:
            return
        
        embeddings = [chunk['embedding'] for chunk in chunks_with_embeddings]
        
        # For text chunks, use content; for images, use caption or empty string
        texts = [
            chunk.get('content', chunk.get('caption', '')) 
            for chunk in chunks_with_embeddings
        ]
        
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
    
    def run(self):
        """
        Run the complete ingestion pipeline
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
                    # Check by file name in metadata
                    existing_count = self.vector_store.collection.query(
                        expr=f'file_name == "{pdf_blob.name}"',
                        output_fields=["primary_key"],
                        limit=1
                    )
                    
                    if existing_count:
                        logger.info(f"Skipping {pdf_blob.name} - already processed")
                        continue
                    
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
    logger.info("Starting data ingestion pipeline")
    
    try:
        pipeline = IngestionPipeline()
        pipeline.run()
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

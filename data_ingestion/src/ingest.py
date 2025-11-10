"""Main ingestion orchestrator"""

import logging
from typing import List, Dict, Any
from tqdm import tqdm

from google.cloud import storage
from google.oauth2 import service_account

from .config import Config
from .mineru_parser import MinerUParser
from .embedder import TextEmbedder
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
        
        # MinerU parser for PDF extraction (with Gemini Vision for images)
        parser_config = {
            'chunk_size': Config.CHUNK_SIZE,
            'overlap': Config.CHUNK_OVERLAP
        }
        self.parser = MinerUParser(
            parser_config, 
            Config.OUTPUT_DIR,
            gemini_api_key=Config.GEMINI_API_KEY,
            vision_model=Config.GEMINI_VISION_MODEL,
            image_prompt=Config.IMAGE_DESCRIPTION_PROMPT
        )
        
        # Gemini embedder for text embeddings
        self.embedder = TextEmbedder(
            model_name=Config.EMBEDDING_MODEL,
            api_key=Config.GEMINI_API_KEY,
            embedding_dimension=Config.EMBEDDING_DIMENSION
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
    
    def _upload_extracted_content_to_gcs(self):
        """Upload extracted content (images, metadata) back to GCS bucket"""
        if not Config.UPLOAD_OUTPUT_TO_GCS:
            logger.info("Skipping GCS upload (UPLOAD_OUTPUT_TO_GCS=false)")
            return
        
        try:
            from pathlib import Path
            
            output_dir = Path(Config.OUTPUT_DIR)
            if not output_dir.exists():
                logger.warning(f"Output directory does not exist: {output_dir}")
                return
            
            logger.info("Uploading extracted content to GCS...")
            uploaded_count = 0
            
            # Upload all files in output directory (except temp)
            for local_file in output_dir.rglob('*'):
                if local_file.is_file() and 'temp' not in local_file.parts:
                    # Create relative path for GCS
                    relative_path = local_file.relative_to(output_dir)
                    gcs_path = f"{Config.GCS_OUTPUT_PREFIX}/{relative_path}".replace('\\', '/')
                    
                    # Upload to GCS
                    blob = self.bucket.blob(gcs_path)
                    blob.upload_from_filename(str(local_file))
                    uploaded_count += 1
                    
                    if uploaded_count % 10 == 0:
                        logger.debug(f"Uploaded {uploaded_count} files...")
            
            logger.info(f"Successfully uploaded {uploaded_count} files to gs://{Config.GCS_BUCKET_NAME}/{Config.GCS_OUTPUT_PREFIX}/")
            
        except Exception as e:
            logger.error(f"Error uploading to GCS: {e}")
            # Don't fail the pipeline, just log the error
    
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
        Process a single PDF blob from GCS using MinerU + Gemini Vision
        
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
            # Parse PDF with MinerU (extracts text + images converted to text descriptions)
            chunks = self.parser.parse_pdf(temp_path)
            
            if not chunks:
                logger.warning(f"No chunks extracted from {pdf_blob.name}, skipping")
                return []
            
            logger.info(f"Extracted {len(chunks)} chunks from {pdf_blob.name}")
            
            # All chunks are now text (including image descriptions)
            # Extract text for embedding
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings using Gemini
            logger.info("Generating Gemini embeddings...")
            embeddings = self.embedder.embed_text(texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding.tolist()
                chunk['text'] = chunk['content']  # Add 'text' field for compatibility
            
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
            
            # Upload extracted content to GCS
            self._upload_extracted_content_to_gcs()
            
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

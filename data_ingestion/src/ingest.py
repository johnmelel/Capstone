"""Main ingestion orchestrator"""

import logging
import json
import gc
from typing import List, Dict, Any
from tqdm import tqdm

from google.cloud import storage
from google.oauth2 import service_account

from .config import Config
from .pdf_extractor import PDFExtractor
from .chunker import RecursiveTokenChunker, ImageCaptionChunker, chunk_with_metadata
from .vector_store import MilvusVectorStore
from .gcs_image_uploader import GCSImageUploader
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
        """Initialize the ingestion pipeline"""
        Config.validate()
        self._init_gcs_client()
        
        logger.info("Initializing pipeline components...")
        
        self.pdf_extractor = PDFExtractor(extract_images=Config.ENABLE_MULTIMODAL)
        
        # Initialize RecursiveTokenChunker for text
        self.text_chunker = RecursiveTokenChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Initialize ImageCaptionChunker for images (uses text chunker for caption splitting)
        self.image_chunker = ImageCaptionChunker(self.text_chunker)
        
        if Config.ENABLE_MULTIMODAL:
            logger.info("🎨 Multimodal mode enabled - will extract and process images")
            self.image_uploader = GCSImageUploader(
                bucket_name=Config.GCS_BUCKET_NAME,
                service_account_file=Config.GOOGLE_SERVICE_ACCOUNT_JSON,
                images_prefix=Config.GCS_IMAGES_PREFIX
            )
        else:
            logger.info("📝 Text-only mode - images will be skipped")
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
        try:
            if hasattr(self, 'gcs_client'):
                pass
        except Exception as e:
            logger.warning(f"Error closing GCS client: {e}")
    
    def _init_gcs_client(self):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                Config.GOOGLE_SERVICE_ACCOUNT_JSON
            )
            self.gcs_client = storage.Client(credentials=credentials)
            self.bucket = self.gcs_client.bucket(Config.GCS_BUCKET_NAME)
            
            if not self.bucket.exists():
                raise ValueError(f"Bucket '{Config.GCS_BUCKET_NAME}' does not exist")
            
            logger.info(f"Connected to GCS bucket: {Config.GCS_BUCKET_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise
    
    def list_pdf_blobs(self) -> List[storage.Blob]:
        try:
            delimiter = None if Config.GCS_RECURSIVE else '/'
            prefix = f"{Config.GCS_BUCKET_PREFIX}/" if Config.GCS_BUCKET_PREFIX else None
            
            blobs = list(self.bucket.list_blobs(prefix=prefix, delimiter=delimiter))
            
            pdf_blobs = [
                blob for blob in blobs 
                if blob.name.lower().endswith('.pdf') and not blob.name.endswith('/')
            ]
            
            logger.info(f"Found {len(pdf_blobs)} PDF files in bucket")
            return pdf_blobs
            
        except Exception as e:
            logger.error(f"Error listing PDF blobs: {e}")
            raise
    
    def process_pdf_blob(self, pdf_blob: storage.Blob) -> List[Dict[str, Any]]:
        """
        Process a single PDF blob: Text Stream + Image Stream
        """
        logger.info(f"Processing: {pdf_blob.name}")
        
        class MockFilePath:
            def __init__(self, name):
                self.name = name
                self.etag = getattr(pdf_blob, 'etag', None)
                self.size = getattr(pdf_blob, 'size', None)
        
        mock_file_path = MockFilePath(pdf_blob.name)
        all_chunks = []
        
        # 1. Extract Content (Text + Images)
        if Config.ENABLE_MULTIMODAL:
            result = self.pdf_extractor.extract_with_images(pdf_blob)
            if not result:
                return []
            # result has 'text', 'pages', 'images', 'metadata'
            pages = result.get('pages', [])
            images = result.get('images', [])
            
            # Fallback if pages not present (legacy extraction)
            if not pages and result.get('text'):
                pages = [{'text': result['text'], 'page_num': 1}]
                
            logger.info(f"📸 Extracted {len(images)} images and {len(pages)} pages from {pdf_blob.name}")
        else:
            text = self.pdf_extractor.extract_text(pdf_blob)
            images = []
            if not text:
                return []
            pages = [{'text': text, 'page_num': 1}] # Default to page 1 for text-only
        
        # 2. Text Stream Processing
        text_chunks = []
        if pages:
            text_chunks = chunk_with_metadata(
                text_or_pages=pages,
                file_path=mock_file_path,
                chunker=self.text_chunker,
                embedding_type='text',
                has_image=False
            )
            
            # Embed Text Chunks
            if text_chunks:
                logger.info(f"📝 Embedding {len(text_chunks)} text chunks...")
                chunk_texts = [c['text'] for c in text_chunks]
                embeddings = self.embedder.embed_batch(chunk_texts)
                
                for i, chunk in enumerate(text_chunks):
                    chunk['embedding'] = embeddings[i].tolist()
                
                all_chunks.extend(text_chunks)
        
        # 3. Image Stream Processing
        if images and Config.ENABLE_MULTIMODAL:
            logger.info(f"🖼️ Processing {len(images)} images...")
            
            # Upload images to GCS first (needed for metadata)
            # We do this in batches
            uploaded_images_map = {} # Map path -> GCS url
            
            # Group images for upload
            for i in range(0, len(images), Config.IMAGE_UPLOAD_BATCH_SIZE):
                batch = images[i:i + Config.IMAGE_UPLOAD_BATCH_SIZE]
                # We use a dummy file_hash/chunk_index for initial upload since they aren't tied to text chunks yet
                # Or better: use the image hash/name
                
                # For now, we reuse the uploader logic but we might need to adapt it
                # The uploader expects file_hash and chunk_index. Let's use file_hash of PDF and image index.
                file_hash = text_chunks[0]['metadata']['file_hash'] if text_chunks else "unknown_hash"
                
                uploaded_batch = self.image_uploader.upload_images_batch(
                    images=batch,
                    file_hash=file_hash,
                    chunk_index=9999 # Special index for independent images
                )
                
                for img_data, uploaded_info in zip(batch, uploaded_batch):
                    img_data['gcs_path'] = uploaded_info['gcs_path']
            
            # Process each image
            image_chunks = []
            for img in images:
                # Chunk image (split caption if needed)
                # Returns list of (image_data, caption_chunk_text)
                caption_splits = self.image_chunker.chunk_image(img)
                
                for img_data, caption_text in caption_splits:
                    # Create metadata
                    metadata = text_chunks[0]['metadata'].copy() if text_chunks else {}
                    metadata.update({
                        'chunk_index': len(all_chunks) + len(image_chunks), # Continue index
                        'has_image': True,
                        'image_count': 1,
                        'image_gcs_paths': json.dumps([img_data['gcs_path']]),
                        'image_metadata': json.dumps([{
                            'page_num': img_data.get('page_num'),
                            'bbox': img_data.get('bbox'),
                            'caption': img_data.get('caption') # Original full caption
                        }])
                    })
                    
                    chunk_data = {
                        'text': caption_text if caption_text else "", # Caption text or empty
                        'metadata': metadata,
                        'image_bytes': img_data['bytes'] # For embedding
                    }
                    
                    if caption_text:
                        chunk_data['embedding_type'] = 'multimodal'
                        chunk_data['metadata']['embedding_type'] = 'multimodal'
                    else:
                        chunk_data['embedding_type'] = 'image'
                        chunk_data['metadata']['embedding_type'] = 'image'
                        
                    image_chunks.append(chunk_data)
            
            # Embed Image Chunks
            # We need to batch these carefully: Multimodal vs Image-only
            multimodal_batch = []
            image_only_batch = []
            
            for chunk in image_chunks:
                if chunk['embedding_type'] == 'multimodal':
                    multimodal_batch.append(chunk)
                else:
                    image_only_batch.append(chunk)
            
            # Process Multimodal Batches
            if multimodal_batch:
                logger.info(f"🎨 Embedding {len(multimodal_batch)} multimodal chunks (Image + Caption)...")
                # Process in small batches
                batch_size = 10
                for i in range(0, len(multimodal_batch), batch_size):
                    batch = multimodal_batch[i:i+batch_size]
                    texts = [c['text'] for c in batch]
                    imgs = [c['image_bytes'] for c in batch]
                    
                    embeddings = self.embedder.embed_multimodal(texts=texts, images=imgs)
                    
                    for j, chunk in enumerate(batch):
                        chunk['embedding'] = embeddings[j].tolist()
                        del chunk['image_bytes'] # Free memory
                        all_chunks.append(chunk)
            
            # Process Image-only Batches
            if image_only_batch:
                logger.info(f"🖼️ Embedding {len(image_only_batch)} image-only chunks...")
                batch_size = 10
                for i in range(0, len(image_only_batch), batch_size):
                    batch = image_only_batch[i:i+batch_size]
                    imgs = [c['image_bytes'] for c in batch]
                    
                    embeddings = self.embedder.embed_image(image=imgs)
                    
                    for j, chunk in enumerate(batch):
                        chunk['embedding'] = embeddings[j].tolist()
                        del chunk['image_bytes'] # Free memory
                        all_chunks.append(chunk)
                        
        return all_chunks

    def ingest_chunks(self, chunks_with_embeddings: List[Dict[str, Any]], file_name: str = "unknown"):
        """Insert chunks into Milvus"""
        if not chunks_with_embeddings:
            return
        
        # Validate chunks before insertion
        valid_chunks = []
        for chunk in chunks_with_embeddings:
            text_len = len(chunk['text'])
            if text_len > 9000:
                logger.warning(f"⚠️ Skipping chunk with text length {text_len} > 9000 chars (Milvus limit 10k). File: {file_name}")
                continue
            valid_chunks.append(chunk)
            
        if not valid_chunks:
            logger.warning(f"No valid chunks to insert for {file_name}")
            return
        
        embeddings = [chunk['embedding'] for chunk in valid_chunks]
        texts = [chunk['text'] for chunk in valid_chunks]
        metadatas = [chunk['metadata'] for chunk in valid_chunks]
        
        # Insert in batches
        batch_size = Config.BATCH_SIZE
        total_batches = (len(embeddings) + batch_size - 1) // batch_size
        
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            logger.info(f"💾 Milvus batch {i // batch_size + 1}/{total_batches} for {file_name}")
            
            try:
                self.vector_store.insert(
                    embeddings=batch_embeddings,
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
            except Exception as e:
                logger.error(f"Failed to insert batch: {e}")
                raise

    def run(self, force: bool = False):
        """Run the complete ingestion pipeline"""
        try:
            logger.info("Listing PDF files from Google Cloud Storage...")
            pdf_blobs = self.list_pdf_blobs()
            
            if not pdf_blobs:
                logger.warning("No PDF files found in GCS bucket")
                return
            
            logger.info(f"Starting ingestion of {len(pdf_blobs)} PDF files")
            
            successful_uploads = 0
            failed_uploads = 0
            
            for pdf_blob in tqdm(pdf_blobs, desc="Processing PDFs"):
                try:
                    # Deduplication check
                    if not force:
                        safe_filename = pdf_blob.name.replace('"', '\\"')
                        existing = self.vector_store.collection.query(
                            expr=f'file_name == "{safe_filename}"',
                            output_fields=["primary_key"],
                            limit=1
                        )
                        if existing:
                            logger.info(f"✓ Skipping {pdf_blob.name} - already processed")
                            successful_uploads += 1
                            continue
                    else:
                        # Delete existing
                        safe_filename = pdf_blob.name.replace('"', '\\"')
                        self.vector_store.collection.delete(f'file_name == "{safe_filename}"')
                    
                    # Process
                    chunks = self.process_pdf_blob(pdf_blob)
                    if chunks:
                        self.ingest_chunks(chunks, file_name=pdf_blob.name)
                        successful_uploads += 1
                        logger.info(f"✅ Successfully ingested {len(chunks)} chunks from {pdf_blob.name}")
                    else:
                        failed_uploads += 1
                        logger.warning(f"No chunks generated from {pdf_blob.name}")
                        
                except Exception as e:
                    failed_uploads += 1
                    logger.error(f"Error processing {pdf_blob.name}: {e}", exc_info=True)
                    continue
            
            logger.info("=" * 60)
            logger.info("Ingestion completed!")
            logger.info(f"Successfully processed: {successful_uploads}")
            logger.info(f"Failed to process: {failed_uploads}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error in ingestion pipeline: {e}", exc_info=True)
            raise
        finally:
            self.vector_store.close()
            self._close_gcs_client()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--recreate-collection", action="store_true")
    args = parser.parse_args()
    
    try:
        pipeline = IngestionPipeline(recreate_collection=args.recreate_collection)
        pipeline.run(force=args.force)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import logging
import asyncio
from pathlib import Path

from .config import Config
from .gcs_downloader import download_pdfs_from_gcs
from .rag_processor import RAGProcessor
from .vector_store import MilvusVectorStore

# Configure logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Config.LOG_FILE)
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Main function to run the PDF ingestion pipeline."""
    try:
        # Step 1: Download PDFs from GCS
        logger.info("Starting PDF download from GCS...")
        downloaded_files = download_pdfs_from_gcs()
        if not downloaded_files:
            logger.info("No new PDF files to process.")
            return
        logger.info(f"Successfully downloaded {len(downloaded_files)} files.")

        # Step 2: Process PDFs with RAG-Anything
        logger.info("Starting PDF processing...")
        rag_processor = RAGProcessor()
        processed_files = []
        for file_path in downloaded_files:
            processed_file = await rag_processor.process_document(file_path)
            processed_files.append(processed_file)
        logger.info(f"Successfully processed {len(processed_files)} files.")

        # Step 3: Insert data into Milvus
        logger.info("Starting data insertion into Milvus...")
        vector_store = MilvusVectorStore()
        for file_path in processed_files:
            vector_store.insert_from_parquet(str(file_path))
        vector_store.close()
        logger.info("Successfully inserted data into Milvus.")

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python
import logging
import asyncio

from config import Config
from gcs_downloader import download_pdfs_from_gcs
from rag_processor import RAGProcessor
from vector_store import MilvusVectorStore

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
        # Validate configuration first
        logger.info("=" * 60)
        logger.info("PDF Ingestion Pipeline - Starting")
        logger.info("=" * 60)
        
        try:
            Config.validate()
            Config.print_config()
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return
        
        # Step 1: Download PDFs from GCS
        logger.info("Step 1/3: Downloading PDFs from GCS...")
        downloaded_files = download_pdfs_from_gcs()
        if not downloaded_files:
            logger.info("No new PDF files to process.")
            return
        logger.info(f"✓ Successfully downloaded {len(downloaded_files)} files.")

        # Step 2: Process PDFs with RAG-Anything
        logger.info("Step 2/3: Processing PDFs with RAG-Anything...")
        rag_processor = RAGProcessor()
        processed_files = []
        for idx, file_path in enumerate(downloaded_files, 1):
            logger.info(f"Processing file {idx}/{len(downloaded_files)}: {file_path.name}")
            try:
                processed_file = await rag_processor.process_document(file_path)
                processed_files.append(processed_file)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}", exc_info=True)
                # Continue with other files
                continue
        
        if not processed_files:
            logger.error("No files were successfully processed.")
            return
            
        logger.info(f"✓ Successfully processed {len(processed_files)}/{len(downloaded_files)} files.")

        # Step 3: Insert data into Milvus
        logger.info("Step 3/3: Inserting data into Milvus...")
        vector_store = MilvusVectorStore()
        total_inserted = 0
        for idx, file_path in enumerate(processed_files, 1):
            logger.info(f"Inserting data from file {idx}/{len(processed_files)}: {file_path.name}")
            try:
                vector_store.insert_from_parquet(str(file_path))
                total_inserted += 1
            except Exception as e:
                logger.error(f"Failed to insert data from {file_path.name}: {e}", exc_info=True)
                # Continue with other files
                continue
        
        vector_store.close()
        logger.info(f"✓ Successfully inserted data from {total_inserted}/{len(processed_files)} files into Milvus.")

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Summary: Downloaded {len(downloaded_files)}, Processed {len(processed_files)}, Inserted {total_inserted}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())

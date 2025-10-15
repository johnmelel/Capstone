"""Main entry point for the pipeline."""

import argparse
from pathlib import Path
from src.utils import load_config, setup_logging, create_directories
from src.pipeline import Pipeline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # supress warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'    # only show real errors
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Medical Textbook Embedding Pipeline")
    parser.add_argument("--test", action="store_true", help="Test mode (3 PDFs, 5 pages each)")
    parser.add_argument("--full", action="store_true", help="Process all PDFs")
    
    args = parser.parse_args()
    
    # Setup
    load_dotenv()
    create_directories()
    setup_logging()
    config = load_config()
    
    # Get Google Cloud credentials
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT not set in .env file")
    
    # Create pipeline
    pipeline = Pipeline(config, project_id, location)
    
    # Process
    pdf_dir = Path("test_dataset")
    test_mode = args.test or not args.full
    
    pipeline.process_directory(pdf_dir, test_mode=test_mode)

if __name__ == "__main__":
    main()
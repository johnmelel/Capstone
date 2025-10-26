
"""
Main entry point for the pipeline.

Steps:
1. Loads .env file (Google Cloud credentials)
2. Creates necessary folders (extracted_content, logs, etc.)
3. Loads config.yaml settings
4. Creates a Pipeline object
5. Instructs pipeline to process PDFs in the specific folder
6. Handles --test or --full mode
"""

import os

# must come before ANY google/vertex/grpc imports
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = "none"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"


import argparse
from pathlib import Path
from src.utils import load_config, setup_logging, create_directories
from src.pipeline import Pipeline
from dotenv import load_dotenv

def main():
    """Main entry point."""
    # Parse command line argument

    parser = argparse.ArgumentParser(description="Medical Document RAG Pipeline")
    parser.add_argument("--test", action="store_true", 
                       help="Test mode: Process 1 PDF, 5 pages max")
    parser.add_argument("--full", action="store_true", 
                       help="Full mode: Process all PDFs, all pages")
    
    # NEW: Add cloud/local flags
    parser.add_argument("--cloud", action="store_true",
                       help="Use Cloud Milvus (requires MILVUS_HOST in .env)")
    parser.add_argument("--local", action="store_true",
                       help="Use Local Embedded Milvus (ignores MILVUS_HOST)")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" MEDICAL DOCUMENT RAG PIPELINE")
    print("="*70)
    
    # Setup environment
    print("\n[Setup] Loading environment variables...")
    load_dotenv()
    
    # NEW: Handle cloud/local override
    if args.cloud and args.local:
        print("[Error] Cannot use both --cloud and --local. Pick one!")
        return
    
    if args.local:
        # Force local mode by removing MILVUS_HOST temporarily
        print("[Mode] FORCING LOCAL EMBEDDED MILVUS")
        if "MILVUS_HOST" in os.environ:
            del os.environ["MILVUS_HOST"]
    elif args.cloud:
        print("[Mode] FORCING CLOUD MILVUS")
        if not os.getenv("MILVUS_HOST"):
            print("[Error] --cloud specified but MILVUS_HOST not in .env!")
            print("[Error] Add MILVUS_HOST=34.58.8.31 to your .env file")
            return
    
    print("[Setup] Creating directories...")
    create_directories()
    
    print("[Setup] Configuring logging...")
    setup_logging()
    
    print("[Setup] Loading configuration...")
    config = load_config()
    
    # Get Google Cloud credentials
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not project_id:
        print("\n[Error] GOOGLE_CLOUD_PROJECT not set in .env file")
        print("[Error] Please create .env file with:")
        print("        GOOGLE_CLOUD_PROJECT=your-project-id")
        return
    
    print(f"[Setup] Google Cloud Project: {project_id}")
    print(f"[Setup] Location: {location}")
    
    # Create pipeline
    print("\n[Setup] Initializing pipeline...")
    pipeline = Pipeline(config, project_id, location)
    
    # Determine mode
    if args.full:
        test_mode = False
        print("\n[Mode] FULL MODE - Processing all PDFs")
    else:
        test_mode = True
        print("\n[Mode] TEST MODE - Processing 1 PDF, 5 pages max")
        print("[Mode] (Use --full for complete processing)")
    
    existing_data = pipeline.check_existing_data()

    if existing_data['has_data']:
        print("\n" + "="*70)
        print(f" EXISTING DATA FOUND: {existing_data['count']} embeddings")
        print("="*70)
        print("\nWhat would you like to do?")
        print("  [1] Delete all embeddings and start fresh")
        print("  [2] Continue from where we left off (skip existing)")
        print("  [3] Cancel and exit")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            print("\n[Warning] Deleting all existing embeddings...")
            # Recreate pipeline with drop_if_exists=True
            pipeline = Pipeline(config, project_id, location)
            pipeline.vector_store._setup_collection(drop_if_exists=True)
            pipeline.existing_chunks = set()
            print("[Info] Database cleared. Starting fresh.")
        elif choice == "2":
            print("\n[Info] Continuing from previous run...")
            print("[Info] Will skip already-processed chunks")
        elif choice == "3":
            print("\n[Info] Exiting...")
            return
        else:
            print("\n[Error] Invalid choice. Exiting.")
            return

    # Process PDFs in data/ folder
    pdf_dir = Path("data")
    
    if not pdf_dir.exists():
        print(f"\n[Error] Data directory not found: {pdf_dir}")
        print("[Error] Please create data/ folder with strategy subfolders")
        return
    
    pipeline.process_directory(pdf_dir, test_mode=test_mode)


    
    print("\n[Done] Pipeline completed successfully!")

if __name__ == "__main__":
    main()
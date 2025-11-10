"""
Quick test script for the MinerU-based pipeline.
Tests processing a single PDF from GCS bucket.
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_env_result = load_dotenv()
print(f"Loaded .env: {load_env_result}")
print(f"Current directory: {os.getcwd()}")

from src.config import Config
from google.cloud import storage

def test_mineru_pipeline():
    """Test the MinerU pipeline with a single PDF"""
    
    print("\n" + "="*70)
    print("TESTING MINERU PIPELINE")
    print("="*70)
    
    # Validate configuration
    print("\n1. Validating Configuration...")
    try:
        Config.validate()
        print("✓ Configuration valid")
        print(f"  - Bucket: {Config.GCS_BUCKET_NAME}")
        print(f"  - Prefix: {Config.GCS_BUCKET_PREFIX}")
        print(f"  - Collection: {Config.MILVUS_COLLECTION_NAME}")
        print(f"  - Output Dir: {Config.OUTPUT_DIR}")
        print(f"  - Upload to GCS: {Config.UPLOAD_OUTPUT_TO_GCS}")
        print(f"  - Vision Model: {Config.GEMINI_VISION_MODEL}")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    
    # Check MinerU installation
    print("\n2. Checking MinerU Installation...")
    try:
        import subprocess
        result = subprocess.run(['magic-pdf', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ MinerU installed: {result.stdout.strip()}")
        else:
            print(f"⚠️  MinerU command found but returned error: {result.stderr}")
    except FileNotFoundError:
        print("❌ MinerU not found! Install with: pip install mineru==0.2.6")
        print("   See: https://github.com/opendatalab/MinerU#installation")
        return False
    except Exception as e:
        print(f"⚠️  Could not check MinerU: {e}")
    
    # List PDFs from GCS
    print("\n3. Listing PDFs from GCS...")
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = Config.GOOGLE_SERVICE_ACCOUNT_JSON
        client = storage.Client()
        bucket = client.bucket(Config.GCS_BUCKET_NAME)
        
        blobs = list(bucket.list_blobs(prefix=Config.GCS_BUCKET_PREFIX, max_results=10))
        pdf_blobs = [b for b in blobs if b.name.lower().endswith('.pdf')]
        
        if not pdf_blobs:
            print(f"❌ No PDFs found in gs://{Config.GCS_BUCKET_NAME}/{Config.GCS_BUCKET_PREFIX}")
            return False
        
        print(f"✓ Found {len(pdf_blobs)} PDF(s)")
        test_pdf = pdf_blobs[0]
        print(f"  Testing with: {test_pdf.name}")
        print(f"  Size: {test_pdf.size / 1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ GCS access failed: {e}")
        return False
    
    # Process one PDF
    print("\n4. Processing PDF with MinerU Pipeline...")
    try:
        from src.ingest import IngestionPipeline
        
        pipeline = IngestionPipeline()
        print("✓ Pipeline initialized")
        
        print(f"\nProcessing: {test_pdf.name}")
        print("This may take a few minutes...")
        print("-" * 70)
        
        chunks = pipeline.process_pdf_blob(test_pdf)
        
        print("-" * 70)
        print(f"\n✓ Successfully processed!")
        print(f"  - Extracted {len(chunks)} chunks")
        print(f"  - Sample chunk: {chunks[0]['text'][:100]}...")
        
        if Config.UPLOAD_OUTPUT_TO_GCS:
            print(f"\n5. Checking GCS Upload...")
            output_prefix = f"{Config.GCS_OUTPUT_PREFIX}/{Path(test_pdf.name).stem}/"
            output_blobs = list(bucket.list_blobs(prefix=output_prefix, max_results=5))
            if output_blobs:
                print(f"✓ Uploaded {len(output_blobs)} files to GCS")
                for blob in output_blobs[:3]:
                    print(f"  - {blob.name}")
            else:
                print("⚠️  No files found in GCS output location")
        
        print("\n" + "="*70)
        print("✓ MINERU PIPELINE TEST PASSED!")
        print("="*70)
        print("\nYou can now run the full pipeline:")
        print("  python -m src.ingest")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_mineru_pipeline()
    sys.exit(0 if success else 1)

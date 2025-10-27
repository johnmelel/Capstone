"""
Test script to verify all components are working before running the full pipeline
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_env_variables():
    """Test that all required environment variables are set"""
    print("\n" + "="*60)
    print("TEST 1: Environment Variables")
    print("="*60)
    
    required_vars = [
        'GCS_BUCKET_NAME',
        'GOOGLE_SERVICE_ACCOUNT_JSON',
        'MILVUS_URI',
        'MILVUS_API_KEY',
        'GEMINI_API_KEY'
    ]
    
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)
            print(f"❌ {var}: NOT SET")
        else:
            # Mask sensitive values
            if 'KEY' in var or 'API' in var:
                display_value = value[:10] + "..." if len(value) > 10 else "***"
            else:
                display_value = value
            print(f"✓ {var}: {display_value}")
    
    if missing:
        print(f"\n❌ Missing variables: {', '.join(missing)}")
        return False
    
    # Check if service account file exists
    service_account_path = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
    if not Path(service_account_path).exists():
        print(f"❌ Service account file not found: {service_account_path}")
        return False
    else:
        print(f"✓ Service account file found: {service_account_path}")
    
    print("\n✓ All environment variables are set!")
    return True


def test_gemini_api():
    """Test Gemini API connection and embedding"""
    print("\n" + "="*60)
    print("TEST 2: Gemini API Connection")
    print("="*60)
    
    try:
        import google.generativeai as genai
        
        # Configure API
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        print("✓ Gemini API configured")
        
        # Test embedding
        result = genai.embed_content(
            model='models/text-embedding-005',
            content='This is a test sentence for embedding.',
            task_type='retrieval_document'
        )
        print(f"✓ Embedding generated successfully")
        print(f"  Embedding dimension: {len(result['embedding'])}")
        
        # Test token counting
        model = genai.GenerativeModel('gemini-pro')
        token_result = model.count_tokens('This is a test sentence')
        print(f"✓ Token counting working")
        print(f"  Token count: {token_result.total_tokens}")
        
        print("\n✓ Gemini API is working!")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False


def test_gcs_connection():
    """Test Google Cloud Storage connection"""
    print("\n" + "="*60)
    print("TEST 3: Google Cloud Storage Connection")
    print("="*60)
    
    try:
        from google.cloud import storage
        
        # Set credentials
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
        
        # Create client
        client = storage.Client()
        print("✓ GCS client created")
        
        # Get bucket
        bucket_name = os.getenv('GCS_BUCKET_NAME')
        bucket = client.bucket(bucket_name)
        
        # List files
        prefix = os.getenv('GCS_BUCKET_PREFIX', '')
        blobs = list(bucket.list_blobs(max_results=5, prefix=prefix))
        print(f"✓ Connected to bucket: {bucket_name}")
        print(f"  Found {len(blobs)} files (showing max 5)")
        
        # Count PDFs
        pdf_blobs = [b for b in blobs if b.name.lower().endswith('.pdf')]
        print(f"  PDF files found: {len(pdf_blobs)}")
        if pdf_blobs:
            print(f"  Example: {pdf_blobs[0].name}")
        
        print("\n✓ Google Cloud Storage is working!")
        return True
        
    except Exception as e:
        print(f"❌ GCS test failed: {e}")
        return False


def test_milvus_connection():
    """Test Milvus connection"""
    print("\n" + "="*60)
    print("TEST 4: Milvus Connection")
    print("="*60)
    
    try:
        from pymilvus import connections, utility
        
        # Connect
        connections.connect(
            uri=os.getenv('MILVUS_URI'),
            token=os.getenv('MILVUS_API_KEY')
        )
        print("✓ Connected to Milvus")
        
        # List collections
        collections = utility.list_collections()
        print(f"  Existing collections: {collections if collections else 'None'}")
        
        # Disconnect
        connections.disconnect("default")
        
        print("\n✓ Milvus is working!")
        return True
        
    except Exception as e:
        print(f"❌ Milvus test failed: {e}")
        return False


def test_components():
    """Test individual pipeline components"""
    print("\n" + "="*60)
    print("TEST 5: Pipeline Components")
    print("="*60)
    
    try:
        # Test config
        from src.config import Config
        Config.validate()
        print("✓ Configuration validated")
        
        # Test chunker
        from src.chunker import TextChunker
        chunker = TextChunker()
        test_text = "This is a test sentence. " * 100
        chunks = chunker.chunk_text(test_text)
        print(f"✓ Chunker working: Created {len(chunks)} chunks")
        
        # Test embedder
        from src.embedder import TextEmbedder
        embedder = TextEmbedder()
        embeddings = embedder.embed_text(["Test sentence 1", "Test sentence 2"])
        print(f"✓ Embedder working: Generated embeddings with shape {embeddings.shape}")
        
        print("\n✓ All components are working!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_pdf():
    """Test processing a single PDF"""
    print("\n" + "="*60)
    print("TEST 6: Single PDF Processing (Optional)")
    print("="*60)
    
    response = input("Do you want to test processing one PDF? (y/n): ").lower()
    if response != 'y':
        print("Skipped")
        return True
    
    try:
        from src.ingest import IngestionPipeline
        from google.cloud import storage
        
        # Set credentials
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
        
        # Get first PDF
        client = storage.Client()
        bucket = client.bucket(os.getenv('GCS_BUCKET_NAME'))
        prefix = os.getenv('GCS_BUCKET_PREFIX', '')
        blobs = list(bucket.list_blobs(prefix=prefix, max_results=10))
        pdf_blobs = [b for b in blobs if b.name.lower().endswith('.pdf')]
        
        if not pdf_blobs:
            print("❌ No PDF files found in bucket")
            return False
        
        print(f"Testing with: {pdf_blobs[0].name}")
        
        # Initialize pipeline
        pipeline = IngestionPipeline()
        
        # Process single PDF
        pipeline.process_pdf_blob(pdf_blobs[0])
        
        print("\n✓ Single PDF processed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("DATA INGESTION PIPELINE - SETUP TEST")
    print("="*60)
    
    tests = [
        ("Environment Variables", test_env_variables),
        ("Gemini API", test_gemini_api),
        ("Google Cloud Storage", test_gcs_connection),
        ("Milvus Database", test_milvus_connection),
        ("Pipeline Components", test_components),
        ("Single PDF Processing", test_single_pdf),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou're ready to run the full pipeline:")
        print("  python src/ingest.py")
    else:
        print("\n" + "="*60)
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before running the pipeline.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

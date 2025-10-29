"""
Test script to verify all components are working before running the full pipeline
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Config after loading env
from src.config import Config

def test_env_variables():
    """Test that all required environment variables are set"""
    print("\n" + "="*60)
    print("TEST 1: Environment Variables & Configuration")
    print("="*60)
    
    config_vars = [
        ('GCS_BUCKET_NAME', Config.GCS_BUCKET_NAME),
        ('GCS_BUCKET_PREFIX', Config.GCS_BUCKET_PREFIX),
        ('GOOGLE_SERVICE_ACCOUNT_JSON', Config.GOOGLE_SERVICE_ACCOUNT_JSON),
        ('MILVUS_URI', Config.MILVUS_URI),
        ('MILVUS_API_KEY', Config.MILVUS_API_KEY),
        ('MILVUS_COLLECTION_NAME', Config.MILVUS_COLLECTION_NAME),
        ('GEMINI_API_KEY', Config.GEMINI_API_KEY),
        ('EMBEDDING_MODEL', Config.EMBEDDING_MODEL),
        ('MAX_TOKENS_PER_CHUNK', Config.MAX_TOKENS_PER_CHUNK),
        ('CHUNK_SIZE', Config.CHUNK_SIZE),
        ('CHUNK_OVERLAP', Config.CHUNK_OVERLAP),
        ('BATCH_SIZE', Config.BATCH_SIZE),
    ]
    
    missing = []
    for var_name, value in config_vars:
        if value is None or (isinstance(value, str) and not value):
            missing.append(var_name)
            print(f"❌ {var_name}: NOT SET")
        else:
            # Mask sensitive values
            if 'KEY' in var_name or 'API' in var_name:
                display_value = str(value)[:10] + "..." if len(str(value)) > 10 else "***"
            else:
                display_value = value
            print(f"✓ {var_name}: {display_value}")
    
    if missing:
        print(f"\n❌ Missing variables: {', '.join(missing)}")
        return False
    
    # Validate configuration using Config.validate()
    try:
        Config.validate()
        print("\n✓ Configuration validation passed!")
    except Exception as e:
        print(f"\n❌ Configuration validation failed: {e}")
        return False
    
    print("✓ All environment variables are set!")
    return True


def test_gemini_api():
    """Test Gemini API connection and embedding"""
    print("\n" + "="*60)
    print("TEST 2: Gemini API Connection")
    print("="*60)
    
    try:
        import google.generativeai as genai
        
        # Configure API using Config
        genai.configure(api_key=Config.GEMINI_API_KEY)
        print("✓ Gemini API configured")
        
        # Test embedding using configured model
        result = genai.embed_content(
            model=f'models/{Config.EMBEDDING_MODEL}',
            content='This is a test sentence for embedding.',
            task_type='retrieval_document'
        )
        print("✓ Embedding generated successfully")
        print(f"  Embedding dimension: {len(result['embedding'])}")
        
        # Test token counting
        model = genai.GenerativeModel('gemini-2.5-flash')
        token_result = model.count_tokens('This is a test sentence')
        print("✓ Token counting working")
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
        
        # Set credentials using Config
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = Config.GOOGLE_SERVICE_ACCOUNT_JSON
        
        # Create client
        client = storage.Client()
        print("✓ GCS client created")
        
        # Get bucket using Config
        bucket = client.bucket(Config.GCS_BUCKET_NAME)
        
        # List files using Config prefix
        blobs = list(bucket.list_blobs(max_results=5, prefix=Config.GCS_BUCKET_PREFIX))
        print(f"✓ Connected to bucket: {Config.GCS_BUCKET_NAME}")
        print(f"  Prefix: {Config.GCS_BUCKET_PREFIX or '(root)'}")
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
        
        # Connect using Config
        connections.connect(
            uri=Config.MILVUS_URI,
            token=Config.MILVUS_API_KEY
        )
        print(f"✓ Connected to Milvus: {Config.MILVUS_URI}")
        
        # List collections
        collections = utility.list_collections()
        print(f"  Existing collections: {collections if collections else 'None'}")
        print(f"  Target collection: {Config.MILVUS_COLLECTION_NAME}")
        
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
        # Test config - already loaded, just validate again
        Config.validate()
        print("✓ Configuration validated")
        
        # Test chunker using Config values
        from src.chunker import TextChunker
        chunker = TextChunker()
        test_text = "This is a test sentence. " * 100
        chunks = chunker.chunk_text(test_text)
        print(f"✓ Chunker working: Created {len(chunks)} chunks")
        print(f"  Chunk size: {Config.CHUNK_SIZE} chars, Overlap: {Config.CHUNK_OVERLAP} chars")
        print(f"  Max tokens per chunk: {Config.MAX_TOKENS_PER_CHUNK}")
        
        # Test embedder using Config
        from src.embedder import TextEmbedder
        embedder = TextEmbedder()
        embeddings = embedder.embed_text(["Test sentence 1", "Test sentence 2"])
        print(f"✓ Embedder working: Generated embeddings with shape {embeddings.shape}")
        print(f"  Using model: {Config.EMBEDDING_MODEL}")
        
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
        
        # Set credentials using Config
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = Config.GOOGLE_SERVICE_ACCOUNT_JSON
        
        # Get first PDF using Config values
        client = storage.Client()
        bucket = client.bucket(Config.GCS_BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=Config.GCS_BUCKET_PREFIX, max_results=10))
        pdf_blobs = [b for b in blobs if b.name.lower().endswith('.pdf')]
        
        if not pdf_blobs:
            print("❌ No PDF files found in bucket")
            print(f"  Bucket: {Config.GCS_BUCKET_NAME}")
            print(f"  Prefix: {Config.GCS_BUCKET_PREFIX or '(root)'}")
            return False
        
        print(f"Testing with: {pdf_blobs[0].name}")
        print(f"  File size: {pdf_blobs[0].size / 1024:.2f} KB")
        
        # Initialize pipeline (uses Config internally)
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

"""
Test script to verify all components are working before running the full pipeline
Updated for the new google-genai SDK
"""

import os
import sys
import pytest
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
        pytest.fail(f"Missing variables: {', '.join(missing)}")
    
    # Validate configuration using Config.validate()
    try:
        Config.validate()
        print("\n✓ Configuration validation passed!")
    except Exception as e:
        print(f"\n❌ Configuration validation failed: {e}")
        pytest.fail(f"Configuration validation failed: {e}")
    
    print("✓ All environment variables are set!")


def test_gemini_api():
    """Test Gemini API connection and embedding with new SDK"""
    print("\n" + "="*60)
    print("TEST 2: Gemini API Connection (New SDK)")
    print("="*60)
    
    try:
        from google import genai
        from google.genai import types
        
        # Create client using new SDK
        client = genai.Client(api_key=Config.GEMINI_API_KEY)
        print("✓ Gemini client created with new SDK")
        
        # Test embedding using new SDK
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
        
        result = client.models.embed_content(
            model=Config.EMBEDDING_MODEL,
            contents='This is a test sentence for embedding.',
            config=config
        )
        print("✓ Embedding generated successfully")
        print(f"  Embedding dimension: {len(result.embeddings[0].values)}")
        
        # Test token counting with tokenizer
        try:
            tokenizer = genai.LocalTokenizer(model_name='gemini-2.0-flash-exp')
            token_result = tokenizer.compute_tokens('This is a test sentence')
            print("✓ Token counting working")
            print(f"  Token count: {token_result.token_count}")
        except Exception as e:
            print(f"⚠️  Token counting unavailable: {e}")
            print("  (This is optional - estimation will be used)")
        
        print("\n✓ Gemini API (new SDK) is working!")
        
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Gemini API test failed: {e}")


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
        
    except Exception as e:
        print(f"❌ GCS test failed: {e}")
        pytest.fail(f"GCS test failed: {e}")


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
        
    except Exception as e:
        print(f"❌ Milvus test failed: {e}")
        pytest.fail(f"Milvus test failed: {e}")


def test_components():
    """Test individual pipeline components with new SDK"""
    print("\n" + "="*60)
    print("TEST 5: Pipeline Components (New SDK)")
    print("="*60)
    
    try:
        # Test config - already loaded, just validate again
        Config.validate()
        print("✓ Configuration validated")
        
        # Test chunker using Config values
        from src.chunker import TextChunker
        import time
        chunker = TextChunker()
        test_text = "This is a test sentence. " * 100
        print(f"Testing chunker with input size: {len(test_text)} characters")
        print(f"  Chunk size: {Config.CHUNK_SIZE} chars, Overlap: {Config.CHUNK_OVERLAP} chars, Max tokens: {Config.MAX_TOKENS_PER_CHUNK}")
        start_time = time.time()
        print("Calling chunker.chunk_text()...")
        try:
            chunks = chunker.chunk_text(test_text)
            print("chunker.chunk_text() returned!")
        except Exception as e:
            print(f"Exception during chunking: {e}")
            import traceback
            traceback.print_exc()
            return False
        elapsed = time.time() - start_time
        print(f"✓ Chunker working: Created {len(chunks)} chunks in {elapsed:.3f} seconds")
        if len(chunks) == 0:
            print("❌ No chunks produced! Check input and config.")
            pytest.fail("No chunks produced! Check input and config.")
        else:
            print(f"  First chunk: '{chunks[0][:50]}...' ({len(chunks[0])} chars)")
            print(f"  Last chunk: '{chunks[-1][:50]}...' ({len(chunks[-1])} chars)")
            if chunks:
                print("  Token counts for first 3 chunks (estimated):")
                for i, chunk in enumerate(chunks[:3]):
                    try:
                        tokens = chunker._estimate_tokens(chunk)
                    except Exception as e:
                        tokens = f"Error: {e}"
                    print(f"    Chunk {i+1}: ~{tokens} tokens, {len(chunk)} chars")
        if elapsed > 5:
            print(f"⚠️ Chunking took longer than expected: {elapsed:.2f} seconds")

        # Test embedder using new SDK
        from src.embedder import TextEmbedder
        embedder = TextEmbedder()
        print(f"Testing embedder with new SDK (model: {Config.EMBEDDING_MODEL})...")
        embeddings = embedder.embed_text(["Test sentence 1", "Test sentence 2"])
        print(f"✓ Embedder working: Generated embeddings with shape {embeddings.shape}")
        print(f"  Using model: {Config.EMBEDDING_MODEL}")
        print(f"  Embedding dimension: {embedder.get_embedding_dimension()}")

        print("\n✓ All components are working with new SDK!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Component test failed: {e}")


def test_single_pdf():
    """Test processing a single PDF"""
    print("\n" + "="*60)
    print("TEST 6: Single PDF Processing (Optional)")
    print("="*60)
    
    response = input("Do you want to test processing one PDF? (y/n): ").lower()
    if response != 'y':
        print("Skipped")
        pytest.skip("User skipped PDF processing test")
    
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
            pytest.fail(f"No PDF files found in bucket {Config.GCS_BUCKET_NAME} with prefix {Config.GCS_BUCKET_PREFIX}")
        
        print(f"Testing with: {pdf_blobs[0].name}")
        print(f"  File size: {pdf_blobs[0].size / 1024:.2f} KB")
        
        # Initialize pipeline (uses Config and new SDK internally)
        pipeline = IngestionPipeline()
        
        # Process single PDF
        pipeline.process_pdf_blob(pdf_blobs[0])
        
        print("\n✓ Single PDF processed successfully with new SDK!")
        
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"PDF processing test failed: {e}")


def main():
    """Run all tests"""
    print("="*60)
    print("DATA INGESTION PIPELINE - SETUP TEST (NEW SDK)")
    print("="*60)
    print("Now using google-genai SDK (new unified SDK)")
    print("="*60)
    
    tests = [
        ("Environment Variables", test_env_variables),
        ("Gemini API (New SDK)", test_gemini_api),
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


import sys
import logging
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
import tempfile

# Add data_ingestion to path to allow src imports if running from different location
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.pdf_extractor import PDFExtractor
from src.chunker import ExactTokenChunker, ImageCaptionChunker, chunk_with_metadata

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_from_gcs(gcs_path: str) -> Path:
    if not gcs_path.startswith("gs://"):
        return Path(gcs_path)
    
    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    
    print(f"Downloading from Bucket: {bucket_name}, Blob: {blob_name}")
    
    credentials = service_account.Credentials.from_service_account_file(
        Config.GOOGLE_SERVICE_ACCOUNT_JSON
    )
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    blob.download_to_filename(temp_pdf.name)
    temp_pdf.close()
    
    return Path(temp_pdf.name)

def verify_pipeline(input_path: str):
    print(f"\n{'='*80}")
    print(f"VERIFYING PIPELINE FOR: {input_path}")
    print(f"{'='*80}\n")
    
    # 1. Setup
    pdf_path = download_from_gcs(input_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        return

    # Mock Blob for PDFExtractor
    class MockBlob:
        def __init__(self, path):
            self.name = path.name
            self.bucket = None
            self._path = path
        def download_to_filename(self, filename):
            import shutil
            shutil.copy(self._path, filename)
            
    mock_blob = MockBlob(pdf_path)
    
    # 2. Extraction
    print(">>> Running PDFExtractor (with CaptionTagger and Markdown Replacement)...")
    extractor = PDFExtractor(extract_images=True)
    result = extractor.extract_with_images(mock_blob)
    
    text = result['text']
    images = result['images']
    
    print(f"\n[Extraction Result]")
    print(f"Text Length: {len(text)} chars")
    print(f"Image Count: {len(images)}")
    
    # Check for replaced tags
    print("\n[Checking Text for Image Captions]")
    if "[Image:" in text:
        print("SUCCESS: Found '[Image: ...]' tags in text!")
        # Show a sample
        import re
        matches = re.findall(r'\[Image: .*?\]', text)
        for i, match in enumerate(matches[:3]):
            print(f"  Match {i+1}: {match[:100]}...")
    else:
        print("WARNING: No '[Image: ...]' tags found. (Might be expected if PDF has no captioned images)")

    # 3. Chunking
    print("\n>>> Running Chunkers...")
    
    # Text Chunking
    text_chunker = ExactTokenChunker(chunk_size=512, chunk_overlap=50)
    text_chunks = chunk_with_metadata(text, pdf_path, text_chunker)
    print(f"Generated {len(text_chunks)} text chunks.")
    
    # Image Chunking
    image_chunker = ImageCaptionChunker(text_chunker)
    image_chunks = []
    
    print("\n[Checking Image Chunking & Caption Splitting]")
    for i, img in enumerate(images):
        caption = img.get('caption')
        chunks = image_chunker.chunk_image(img)
        
        if i < 3 or len(chunks) > 1: # Show first few or any split ones
            print(f"Image {i}: Caption Length={len(caption) if caption else 0}")
            if len(chunks) > 1:
                print(f"  -> SPLIT into {len(chunks)} chunks!")
                for j, (img_data, cap_chunk) in enumerate(chunks):
                    print(f"     Chunk {j}: {cap_chunk[:50]}...")
            else:
                 print(f"  -> Single chunk. Caption: {chunks[0][1][:50] if chunks[0][1] else 'None'}")
                 
        if chunks:
             # Flatten for count
             for c in chunks:
                 image_chunks.append(c)

    print(f"\nTotal Image Chunks generated: {len(image_chunks)}")
    
    # Cleanup
    if input_path.startswith("gs://"):
        pdf_path.unlink()

if __name__ == "__main__":
    # Default from debug script
    DEFAULT_PATH = "gs://adsp-34002-ip09-team-2/Data/LIRADSpapers-selected/m-cunha-et-al-2021-how-to-use-li-rads-to-report-liver-ct-and-mri-observations.pdf"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", nargs='?', default=DEFAULT_PATH)
    args = parser.parse_args()
    
    verify_pipeline(args.input_path)

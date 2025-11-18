"""Debug script for MinerU image extraction"""

import sys
import logging
from pathlib import Path
import tempfile
import shutil
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.pdf_extractor import PDFExtractor
from src.utils import setup_logging

# Load environment
load_dotenv()

# Setup logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def list_directory_tree(directory: Path, prefix="", max_depth=5, current_depth=0):
    """Recursively list directory structure"""
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            
            if item.is_dir():
                print(f"{prefix}{connector}{item.name}/")
                extension = "    " if is_last else "│   "
                list_directory_tree(item, prefix + extension, max_depth, current_depth + 1)
            else:
                size_kb = item.stat().st_size / 1024
                print(f"{prefix}{connector}{item.name} ({size_kb:.2f} KB)")
    except PermissionError:
        print(f"{prefix}[Permission Denied]")


def debug_pdf_extraction(pdf_path: Path):
    """
    Debug PDF extraction with detailed output at each stage
    
    Args:
        pdf_path: Path to test PDF file
    """
    print_section("MinerU Image Extraction Debug")
    
    # Check configuration
    print_section("Configuration Check")
    print(f"ENABLE_MULTIMODAL: {Config.ENABLE_MULTIMODAL}")
    print(f"MINERU_BACKEND: {Config.MINERU_BACKEND}")
    print(f"MINERU_MODEL_SOURCE: {Config.MINERU_MODEL_SOURCE}")
    print(f"MINERU_LANG: {Config.MINERU_LANG}")
    print(f"MINERU_DEBUG_MODE: {Config.MINERU_DEBUG_MODE}")
    print(f"MINERU_ENABLE_TABLES: {Config.MINERU_ENABLE_TABLES}")
    print(f"MINERU_ENABLE_FORMULAS: {Config.MINERU_ENABLE_FORMULAS}")
    print(f"GCS_IMAGES_PREFIX: {Config.GCS_IMAGES_PREFIX}")
    
    if not Config.ENABLE_MULTIMODAL:
        print("\n⚠️  WARNING: ENABLE_MULTIMODAL is set to False!")
        print("   Images will not be extracted. To enable, set ENABLE_MULTIMODAL=true in .env")
    
    # Check PDF file
    print_section("PDF File Check")
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    print(f"✓ PDF found: {pdf_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    
    # Initialize extractor with image extraction enabled
    print_section("Initializing PDF Extractor")
    extractor = PDFExtractor(extract_images=True)
    print(f"✓ Extractor initialized")
    print(f"  extract_images: {extractor.extract_images}")
    print(f"  use_gpu: {extractor.use_gpu}")
    print(f"  backend: {extractor.backend}")
    
    # Create temporary output directory (for debugging, we won't auto-delete)
    temp_output_dir = Path(tempfile.mkdtemp(prefix="mineru_debug_"))
    print(f"✓ Created temp directory: {temp_output_dir}")
    
    try:
        # Process PDF with MinerU
        print_section("Processing PDF with MinerU")
        print("This may take a while depending on PDF size...")
        
        # Call the internal method directly to get more control
        pdf_bytes = pdf_path.read_bytes()
        pdf_name = pdf_path.stem
        
        print(f"  Processing: {pdf_name}")
        print(f"  Output dir: {temp_output_dir}")
        
        # Import MinerU here
        from mineru.cli.common import do_parse, read_fn
        
        # Read PDF
        print("\n📖 Reading PDF...")
        pdf_bytes = read_fn(pdf_path)
        print(f"  ✓ Read {len(pdf_bytes)} bytes")
        
        # Parse with MinerU
        print("\n🔄 Parsing with MinerU...")
        print(f"  f_dump_content_list: {True}")  # Enable content list for image metadata
        
        do_parse(
            output_dir=str(temp_output_dir),
            pdf_file_names=[pdf_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[extractor.lang],
            backend=extractor.backend,
            parse_method='auto',
            formula_enable=extractor.enable_formulas,
            table_enable=extractor.enable_tables,
            f_dump_md=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,  # Enable content list for images
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False
        )
        
        print("  ✓ MinerU processing complete")
        
        # Show output directory structure
        print_section("MinerU Output Directory Structure")
        list_directory_tree(temp_output_dir)
        
        # Check for markdown file
        print_section("Text Extraction Check")
        md_file_path = temp_output_dir / pdf_name / 'auto' / f"{pdf_name}.md"
        
        if md_file_path.exists():
            print(f"✓ Markdown file found: {md_file_path}")
            with open(md_file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            print(f"  Text length: {len(md_content)} characters")
            print(f"  Preview (first 500 chars):")
            print("-" * 80)
            print(md_content[:500])
            print("-" * 80)
        else:
            print(f"❌ Markdown file not found at: {md_file_path}")
        
        # Check for images directory
        print_section("Image Extraction Check")
        images_dir = temp_output_dir / pdf_name / 'auto' / 'images'
        
        if images_dir.exists():
            print(f"✓ Images directory found: {images_dir}")
            
            # List all image files
            image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg'))
            
            if image_files:
                print(f"  ✓ Found {len(image_files)} images:")
                for img_file in image_files:
                    img_size_kb = img_file.stat().st_size / 1024
                    print(f"    - {img_file.name} ({img_size_kb:.2f} KB)")
                
                # Show first image details
                if len(image_files) > 0:
                    first_img = image_files[0]
                    print(f"\n  First image details:")
                    print(f"    Path: {first_img}")
                    print(f"    Size: {first_img.stat().st_size / 1024:.2f} KB")
                    
                    try:
                        from PIL import Image
                        with Image.open(first_img) as img:
                            print(f"    Dimensions: {img.size[0]}x{img.size[1]} pixels")
                            print(f"    Format: {img.format}")
                            print(f"    Mode: {img.mode}")
                    except Exception as e:
                        print(f"    Could not load image: {e}")
            else:
                print(f"  ⚠️  Images directory exists but is empty!")
                print(f"     This might mean:")
                print(f"     1. The PDF has no extractable images")
                print(f"     2. MinerU failed to extract images")
                print(f"     3. Images are in a different location")
        else:
            print(f"❌ Images directory not found at: {images_dir}")
            print(f"   Expected location: {images_dir}")
            print(f"\n   Checking for alternate locations...")
            
            # Check alternate locations
            possible_locations = [
                temp_output_dir / pdf_name / 'images',
                temp_output_dir / pdf_name / 'auto',
                temp_output_dir / 'images',
            ]
            
            for loc in possible_locations:
                if loc.exists() and loc.is_dir():
                    print(f"   Found directory: {loc}")
                    imgs = list(loc.glob('*.png')) + list(loc.glob('*.jpg'))
                    if imgs:
                        print(f"     Contains {len(imgs)} images!")
        
        # Check content_list.json
        print_section("Content List Check")
        content_list_path = temp_output_dir / pdf_name / 'auto' / f"{pdf_name}_content_list.json"
        
        if content_list_path.exists():
            print(f"✓ Content list found: {content_list_path}")
            
            try:
                import json
                with open(content_list_path, 'r', encoding='utf-8') as f:
                    content_data = json.load(f)
                
                print(f"  Type: {type(content_data)}")
                
                if isinstance(content_data, list):
                    print(f"  Total items: {len(content_data)}")
                    
                    # Count items by type
                    type_counts = {}
                    image_items = []
                    
                    for item in content_data:
                        if isinstance(item, dict):
                            item_type = item.get('type', 'unknown')
                            type_counts[item_type] = type_counts.get(item_type, 0) + 1
                            
                            if item_type == 'image':
                                image_items.append(item)
                    
                    print(f"  Items by type:")
                    for item_type, count in sorted(type_counts.items()):
                        print(f"    {item_type}: {count}")
                    
                    if image_items:
                        print(f"\n  First image metadata:")
                        import json
                        print(json.dumps(image_items[0], indent=2))
                    else:
                        print(f"\n  ⚠️  No image items found in content list!")
                elif isinstance(content_data, dict):
                    print(f"  Keys: {list(content_data.keys())}")
                
            except Exception as e:
                print(f"  ❌ Error parsing content list: {e}")
        else:
            print(f"❌ Content list not found at: {content_list_path}")
        
        # Now test the extraction method
        print_section("Testing extract_with_images() Method")
        
        result = extractor.extract_with_images(pdf_path)
        
        if result:
            print(f"✓ Extraction successful!")
            print(f"  Text length: {len(result['text'])} characters")
            print(f"  Images extracted: {len(result['images'])}")
            
            if result['images']:
                print(f"\n  Image details:")
                for i, img_data in enumerate(result['images']):
                    print(f"    Image {i+1}:")
                    print(f"      Page: {img_data['page_num']}")
                    print(f"      Index: {img_data['image_index']}")
                    print(f"      Size: {img_data['size']}")
                    print(f"      Bytes: {len(img_data['bytes'])} bytes")
            else:
                print(f"\n  ⚠️  No images in result!")
        else:
            print(f"❌ Extraction returned None!")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ask user if they want to keep the temp directory
        print_section("Cleanup")
        print(f"Temporary directory: {temp_output_dir}")
        
        keep = input("Keep temporary directory for inspection? (y/n): ").lower().strip()
        
        if keep == 'y':
            print(f"✓ Kept directory: {temp_output_dir}")
            print(f"  You can inspect the files manually")
        else:
            try:
                shutil.rmtree(temp_output_dir)
                print(f"✓ Deleted temporary directory")
            except Exception as e:
                print(f"⚠️  Failed to delete directory: {e}")


def main():
    """Main entry point"""
    import argparse
    from google.cloud import storage
    from google.oauth2 import service_account
    
    parser = argparse.ArgumentParser(
        description="Debug MinerU image extraction - tests first PDF from GCS bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        nargs='?',  # Make optional
        help="Path to PDF file to test (optional, uses first PDF from GCS bucket if not provided)"
    )
    
    args = parser.parse_args()
    
    # Use command line arg if provided, otherwise get first PDF from GCS
    if args.pdf_path:
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            print(f"Error: PDF file not found: {pdf_path}")
            sys.exit(1)
    else:
        print("No PDF path provided, fetching first PDF from GCS bucket...")
        print(f"Bucket: {Config.GCS_BUCKET_NAME}")
        print(f"Prefix: {Config.GCS_BUCKET_PREFIX or '(none)'}")
        
        try:
            # Initialize GCS client
            credentials = service_account.Credentials.from_service_account_file(
                Config.GOOGLE_SERVICE_ACCOUNT_JSON
            )
            gcs_client = storage.Client(credentials=credentials)
            bucket = gcs_client.bucket(Config.GCS_BUCKET_NAME)
            
            # List PDFs
            delimiter = None if Config.GCS_RECURSIVE else '/'
            prefix = f"{Config.GCS_BUCKET_PREFIX}/" if Config.GCS_BUCKET_PREFIX else None
            
            blobs = list(bucket.list_blobs(prefix=prefix, delimiter=delimiter))
            pdf_blobs = [
                blob for blob in blobs 
                if blob.name.lower().endswith('.pdf') and not blob.name.endswith('/')
            ]
            
            if not pdf_blobs:
                print("Error: No PDF files found in GCS bucket")
                sys.exit(1)
            
            # Use first PDF
            first_blob = pdf_blobs[0]
            print(f"\nFound {len(pdf_blobs)} PDFs, using first: {first_blob.name}")
            print(f"Size: {first_blob.size / (1024*1024):.2f} MB")
            
            # Download to temp file
            temp_pdf_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            pdf_path = Path(temp_pdf_file.name)
            
            print(f"Downloading to: {pdf_path}")
            blob_data = first_blob.download_as_bytes()
            temp_pdf_file.write(blob_data)
            temp_pdf_file.flush()
            temp_pdf_file.close()
            print("Download complete!\n")
            
        except Exception as e:
            print(f"Error fetching PDF from GCS: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    debug_pdf_extraction(pdf_path)


if __name__ == "__main__":
    main()

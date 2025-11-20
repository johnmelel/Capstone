"""
Debug script for Mineru parsing and image caption tagging.
This script runs Mineru with visualization options enabled and applies the CaptionTagger.
It supports downloading PDFs from GCS and generating a custom text output file.
"""

import sys
import logging
from pathlib import Path
import tempfile
import shutil
import json
import argparse
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.caption_tagger import CaptionTagger
from mineru.cli.common import do_parse, read_fn

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def download_from_gcs(gcs_path: str) -> Path:
    """
    Download a file from GCS given a full path (gs://bucket/path/to/file.pdf)
    """
    if not gcs_path.startswith("gs://"):
        raise ValueError("GCS path must start with gs://")
    
    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    
    print(f"Downloading from Bucket: {bucket_name}, Blob: {blob_name}")
    
    try:
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
    except Exception as e:
        logger.error(f"Failed to download from GCS: {e}")
        raise

def generate_debug_text_file(content_list: list, tagged_images: list, output_path: Path):
    """
    Generate a text file with all texts indexed and image placeholders with caption IDs.
    """
    # Create a map of image index to caption info for easy lookup
    # We need to map from the index in content_list to the tagged info
    # But tagged_images is a subset.
    # Let's just iterate content_list and check if it's an image.
    
    # First, build a lookup for tagged images based on their original index if possible,
    # or just re-run tagging logic or use the fact that we iterate content_list.
    
    # Actually, CaptionTagger returns a list of tagged images. 
    # To map back to the content_list, we might need to know which item in content_list corresponds to which tagged image.
    # The current CaptionTagger implementation iterates content_list and appends to tagged_images.
    # So we can just re-run the logic or iterate content_list and use the tagger's helper.
    
    tagger = CaptionTagger()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("DEBUG OUTPUT: Text and Image Placeholders\n")
        f.write("=========================================\n\n")
        
        for i, block in enumerate(content_list):
            block_type = block.get('type')
            
            if block_type == 'text':
                text = block.get('text', '').strip()
                f.write(f"[BLOCK ID: {i}] {text}\n\n")
            
            elif block_type == 'image':
                # Find caption for this image
                caption, caption_id = tagger._find_caption_for_image(content_list, i)
                
                f.write(f"[IMAGE BLOCK ID: {i}]\n")
                if caption_id is not None:
                    f.write(f"  -> CAPTION FOUND AT BLOCK ID: {caption_id}\n")
                    f.write(f"  -> CAPTION TEXT: {caption}\n")
                else:
                    f.write("  -> NO CAPTION FOUND\n")
                f.write("\n")
            
            # Handle other types if necessary (tables, etc.)
            elif block_type == 'table':
                 f.write(f"[TABLE BLOCK ID: {i}] (Table content omitted)\n\n")

def debug_parsing(input_path: str):
    """
    Run Mineru parsing with visualization and test caption tagging.
    """
    print_section(f"Debugging Parsing for: {input_path}")
    
    # Handle GCS or local path
    if input_path.startswith("gs://"):
        pdf_path = download_from_gcs(input_path)
        is_temp = True
        original_name = Path(input_path).stem
    else:
        pdf_path = Path(input_path)
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return
        is_temp = False
        original_name = pdf_path.stem

    # Create output directory
    output_base = Path(__file__).parent / "debug_output"
    output_base.mkdir(exist_ok=True)
    
    # Create a specific folder for this run
    output_dir = output_base / original_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    logger.info(f"Output directory: {output_dir}")

    try:
        # 1. Run Mineru Parsing
        print_section("Running Mineru do_parse")
        
        pdf_bytes = read_fn(pdf_path)
        
        # Configure Mineru to draw bounding boxes for visualization
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[original_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=['en'],
            backend=Config.MINERU_BACKEND,
            parse_method='auto',
            formula_enable=Config.MINERU_ENABLE_FORMULAS,
            table_enable=Config.MINERU_ENABLE_TABLES,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,
            f_draw_layout_bbox=True, # Highlight sections
            f_draw_span_bbox=True    # Highlight spans
        )
        
        print("✓ Mineru parsing complete")
        
        # 2. Inspect Output and Generate Text File
        auto_dir = output_dir / original_name / 'auto'
        content_list_path = auto_dir / f"{original_name}_content_list.json"
        
        if not content_list_path.exists():
            logger.error("content_list.json not found!")
            return

        with open(content_list_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
            
        # Generate the requested text file
        text_output_path = output_dir / f"{original_name}_debug_content.txt"
        
        # Get tagged images for JSON dump
        tagger = CaptionTagger()
        tagged_images = tagger.tag_images(content_list)
        
        generate_debug_text_file(content_list, tagged_images, text_output_path)
        
        print(f"✓ Debug text file generated: {text_output_path}")
        
        # 3. Save Tagged Results JSON
        tagged_output_path = output_dir / "tagged_images.json"
        with open(tagged_output_path, 'w', encoding='utf-8') as f:
            json.dump(tagged_images, f, indent=2, ensure_ascii=False)
            
        print(f"✓ Tagged images JSON saved to: {tagged_output_path}")
        
        print_section("Debug Complete")
        print(f"Inspect results in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if is_temp and pdf_path.exists():
            pdf_path.unlink()

# Default path for debugging - EDIT THIS
DEFAULT_INPUT_PATH = "gs://adsp-34002-ip09-team-2/Data/LIRADSpapers-selected/m-cunha-et-al-2021-how-to-use-li-rads-to-report-liver-ct-and-mri-observations.pdf"

def main():
    parser = argparse.ArgumentParser(description="Debug Mineru parsing and caption tagging")
    parser.add_argument("input_path", type=str, nargs='?', default=DEFAULT_INPUT_PATH, help="Path to PDF file (local or gs://)")
    args = parser.parse_args()
    
    print(f"Using input path: {args.input_path}")
    debug_parsing(args.input_path)

if __name__ == "__main__":
    main()

"""
=====================================================================
TEST SCRIPT: Docling Image Extraction Diagnostic
=====================================================================

PURPOSE:
This script tests whether Docling is actually extracting images from PDFs.
It performs multiple diagnostic checks to identify where the problem might be.

WHAT THIS SCRIPT DOES:
1. Initializes Docling with pipeline options for image generation
2. Converts a test PDF
3. Inspects the document structure for images
4. Checks multiple ways to access images (pictures, figures, etc.)
5. Attempts to save any found images to disk
6. Provides detailed diagnostic output

USAGE:
    python test_docling_images.py <path_to_pdf>

ARCHITECTURE NOTES:
- Docling's PdfPipelineOptions.generate_picture_images must be True
- Images are accessed via: result.document.pictures
- Each picture object should have: .image, .caption, .page attributes
=====================================================================
"""

from pathlib import Path
import sys

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.document import PdfFormatOption
    DOCLING_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] Docling not installed: {e}")
    print("[INFO] Install with: pip install docling")
    sys.exit(1)


def test_docling_image_extraction(pdf_path: str):
    """
    Test docling image extraction with detailed diagnostics.
    
    PROCESS:
    1. Setup pipeline options with image generation enabled
    2. Convert PDF with docling
    3. Inspect document structure
    4. Check for images in multiple locations
    5. Attempt to save images
    6. Print comprehensive diagnostic report
    """
    
    print("\n" + "="*70)
    print("DOCLING IMAGE EXTRACTION DIAGNOSTIC TEST")
    print("="*70)
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        return
    
    print(f"\n📄 Testing PDF: {pdf_path.name}")
    print(f"   Location: {pdf_path}")
    
    # ========================================================================
    # STEP 1: Setup Pipeline Options
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 1: Setting up Docling Pipeline Options")
    print("-"*70)
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True  # CRITICAL FOR IMAGE EXTRACTION
    
    print(f"✓ generate_picture_images: {pipeline_options.generate_picture_images}")
    print(f"  images_scale: {getattr(pipeline_options, 'images_scale', 'default')}")
    print(f"  do_picture_description: {getattr(pipeline_options, 'do_picture_description', False)}")
    
    # ========================================================================
    # STEP 2: Initialize Converter
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 2: Initializing Document Converter")
    print("-"*70)
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    print("✓ Converter initialized with image pipeline options")
    
    # ========================================================================
    # STEP 3: Convert PDF
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 3: Converting PDF (this may take a moment...)")
    print("-"*70)
    
    try:
        result = converter.convert(str(pdf_path))
        print("✓ PDF conversion complete")
    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # STEP 4: Inspect Document Structure
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 4: Inspecting Document Structure")
    print("-"*70)
    
    doc = result.document
    
    # Check document attributes
    print("\n📋 Document Attributes:")
    attrs = dir(doc)
    image_related = [a for a in attrs if 'picture' in a.lower() or 'image' in a.lower() or 'figure' in a.lower()]
    print(f"   Image-related attributes: {image_related}")
    
    # ========================================================================
    # STEP 5: Check Pictures Collection
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 5: Checking Pictures Collection")
    print("-"*70)
    
    print(f"\n🖼️  Checking: result.document.pictures")
    
    if not hasattr(doc, 'pictures'):
        print("   ❌ PROBLEM: Document has no 'pictures' attribute!")
        print("   This suggests images weren't extracted.")
    else:
        pictures = doc.pictures
        print(f"   ✓ pictures attribute exists")
        print(f"   Type: {type(pictures)}")
        
        try:
            pic_list = list(pictures)
            print(f"   Count: {len(pic_list)} pictures found")
            
            if len(pic_list) == 0:
                print("\n   ⚠️  WARNING: Zero pictures found!")
                print("   Possible reasons:")
                print("      1. PDF has no images")
                print("      2. Images are embedded in a way Docling can't extract")
                print("      3. Pipeline options not configured correctly")
                print("      4. PDF is scanned/image-based (not vector)")
            else:
                print(f"\n   ✅ SUCCESS: {len(pic_list)} pictures extracted!")
                
                # ========================================================================
                # STEP 6: Inspect First Picture
                # ========================================================================
                print("\n" + "-"*70)
                print("STEP 6: Inspecting First Picture Object")
                print("-"*70)
                
                first_pic = pic_list[0]
                print(f"\n🔍 First Picture Attributes:")
                print(f"   Type: {type(first_pic)}")
                print(f"   Attributes: {[a for a in dir(first_pic) if not a.startswith('_')]}")
                
                # Check for image attribute
                if hasattr(first_pic, 'image'):
                    print(f"   ✓ Has .image attribute")
                    print(f"     Image type: {type(first_pic.image)}")
                    
                    # Check if it's a PIL image
                    try:
                        from PIL import Image
                        if isinstance(first_pic.image, Image.Image):
                            print(f"     ✓ Is PIL Image")
                            print(f"     Size: {first_pic.image.size}")
                            print(f"     Mode: {first_pic.image.mode}")
                    except:
                        pass
                else:
                    print(f"   ❌ No .image attribute!")
                
                # Check for caption
                if hasattr(first_pic, 'caption'):
                    caption_text = first_pic.caption.text if hasattr(first_pic.caption, 'text') else str(first_pic.caption)
                    print(f"   Caption: {caption_text[:100] if caption_text else 'None'}")
                
                # Check for page number
                if hasattr(first_pic, 'page'):
                    print(f"   Page: {first_pic.page}")
                
                # ========================================================================
                # STEP 7: Try to Save Images
                # ========================================================================
                print("\n" + "-"*70)
                print("STEP 7: Attempting to Save Images")
                print("-"*70)
                
                output_dir = Path("test_outputs")
                output_dir.mkdir(exist_ok=True)
                
                saved_count = 0
                for idx, picture in enumerate(pic_list):
                    try:
                        if hasattr(picture, 'image'):
                            output_path = output_dir / f"test_image_{idx}.png"
                            picture.image.save(output_path)
                            print(f"   ✓ Saved: {output_path.name}")
                            saved_count += 1
                        else:
                            print(f"   ❌ Picture {idx} has no .image attribute")
                    except Exception as e:
                        print(f"   ❌ Failed to save picture {idx}: {e}")
                
                print(f"\n   📊 Saved {saved_count}/{len(pic_list)} images to {output_dir}/")
                
        except Exception as e:
            print(f"   ❌ Error iterating pictures: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # STEP 8: Check Alternative Image Sources
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 8: Checking Alternative Image Sources")
    print("-"*70)
    
    # Check if there's a 'figures' attribute
    if hasattr(doc, 'figures'):
        print(f"\n🖼️  Found 'figures' attribute")
        try:
            fig_list = list(doc.figures)
            print(f"   Count: {len(fig_list)}")
        except Exception as e:
            print(f"   Error accessing figures: {e}")
    else:
        print(f"\n   No 'figures' attribute")
    
    # Check document elements
    if hasattr(doc, 'elements'):
        print(f"\n📦 Checking document.elements")
        try:
            elements = list(doc.elements)
            print(f"   Total elements: {len(elements)}")
            
            # Count element types
            from collections import Counter
            type_counts = Counter([type(e).__name__ for e in elements])
            print(f"   Element types: {dict(type_counts)}")
            
            # Look for picture/image elements
            image_elements = [e for e in elements if 'picture' in type(e).__name__.lower() or 'image' in type(e).__name__.lower()]
            print(f"   Image-like elements: {len(image_elements)}")
            
        except Exception as e:
            print(f"   Error accessing elements: {e}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    if hasattr(doc, 'pictures'):
        pic_count = len(list(doc.pictures))
        if pic_count > 0:
            print(f"✅ SUCCESS: Docling extracted {pic_count} images")
            print(f"   Images saved to: test_outputs/")
        else:
            print(f"⚠️  WARNING: No images extracted from PDF")
            print(f"   Possible reasons:")
            print(f"   • PDF contains no extractable images")
            print(f"   • Images are embedded as raster/bitmap in complex ways")
            print(f"   • Try a different PDF with clear figures")
    else:
        print(f"❌ PROBLEM: Document has no 'pictures' attribute")
        print(f"   This indicates the pipeline didn't extract images")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_docling_images.py <path_to_pdf>")
        print("\nExample:")
        print("  python test_docling_images.py test_dataset/sample.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    test_docling_image_extraction(pdf_path)
"""
=====================================================================
SIMPLE TEST: Compare Docling Image Extraction Methods
=====================================================================

PURPOSE:
Quick test to compare what images your current code extracts vs what
the document actually contains.

WHAT THIS DOES:
1. Mimics your exact strategy_5_research implementation
2. Checks both doc.pictures and result.document.pictures
3. Compares the two methods
4. Shows you exactly what's being extracted

USAGE:
    python simple_image_test.py <path_to_pdf>
=====================================================================
"""

from pathlib import Path
import sys

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] {e}")
    sys.exit(1)


def simple_image_test(pdf_path: str):
    """Test both ways to access images in docling."""
    
    print("\n" + "="*70)
    print("SIMPLE IMAGE EXTRACTION TEST")
    print("="*70)
    
    pdf_path = Path(pdf_path)
    print(f"\n📄 PDF: {pdf_path.name}\n")
    
    # Setup (matching your code)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    
    print("🔄 Converting PDF with Docling...")
    converter = DocumentConverter(pipeline_options=pipeline_options)
    result = converter.convert(str(pdf_path))
    
    print("✓ Conversion complete\n")
    
    # Get the document
    doc = result.document
    
    # ========================================================================
    # METHOD 1: Using result.document.pictures (Strategy 1 approach)
    # ========================================================================
    print("-"*70)
    print("METHOD 1: result.document.pictures")
    print("-"*70)
    
    if hasattr(result.document, 'pictures'):
        pics_1 = list(result.document.pictures)
        print(f"✓ Found attribute 'pictures'")
        print(f"  Images found: {len(pics_1)}")
        
        for idx, pic in enumerate(pics_1):
            print(f"\n  Picture {idx}:")
            print(f"    Has .image: {hasattr(pic, 'image')}")
            print(f"    Has .caption: {hasattr(pic, 'caption')}")
            print(f"    Has .page: {hasattr(pic, 'page')}")
            
            if hasattr(pic, 'image'):
                print(f"    Image type: {type(pic.image)}")
                if hasattr(pic.image, 'size'):
                    print(f"    Image size: {pic.image.size}")
    else:
        print("❌ No 'pictures' attribute found")
        pics_1 = []
    
    # ========================================================================
    # METHOD 2: Using doc.pictures (Strategy 5 approach)
    # ========================================================================
    print("\n" + "-"*70)
    print("METHOD 2: doc.pictures")
    print("-"*70)
    
    if hasattr(doc, 'pictures'):
        pics_2 = list(doc.pictures)
        print(f"✓ Found attribute 'pictures'")
        print(f"  Images found: {len(pics_2)}")
        
        for idx, pic in enumerate(pics_2):
            print(f"\n  Picture {idx}:")
            print(f"    Has .image: {hasattr(pic, 'image')}")
            print(f"    Has .caption: {hasattr(pic, 'caption')}")
            print(f"    Has .page: {hasattr(pic, 'page')}")
    else:
        print("❌ No 'pictures' attribute found")
        pics_2 = []
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*70)
    print("COMPARISON & DIAGNOSIS")
    print("="*70)
    
    print(f"\nMethod 1 (result.document.pictures): {len(pics_1)} images")
    print(f"Method 2 (doc.pictures):              {len(pics_2)} images")
    
    if len(pics_1) == len(pics_2) == 0:
        print("\n⚠️  NO IMAGES FOUND BY EITHER METHOD")
        print("\nPossible reasons:")
        print("  1. PDF genuinely has no extractable images")
        print("  2. Images are in a format Docling can't extract")
        print("  3. PDF is scanned (image-based PDF)")
        print("  4. Need to check Docling version/installation")
        print("\nTry:")
        print("  • Testing with a different PDF that definitely has images")
        print("  • Checking: pip list | grep docling")
        print("  • Running: pip install --upgrade docling")
        
    elif len(pics_1) > 0 or len(pics_2) > 0:
        print("\n✅ SUCCESS: Images are being extracted!")
        
        if len(pics_1) != len(pics_2):
            print(f"\n⚠️  Methods returned different counts!")
            print(f"    This is unexpected - both should access the same data")
        else:
            print(f"\n✓ Both methods returned same count")
        
        # Try to save one
        if len(pics_1) > 0 and hasattr(pics_1[0], 'image'):
            output_dir = Path("test_outputs")
            output_dir.mkdir(exist_ok=True)
            test_path = output_dir / f"extracted_test_image.png"
            
            try:
                pics_1[0].image.save(test_path)
                print(f"\n✓ Successfully saved test image to: {test_path}")
            except Exception as e:
                print(f"\n❌ Failed to save image: {e}")
    
    print("\n" + "="*70 + "\n")
    
    # Return for programmatic use
    return {
        'method_1_count': len(pics_1),
        'method_2_count': len(pics_2),
        'working': len(pics_1) > 0 or len(pics_2) > 0
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_image_test.py <path_to_pdf>")
        print("\nExample:")
        print("  python simple_image_test.py test_dataset/sample.pdf")
        sys.exit(1)
    
    result = simple_image_test(sys.argv[1])
    
    # Exit with status
    sys.exit(0 if result['working'] else 1)
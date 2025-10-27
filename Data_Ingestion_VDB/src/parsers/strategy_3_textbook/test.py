"""
Strategy 3 - test.py
python3 -m src.parsers.strategy_3_textbook.test | tee src/parsers/strategy_3_textbook/output.txt
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path so we can import from parsers
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from docling.document_converter import DocumentConverter
from src.parsers.strategy_3_textbook.post_processor import clean_and_chunk


def print_separator():
    """Print visual separator line."""
    print("=" * 80)


def print_header(text: str):
    """Print section header."""
    print_separator()
    print(text.center(80))
    print_separator()


def print_subheader(text: str):
    """Print subsection header."""
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print('─' * 80)


def display_chunk(chunk: Dict, index: int):
    """
    Display a single chunk with formatting.
    
    Shows:
    - Chunk number and ID
    - Type (text/image/table)
    - Content preview (first 200 chars)
    - Metadata
    """
    print(f"\n[Chunk {index}] ID: {chunk['chunk_id']}")
    print(f"Type: {chunk['type'].upper()}")
    
    if chunk['type'] == 'text':
        content = chunk.get('content', '')
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"Content: {preview}")
        print(f"Length: {len(content)} chars, {len(content.split())} words")
        
        # Show references if present
        metadata = chunk.get('metadata', {})
        if metadata.get('figure_refs'):
            print(f"Figure refs: {metadata['figure_refs']}")
        if metadata.get('table_refs'):
            print(f"Table refs: {metadata['table_refs']}")
    
    elif chunk['type'] == 'image':
        print(f"Image path: {chunk.get('image_path')}")
        if chunk.get('caption'):
            print(f"Caption: {chunk['caption']}")
    
    elif chunk['type'] == 'table':
        print(f"Table content: {chunk.get('content', '')[:100]}...")
        if chunk.get('caption'):
            print(f"Caption: {chunk['caption']}")
    
    # Show metadata
    metadata = chunk.get('metadata', {})
    print(f"Metadata: Page {metadata.get('page', 'N/A')}")


def display_statistics(chunks: List[Dict], pdf_name: str):
    """
    Display statistics about parsed chunks.
    """
    print_subheader(f"STATISTICS FOR {pdf_name}")
    
    total = len(chunks)
    text_chunks = [c for c in chunks if c['type'] == 'text']
    image_chunks = [c for c in chunks if c['type'] == 'image']
    table_chunks = [c for c in chunks if c['type'] == 'table']
    
    print(f"  Total chunks: {total}")
    print(f"    - Text: {len(text_chunks)}")
    print(f"    - Images: {len(image_chunks)}")
    print(f"    - Tables: {len(table_chunks)}")
    
    if text_chunks:
        avg_length = sum(len(c['content']) for c in text_chunks) / len(text_chunks)
        print(f"  Average text chunk length: {avg_length:.0f} characters")


def organize_chunks_by_page(chunks: List[Dict]) -> Dict[int, List[Dict]]:
    """Group chunks by page number."""
    by_page = {}
    for chunk in chunks:
        page = chunk.get('metadata', {}).get('page', 0)
        if page not in by_page:
            by_page[page] = []
        by_page[page].append(chunk)
    return by_page


def test_single_pdf(pdf_path: Path, max_pages: int = None):
    """
    Test parsing on a single PDF.
    
    Process:
    1. Load PDF with docling
    2. Extract markdown
    3. Clean and chunk with post_processor
    4. Display results
    """
    print_header("CLINICAL IMAGE PARSER TEST")
    print(f"\nProcessing: {pdf_path.name}")
    if max_pages:
        print(f"Page limit: {max_pages}")
    
    # Step 1: Convert PDF
    print("\nConverting PDF with docling...")
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    
    # Step 2: Get markdown
    md_content = result.document.export_to_markdown()
    print(f"Extracted {len(md_content)} characters (raw)")
    
    # Step 3: Clean and chunk
    print("\nCleaning and chunking with post_processor...")
    chunks = clean_and_chunk(md_content, pdf_path.stem, page_num=1)
    
    if not chunks:
        print("\n[ERROR] No chunks created!")
        print("\nPossible reasons:")
        print("  - PDF is empty or corrupted")
        print("  - Parser failed to extract content")
        print("  - PDF format not supported")
        return
    
    print(f"\n✓ Created {len(chunks)} clean chunks")
    
    # Show statistics
    display_statistics(chunks, pdf_path.name)
    
    # Display each chunk
    print_subheader("CHUNKS")
    for i, chunk in enumerate(chunks, 1):
        display_chunk(chunk, i)
    
    # Final summary
    print()
    print_separator()
    print("PARSING COMPLETE".center(80))
    print_separator()
    print()
    print("Next steps:")
    print("  1. Review extracted content above")
    print("  2. Check if chunking looks appropriate")
    print("  3. Verify cleaning rules removed unwanted text")
    print("  4. Update cleaning_rules.yaml if needed")
    print("  5. Run this test again to verify improvements")
    print()


def select_pdf() -> Path:
    """
    Prompt user to select a PDF file from data/strategy_3_textbook/.
    
    Returns:
        Path to selected PDF
    """
    data_dir = Path("data/strategy_3_textbook")
    
    if not data_dir.exists():
        print(f"\n[ERROR] Directory not found: {data_dir}")
        print("Please create it and add clinical PDFs.")
        sys.exit(1)
    
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n[ERROR] No PDFs found in {data_dir}")
        print("Please add at least one clinical document.")
        sys.exit(1)
    
    print(f"\nAvailable PDFs:")
    for i, pdf in enumerate(pdf_files, 1):
        size_mb = pdf.stat().st_size / 1024 / 1024
        print(f"  [{i}] {pdf.name} ({size_mb:.1f} MB)")
    
    while True:
        choice = input("\nSelect PDF number (or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            print("Exiting.")
            sys.exit(0)
        
        try:
            index = int(choice) - 1
            if 0 <= index < len(pdf_files):
                return pdf_files[index]
            else:
                print(f"Please enter 1-{len(pdf_files)}")
        except ValueError:
            print("Please enter a valid number")


def get_page_limit() -> int:
    """
    Prompt user for page limit.
    
    Returns:
        Page limit (or None for all pages)
    """
    choice = input("\nHow many pages to process? (number or 'all'): ").strip().lower()
    
    if choice == 'all':
        return None
    
    try:
        return int(choice)
    except ValueError:
        print("[INFO] Invalid input, processing all pages")
        return None


def main():
    """Main entry point for the test tool."""
    print_header("STRATEGY 1: CLINICAL IMAGE PARSER TEST TOOL")
    
    # Select PDF
    pdf_path = select_pdf()
    
    # Get page limit
    max_pages = get_page_limit()
    
    # Run test
    print("\n")
    test_single_pdf(pdf_path, max_pages)


if __name__ == "__main__":
    main()
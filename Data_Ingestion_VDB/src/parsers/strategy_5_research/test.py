"""
Strategy 5 - test.py
python3 -m src.parsers.strategy_5_research.test | tee src/parsers/strategy_5_research/output.txt
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from tqdm import tqdm

# Add src to path so we can import from parsers
current_dir = Path(__file__).parent      # Path(__file__).parent = /full/path/to/Data_Ingestion_VDB/src/parsers/
project_root = current_dir.parent.parent # current_dir.parent = /full/path/to/Data_Ingestion_VDB/src/
                                         # current_dir.parent.parent = /full/path/to/Data_Ingestion_VDB/
sys.path.insert(0, str(project_root))

from docling.document_converter import DocumentConverter
from .post_processor import clean_and_chunk

if __name__ == "__main__":
    # Get PDF
    pdf_dir = Path("data/strategy_5_research")
    pdfs = list(pdf_dir.glob("*.pdf"))
    
    if not pdfs:
        print("No PDFs found!")
        exit()
    
    # QUESTIONS
    # QUESTION 1: Pick the pdf
    print("\nAvailable PDFs:")
    for i, pdf in enumerate(pdfs, 1):
        print(f"  [{i}] {pdf.name}")
    
    choice = int(input("\nSelect PDF: ")) - 1
    pdf_path = pdfs[choice]
    
    page_choice = input("\nProcess all pages? (y/n): ").strip().lower()
    
    # QUESTION 2: how many pages?
    if page_choice == 'n':
        max_pages = int(input("How many pages?: "))
    else:
        max_pages = None
    
    # QUESTION 3: print chunks or full markdown? 
    output_choice = input("\nView chunked or full markdown? (c/f): ").strip().lower()
    print(f"\nLoading {pdf_path.name}...")
    

    converter = DocumentConverter()
    
    if max_pages:
        result = converter.convert(str(pdf_path), max_num_pages=max_pages)
    else:
        result = converter.convert(str(pdf_path))

    markdown = result.document.export_to_markdown()
    
    print(f"Extracted {len(markdown):,} characters (raw)\n")
    
    # === USE CLEAN_AND_CHUNK ===
    print("Cleaning and chunking with post_processor...\n")
    
    chunks = clean_and_chunk(
        text=markdown, 
        pdf_name=pdf_path.stem, 
        page_num=1
    )
    
    print(f"✓ Created {len(chunks)} clean chunks\n")
    
    # Show how much was removed
    total_cleaned_chars = sum(len(c['content']) for c in chunks)
    removed = len(markdown) - total_cleaned_chars
    pct = (removed / len(markdown) * 100) if len(markdown) > 0 else 0
    print(f"✓ Removed {removed:,} characters of noise ({pct:.1f}%)\n")
    
    # === SHOW OUTPUT ===
    if output_choice == 'f':
        # Show full cleaned text (combine all chunks)
        print(f"\n{'='*80}")
        print("FULL TEXT (CLEANED)")
        print(f"{'='*80}\n")
        
        for chunk in chunks:
            print(chunk['content'])
            print()
    else:
        # Show individual chunks
        print(f"\n{'='*80}")
        print(f"CHUNKS (CLEANED) - Total: {len(chunks)}")
        print(f"{'='*80}\n")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"--- CHUNK {i} ---\n")
            
            # Show content
            for line in chunk['content'].split('\n'):
                print(f"\t{line}")
            
            # Show metadata
            meta = chunk['metadata']
            print(f"\n\t[Metadata]")
            print(f"\t  chunk_id: {chunk['chunk_id']}")
            print(f"\t  chars: {meta['char_count']}, words: {meta['word_count']}")
            if meta.get('figure_refs'):
                print(f"\t  figures: {meta['figure_refs']}")
            if meta.get('table_refs'):
                print(f"\t  tables: {meta['table_refs']}")
            print()

#     """Print a visual separator line."""
#     print(char * length)


# def print_header(text: str, char="="):
#     """Print a formatted section header."""
#     print_separator(char=char)
#     print(text.center(80))
#     print_separator(char=char)

# def print_subheader(text: str):
#     """Print a subheader with plus signs."""
#     print(f"\n{'_' * 80}\n{' ' *80}\n{text}\n{' ' *80}\n{'_' * 80}")

# def print_chunk_separator():
#     """Print separator between chunks."""
#     print(f"\n {'_' * 62}CHUNK{'_' * 62}\n")

# def display_text_chunk(chunk: Dict[str, Any], index: int):
#     """
#     Display a text chunk with content preview.
    
#     Args:
#         chunk: Text chunk dictionary
#         index: Display index number
#     """
#     content = chunk.get('content', '')
#     metadata = chunk.get('metadata', {})
    
#     print_chunk_separator()
#     print(f"          CHUNK {index}: TEXT")
#     print(f"          Length: {len(content)} chars, {len(content.split())} words")
    
#     # Show references found
#     refs = []
#     if metadata.get('figure_refs'):
#         refs.append(f"Figure refs: {metadata['figure_refs']}")
#     if metadata.get('table_refs'):
#         refs.append(f"Table refs: {metadata['table_refs']}")
#     if refs:
#         print(f"          {', '.join(refs)}")
    
#     print()
#     print("          CONTENT:")
#     print()
    
#     # Show beginning (first 400 chars)
#     start_text = content[:400]
#     print(f"          {start_text}...")
    
#     # Show ending if content is long
#     if len(content) > 500:
#         print()
#         print("          [... middle content omitted ...]")
#         print()
#         end_text = content[-200:]
#         print(f"          ...{end_text}")
    
#     print()

# def display_image_chunk(chunk: Dict[str, Any], index: int):
#     """
#     Display an image chunk with metadata.
    
#     Args:
#         chunk: Image chunk dictionary
#         index: Display index number
#     """
#     metadata = chunk.get('metadata', {})
#     caption = chunk.get('caption', '')
    
#     print_chunk_separator()
#     print(f"          CHUNK {index}: IMAGE")
#     print(f"          Path {chunk.get('image_path', 'N/A')}")
#     print(f"          Size: {metadata.get('width', '?')}x{metadata.get('height', '?')} pixels")
    
    
#     if metadata.get('figure_id'):
#         print(f"          Figure ID: {metadata['figure_id']}")

#     print()

    
#     if caption:
#         print(f"          CAPTION:")
#         print()
#         import textwrap
#         wrapped = textwrap.fill(caption[:300], width=70, initial_indent='          ', subsequent_indent='          ')
#         print(wrapped)
#     else:
#         print(f"          CAPTION: [None detected]")

#     print()

# def display_table_chunk(chunk: Dict[str, Any], index: int):
#     """
#     Display a table chunk with clean formatting.
    
#     Args:
#         chunk: Table chunk dictionary
#         index: Display index number
#     """
#     content = chunk.get('content', '')
#     caption = chunk.get('caption', '')
#     metadata = chunk.get('metadata', {})
    
#     print_chunk_separator()
#     print(f"          CHUNK {index}: TABLE")
    
#     if metadata.get('table_id'):
#         print(f"          Table ID: {metadata['table_id']}")
    
#     print()
    
#     if caption:
#         print("          CAPTION:")
#         print()
#         import textwrap
#         wrapped = textwrap.fill(caption, width=70, initial_indent='          ', subsequent_indent='          ')
#         print(wrapped)
#         print()
    
#     print("          CONTENT:")
#     print()
#     print(f"          {content[:400]}...")
#     print()

# def display_chunk(chunk: Dict[str, Any], index: int):
#     """
#     Display chunk based on its type.
    
#     Args:
#         chunk: Chunk dictionary
#         index: Display index number
#     """
#     chunk_type = chunk.get('type', 'unknown')
    
#     if chunk_type == 'text':
#         display_text_chunk(chunk, index)
#     elif chunk_type == 'image':
#         display_image_chunk(chunk, index)
#     elif chunk_type == 'table':
#         display_table_chunk(chunk, index)
#     else:
#         print(f"\n          CHUNK {index}: UNKNOWN TYPE ({chunk_type})")

# def organize_chunks_by_page(chunks: List[Dict]) -> Dict[int, List[Dict]]:
#     """
#     Organize chunks by page number.
    
#     Args:
#         chunks: List of all chunks
        
#     Returns:
#         Dictionary mapping page_num -> list of chunks
#     """
#     by_page = {}
#     for chunk in chunks:
#         page_num = chunk['metadata']['page']
#         if page_num not in by_page:
#             by_page[page_num] = []
#         by_page[page_num].append(chunk)
#     return by_page

# def display_statistics(chunks: List[Dict], pdf_name: str):
#     """
#     Display parsing statistics.
    
#     Args:
#         chunks: List of all chunks
#         pdf_name: Name of PDF file
#     """
#     total_text = sum(1 for c in chunks if c['type'] == 'text')
#     total_images = sum(1 for c in chunks if c['type'] == 'image')
#     total_tables = sum(1 for c in chunks if c['type'] == 'table')
    
#     print_header(f"DOCUMENT: {pdf_name}")
    
#     print()
#     print(f"          Total chunks extracted: {len(chunks)}")
#     print(f"            - Text chunks: {total_text}")
#     print(f"            - Images: {total_images}")
#     print(f"            - Tables: {total_tables}")
#     print()
    
#     # Character count for text
#     if total_text > 0:
#         total_chars = sum(len(c['content']) for c in chunks if c['type'] == 'text')
#         avg_chars = total_chars / total_text
#         print(f"          Text statistics:")
#         print(f"            - Total characters: {total_chars:,}")
#         print(f"            - Average chunk size: {avg_chars:.0f} characters")
#         print()

# def test_single_pdf(pdf_path: Path, max_pages: int = None):
#     """Test parsing on a single PDF and display results."""
    
#     import time
    
#     # Step 1: Load config
#     print("\n[Setup] Loading configuration...")
#     config = load_config()
    
#     # Step 2: Initialize parser
#     print("[Setup] Initializing parser...")
#     output_dir = Path("extracted_content")
#     output_dir.mkdir(exist_ok=True)
    
#     try:
#         parser = ResearchParser(config, output_dir)
#     except Exception as e:
#         print(f"\n[ERROR] Failed to initialize parser: {e}")
#         import traceback
#         traceback.print_exc()
#         return
    
#     # Step 3: Parse PDF
#     print(f"[Setup] Parsing PDF...")
#     if max_pages:
#         print(f"        Processing first {max_pages} pages only")
#     else:
#         print(f"        Processing all pages")
    
#     parse_start = time.time()
    
#     try:
#         chunks = parser.parse_pdf(pdf_path, max_pages=max_pages)
#     except Exception as e:
#         print(f"\n[ERROR] Parsing failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return
    
#     parse_duration = time.time() - parse_start
    
#     # Step 4: Display results
#     print(f"[Setup] Parsing complete in {parse_duration:.2f} seconds")
#     print()
    
#     if not chunks:
#         print("\n[WARNING] No chunks extracted!")
#         print("\nPossible reasons:")
#         print("  - PDF is empty or corrupted")
#         print("  - Parser failed to extract content")
#         print("  - PDF format not supported")
#         return
    
#     # Show statistics
#     display_statistics(chunks, pdf_path.name)
    
#     # Organize by page
#     chunks_by_page = organize_chunks_by_page(chunks)
    
#     # Display each page
#     for page_num in sorted(chunks_by_page.keys()):
#         print_subheader(f"PAGE {page_num}")
        
#         page_chunks = chunks_by_page[page_num]
        
#         # Count types
#         text_count = sum(1 for c in page_chunks if c['type'] == 'text')
#         image_count = sum(1 for c in page_chunks if c['type'] == 'image')
#         table_count = sum(1 for c in page_chunks if c['type'] == 'table')
        
#         print(f"          {text_count} text, {image_count} images, {table_count} tables")
        
#         # Display each chunk
#         for i, chunk in enumerate(page_chunks, 1):
#             display_chunk(chunk, i)
    
#     # Final summary
#     print()
#     print_separator()
#     print("PARSING COMPLETE".center(80))
#     print_separator()
#     print()
#     print("Next steps:")
#     print("  1. Review extracted content above")
#     print("  2. Check if chunking looks appropriate")
#     print("  3. Verify captions are detected correctly")
#     print("  4. Identify any missing content")
#     print("  5. Update strategy_5_research.py if needed")
#     print("  6. Run this test again to verify improvements")
#     print()

# def select_pdf() -> Path:
#     """
#     Prompt user to select a PDF file from data/strategy_5_research/.
    
#     Returns:
#         Path to selected PDF
#     """
#     data_dir = Path("data/strategy_5_research")
    
#     if not data_dir.exists():
#         print(f"\n[ERROR] Directory not found: {data_dir}")
#         print("Please create it and add research PDFs.")
#         sys.exit(1)
    
#     pdf_files = list(data_dir.glob("*.pdf"))
    
#     if not pdf_files:
#         print(f"\n[ERROR] No PDFs found in {data_dir}")
#         print("Please add at least one research paper.")
#         sys.exit(1)
    
#     print(f"\nAvailable PDFs:")
#     for i, pdf in enumerate(pdf_files, 1):
#         size_mb = pdf.stat().st_size / 1024 / 1024
#         print(f"  [{i}] {pdf.name} ({size_mb:.1f} MB)")
    
#     while True:
#         choice = input("\nSelect PDF number (or 'q' to quit): ").strip()
        
#         if choice.lower() == 'q':
#             print("Exiting.")
#             sys.exit(0)
        
#         try:
#             index = int(choice) - 1
#             if 0 <= index < len(pdf_files):
#                 return pdf_files[index]
#             else:
#                 print(f"Please enter 1-{len(pdf_files)}")
#         except ValueError:
#             print("Please enter a valid number")

# def get_page_limit() -> int:
#     """
#     Prompt user for page limit.
    
#     Returns:
#         Page limit (or None for all pages)
#     """
#     choice = input("\nHow many pages to process? (number or 'all'): ").strip().lower()
    
#     if choice == 'all':
#         return None
    
#     try:
#         return int(choice)
#     except ValueError:
#         print("[INFO] Invalid input, processing all pages")
#         return None

# def main():
#     """Main entry point for the test tool."""
#     print_header("RESEARCH PAPER PARSING TEST TOOL")
    
#     # Select PDF
#     pdf_path = select_pdf()
    
#     # Get page limit
#     max_pages = get_page_limit()
    
#     # Run test
#     print("\n")
#     test_single_pdf(pdf_path, max_pages)


# if __name__ == "__main__":
#     main()
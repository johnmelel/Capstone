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
    


    from ..strategy_5_research.main import ResearchParser

    config = {'chunk_size':1000}
    output_dir = Path("temp/test_output")
    parser = ResearchParser(config, output_dir)

    chunks = parser.parse_pdf(pdf_path, max_pages=max_pages)

    mineru_output = parser.mineru_temp_dir / pdf_path.stem
    md_files = list(mineru_output.glob("**/*.md"))
    if md_files:
        markdown = md_files[0].read_text(encoding='utf-8')
    else:
        markdown = ""



    print(f"Extracted {len(markdown):,} characters (raw)\n")
    
    # === USE CLEAN_AND_CHUNK ===
    print("Cleaning and chunking with post_processor...\n")

    chunks_from_markdown = clean_and_chunk(
        text=markdown, 
        pdf_name=pdf_path.stem, 
        page_num=1
    )

    print(f"✓ Created {len(chunks_from_markdown)} clean chunks\n")
    
    # Show how much was removed
    total_cleaned_chars = sum(len(c['content']) for c in chunks_from_markdown)
    removed = len(markdown) - total_cleaned_chars
    pct = (removed / len(markdown) * 100) if len(markdown) > 0 else 0
    print(f"✓ Removed {removed:,} characters of noise ({pct:.1f}%)\n")
    
    # === SHOW OUTPUT ===
    if output_choice == 'f':
        # Show full cleaned text (combine all chunks)
        print(f"\n{'='*80}")
        print("FULL TEXT (CLEANED)")
        print(f"{'='*80}\n")
        
        print(markdown)
    else:
        # Show individual chunks
        print(f"\n{'='*80}")
        print(f"CHUNKS (CLEANED) - Total: {len(chunks_from_markdown)}")
        print(f"{'='*80}\n")
        
        for i, chunk in enumerate(chunks_from_markdown, 1):
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

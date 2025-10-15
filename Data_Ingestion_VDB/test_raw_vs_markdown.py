# test_raw_vs_markdown.py - UPDATED to limit to 10 pages
"""A/B test: Raw text vs Markdown for RAG quality - First 10 pages only."""

import fitz
import pymupdf4llm
from pathlib import Path
import json

def extract_raw(pdf_path: str, page_num: int) -> str:
    """Extract using current method - raw text."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    text = page.get_text("text", sort=True)
    doc.close()
    return text


def extract_markdown(pdf_path: str, page_num: int) -> str:
    """Extract using pymupdf4llm - markdown."""
    md_dict = pymupdf4llm.to_markdown(pdf_path, pages=[page_num])
    
    if isinstance(md_dict, dict) and 'text' in md_dict:
        return md_dict['text']
    else:
        return md_dict


def chunk_text(text: str, max_size: int = 950) -> list:
    """Simple chunking (same logic for both)."""
    chunks = []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 10]
    
    current = ""
    chunk_id = 0
    
    for para in paragraphs:
        if len(para) > max_size:
            if current.strip():
                chunks.append({'text': current.strip()})
                chunk_id += 1
                current = ""
            
            sentences = [s.strip() + '.' for s in para.split('.') if s.strip()]
            for sent in sentences:
                if len(current) + len(sent) > max_size:
                    if current.strip():
                        chunks.append({'text': current.strip()})
                        chunk_id += 1
                    current = sent + " "
                else:
                    current += sent + " "
        
        elif len(current) + len(para) > max_size:
            if current.strip():
                chunks.append({'text': current.strip()})
                chunk_id += 1
            current = para + "\n\n"
        else:
            current += para + "\n\n"
    
    if current.strip():
        chunks.append({'text': current.strip()})
    
    return chunks


# Process all PDFs in test_dataset folder
test_dataset_path = Path("test_dataset")
pdf_files = list(test_dataset_path.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files\n")

all_raw_chunks = []
all_md_chunks = []
chunk_id_counter = 0

MAX_PAGES = 10  # Only process first 10 pages

for pdf_file in pdf_files:
    print(f"{'='*70}")
    print(f"Processing: {pdf_file.name}")
    print('='*70)
    
    # Open to get page count
    doc = fitz.open(str(pdf_file))
    num_pages = len(doc)
    doc.close()
    
    # Limit to first 10 pages
    pages_to_process = min(num_pages, MAX_PAGES)
    print(f"Total pages: {num_pages} | Processing: {pages_to_process} pages")
    
    # Process each page
    for page_num in range(pages_to_process):
        print(f"  📄 Extracting page {page_num + 1}/{pages_to_process}...", end="")
        
        # RAW extraction
        try:
            raw_text = extract_raw(str(pdf_file), page_num)
            raw_chunks = chunk_text(raw_text)
            
            # Add metadata and update IDs
            for chunk in raw_chunks:
                chunk['id'] = f'raw_{pdf_file.stem}_p{page_num+1}_c{len(all_raw_chunks)}'
                chunk['source'] = str(pdf_file.name)
                chunk['page'] = page_num + 1
            
            all_raw_chunks.extend(raw_chunks)
            print(f" ✅ {len(raw_chunks)} raw chunks", end="")
            
        except Exception as e:
            print(f" ⚠️  RAW error: {e}", end="")
        
        # MARKDOWN extraction
        try:
            md_text = extract_markdown(str(pdf_file), page_num)
            md_chunks = chunk_text(md_text)
            
            # Add metadata and update IDs
            for chunk in md_chunks:
                chunk['id'] = f'md_{pdf_file.stem}_p{page_num+1}_c{len(all_md_chunks)}'
                chunk['source'] = str(pdf_file.name)
                chunk['page'] = page_num + 1
            
            all_md_chunks.extend(md_chunks)
            print(f" | {len(md_chunks)} md chunks")
            
        except Exception as e:
            print(f" | ⚠️  MD error: {e}")
    
    print(f"✅ Completed {pdf_file.name}\n")

# Save combined chunks
with open('raw_chunks.json', 'w') as f:
    json.dump(all_raw_chunks, f, indent=2)

with open('md_chunks.json', 'w') as f:
    json.dump(all_md_chunks, f, indent=2)

print(f"{'='*70}")
print(f"📊 EXTRACTION SUMMARY")
print(f"{'='*70}")
print(f"✅ TOTAL RAW CHUNKS: {len(all_raw_chunks)}")
print(f"✅ TOTAL MARKDOWN CHUNKS: {len(all_md_chunks)}")
print(f"{'='*70}\n")

# Show breakdown by source
print("Breakdown by PDF:")
for pdf_file in pdf_files:
    raw_count = len([c for c in all_raw_chunks if c['source'] == pdf_file.name])
    md_count = len([c for c in all_md_chunks if c['source'] == pdf_file.name])
    print(f"  📄 {pdf_file.name}")
    print(f"     Raw: {raw_count} chunks | Markdown: {md_count} chunks")

print(f"\n{'='*70}")
print("✅ Ready for testing! Run 'python3 rag_tester.py' next")
print(f"{'='*70}")
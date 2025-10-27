"""
Strategy 5 - post_processor.py
"""

import re
import yaml
from pathlib import Path
from typing import List, Dict


def load_rules(rules_file: str = "cleaning_rules.yaml") -> Dict:
    """Load cleaning rules from YAML file."""
    rules_path = Path(__file__).parent / rules_file
    
    if not rules_path.exists():
        print(f"[Warning] No cleaning rules found at {rules_path}")
        return {'remove_patterns': [], 'normalization': {}, 'manual_fixes': [], 
                'chunking': {'max_size': 950, 'overlap': 200}}
    
    with open(rules_path, 'r') as f:
        return yaml.safe_load(f)


def remove_patterns(text: str, patterns: List[Dict]) -> str:
    """Remove unwanted patterns from text."""
    for rule in patterns:
        pattern = rule.get('pattern')
        if not pattern:
            continue
        
        multiline = rule.get('multiline', False)
        flags = re.MULTILINE | re.DOTALL if multiline else 0
        
        text = re.sub(pattern, '', text, flags=flags)
    
    return text


def normalize_text(text: str, normalization: Dict) -> str:
    """
    Fix formatting issues:
    1. Join hyphenated words: "diagnos-\ntic" → "diagnostic"
    2. Clean multiple spaces
    3. Clean excessive newlines
    """
    # 1. Dehyphenate
    if normalization.get('dehyphenate', {}).get('enabled', True):
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # 2. Clean whitespace
    if normalization.get('clean_whitespace', {}).get('enabled', True):
        text = re.sub(r' +', ' ', text)
    
    # 3. Clean newlines (max 2 consecutive)
    if normalization.get('clean_newlines', {}).get('enabled', True):
        text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def apply_manual_fixes(text: str, fixes: List[Dict]) -> str:
    """Apply specific find-replace corrections."""
    for fix in fixes:
        find = fix.get('find')
        replace = fix.get('replace')
        
        if not find or replace is None:
            continue
        
        if fix.get('apply_to') == 'first_occurrence':
            text = text.replace(find, replace, 1)
        else:
            text = text.replace(find, replace)
    
    return text


def create_chunks_with_overlap(text: str, pdf_name: str, page_num: int, 
                                max_size: int = 950, overlap: int = 150):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current = ""
    chunk_idx = 0
    
    for para in paragraphs:
        # Check if adding para would exceed limit
        if len(current) + len(para) > max_size:
            # Save current chunk (if substantial)
            if len(current) >= 50:
                chunks.append(_make_chunk_dict(current.strip(), pdf_name, page_num, chunk_idx))
                chunk_idx += 1
                
                # Get overlap for next chunk
                overlap_text = _get_overlap(current, overlap)
            else:
                overlap_text = ""
            
            # Start new chunk with overlap + new paragraph
            current = overlap_text + para + "\n\n"
            
            # ✅ NEW: Enforce size limit while preserving overlap
            if len(current) > max_size:
                # Option A: Split the paragraph if it's too long
                if len(para) > (max_size - overlap):
                    # Paragraph itself is too big, need to split it
                    para_pieces = _split_long_text(para, max_size - overlap)
                    
                    # Use first piece with overlap
                    current = overlap_text + para_pieces[0] + "\n\n"
                    
                    # Save this chunk
                    chunks.append(_make_chunk_dict(current.strip(), pdf_name, page_num, chunk_idx))
                    chunk_idx += 1
                    
                    # Process remaining pieces
                    for piece in para_pieces[1:]:
                        overlap_text = _get_overlap(current, overlap)
                        current = overlap_text + piece + "\n\n"
                        
                        if len(current) > max_size:
                            current = current[:max_size]
                        
                        chunks.append(_make_chunk_dict(current.strip(), pdf_name, page_num, chunk_idx))
                        chunk_idx += 1
                    
                    current = ""  # Reset for next paragraph
                else:
                    # Truncate while keeping overlap intact
                    current = current[:max_size]
        else:
            # Add paragraph to current chunk
            current += para + "\n\n"
            
            # ✅ NEW: Safety check even when adding normally
            if len(current) > max_size:
                current = current[:max_size]
    
    # Save final chunk
    if len(current) >= 50:
        if len(current) > max_size:
            current = current[:max_size]
        chunks.append(_make_chunk_dict(current.strip(), pdf_name, page_num, chunk_idx))
    
    return chunks


def _split_long_text(text: str, max_size: int) -> List[str]:
    """Split text into pieces that fit within max_size."""
    if len(text) <= max_size:
        return [text]
    
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    pieces = []
    current = ""
    
    for sent in sentences:
        if len(current) + len(sent) > max_size:
            if current:
                pieces.append(current.strip())
            current = sent + " "
        else:
            current += sent + " "
    
    if current:
        pieces.append(current.strip())
    
    return pieces

def _get_overlap(text: str, overlap_size: int) -> str:
    """Get last N characters for overlap, prefer sentence breaks."""
    if len(text) <= overlap_size:
        return text
    
    overlap = text[-overlap_size:]
    
    # Try to start at sentence boundary
    sentence_start = overlap.find('. ')
    if sentence_start != -1 and sentence_start < overlap_size * 0.5:
        overlap = overlap[sentence_start + 2:]
    
    return overlap + "\n\n"


def _make_chunk_dict(text: str, pdf_name: str, page_num: int, chunk_idx: int) -> Dict:
    """Create chunk dictionary with metadata."""
    chunk_id = f"{pdf_name}_p{page_num}_text_{chunk_idx}"
    
    # Find references
    fig_refs = re.findall(r'Fig\.?\s*\d+[A-Z]?', text, re.IGNORECASE)
    table_refs = re.findall(r'Table\s*\d+', text, re.IGNORECASE)
    
    return {
        'chunk_id': chunk_id,
        'type': 'text',
        'content': text,
        'metadata': {
            'pdf_name': pdf_name,
            'page': page_num,
            'chunk_index': chunk_idx,
            'char_count': len(text),
            'word_count': len(text.split()),
            'figure_refs': ', '.join(fig_refs) if fig_refs else None,
            'table_refs': ', '.join(table_refs) if table_refs else None,
        }
    }


def clean_and_chunk(text: str, pdf_name: str, page_num: int, 
                    rules_file: str = "cleaning_rules.yaml") -> List[Dict]:
    """
    Main function: Clean text and create chunks.
    
    Args:
        text: Raw text from docling
        pdf_name: Name of PDF
        page_num: Page number
        rules_file: Path to cleaning rules YAML
    
    Returns:
        List of clean chunk dictionaries
    """
    # Load rules
    rules = load_rules(rules_file)
    
    # Step 1: Remove patterns
    text = remove_patterns(text, rules.get('remove_patterns', []))
    
    # Step 2: Normalize
    text = normalize_text(text, rules.get('normalization', {}))
    
    # Step 3: Manual fixes
    text = apply_manual_fixes(text, rules.get('manual_fixes', []))
    
    # Step 4: Create chunks
    chunking = rules.get('chunking', {})
    chunks = create_chunks_with_overlap(
        text, pdf_name, page_num,
        max_size=chunking.get('max_size', 950),
        overlap=chunking.get('overlap', 200)
    )
    
    return chunks
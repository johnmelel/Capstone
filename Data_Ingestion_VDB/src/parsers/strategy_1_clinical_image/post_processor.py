

import re
import yaml
from pathlib import Path
from typing import List, Dict


def load_rules(rules_file: str = "cleaning_rules.yaml") -> Dict:
    """
    Load cleaning rules from YAML file.
    
    PROCESS:
    1. Locate cleaning_rules.yaml in same directory
    2. Parse YAML into dictionary
    3. Return rules or default config if file not found
    
    Returns:
        Dictionary with cleaning rules
    """
    rules_path = Path(__file__).parent / rules_file
    
    if not rules_path.exists():
        print(f"[Warning] No cleaning rules found at {rules_path}")
        return {
            'remove_patterns': [],
            'normalization': {},
            'manual_fixes': [],
            'chunking': {'max_size': 950, 'overlap': 200}
        }
    
    with open(rules_path, 'r') as f:
        return yaml.safe_load(f)


def remove_patterns(text: str, patterns: List[Dict]) -> str:
    """
    Remove unwanted patterns from text using regex.
    
    PROCESS:
    1. Iterate through each pattern in rules
    2. Apply regex substitution
    3. Remove matched text
    
    Args:
        text: Raw text
        patterns: List of pattern dictionaries from YAML
        
    Returns:
        Text with patterns removed
    """
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
    Fix formatting issues in text.
    
    FIXES APPLIED:
    1. Join hyphenated words: "diagnos-\ntic" → "diagnostic"
    2. Clean multiple spaces → single space
    3. Clean excessive newlines (keep max 2 consecutive)
    
    Args:
        text: Text to normalize
        normalization: Normalization rules from YAML
        
    Returns:
        Normalized text
    """
    # 1. Dehyphenate (fix words split across lines)
    if normalization.get('dehyphenate', {}).get('enabled', True):
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # 2. Clean whitespace (multiple spaces → single space)
    if normalization.get('clean_whitespace', {}).get('enabled', True):
        text = re.sub(r' +', ' ', text)
    
    # 3. Clean newlines (max 2 consecutive)
    if normalization.get('clean_newlines', {}).get('enabled', True):
        text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def apply_manual_fixes(text: str, fixes: List[Dict]) -> str:
    """
    Apply specific find-replace corrections.
    
    Used for:
    - Fixing known formatting errors
    - Correcting column order issues
    - Inserting missing section headers
    
    Args:
        text: Text to fix
        fixes: List of fix dictionaries from YAML
        
    Returns:
        Text with fixes applied
    """
    for fix in fixes:
        find_str = fix.get('find')
        replace_str = fix.get('replace')
        
        if not find_str:
            continue
        
        # Apply first occurrence only if specified
        if fix.get('apply_to') == 'first_occurrence':
            text = text.replace(find_str, replace_str, 1)
        else:
            text = text.replace(find_str, replace_str)
    
    return text


def create_chunks_with_overlap(text: str, pdf_name: str, page_num: int,
                               max_size: int = 950, overlap: int = 200) -> List[Dict]:
    """
    Create semantic chunks with overlap.
    
    CHUNKING STRATEGY:
    1. Split text into sentences
    2. Group sentences into chunks under max_size
    3. Add overlap from previous chunk
    4. Prefer sentence boundaries for splits
    5. Detect figure/table references in each chunk
    
    Args:
        text: Clean text to chunk
        pdf_name: PDF name for IDs
        page_num: Page number
        max_size: Max characters per chunk
        overlap: Number of characters to overlap
        
    Returns:
        List of chunk dictionaries
    """
    if not text or not text.strip():
        return []
    
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    chunk_idx = 0
    overlap_text = ""
    
    for sentence in sentences:
        # Start new chunk if adding sentence exceeds max_size
        if len(current_chunk) + len(sentence) > max_size and current_chunk:
            # Save current chunk
            chunk_text = overlap_text + current_chunk
            chunks.append(_make_chunk_dict(chunk_text, pdf_name, page_num, chunk_idx))
            chunk_idx += 1
            
            # Get overlap for next chunk
            overlap_text = _get_overlap(current_chunk, overlap)
            current_chunk = sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence
    
    # Add final chunk
    if current_chunk:
        chunk_text = overlap_text + current_chunk
        chunks.append(_make_chunk_dict(chunk_text, pdf_name, page_num, chunk_idx))
    
    return chunks


def _get_overlap(text: str, overlap_size: int) -> str:
    """
    Extract overlap text from end of chunk.
    
    STRATEGY:
    - Get last overlap_size characters
    - Try to start at sentence boundary
    - Fallback to character boundary if no sentence found
    
    Args:
        text: Text to extract from
        overlap_size: Number of characters to extract
        
    Returns:
        Overlap text
    """
    if len(text) <= overlap_size:
        return text
    
    overlap = text[-overlap_size:]
    
    # Try to start at sentence boundary
    sentence_start = overlap.find('. ')
    if sentence_start != -1 and sentence_start < overlap_size * 0.5:
        overlap = overlap[sentence_start + 2:]
    
    return overlap + "\n\n"


def _make_chunk_dict(text: str, pdf_name: str, page_num: int, chunk_idx: int) -> Dict:
    """
    Create chunk dictionary with metadata.
    
    METADATA INCLUDES:
    - Chunk ID (unique identifier)
    - PDF name and page number
    - Chunk index
    - Character and word counts
    - Figure/table references found in text
    
    Args:
        text: Chunk text
        pdf_name: PDF name
        page_num: Page number
        chunk_idx: Chunk index
        
    Returns:
        Chunk dictionary
    """
    chunk_id = f"{pdf_name}_p{page_num}_text_{chunk_idx}"
    
    # Find references to figures and tables
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
    
    FULL PROCESS:
    1. Load cleaning rules from YAML
    2. Remove unwanted patterns (headers, page numbers, etc.)
    3. Normalize text (fix line breaks, spacing)
    4. Apply manual fixes for known issues
    5. Create semantic chunks with overlap
    6. Add metadata to each chunk
    7. Return clean chunks ready for embedding
    
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
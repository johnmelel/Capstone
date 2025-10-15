"""Main processing pipeline."""

from pathlib import Path
from typing import List, Dict, Any
from .extractors import PDFExtractor
from .embedders import VertexAIEmbedder
from .vector_store import VectorStore

class Pipeline:
    """Orchestrate the entire embedding pipeline."""
    
    def __init__(self, config: Dict[str, Any], project_id: str, location: str):
        print("\n" + "="*70)
        print(" INITIALIZING PIPELINE")
        print("="*70)
        
        self.config = config
        self.extractor = PDFExtractor(config)
        self.embedder = VertexAIEmbedder(config, project_id, location)
        self.vector_store = VectorStore(config, drop_if_exists=True)
        
        print("\n Pipeline ready!")
    
    def process_pdf(self, pdf_path: Path, max_pages: int = None):
        """Process a single PDF."""
        
        # Step 1: Extract
        items = self.extractor.extract_from_pdf(pdf_path, max_pages)
        
        if not items:
            print("  No items extracted!")
            return
        
        # Step 2: Embed
        embeddings = self.embedder.embed_batch(items)
        
        # Step 3: Upload
        self.vector_store.add_items(items, embeddings)
        
        print(f"\n Successfully processed {pdf_path.name}")
    
    def process_directory(self, pdf_dir: Path, test_mode: bool = False):
        """Process all PDFs in a directory."""
        pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
        
        if not pdf_files:
            print(f" No PDF files found in {pdf_dir}")
            return
        
        if test_mode:
            pdf_files = pdf_files[:3]  # Process 3 PDFs in test mode
            max_pages = self.config['processing']['max_test_pages']
            print(f"\n TEST MODE: Processing first 3 PDFs, first {max_pages} pages each")
        else:
            max_pages = None
            print(f"\n FULL MODE: Processing {len(pdf_files)} PDFs")
        
        for pdf_path in pdf_files:
            self.process_pdf(pdf_path, max_pages)
        
        print("\n" + "="*70)
        print(" ALL DONE!")
        print("="*70)
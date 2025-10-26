"""
Clinical Image Parser (Strategy 1)

PURPOSE:
Parse clinical imaging documents/reports.

TODO: Implement specific parsing logic for clinical imaging documents.
"""

from pathlib import Path
from typing import List, Dict, Any
from .base_parser import BaseParser


class ClinicalImageParser(BaseParser):
    """Parser for clinical imaging documents."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config, output_dir)
        self.strategy_name = "clinical_image"
        print(f"[ClinicalImageParser] Initialized")
    
    def parse_pdf(self, pdf_path: Path, max_pages: int = None) -> List[Dict]:
        """
        Parse clinical imaging document.
        
        TODO: Implement parsing logic.
        """
        print(f"[ClinicalImageParser] Parsing {pdf_path.name}")
        print("[Warning] This parser is not yet implemented!")
        return []
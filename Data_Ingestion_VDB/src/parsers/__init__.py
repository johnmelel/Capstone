"""
Parser package - Contains all document parsing strategies.

Each parser inherits from BaseParser and implements custom logic
for different document types.
"""

from .base_parser import BaseParser
from .strategy_1_clinical_image.main import ClinicalImageParser
from .strategy_2_clinical_qa.main import ClinicalQAParser
from .strategy_3_textbook.main import TextbookParser
from .strategy_4_lexicon.main import LexiconParser
from .strategy_5_research.main import ResearchParser

__all__ = [
    'BaseParser',
    'ClinicalImageParser',
    'ClinicalQAParser',
    'TextbookParser',
    'LexiconParser',
    'ResearchParser',
]
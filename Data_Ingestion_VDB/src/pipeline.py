"""
Main processing pipeline:
1. Initialize: Create extractor, embedder, vector_store
2. Check existing embeddings on Milvus
3. For each PDF:
    a. Detect which parser to use based on PDF folder
    b. Extract content (text, images, tables)
    c. Embed everything (convert to vectors)
    d. Store in Milvus database
"""

from pathlib import Path
from typing import List, Dict, Any
import yaml
from .embedders import VertexAIEmbedder
from .vector_store import VectorStore

# Import all parsers
from .parsers import (
    ClinicalImageParser,
    ClinicalQAParser,
    TextbookParser,
    LexiconParser,
    ResearchParser
)

# ============================================================================
# PRINTING UTILITIES - Clean, consistent formatting
# ============================================================================

def print_header(text: str, width: int = 80):
    """Print a major section header."""
    print("\n" + "=" * width)
    print(f" {text}")
    print("=" * width)

def print_subheader(text: str, width: int = 80):
    """Print a minor section header."""
    print("\n" + "-" * width)
    print(f" {text}")
    print("-" * width)

def print_info(label: str, value: str = "", indent: int = 0):
    """Print an info line with optional indentation."""
    spaces = "  " * indent
    if value:
        print(f"{spaces}[{label}] {value}")
    else:
        print(f"{spaces}[{label}]")

def print_success(message: str, indent: int = 0):
    """Print a success message."""
    spaces = "  " * indent
    print(f"{spaces}✓ {message}")

def print_warning(message: str, indent: int = 0):
    """Print a warning message."""
    spaces = "  " * indent
    print(f"{spaces}⚠ {message}")

def print_error(message: str, indent: int = 0):
    """Print an error message."""
    spaces = "  " * indent
    print(f"{spaces}✗ {message}")

def print_progress(current: int, total: int, item_name: str = "item"):
    """Print progress counter."""
    print(f"\n┌─ Processing {item_name} {current}/{total} ─┐")

def print_step(step_num: int, total_steps: int, description: str):
    """Print a processing step."""
    print(f"\n[Step {step_num}/{total_steps}] {description}")

# ============================================================================
# PIPELINE CLASS
# ============================================================================

class Pipeline:

    def __init__(self, config: Dict[str, Any], project_id: str, location: str):
        print_header("INITIALIZING PIPELINE")
        
        self.config = config
        self.project_id = project_id
        self.location = location

        # Load parser mapping (folder name -> parser class)
        self.parser_mapping = self._load_parser_mapping()

        # Initialize embedder and vector store (same for all parsers)
        self.embedder = VertexAIEmbedder(config, project_id, location)
        self.vector_store = VectorStore(config, drop_if_exists=False)

        # Store parser instances (created on-demand)
        self.parser_instances = {}
        self.existing_chunks = set()
        
        print_success(f"Pipeline ready with {len(self.parser_mapping)} parser strategies", indent=1)

    def check_existing_data(self) -> dict:
        """Check what data already exists in vector store."""
        count = self.vector_store.count_entities()
        
        if count > 0:
            print_info("Existing Data", f"Found {count} embeddings in database", indent=1)
            self.existing_chunks = self.vector_store.get_existing_chunk_ids()
            print_info("Loaded", f"{len(self.existing_chunks)} chunk IDs for deduplication", indent=1)
        
        return {
            'count': count,
            'has_data': count > 0
        }

    def _load_parser_mapping(self) -> Dict[str, type]:
        """
        Load parser mapping from config/parser_mapping.yaml
        
        Maps folder names to parser classes:
        'strategy_5_research' -> ResearchParser
        """
        mapping_file = Path("config/parser_mapping.yaml")
        
        if not mapping_file.exists():
            print_warning("No parser_mapping.yaml found, using defaults", indent=1)
            return {
                'strategy_1_clinical_image': ClinicalImageParser,
                'strategy_2_clinical_qa': ClinicalQAParser,
                'strategy_3_textbook': TextbookParser,
                'strategy_4_lexicon': LexiconParser,
                'strategy_5_research': ResearchParser,
            }
        
        with open(mapping_file, 'r') as f:
            yaml_mapping = yaml.safe_load(f)
        
        # Convert string names to actual classes
        class_map = {
            'ClinicalImageParser': ClinicalImageParser,
            'ClinicalQAParser': ClinicalQAParser,
            'TextbookParser': TextbookParser,
            'LexiconParser': LexiconParser,
            'ResearchParser': ResearchParser,
        }
        
        mapping = {}
        for folder_name, class_name in yaml_mapping.items():
            if class_name in class_map:
                mapping[folder_name] = class_map[class_name]
            else:
                print_warning(f"Unknown parser class: {class_name}", indent=1)
        
        return mapping

    def _get_parser_for_pdf(self, pdf_path: Path):
        """
        Determine which parser to use based on PDF location.
            
        Example:
        - PDF in data/strategy_5_research/ -> Use ResearchParser
        - PDF in data/strategy_1_clinical_image/ -> Use ClinicalImageParser
            
        Args:
             pdf_path: Path to PDF file
                
        Returns:
            Parser instance
        """
        # Get parent folder name
        parent_folder = pdf_path.parent.name
            
        print_info("Strategy", parent_folder, indent=1)
            
        # Find matching parser
        if parent_folder in self.parser_mapping:
            parser_class = self.parser_mapping[parent_folder]
                
            # Create parser instance if not already created
            if parent_folder not in self.parser_instances:
                output_dir = Path("extracted_content")
                self.parser_instances[parent_folder] = parser_class(
                    self.config, 
                    output_dir
                )
                
            parser = self.parser_instances[parent_folder]
            print_info("Parser", parser.__class__.__name__, indent=1)
            return parser
        else:
            print_error(f"No parser found for folder: {parent_folder}", indent=1)
            print_error(f"Available: {', '.join(self.parser_mapping.keys())}", indent=1)
            raise ValueError(f"No parser configured for folder: {parent_folder}")

    def process_pdf(self, pdf_path: Path, max_pages: int = None):
        """
        Process a single PDF.

        Steps:
        1. Detect which parser to use (based on folder)
        2. Parse the PDF (extract chunks)
        3. Embed all chunks (text, images, tables)
        4. Store in Milvus vector database
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Optional limit on pages (for testing)
        """
        
        print_subheader(f"📄 {pdf_path.name}")
        
        # Step 1: Get the right parser
        parser = self._get_parser_for_pdf(pdf_path)
        
        # Step 2: Parse PDF (extract chunks)
        print_step(1, 3, "Parsing PDF")
        chunks = parser.parse_pdf(pdf_path, max_pages)

        if not chunks:
            print_warning("No chunks extracted from PDF!", indent=1)
            return

        # Filter out already-processed chunks
        if self.existing_chunks:
            original_count = len(chunks)
            chunks = [c for c in chunks if c['chunk_id'] not in self.existing_chunks]
            skipped = original_count - len(chunks)
            
            if skipped > 0:
                print_info("Skipped", f"{skipped} already-processed chunks", indent=1)
            
            if not chunks:
                print_info("Status", "All chunks already in database - skipping", indent=1)
                return
            
            print_info("New chunks", f"{len(chunks)} to process", indent=1)
        else:
            print_info("Extracted", f"{len(chunks)} chunks", indent=1)
        
        # Step 3: Embed all chunks
        print_step(2, 3, "Embedding chunks")
        embeddings = self.embedder.embed_batch(chunks)
        print_success(f"Created {len(embeddings)} embeddings", indent=1)
        
        # Step 4: Store in Milvus
        print_step(3, 3, "Storing in vector database")
        self.vector_store.add_items(chunks, embeddings)
        print_success(f"Stored successfully", indent=1)
    
    def process_directory(self, data_dir: Path, test_mode: bool = False):
        """
        Process all PDFs in data directory.
        
        Automatically searches all strategy subfolders:
        - data/strategy_1_clinical_image/*.pdf
        - data/strategy_2_clinical_text/*.pdf
        - data/strategy_5_research/*.pdf
        - etc.
        
        Args:
            data_dir: Path to data directory (e.g., "data/")
            test_mode: If True, process only first PDF from each folder, 5 pages max
        """
        print_header("SCANNING DATA DIRECTORY")
        print_info("Directory", str(data_dir), indent=1)
        
        # Find all PDFs in strategy subfolders
        all_pdfs = []
        
        # Look in each strategy subfolder
        for strategy_folder in self.parser_mapping.keys():
            folder_path = data_dir / strategy_folder
            
            if not folder_path.exists():
                print_warning(f"Folder not found: {strategy_folder}", indent=1)
                continue
            
            pdfs = list(folder_path.glob("*.pdf"))
            if pdfs:
                print_success(f"{len(pdfs)} PDFs in {strategy_folder}/", indent=1)
                all_pdfs.extend(pdfs)
        
        if not all_pdfs:
            print_error(f"No PDF files found in {data_dir}", indent=1)
            print_info("Expected folders", "", indent=1)
            for folder in self.parser_mapping.keys():
                print(f"      - {data_dir}/{folder}/")
            return
        
        # Test mode: limit PDFs and pages
        if test_mode:
            all_pdfs = all_pdfs[:1]
            max_pages = self.config['processing']['max_test_pages']
            print_header(f"TEST MODE: 1 PDF, {max_pages} pages max")
        else:
            max_pages = None
            print_header(f"FULL MODE: Processing {len(all_pdfs)} PDFs")
        
        # Process each PDF
        for i, pdf_path in enumerate(all_pdfs, 1):
            print_progress(i, len(all_pdfs), "PDF")
            try:
                self.process_pdf(pdf_path, max_pages)
            except Exception as e:
                print_error(f"Failed to process {pdf_path.name}: {e}", indent=1)
                print_warning("Continuing with next PDF...", indent=1)
                import traceback
                traceback.print_exc()
        
        # Final summary
        print_header("PIPELINE COMPLETE")
        print_success(f"Processed {len(all_pdfs)} PDFs", indent=1)
        print_info("Total in database", f"{self.vector_store.collection.num_entities} chunks", indent=1)



# """
# Main processing pipeline:
# 1. Initialize: Create extractor, embedder, vector_store
# 2. Check existing embeddings on Milvus
# 3. For each PDF:
#     a. Detect which parser to use based on PDF folder
#     b. Extract content (text, images, tables)
#     c. Embed everything (convert to vectors)
#     d. Store in Milvus database
# """

# from pathlib import Path
# from typing import List, Dict, Any
# import yaml
# from .embedders import VertexAIEmbedder
# from .vector_store import VectorStore

# # Import all parsers
# from .parsers import (
#     ClinicalImageParser,
#     ClinicalQAParser,
#     TextbookParser,
#     LexiconParser,
#     ResearchParser
# )

# class Pipeline:

#     def __init__(self, config: Dict[str, Any], project_id: str, location: str):
#         print("\n" + "="*70)
#         print(" INITIALIZING PIPELINE")
#         print("="*70)
        
#         self.config = config
#         self.project_id = project_id
#         self.location = location

#         # Load parser mapping (folder name -> parser class)
#         self.parser_mapping = self._load_parser_mapping()

#         # Initialize embedder and vector store (same for all parsers)
#         self.embedder = VertexAIEmbedder(config, project_id, location)
#         self.vector_store = VectorStore(config, drop_if_exists=False)

#         # Store parser instances (created on-demand)
#         self.parser_instances = {}

#         self.existing_chunks = set()
        
#         print("\n[Pipeline] ready!")
#         print(f"[Pipeline] Loaded {len(self.parser_mapping)} parser strategies")

#     def check_existing_data(self) -> dict:
#         """Check what data already exists in vector store."""
#         count = self.vector_store.count_entities() # How many in total?
        
#         if count > 0:
#             # Get all chunk_ids from Milvus database
#             print(f"\n[Info] Found {count} existing embeddings in database")
#             self.existing_chunks = self.vector_store.get_existing_chunk_ids()
#             print(f"[Info] Loaded {len(self.existing_chunks)} existing chunk IDs")
        
#         return {
#             'count': count,
#             'has_data': count > 0
#         }

#     def _load_parser_mapping(self) -> Dict[str, type]:
#         """
#         Load parser mapping from config/parser_mapping.yaml
        
#         Maps folder names to parser classes:
#         'strategy_5_research' -> ResearchParser
#         """
#         mapping_file = Path("config/parser_mapping.yaml")
        
#         if not mapping_file.exists():
#             print(f"[Warning] No parser_mapping.yaml found, using defaults")
#             return {
#                 'strategy_1_clinical_image': ClinicalImageParser,
#                 'strategy_2_clinical_qa': ClinicalQAParser,
#                 'strategy_3_textbook': TextbookParser,
#                 'strategy_4_lexicon': LexiconParser,
#                 'strategy_5_research': ResearchParser,
#             }
        
#         with open(mapping_file, 'r') as f:
#             yaml_mapping = yaml.safe_load(f)
        
#         # Convert string names to actual classes
#         class_map = {
#             'ClinicalImageParser': ClinicalImageParser,
#             'ClinicalQAParser': ClinicalQAParser,
#             'TextbookParser': TextbookParser,
#             'LexiconParser': LexiconParser,
#             'ResearchParser': ResearchParser,
#         }
        
#         mapping = {}
#         for folder_name, class_name in yaml_mapping.items():
#             if class_name in class_map:
#                 mapping[folder_name] = class_map[class_name]
#             else:
#                 print(f"[Warning] Unknown parser class: {class_name}")
        
#         return mapping

#     def _get_parser_for_pdf(self, pdf_path: Path):
#         """
#         Determine which parser to use based on PDF location.
            
#         Example:
#         - PDF in data/strategy_5_research/ -> Use ResearchParser
#         - PDF in data/strategy_1_clinical_image/ -> Use ClinicalImageParser
            
#         Args:
#              pdf_path: Path to PDF file
                
#         Returns:
#             Parser instance
#         """
#         # Get parent folder name
#         parent_folder = pdf_path.parent.name
            
#         print(f"\n  [Pipeline] PDF folder: {parent_folder}")
            
#         # Find matching parser
#         if parent_folder in self.parser_mapping:
#             parser_class = self.parser_mapping[parent_folder]
                
#             # Create parser instance if not already created
#             if parent_folder not in self.parser_instances:
#                 output_dir = Path("extracted_content")
#                 self.parser_instances[parent_folder] = parser_class(
#                     self.config, 
#                     output_dir
#                 )
                
#             parser = self.parser_instances[parent_folder]
#             print(f"    [Pipeline] Using parser: {parser.__class__.__name__}")
#             return parser
#         else:
#             print(f"[Error] No parser found for folder: {parent_folder}")
#             print(f"[Error] Available folders: {list(self.parser_mapping.keys())}")
#             raise ValueError(f"No parser configured for folder: {parent_folder}")

#     def process_pdf(self, pdf_path: Path, max_pages: int = None):
#         """
#         Process a single PDF.

#         Steps:
#         1. Detect which parser to use (based on folder)
#         2. Parse the PDF (extract chunks)
#         3. Embed all chunks (text, images, tables)
#         4. Store in Milvus vector database
        
#         Args:
#             pdf_path: Path to PDF file
#             max_pages: Optional limit on pages (for testing)
#         """
        
        
#         print(f"\n{'-'*70}")
#         print(f"[Pipeline] Processing: {pdf_path.name}")
#         print(f"{'-'*70}")
        
#         # Step 1: Get the right parser
#         parser = self._get_parser_for_pdf(pdf_path)
        
#         # Step 2: Parse PDF (extract chunks)
#         print(f"\n[Step 1/3] Parsing PDF...")
#         chunks = parser.parse_pdf(pdf_path, max_pages)

#         if not chunks:
#             print("[Warning] No chunks extracted from PDF!")
#             return

#         #print(f"[Step 1/3] Extracted {len(chunks)} chunks")

#         # Filter out already-processed chunks
#         if self.existing_chunks:
#             original_count = len(chunks)
#             chunks = [c for c in chunks if c['chunk_id'] not in self.existing_chunks]
#             skipped = original_count - len(chunks)
            
#             if skipped > 0:
#                 print(f"[Step 1/3] Skipped {skipped} already-processed chunks")
            
#             if not chunks:
#                 print(f"[Info] All chunks from this PDF already exist in database. Skipping.")
#                 return
            
#             print(f"[Step 1/3] Processing {len(chunks)} new chunks")
        
#         # Step 3: Embed all chunks
#         print(f"\n[Step 2/3] Embedding chunks...")
#         embeddings = self.embedder.embed_batch(chunks)
#         #print(f"[Step 2/3] Created {len(embeddings)} embeddings")
        
#         # Step 4: Store in Milvus
#         print(f"\n[Step 3/3] Storing in vector database...")
#         self.vector_store.add_items(chunks, embeddings)
#         #print(f"[Step 3/3] Stored successfully")
        
#         print(f"\n{'='*70}")
#         print(f"[Success] Finished processing: {pdf_path.name}")
#         print(f"{'='*70}")
    
#     def process_directory(self, data_dir: Path, test_mode: bool = False):
        """
        Process all PDFs in data directory.
        
        Automatically searches all strategy subfolders:
        - data/strategy_1_clinical_image/*.pdf
        - data/strategy_2_clinical_text/*.pdf
        - data/strategy_5_research/*.pdf
        - etc.
        
        Args:
            data_dir: Path to data directory (e.g., "data/")
            test_mode: If True, process only first PDF from each folder, 5 pages max
        """
        print(f"\n{'='*70}")
        print(f"[Pipeline] Scanning directory: {data_dir}")
        print(f"{'='*70}")
        
        # Find all PDFs in strategy subfolders
        all_pdfs = []
        
        # Look in each strategy subfolder
        for strategy_folder in self.parser_mapping.keys():
            folder_path = data_dir / strategy_folder
            
            if not folder_path.exists():
                print(f"[Info] Folder not found: {strategy_folder}")
                continue
            
            pdfs = list(folder_path.glob("*.pdf"))
            if pdfs:
                print(f"        [Found] {len(pdfs)} PDFs in {strategy_folder}/")
                all_pdfs.extend(pdfs)
        
        if not all_pdfs:
            print(f"\n[Error] No PDF files found in {data_dir}")
            print(f"[Error] Make sure PDFs are in strategy subfolders:")
            for folder in self.parser_mapping.keys():
                print(f"  - {data_dir}/{folder}/")
            return
        
        # Test mode: limit PDFs and pages
        if test_mode:
            all_pdfs = all_pdfs[:1]  # Process only 1 PDF in test mode
            max_pages = self.config['processing']['max_test_pages']
            print(f"[TEST MODE] Processing 1 PDF, {max_pages} pages max")
        else:
            max_pages = None
            print(f"[FULL MODE] Processing {len(all_pdfs)} PDFs")
        
        
        # Process each PDF
        for i, pdf_path in enumerate(all_pdfs, 1):
            print(f"[Pipeline] PDF {i}/{len(all_pdfs)}")
            try:
                self.process_pdf(pdf_path, max_pages)
            except Exception as e:
                print(f"\n[Error] Failed to process {pdf_path.name}: {e}")
                print(f"[Error] Continuing with next PDF...")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*70)
        print("[Pipeline] ALL DONE!")
        print("="*70)
        print(f"[Summary] Processed {len(all_pdfs)} PDFs")
        print(f"[Summary] Total chunks in database: {self.vector_store.collection.num_entities}")
        print("="*70)
"""
PDF Processing Pipeline - Text, Image, and Table Extraction

Usage:
    python pdf_pipeline.py --test-mode              # Process first PDF only
    python pdf_pipeline.py --full                   # Process all PDFs
    python pdf_pipeline.py --test-mode --embed      # With embeddings
"""

import fitz  # PyMuPDF
import json
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import io
import argparse


# ============================================================================
# TEXT EXTRACTION & CHUNKING
# ============================================================================

class TextExtractor:
    """
    Extracts text from PDFs and chunks it with overlap
    
    WHY OVERLAP? Prevents cutting sentences/ideas at chunk boundaries
    """
    
    def __init__(self, chunk_size=512, overlap=75):
        self.chunk_size = chunk_size  # tokens (characters / 4)
        self.overlap = overlap
        self.max_chars = chunk_size * 4
        self.overlap_chars = overlap * 4
    
    def extract_and_chunk(self, page, page_num):
        """
        Extract text from a page and split into overlapping chunks
        
        Returns: List of text chunks with metadata
        """
        print(f"    📝 Extracting text from page {page_num + 1}...")
        
        # Get structured text
        text_dict = page.get_text("dict")
        
        # Extract text blocks with position info
        text_blocks = []
        for block in text_dict["blocks"]:
            if block["type"] == 0:  # Text block (not image)
                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"]
                
                if block_text.strip():
                    text_blocks.append({
                        "text": block_text.strip(),
                        "bbox": block["bbox"]
                    })
        
        # Combine all text
        full_text = " ".join([b["text"] for b in text_blocks])
        
        if not full_text.strip():
            print(f"       ⚠️  No text found on page {page_num + 1}")
            return []
        
        # Chunk with overlap
        chunks = self._chunk_with_overlap(full_text, page_num)
        
        print(f"       ✅ Created {len(chunks)} text chunks")
        return chunks
    
    def _chunk_with_overlap(self, text, page_num):
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        chunk_start_word = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            word_length = len(word) + 1  # +1 for space
            
            # Check if adding this word exceeds limit
            if current_length + word_length > self.max_chars and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "chunk_id": f"page{page_num + 1}_text_{len(chunks)}",
                    "type": "text",
                    "content": chunk_text,
                    "metadata": {
                        "page": page_num + 1,
                        "char_count": len(chunk_text),
                        "token_estimate": len(chunk_text) // 4,
                        "word_range": [chunk_start_word, i - 1],
                        "has_overlap": len(chunks) > 0
                    }
                })
                
                # Calculate overlap for next chunk
                overlap_words = []
                overlap_len = 0
                for w in reversed(current_chunk):
                    if overlap_len + len(w) + 1 <= self.overlap_chars:
                        overlap_words.insert(0, w)
                        overlap_len += len(w) + 1
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_words
                current_length = overlap_len
                chunk_start_word = i - len(overlap_words)
            else:
                current_chunk.append(word)
                current_length += word_length
                i += 1
        
        # Save final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "chunk_id": f"page{page_num + 1}_text_{len(chunks)}",
                "type": "text",
                "content": chunk_text,
                "metadata": {
                    "page": page_num + 1,
                    "char_count": len(chunk_text),
                    "token_estimate": len(chunk_text) // 4,
                    "word_range": [chunk_start_word, len(words) - 1],
                    "has_overlap": len(chunks) > 0
                }
            })
        
        return chunks


# ============================================================================
# IMAGE EXTRACTION
# ============================================================================

class ImageExtractor:
    """
    Extracts images from PDFs and saves them as files
    
    Also captures surrounding text as "context" for each image
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / "extracted_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_images(self, page, page_num, doc):
        """
        Extract all images from a page and save them
        
        Returns: List of image chunks with metadata
        """
        print(f"    🖼️  Extracting images from page {page_num + 1}...")
        
        image_list = page.get_images(full=True)
        
        if not image_list:
            print(f"       ℹ️  No images found on page {page_num + 1}")
            return []
        
        image_chunks = []
        
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            
            try:
                # Extract the image
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Save image to file
                image_filename = f"page_{page_num + 1}_image_{img_idx + 1}.{image_ext}"
                image_path = self.output_dir / image_filename
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                # Get image position on page
                rects = page.get_image_rects(xref)
                bbox = list(rects[0]) if rects else None
                
                # Get surrounding text context
                context_text = self._get_image_context(page, bbox)
                
                # Create image chunk
                image_chunks.append({
                    "chunk_id": f"page{page_num + 1}_image_{img_idx}",
                    "type": "image",
                    "image_path": f"extracted_images/{image_filename}",
                    "context_text": context_text,
                    "metadata": {
                        "page": page_num + 1,
                        "dimensions": {
                            "width": base_image["width"],
                            "height": base_image["height"]
                        },
                        "format": image_ext,
                        "colorspace": base_image.get("colorspace", "unknown"),
                        "position": bbox
                    }
                })
                
                print(f"       ✅ Saved: {image_filename} ({base_image['width']}x{base_image['height']})")
                
            except Exception as e:
                print(f"       ⚠️  Could not extract image {img_idx + 1}: {e}")
        
        return image_chunks
    
    def _get_image_context(self, page, img_bbox, max_chars=150):
        """
        Get text surrounding an image for context
        
        This helps with multimodal embeddings later!
        """
        if not img_bbox:
            return ""
        
        text_dict = page.get_text("dict")
        contexts = []
        
        img_y = img_bbox[1]  # Top y-coordinate of image
        
        for block in text_dict["blocks"]:
            if block["type"] == 0:  # Text block
                block_y = block["bbox"][1]
                
                # Get text within 100 pixels of image
                if abs(block_y - img_y) < 100:
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]
                    
                    if block_text.strip():
                        contexts.append(block_text.strip())
        
        combined = " ".join(contexts)
        return combined[:max_chars] if combined else "No surrounding text found"


# ============================================================================
# TABLE DETECTION (BASIC)
# ============================================================================

class TableDetector:
    """
    Detects table-like structures in PDFs
    
    NOTE: This is basic detection using text alignment.
    For production, consider using pdfplumber or camelot.
    """
    
    def detect_tables(self, page, page_num):
        """
        Detect tables on a page using text alignment
        
        Returns: List of table chunks with structured data
        """
        print(f"    📊 Detecting tables on page {page_num + 1}...")
        
        blocks = page.get_text("blocks")
        
        # Group text blocks by y-coordinate (rows)
        rows = {}
        tolerance = 5  # pixels
        
        for block in blocks:
            if block[6] == 0:  # Text block
                x0, y0, x1, y1, text = block[:5]
                
                # Find or create row
                row_key = None
                for y in rows.keys():
                    if abs(y - y0) < tolerance:
                        row_key = y
                        break
                
                if row_key is None:
                    row_key = y0
                    rows[row_key] = []
                
                rows[row_key].append({
                    "x": x0,
                    "text": text.strip()
                })
        
        # Sort rows by y-coordinate
        sorted_rows = sorted(rows.items(), key=lambda x: x[0])
        
        # Check if this looks like a table (multiple aligned columns)
        table_chunks = []
        
        if len(sorted_rows) >= 3:  # At least 3 rows
            # Sort cells in each row by x-coordinate
            table_data = []
            for y, cells in sorted_rows:
                sorted_cells = sorted(cells, key=lambda x: x["x"])
                row_data = [cell["text"] for cell in sorted_cells]
                
                # Only include rows with multiple columns
                if len(row_data) >= 2:
                    table_data.append(row_data)
            
            # If we have consistent multi-column data, it's likely a table
            if len(table_data) >= 3:
                table_chunks.append({
                    "chunk_id": f"page{page_num + 1}_table_0",
                    "type": "table",
                    "content": table_data,
                    "metadata": {
                        "page": page_num + 1,
                        "num_rows": len(table_data),
                        "num_cols": max(len(row) for row in table_data),
                        "detection_method": "text_alignment"
                    }
                })
                
                print(f"       ✅ Detected table with {len(table_data)} rows")
            else:
                print(f"       ℹ️  No clear table structure detected")
        else:
            print(f"       ℹ️  Not enough rows for table detection")
        
        return table_chunks


# ============================================================================
# EMBEDDING GENERATION (OPTIONAL)
# ============================================================================

class EmbeddingGenerator:
    """
    Generate embeddings for text chunks
    
    Uses sentence-transformers (free, local, no API needed)
    """
    
    def __init__(self):
        self.model = None
        print("\n🔧 Loading embedding model (this may take a moment)...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("   ✅ Model loaded: all-MiniLM-L6-v2 (384 dimensions)")
        except ImportError:
            print("   ⚠️  sentence-transformers not installed")
            print("   Install with: pip install sentence-transformers")
            self.model = None
    
    def generate_embeddings(self, text_chunks):
        """
        Generate embeddings for text chunks
        
        Returns: NumPy array of embeddings
        """
        if not self.model:
            print("   ⚠️  Skipping embeddings (model not available)")
            return None, None
        
        print(f"\n🧮 Generating embeddings for {len(text_chunks)} text chunks...")
        
        # Extract just the text content
        texts = [chunk["content"] for chunk in text_chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create metadata mapping
        metadata = []
        for i, chunk in enumerate(text_chunks):
            metadata.append({
                "embedding_index": i,
                "chunk_id": chunk["chunk_id"],
                "page": chunk["metadata"]["page"],
                "preview": chunk["content"][:100] + "..."
            })
        
        print(f"   ✅ Generated {len(embeddings)} embeddings")
        print(f"   📐 Dimensions: {embeddings.shape}")
        
        return embeddings, metadata


# ============================================================================
# OUTPUT ORGANIZATION
# ============================================================================

class OutputOrganizer:
    """
    Manages the folder structure and saves all outputs
    """
    
    def __init__(self, pdf_name, output_base="output"):
        self.pdf_name = Path(pdf_name).stem
        self.base_dir = Path(output_base) / self.pdf_name
        
        # Create folder structure
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir = self.base_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        self.embeddings_dir = self.base_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
    
    def save_chunks(self, text_chunks, image_chunks, table_chunks, settings):
        """Save all chunks to separate JSON files"""
        print(f"\n💾 Saving chunks for {self.pdf_name}...")
        
        # Save text chunks
        text_output = {
            "_description": "Text chunks extracted with overlap for context preservation",
            "_created_at": datetime.now().isoformat(),
            "_settings": settings,
            "total_chunks": len(text_chunks),
            "chunks": text_chunks
        }
        text_path = self.chunks_dir / "text_chunks.json"
        with open(text_path, "w") as f:
            json.dump(text_output, f, indent=2)
        print(f"   ✅ Saved {len(text_chunks)} text chunks → chunks/text_chunks.json")
        
        # Save image chunks
        image_output = {
            "_description": "Image chunks with file paths and surrounding context",
            "_created_at": datetime.now().isoformat(),
            "total_chunks": len(image_chunks),
            "chunks": image_chunks
        }
        image_path = self.chunks_dir / "image_chunks.json"
        with open(image_path, "w") as f:
            json.dump(image_output, f, indent=2)
        print(f"   ✅ Saved {len(image_chunks)} image chunks → chunks/image_chunks.json")
        
        # Save table chunks
        table_output = {
            "_description": "Table structures detected using text alignment",
            "_created_at": datetime.now().isoformat(),
            "_note": "For better table extraction, consider pdfplumber",
            "total_chunks": len(table_chunks),
            "chunks": table_chunks
        }
        table_path = self.chunks_dir / "table_chunks.json"
        with open(table_path, "w") as f:
            json.dump(table_output, f, indent=2)
        print(f"   ✅ Saved {len(table_chunks)} table chunks → chunks/table_chunks.json")
    
    def save_embeddings(self, embeddings, metadata):
        """Save embeddings and their metadata"""
        if embeddings is None:
            return
        
        print(f"\n💾 Saving embeddings...")
        
        # Save embeddings as NumPy array
        embeddings_path = self.embeddings_dir / "text_embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"   ✅ Saved embeddings → embeddings/text_embeddings.npy")
        
        # Save metadata
        metadata_output = {
            "_description": "Metadata mapping embedding indices to chunks",
            "_created_at": datetime.now().isoformat(),
            "_model": "all-MiniLM-L6-v2",
            "_dimensions": int(embeddings.shape[1]),
            "mappings": metadata
        }
        metadata_path = self.embeddings_dir / "embedding_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata_output, f, indent=2)
        print(f"   ✅ Saved metadata → embeddings/embedding_metadata.json")
    
    def save_document_metadata(self, page_count, stats):
        """Save document-level metadata"""
        metadata = {
            "_description": "Document-level processing metadata",
            "_created_at": datetime.now().isoformat(),
            "pdf_name": f"{self.pdf_name}.pdf",
            "total_pages": page_count,
            "statistics": stats
        }
        
        metadata_path = self.base_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Saved document metadata → metadata.json")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class PDFPipeline:
    """
    Main orchestrator for the entire PDF processing pipeline
    """
    
    def __init__(self, chunk_size=512, overlap=75, generate_embeddings=False):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.generate_embeddings = generate_embeddings
        
        self.text_extractor = TextExtractor(chunk_size, overlap)
        self.table_detector = TableDetector()
        self.embedding_generator = EmbeddingGenerator() if generate_embeddings else None
    
    def process_pdf(self, pdf_path, output_dir="output"):
        """Process a single PDF through the entire pipeline"""
        
        print("\n" + "=" * 70)
        print(f"📄 PROCESSING: {pdf_path.name}")
        print("=" * 70)
        
        # Initialize output organizer
        organizer = OutputOrganizer(pdf_path.name, output_dir)
        image_extractor = ImageExtractor(organizer.base_dir)
        
        # Open PDF
        doc = fitz.open(pdf_path)
        print(f"📊 Total pages: {doc.page_count}")
        
        # Storage for all chunks
        all_text_chunks = []
        all_image_chunks = []
        all_table_chunks = []
        
        # Process each page
        for page_num, page in enumerate(doc):
            print(f"\n  📖 Processing page {page_num + 1}/{doc.page_count}")
            
            # Extract text
            text_chunks = self.text_extractor.extract_and_chunk(page, page_num)
            all_text_chunks.extend(text_chunks)
            
            # Extract images
            image_chunks = image_extractor.extract_images(page, page_num, doc)
            all_image_chunks.extend(image_chunks)
            
            # Detect tables
            table_chunks = self.table_detector.detect_tables(page, page_num)
            all_table_chunks.extend(table_chunks)
        
        doc.close()
        
        # Calculate statistics
        stats = {
            "text_chunks": len(all_text_chunks),
            "image_chunks": len(all_image_chunks),
            "table_chunks": len(all_table_chunks),
            "total_chunks": len(all_text_chunks) + len(all_image_chunks) + len(all_table_chunks)
        }
        
        # Save chunks
        settings = {
            "chunk_size_tokens": self.chunk_size,
            "overlap_tokens": self.overlap,
            "chunking_strategy": "overlap"
        }
        organizer.save_chunks(all_text_chunks, all_image_chunks, all_table_chunks, settings)
        
        # Generate embeddings if requested
        if self.generate_embeddings and all_text_chunks:
            embeddings, metadata = self.embedding_generator.generate_embeddings(all_text_chunks)
            organizer.save_embeddings(embeddings, metadata)
        
        # Save document metadata
        organizer.save_document_metadata(doc.page_count, stats)
        
        # Print summary
        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE")
        print("=" * 70)
        print(f"📁 Output location: {organizer.base_dir}/")
        print(f"\n📊 Summary:")
        print(f"   • Text chunks: {stats['text_chunks']}")
        print(f"   • Image chunks: {stats['image_chunks']}")
        print(f"   • Table chunks: {stats['table_chunks']}")
        print(f"   • Total chunks: {stats['total_chunks']}")
        
        return stats
    
    def process_multiple(self, pdf_dir, test_mode=False, output_dir="output"):
        """Process multiple PDFs"""
        pdf_dir = Path(pdf_dir)
        pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_dir}/")
            return
        
        if test_mode:
            pdf_files = pdf_files[:1]
            print(f"\n🧪 TEST MODE: Processing only first PDF\n")
        else:
            print(f"\n📚 FULL MODE: Processing {len(pdf_files)} PDFs\n")
        
        all_stats = []
        
        for pdf_path in pdf_files:
            stats = self.process_pdf(pdf_path, output_dir)
            all_stats.append({
                "filename": pdf_path.name,
                "statistics": stats
            })
        
        # Save processing summary
        summary = {
            "_description": "Summary of all processed PDFs",
            "_created_at": datetime.now().isoformat(),
            "total_pdfs": len(all_stats),
            "settings": {
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "embeddings_generated": self.generate_embeddings
            },
            "results": all_stats
        }
        
        summary_path = Path(output_dir) / "processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Processing summary saved → {summary_path}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PDF Processing Pipeline - Extract text, images, and tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_pipeline.py --test-mode              # Test with first PDF
  python pdf_pipeline.py --full                   # Process all PDFs
  python pdf_pipeline.py --test-mode --embed      # With embeddings
        """
    )
    
    parser.add_argument('--test-mode', action='store_true',
                        help='Process only the first PDF (for testing)')
    parser.add_argument('--full', action='store_true',
                        help='Process all PDFs in input directory')
    parser.add_argument('--embed', action='store_true',
                        help='Generate embeddings (requires sentence-transformers)')
    parser.add_argument('--input-dir', default='test_dataset',
                        help='Input directory containing PDFs (default: test_dataset)')
    parser.add_argument('--output-dir', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--chunk-size', type=int, default=512,
                        help='Chunk size in tokens (default: 512)')
    parser.add_argument('--overlap', type=int, default=75,
                        help='Overlap size in tokens (default: 75)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 70)
    print("🚀 PDF PROCESSING PIPELINE")
    print("=" * 70)
    print(f"⚙️  Configuration:")
    print(f"   • Mode: {'TEST (first PDF only)' if args.test_mode else 'FULL (all PDFs)'}")
    print(f"   • Chunk size: {args.chunk_size} tokens")
    print(f"   • Overlap: {args.overlap} tokens")
    print(f"   • Generate embeddings: {'Yes' if args.embed else 'No'}")
    print(f"   • Input directory: {args.input_dir}/")
    print(f"   • Output directory: {args.output_dir}/")
    print("=" * 70)
    
    # Create pipeline
    pipeline = PDFPipeline(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        generate_embeddings=args.embed
    )
    
    # Process PDFs
    pipeline.process_multiple(
        pdf_dir=args.input_dir,
        test_mode=args.test_mode,
        output_dir=args.output_dir
    )
    
    print("\n" + "=" * 70)
    print("🎉 ALL DONE!")
    print("=" * 70)
    print(f"📂 Check the {args.output_dir}/ folder for results")
    print("📊 Next step: Open evaluation.ipynb to analyze results")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
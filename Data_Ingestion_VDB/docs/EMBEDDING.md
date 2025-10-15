# Embedding Architecture

## Core Design

One model (Gemini) embeds everything into the same 768-d vector space.

Text → Gemini → 768-d vector
Images → Gemini → 768-d vector  
Tables → Gemini → 768-d vector
All go to Milvus for search

Same dimensional space = directly comparable = cross-modal search works.

## Key Decisions

### 1. Why Gemini for Everything?

Decision: Use one multimodal model instead of specialized models per type.

Alternatives considered:
- BioBERT for text + CLIP for images + custom for tables
- Would create 3 different vector spaces that cannot be compared

Why Gemini wins:
- Text and images map to SAME space (can compare directly)
- Reads text inside images (no separate OCR needed)
- Understands tables from images or markdown
- Simple: one API, one embedding function

Trade-off:
- Cost: ~$0.00001 per item (vs free local models)
- Privacy: data sent to Google (vs local processing)
- We accept this for simplicity and cross-modal capability

### 2. Why NOT PDFPlumber for Tables?

Decision: Simple text-based detection, let Gemini handle tables as text or images.

Why we skip PDFPlumber:
- Only works on native PDFs (fails on scanned textbooks)
- Additional dependency (more code to maintain)
- Complex table to vector logic (need separate embedding strategy)
- Structure preservation is hard

Our approach:
- Native PDF tables: Extract as markdown text, embed with Gemini
- Scanned PDF tables: Extract table region as image, embed with Gemini

Trade-off:
- Lose: Perfect table structure extraction
- Gain: Simpler code, works on all PDFs, unified pipeline

Can add PDFPlumber later if table quality is insufficient.

### 3. Why Milvus Instead of ChromaDB?

Decision: Milvus for vector store.

Comparison:
- ChromaDB: ~1M vectors max, prototyping-focused, basic HNSW, vertical scaling only
- Milvus: Billions of vectors, production-ready, multiple index types, horizontal scaling

Why Milvus:
- Better for 25K+ vectors (our scale)
- Production-grade
- More index options for optimization

Trade-off: Requires Docker/server vs ChromaDB embedded option.

### 4. Chunking Strategy

Decision: Simple character-based chunking with overlap.

Why not semantic/paragraph chunking:
- PDFs have inconsistent structure
- PyMuPDF text extraction is messy
- Simple = easier to debug

Parameters:
- Chunk size: 800 chars (~200 tokens)
- Overlap: 100 chars (prevents losing context at boundaries)

Can improve later with better parsing.

### 5. Image Embedding Strategy

Decision: Embed caption + image together when caption exists.

Just image: embed(image) - OK
Image + caption: embed(["Figure 3.2: Heart anatomy", image]) - Better

Why:
- Captions provide explicit semantics
- Image provides visual information
- Gemini fuses both into richer embedding

### 6. Normalization

Decision: L2 normalize all embeddings.

Why:
Without normalization, vectors with different magnitudes create biased similarity scores.
With normalization, all vectors have length 1.0, so cosine similarity = dot product (faster).

## Architecture Flow

PDF Document
→ PyMuPDF Extraction (text blocks, images, tables)
→ Chunk Organization (chunk_id, type, content, metadata)
→ Gemini Embedding (everything to 768-d)
→ L2 Normalization
→ Milvus Vector Store (with IVF_FLAT index)
→ Cosine similarity search enabled

## Summary of Trade-offs

Decision: Gemini for all
- Lose: Cost, privacy
- Gain: Unified space, cross-modal search

Decision: Skip PDFPlumber
- Lose: Perfect table extraction
- Gain: Simplicity, works on scanned PDFs

Decision: Simple chunking
- Lose: Semantic boundaries
- Gain: Easy to debug

Decision: Milvus
- Lose: Easy embedded setup
- Gain: Production scale

Decision: Normalize embeddings
- Lose: Raw magnitude info
- Gain: Fair similarity comparison

Core philosophy: Simplicity + cross-modal capability over perfect extraction.

## What This Enables

Cross-modal search example:
Query: "heart valve anatomy"
Returns:
- Text chunks about valves
- Images showing valve diagrams
- Tables with valve measurements

All in one query because everything lives in the same vector space.

## Future Improvements

Without major changes:
- Add PDFPlumber for better tables
- Better caption extraction
- Semantic chunking
- Query expansion with medical synonyms
- Hybrid search (semantic + keyword)

If we needed to change:
- Privacy concerns: Switch to local model (PubMed CLIP + BioBERT)
- Cost concerns: Use local models
- Better medical understanding: Fine-tune on medical corpus

But for now: Simple, unified, cross-modal.


┌─────────────────┐
│  PDF Document   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│     PyMuPDF Extraction          │
│  • Text blocks (with overlap)   │
│  • Images (with captions)       │
│  • Tables (basic detection)     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│    Chunk Organization           │
│  {chunk_id, type, content,      │
│   metadata}                     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│    Gemini Embedding             │
│  • Text → 768-d                 │
│  • Image → 768-d                │
│  • Table → 768-d                │
│  • Normalize (L2)               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│    Milvus Vector Store          │
│  • Store embeddings             │
│  • Build IVF_FLAT index         │
│  • Enable cosine search         │
└─────────────────────────────────┘
# Embedding Dimension Mismatch Fix

## Date: November 10, 2025

## Problem
The pipeline was failing with the error:
```
ValueError: all the input array dimensions except for the concatenation axis must match exactly, 
but along dimension 1, the array at index 0 has size 768 and the array at index 1 has size 3072
```

This meant that embeddings of different dimensions (768 and 3072) were being inserted into the same vector database, which requires all vectors to have the same dimension.

---

## Root Causes

### 1. **Multiple Embedding Models**
- Text content may use one model (768-dim)
- Tables/images may use a different model (3072-dim)
- RAGAnything/LightRAG may have internal defaults for multimodal content

### 2. **No Dimension Validation**
- The code didn't validate embedding dimensions before insertion
- Mismatched embeddings were only caught when inserting into the vector DB

### 3. **Hardcoded Values**
- Embedding dimension was hardcoded in multiple places (768)
- If the model changed, dimensions could mismatch

---

## Solutions Implemented

### 1. **Config Centralization**
**File**: `config.py`

Added `EMBEDDING_DIM` configuration:
```python
# Embedding dimension (must match the model)
# models/embedding-001 = 768 dimensions
# text-embedding-004 = 768 dimensions  
# If you change EMBEDDING_MODEL, update this accordingly
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))
```

**Benefits**:
- Single source of truth for embedding dimension
- Easy to update if model changes
- Can be overridden via environment variable

---

### 2. **Embedding Function Validation**
**File**: `rag_processor.py` - `google_embedding_func()`

Added dimension validation after receiving embeddings from API:
```python
# Validate embedding dimensions
for idx, emb in enumerate(batch_embeddings):
    if len(emb) != Config.EMBEDDING_DIM:
        logger.error(
            f"Dimension mismatch! Expected {Config.EMBEDDING_DIM}, got {len(emb)} "
            f"for text: '{batch[idx][:100]}...'"
        )
        raise ValueError(
            f"Embedding dimension mismatch: expected {Config.EMBEDDING_DIM}, got {len(emb)}"
        )
```

**Benefits**:
- Catches dimension mismatches immediately after API call
- Provides detailed error message with the problematic text
- Prevents bad embeddings from propagating through pipeline

---

### 3. **Entity Processing Validation**
**File**: `rag_processor.py` - `process_document()`

Added validation when extracting embeddings from entities:
```python
# Validate embedding dimension before adding to data
if len(embedding) != Config.EMBEDDING_DIM:
    logger.error(
        f"Dimension mismatch in entity {idx}! Expected {Config.EMBEDDING_DIM}, "
        f"got {len(embedding)}. Skipping this entity."
    )
    continue
```

**Benefits**:
- Validates embeddings from RAGAnything/LightRAG storage
- Skips problematic entities instead of crashing
- Logs which entity caused the issue

---

### 4. **Fallback Processing Validation**
**File**: `rag_processor.py` - fallback chunking section

Added validation for embeddings generated in fallback mode:
```python
# Validate all embeddings have correct dimension
for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    if len(embedding) != Config.EMBEDDING_DIM:
        logger.error(
            f"Dimension mismatch in fallback chunk {idx}! Expected {Config.EMBEDDING_DIM}, "
            f"got {len(embedding)}. Skipping this chunk."
        )
        continue
```

**Benefits**:
- Ensures fallback processing also produces correct dimensions
- Continues processing even if some chunks fail
- Logs which chunks had issues

---

### 5. **Vector Store Validation**
**File**: `vector_store.py` - `insert()`

Added dimension validation before inserting into Milvus:
```python
# Validate embedding dimensions
for idx, emb in enumerate(embeddings):
    if len(emb) != self.embedding_dim:
        logger.error(
            f"Embedding dimension mismatch at index {idx}! "
            f"Expected {self.embedding_dim}, got {len(emb)}. "
            f"Text: '{texts[idx][:100]}...'"
        )
        raise ValueError(
            f"Embedding dimension mismatch: expected {self.embedding_dim}, "
            f"got {len(emb)} at index {idx}"
        )
```

**Benefits**:
- Final validation layer before database insertion
- Catches any embeddings that slipped through earlier checks
- Provides context about which text caused the issue

---

### 6. **Config Usage**
Updated all hardcoded dimension values to use `Config.EMBEDDING_DIM`:
- `rag_processor.py`: `EmbeddingFunc(embedding_dim=Config.EMBEDDING_DIM, ...)`
- `vector_store.py`: `embedding_dim = embedding_dim or Config.EMBEDDING_DIM`

---

## How It Works Now

### Flow with Validation
1. **API Call** → `google_embedding_func()` validates dimensions immediately
2. **Entity Extraction** → Validates embeddings from RAGAnything storage
3. **Fallback Processing** → Validates embeddings from direct extraction
4. **Vector Store Insert** → Final validation before database insertion

### Error Handling
- **Early Detection**: Errors caught at the earliest possible point
- **Detailed Logging**: Each error logs the problematic text and expected/actual dimensions
- **Graceful Degradation**: Non-critical errors skip problematic items instead of crashing
- **Critical Failures**: Dimension mismatches in core functions raise exceptions

---

## Testing Recommendations

### 1. Run the Pipeline
```bash
python pdf_pipeline/main.py
```

### 2. Monitor Logs
Look for:
- ✅ No dimension mismatch errors
- ✅ All embeddings are 768-dimensional
- ⚠️ Any warnings about skipped entities (investigate if many)

### 3. Check Output
```python
import pandas as pd
df = pd.read_parquet("/tmp/processed_data/your-file.parquet")

# Check all embeddings have correct dimension
dims = df['vector'].apply(len).unique()
print(f"Embedding dimensions found: {dims}")
# Should only show [768]
```

### 4. Verify Vector Store
After insertion, check Milvus collection:
- All vectors should have dimension 768
- No insertion errors

---

## Configuration Notes

### Current Settings
- **Model**: `models/embedding-001` (Google Gemini)
- **Dimension**: 768
- **Task Type**: `retrieval_document`

### If You Change the Model
1. Update `EMBEDDING_MODEL` in `.env` or `config.py`
2. Update `EMBEDDING_DIM` to match the new model:
   - `models/embedding-001`: 768 dimensions
   - `text-embedding-004`: 768 dimensions
   - Other models: Check documentation
3. Recreate Milvus collection with new dimension

---

## Troubleshooting

### If You Still See Dimension Mismatches

1. **Check the embedding model**:
   ```python
   print(Config.EMBEDDING_MODEL)
   print(Config.EMBEDDING_DIM)
   ```

2. **Test the embedding function directly**:
   ```python
   import asyncio
   from rag_processor import google_embedding_func
   
   async def test():
       embeddings = await google_embedding_func(["test text"])
       print(f"Dimension: {len(embeddings[0])}")
   
   asyncio.run(test())
   ```

3. **Check RAGAnything internal settings**:
   - RAGAnything may use different models for tables/images
   - Review RAGAnything documentation for multimodal embedding configuration

4. **Check LightRAG storage**:
   - Inspect `rag_storage/` directory
   - Look for cached embeddings with wrong dimensions
   - Delete storage and reprocess if needed

---

## Summary

All embedding dimension validation is now in place:
- ✅ Centralized configuration via `Config.EMBEDDING_DIM`
- ✅ Validation at embedding generation
- ✅ Validation at entity extraction
- ✅ Validation at fallback processing
- ✅ Validation before vector store insertion
- ✅ Detailed error logging with context
- ✅ Graceful error handling where appropriate

The pipeline should now catch and report any dimension mismatches immediately, with clear error messages indicating where the problem occurred.

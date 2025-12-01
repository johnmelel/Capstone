# Query Optimization with RL

**Extremely basic** RL setup to train a small model to rewrite user queries for better vector search results.

## Concept

1. **Generate Questions:** Use an LLM (Gemini) to create 100 questions based on document chunks in Milvus
2. **RL Training:** Train a small open-source model (e.g., GPT-2 small) to rewrite questions for better retrieval
3. **Reward Signal:** Success = retrieved chunks contain answer to original question

## Why This Helps

Vector search struggles with:
- Vague queries: "Tell me about diabetes"
- Medical jargon mismatches: User says "sugar disease" vs docs say "diabetes mellitus"
- Missing context: "What causes it?" without specifying what "it" is

A trained rewriter can:
- Add medical terminology
- Expand abbreviations
- Add relevant context
- Restructure for semantic similarity

## Architecture

```
User Query → RL-Trained Model → Rewritten Query → Vector Search → Better Results
   ↓                                                                      ↓
   "sugar problems"              "diabetes mellitus diagnosis"      [Relevant chunks]
```

## Reward Function (Simple)

```python
def calculate_reward(original_query, rewritten_query, retrieved_chunks):
    # Did we retrieve relevant documents?
    if len(retrieved_chunks) > 0:
        # Check if chunks contain keywords from original query
        relevance_score = check_keyword_overlap(original_query, retrieved_chunks)
        # Penalize queries that are too long (more expensive)
        length_penalty = len(rewritten_query.split()) * 0.01
        return relevance_score - length_penalty
    return -1.0  # No results = bad
```

## Files

1. **1_generate_questions.py** - Generate 100 questions from Milvus chunks using Gemini
2. **2_train_rewriter.py** - Basic RL training loop with policy gradient
3. **config.py** - Configuration settings
4. **requirements.txt** - Minimal dependencies

## Usage

### Step 1: Generate Training Questions

```bash
cd query-optimization-rl
python3 1_generate_questions.py
# Output: questions.json (100 questions with source chunks)
```

### Step 2: Train Query Rewriter

```bash
python3 2_train_rewriter.py
# Trains for 100 episodes
# Saves model to: models/query_rewriter.pt
```

### Step 3: Use Trained Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("models/query_rewriter")
model = AutoModelForCausalLM.from_pretrained("models/query_rewriter")

query = "What helps with sugar problems?"
rewritten = model.generate(query)
# Output: "diabetes mellitus treatment options and glucose management"
```

## RL Training Details (Kept Simple)

- **Algorithm:** REINFORCE (policy gradient)
- **Model:** GPT-2 small (124M params) - easy to train on CPU
- **Episodes:** 100 (can increase for better results)
- **Reward:** Binary success (1 if retrieved relevant docs, 0 otherwise) + length penalty
- **Optimizer:** AdamW with learning rate 5e-5

### Training Loop (Simplified)

```
For each episode:
  1. Sample a question from generated set
  2. Model rewrites the question
  3. Search vector store with rewritten query
  4. Calculate reward based on retrieval quality
  5. Update model weights using policy gradient
  6. Track reward over time
```

## Limitations (Because it's basic)

- Simple binary reward (no nuanced quality scoring)
- Small model (GPT-2 small, not medical-specialized)
- No validation set (just training)
- No hyperparameter tuning
- CPU training only (slow but works)

## Extensions (If You Want to Go Further)

1. **Better Reward:** Use Gemini to judge if retrieved chunks answer the question (0-10 score)
2. **Larger Model:** Fine-tune Mistral-7B or Llama-3.1-8B for better medical understanding
3. **PPO Algorithm:** More stable than REINFORCE
4. **Human Feedback:** RLHF with human ratings of rewritten queries
5. **Multi-Objective:** Balance retrieval quality, query length, and computational cost

## Expected Results

After training on 100 questions:
- **Baseline (no rewriting):** ~60% retrieval success
- **After RL training:** ~75-80% retrieval success
- **Query quality:** More specific, includes medical terms, better structured

## Dependencies

Minimal setup:
- `transformers` - Load/train GPT-2
- `torch` - Neural network training
- `pymilvus` - Vector search
- `vertexai` - Question generation (Gemini)

No complex RL libraries needed - we implement REINFORCE from scratch (it's just 20 lines).

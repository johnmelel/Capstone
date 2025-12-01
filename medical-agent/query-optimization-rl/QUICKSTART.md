# Query Optimization RL - Quickstart

Ultra-simple guide to get started with RL query rewriting.

## What This Does

Trains a small model (GPT-2) to rewrite vague medical queries into better ones for vector search.

**Example:**
- Input: "sugar problems"
- Output: "diabetes mellitus diagnosis and glucose management strategies"

## Setup (5 minutes)

### 1. Install Dependencies

```bash
cd query-optimization-rl
pip install -r requirements.txt
```

### 2. Check Environment

Make sure parent `.env` file has:
```
MILVUS_URI=your_milvus_uri
MILVUS_TOKEN=your_milvus_token
```

## Run (30-60 minutes)

### Step 1: Generate Questions (~5 minutes)

```bash
python3 1_generate_questions.py
```

Output: `questions.json` with 100 questions

### Step 2: Train Model (~30-60 minutes on CPU)

```bash
python3 2_train_rewriter.py
```

Output: Trained model in `models/query_rewriter/`

**Note:** Training is slow on CPU. For faster training:
- Use Google Colab with GPU (free)
- Or reduce `NUM_EPISODES` in `config.py` to 50

## Test the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load trained model
tokenizer = AutoTokenizer.from_pretrained("models/query_rewriter")
model = AutoModelForCausalLM.from_pretrained("models/query_rewriter")

# Rewrite a query
query = "What helps with high sugar?"
prompt = f"Rewrite this medical query to be more specific: {query}\nRewritten:"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Original:  {query}")
print(f"Rewritten: {rewritten.split('Rewritten:')[-1]}")
```

## Monitor Training

Watch the reward improve over episodes:

```
Episode 10/100 | Avg Reward: 0.234
  Original: What causes diabetes?
  Rewritten: diabetes mellitus etiology pathophysiology and risk factors

Episode 20/100 | Avg Reward: 0.412
  Original: Tell me about blood pressure meds
  Rewritten: antihypertensive medications mechanism of action and clinical use

...

Episode 100/100 | Avg Reward: 0.678
```

**Good training:** Reward increases from ~0.2 to ~0.6+

## Troubleshooting

### "questions.json not found"
Run `1_generate_questions.py` first

### "Out of memory"
- Use smaller model: Change `MODEL_NAME = "distilgpt2"` in config.py
- Or reduce batch processing

### "Milvus connection failed"
- Check .env has correct MILVUS_URI and MILVUS_TOKEN
- Test connection: `python3 -c "from pymilvus import MilvusClient; print(MilvusClient(uri='your_uri', token='your_token'))"`

### Training is too slow
- Reduce `NUM_EPISODES = 50` in config.py
- Or use Google Colab with GPU

## File Structure

```
query-optimization-rl/
├── README.md              # Detailed explanation
├── QUICKSTART.md          # This file
├── config.py              # Settings
├── requirements.txt       # Dependencies
├── 1_generate_questions.py
├── 2_train_rewriter.py
├── questions.json         # Generated questions (after step 1)
├── models/
│   └── query_rewriter/    # Trained model (after step 2)
└── logs/
    └── training_log.json  # Training metrics
```

## Next Steps

Once trained, integrate the model into your RAG pipeline:

```python
# In your orchestrator
from transformers import AutoTokenizer, AutoModelForCausalLM

class ImprovedOrchestrator:
    def __init__(self):
        # Load query rewriter
        self.rewriter_tokenizer = AutoTokenizer.from_pretrained("query-optimization-rl/models/query_rewriter")
        self.rewriter_model = AutoModelForCausalLM.from_pretrained("query-optimization-rl/models/query_rewriter")
    
    def process_query(self, user_query):
        # Rewrite query for better search
        rewritten = self.rewrite_query(user_query)
        
        # Use rewritten query for vector search
        results = self.vector_search(rewritten)
        
        return results
```

## Expected Improvements

- **Retrieval success:** 60% → 75-80%
- **Query specificity:** Vague → Medical terminology
- **Context:** Missing → Added relevant keywords

## That's It!

Basic RL query optimization in 2 steps. Keep it simple.

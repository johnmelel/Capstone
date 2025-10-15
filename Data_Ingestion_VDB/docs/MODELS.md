# Models Used

## Gemini embedding-001
- **Dimensions:** 768
- **Input:** Text, images, or both
- **Cost:** ~$0.00001 per item
- **API Key:** Get from https://makersuite.google.com/app/apikey

## Why Gemini?
- Handles text, images, AND text-in-images automatically
- Same vector space for everything
- Good for medical content

## Settings
- Batch size: 100 items at a time
- Normalization: L2 (for cosine similarity)
- Task type: retrieval_document
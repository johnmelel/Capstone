# PDF Ingestion Pipeline

This pipeline ingests PDF documents from a Google Cloud Storage (GCS) bucket, processes them using the `rag-anything` library with Google's Generative AI, and stores the resulting embeddings in a Milvus vector database.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment Variables:**
    Create a `.env` file in the root of the `pdf_pipeline` directory and add the following variables:
    ```
    GOOGLE_SERVICE_ACCOUNT_JSON="path/to/your/service-account.json"
    GCS_BUCKET_NAME="your-gcs-bucket-name"
    GCS_BUCKET_PREFIX="your-prefix/"
    MILVUS_URI="your-milvus-uri"
    MILVUS_API_KEY="your-milvus-api-key"
    GOOGLE_API_KEY="your-google-api-key"
    ```
    You can also customize the following optional variables:
    ```
    EMBEDDING_MODEL="models/embedding-001"
    LLM_MODEL="gemini-1.5-pro"
    MINERU_DEVICE="cuda" # or "cpu"
    LOG_FILE="pipeline.log"
    LOG_LEVEL="INFO"
    ```

## Usage

To run the entire pipeline, simply execute the `main.py` script:
```bash
python main.py
```

The script will:
1.  Download all new PDF files from the configured GCS bucket.
2.  Process each PDF using `rag-anything` to extract text, tables, and other content, and generate embeddings using Google's Generative AI.
3.  Store the embeddings and associated metadata in the configured Milvus vector database.

Logs will be printed to the console and saved to the configured log file (`pipeline.log` by default).

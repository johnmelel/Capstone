import asyncio
import logging
import os
from pathlib import Path
from typing import List
import pandas as pd
import google.generativeai as genai
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

from .config import Config

# Configure logging
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Configure Google Generative AI
genai.configure(api_key=Config.GOOGLE_API_KEY)

def google_llm_model_func(prompt: str, **kwargs):
    """Custom LLM model function using Google's Generative AI."""
    model = genai.GenerativeModel(Config.LLM_MODEL)
    response = model.generate_content(prompt, **kwargs)
    return response.text

def google_embedding_func(texts: List[str]):
    """Custom embedding function using Google's Generative AI."""
    response = genai.embed_content(
        model=Config.EMBEDDING_MODEL,
        content=texts,
        task_type="retrieval_document"
    )
    return response['embedding']

class RAGProcessor:
    def __init__(self):
        self.config = RAGAnythingConfig(
            working_dir="./rag_storage",
        )
        self.embedding_func = EmbeddingFunc(
            embedding_dim=768,  # Dimension for Google's embedding-001 model
            func=google_embedding_func,
        )
        self.rag = RAGAnything(
            config=self.config,
            llm_model_func=google_llm_model_func,
            embedding_func=self.embedding_func,
        )

    async def process_document(self, file_path: Path):
        """Processes a single document and saves the output."""
        logger.info(f"Processing document: {file_path}")
        try:
            await self.rag.process_document_complete(
                file_path=str(file_path),
                output_dir=str(Config.PROCESSED_DATA_DIR),
                device=Config.MINERU_DEVICE,
            )

            # Access the processed data from the knowledge graph
            kg = self.rag.lightrag.knowledge_graph
            nodes = kg.get_all_nodes()

            # Convert the nodes to a pandas DataFrame
            data = []
            for node in nodes:
                data.append({
                    "vector": node.embedding,
                    "text": node.text,
                    "file_name": node.metadata.get("file_name", ""),
                    "file_hash": node.metadata.get("file_hash", ""),
                    "chunk_index": node.metadata.get("chunk_index", 0),
                    "total_chunks": node.metadata.get("total_chunks", 0),
                })
            df = pd.DataFrame(data)

            # Save the DataFrame to a parquet file
            processed_file_path = Config.PROCESSED_DATA_DIR / f"{file_path.stem}.parquet"
            df.to_parquet(processed_file_path)

            logger.info(f"Successfully processed {file_path} and saved to {processed_file_path}")
            return processed_file_path
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise

async def main():
    """Main function to test the RAGProcessor."""
    processor = RAGProcessor()
    dummy_pdf_path = Path("dummy.pdf")
    dummy_pdf_path.write_text("This is a dummy PDF for testing.")
    await processor.process_document(dummy_pdf_path)

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python
"""
Quick fix for RAGAnything API compatibility.
This script shows how to properly use RAGAnything based on actual API.
"""

import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_raganything_usage():
    """Test different ways to use RAGAnything"""
    
    try:
        from raganything import RAGAnything, RAGAnythingConfig
        
        # Initialize
        config = RAGAnythingConfig(working_dir="./test_rag_storage")
        rag = RAGAnything(config=config)
        
        logger.info("RAGAnything initialized successfully")
        logger.info(f"Config: {config}")
        
        # Check what methods/attributes are available
        logger.info("\nAvailable callable methods:")
        for attr in dir(rag):
            if not attr.startswith('_'):
                obj = getattr(rag, attr)
                if callable(obj):
                    logger.info(f"  - {attr}")
        
        # The RAGAnything library might work differently
        # Common patterns:
        
        # Pattern 1: Direct call with file path
        logger.info("\nTrying Pattern 1: rag(file_path)")
        try:
            # Some libraries allow calling the object directly
            # result = await rag("path/to/file.pdf")
            logger.info("  Pattern 1 would be: await rag(file_path)")
        except Exception as e:
            logger.info(f"  Pattern 1 not applicable: {e}")
        
        # Pattern 2: Using process method
        logger.info("\nTrying Pattern 2: rag.process()")
        if hasattr(rag, 'process'):
            logger.info("  ✓ Has 'process' method")
        
        # Pattern 3: Using insert_document
        logger.info("\nTrying Pattern 3: rag.insert_document()")
        if hasattr(rag, 'insert_document'):
            logger.info("  ✓ Has 'insert_document' method")
        
        # Pattern 4: Check for storage/database attributes
        logger.info("\nChecking storage attributes:")
        storage_attrs = ['storage', 'db', 'vector_db', 'index', 'documents']
        for attr in storage_attrs:
            if hasattr(rag, attr):
                logger.info(f"  ✓ Has '{attr}' attribute")
                obj = getattr(rag, attr)
                logger.info(f"     Type: {type(obj)}")
                if hasattr(obj, '__dict__'):
                    logger.info(f"     Methods: {[m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m))]}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_raganything_usage())

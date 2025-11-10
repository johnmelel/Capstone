#!/usr/bin/env python
"""Test script to explore RAGAnything API"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from raganything import RAGAnything, RAGAnythingConfig
    
    # Create a simple config
    config = RAGAnythingConfig(working_dir="./test_rag_storage")
    
    # Initialize RAGAnything
    rag = RAGAnything(config=config)
    
    # List all public methods and attributes
    logger.info("\n" + "="*60)
    logger.info("RAGAnything Public Methods and Attributes:")
    logger.info("="*60)
    
    methods = []
    attributes = []
    
    for name in dir(rag):
        if not name.startswith('_'):
            obj = getattr(rag, name)
            if callable(obj):
                methods.append(name)
            else:
                attributes.append(name)
    
    logger.info("\nMethods:")
    for method in sorted(methods):
        logger.info(f"  - {method}()")
    
    logger.info("\nAttributes:")
    for attr in sorted(attributes):
        try:
            value = getattr(rag, attr)
            logger.info(f"  - {attr}: {type(value).__name__}")
        except Exception as e:
            logger.info(f"  - {attr}: <error accessing: {e}>")
    
    # Check for specific methods we're looking for
    logger.info("\n" + "="*60)
    logger.info("Checking for expected methods:")
    logger.info("="*60)
    
    method_checks = [
        'insert_file',
        'process_document',
        'add_document',
        'insert',
        'ainsert',
        'process',
        'ingest',
        'add',
    ]
    
    for method_name in method_checks:
        has_method = hasattr(rag, method_name)
        logger.info(f"  {method_name}: {'✓ YES' if has_method else '✗ NO'}")
    
    # Check RAGAnythingConfig
    logger.info("\n" + "="*60)
    logger.info("RAGAnythingConfig attributes:")
    logger.info("="*60)
    
    for name in dir(config):
        if not name.startswith('_'):
            try:
                value = getattr(config, name)
                if not callable(value):
                    logger.info(f"  - {name}: {value}")
            except:
                pass
    
    logger.info("\n" + "="*60)
    logger.info("Test completed successfully!")
    logger.info("="*60)
    
except Exception as e:
    logger.error(f"Error during test: {e}", exc_info=True)

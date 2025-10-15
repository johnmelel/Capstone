"""Utility functions."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_file: str = "pipeline.log", level: str = "INFO"):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper())
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f"logs/{log_file}"),
            logging.StreamHandler()
        ]
    )

def load_api_key() -> str:
    """Load Gemini API key from environment."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment.\n"
            "Please create a .env file with: GEMINI_API_KEY=your_key_here"
        )
    
    return api_key

def create_directories():
    """Create necessary output directories."""
    dirs = [
        "data/extracted/images",
        "data/extracted/tables",
        "data/extracted/text",
        "data/processed",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
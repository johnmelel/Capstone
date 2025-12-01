"""
Configuration for Query Optimization RL
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
QUESTIONS_FILE = PROJECT_ROOT / "questions.json"
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Milvus connection (from parent .env)
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME = "capstone_group_2"

# GCP credentials
GCP_PROJECT = "adsp-34002-ip09-team-2"
GCP_LOCATION = "us-central1"
SERVICE_ACCOUNT = Path(__file__).parent.parent / "adsp-34002-ip09-team-2-e0cca2d396a9.json"

# Question generation
NUM_QUESTIONS = 100
CHUNKS_TO_SAMPLE = 50  # Sample from random chunks

# RL Training
MODEL_NAME = "gpt2"  # Small model - 124M params
LEARNING_RATE = 5e-5
NUM_EPISODES = 100
MAX_QUERY_LENGTH = 50  # tokens
REWARD_LENGTH_PENALTY = 0.01  # Penalize long queries

# Vector search
TOP_K_RESULTS = 3  # Retrieve top 3 chunks
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity for reward

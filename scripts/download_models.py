"""Download and cache neural models locally for offline inference.
Prepares models so runtime operates without any network connectivity.
"""

import os
import sys
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TARGET_DIR = "models/all-MiniLM-L6-v2"


def download_model(model_name: str = DEFAULT_MODEL_NAME, target_dir: str = DEFAULT_TARGET_DIR):
    """Download model and save to target directory for offline use."""
    print(f"Downloading model '{model_name}' to '{target_dir}'...")
    os.makedirs(target_dir, exist_ok=True)
    
    # Download from huggingface and save locally
    model = SentenceTransformer(model_name)
    model.save(target_dir)
    print(f"Successfully saved model to '{target_dir}'.")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_NAME
    target_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TARGET_DIR
    download_model(model_name, target_dir)

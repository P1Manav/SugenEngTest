"""Download and cache pinned neural models for offline inference at docker build time.
Verifies model weights with a strict SHA-256 checksum.
"""

import hashlib
import os
from huggingface_hub import snapshot_download

PINNED_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PINNED_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EXPECTED_SAFETENSORS_SHA256 = "eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b"
DEFAULT_TARGET_DIR = "models/paraphrase-multilingual-MiniLM-L12-v2"


def download_and_verify(
    repo_id: str = PINNED_MODEL_ID,
    revision: str = PINNED_REVISION,
    target_dir: str = DEFAULT_TARGET_DIR,
    expected_sha256: str = EXPECTED_SAFETENSORS_SHA256,
):
    print(f"Downloading model '{repo_id}' at pinned revision '{revision}' to '{target_dir}'...")
    os.makedirs(target_dir, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target_dir,
        allow_patterns=[
            "config.json",
            "config_sentence_transformers.json",
            "model.safetensors",
            "modules.json",
            "sentence_bert_config.json",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "unigram.json",
            "1_Pooling/*",
            "2_Normalize/*",
            "README.md",
        ],
    )

    safetensors_path = os.path.join(target_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Missing model.safetensors in {target_dir}")

    print(f"Verifying SHA-256 checksum of {safetensors_path}...")
    sha = hashlib.sha256()
    with open(safetensors_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    actual_sha256 = sha.hexdigest()

    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Checksum verification failed for {safetensors_path}!\n"
            f"Expected: {expected_sha256}\n"
            f"Got:      {actual_sha256}"
        )

    print(f"SHA-256 verification PASSED: {actual_sha256}")
    print(f"Model successfully downloaded and verified in '{target_dir}'.")


if __name__ == "__main__":
    download_and_verify()

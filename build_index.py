"""Build BM25 and dense retrieval indexes from enriched_kb.json.

Creates:
- kb/bm25.pkl: Pickled BM25Okapi index and code mapping
- kb/dense_embeddings.npy: Precomputed normalized dense embeddings for all categories
- kb/dense_codes.json: List of codes corresponding to embedding rows
"""

import json
import os
import pickle
import sys

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from text_utils import clean_text, expand_abbreviations, tokenize


def build_indexes(
    kb_path: str = "kb/enriched_kb.json",
    out_dir: str = "kb",
    model_name_or_path: str = "models/all-MiniLM-L6-v2",
    rebuild_dense: bool = True,
):
    """Build BM25 index and precompute dense embeddings for all ICD-11 categories."""
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading enriched KB from {kb_path}...")
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    codes = list(kb.keys())
    print(f"Total categories to index: {len(codes)}")

    # Prepare corpus texts
    corpus_texts = []
    tokenized_corpus = []

    for code in codes:
        entry = kb[code]
        # Include title, search text (ancestors), and expanded abbreviations
        title = entry.get("title", "")
        search_text = entry.get("search_text", "")
        combined = f"{code} {title}. {search_text}"
        expanded = expand_abbreviations(combined)
        full_text = clean_text(f"{combined} {expanded}")

        corpus_texts.append(full_text)
        tokenized_corpus.append(tokenize(full_text))

    # 1. Build BM25 index
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_artifact = {
        "bm25": bm25,
        "codes": codes,
    }

    bm25_path = os.path.join(out_dir, "bm25.pkl")
    print(f"Saving BM25 index to {bm25_path}...")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2. Build Dense index
    dense_emb_path = os.path.join(out_dir, "dense_embeddings.npy")
    dense_codes_path = os.path.join(out_dir, "dense_codes.json")

    if rebuild_dense or not (os.path.exists(dense_emb_path) and os.path.exists(dense_codes_path)):
        print(f"Loading dense model from {model_name_or_path}...")
        if not os.path.exists(model_name_or_path):
            model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name_or_path)

        print(f"Encoding {len(corpus_texts)} categories in batches...")
        embeddings = model.encode(
            corpus_texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,  # Unit length for cosine similarity via dot product
            convert_to_numpy=True,
        ).astype(np.float32)

        print(f"Saving dense embeddings ({embeddings.shape}) to {dense_emb_path}...")
        np.save(dense_emb_path, embeddings)

        print(f"Saving dense code mappings to {dense_codes_path}...")
        with open(dense_codes_path, "w", encoding="utf-8") as f:
            json.dump(codes, f, ensure_ascii=False)
    else:
        print(f"Dense index already exists at {dense_emb_path}. Skipping dense rebuild.")

    print("Index building successfully completed!")


if __name__ == "__main__":
    rebuild_dense = "--bm25-only" not in sys.argv
    kb_path = "kb/enriched_kb.json"
    model_path = "models/all-MiniLM-L6-v2"
    build_indexes(kb_path, "kb", model_path, rebuild_dense=rebuild_dense)

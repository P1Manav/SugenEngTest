"""Retrieval system combining BM25, Dense embeddings, and Reciprocal Rank Fusion (RRF).
Operates completely offline with precomputed indexes and local sentence-transformers.
"""

import json
import os
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from text_utils import tokenize, clean_text, expand_abbreviations


class BM25Retriever:
    """Fast sparse lexical retrieval using BM25Okapi."""

    def __init__(self, bm25_path: str = "kb/bm25.pkl"):
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}. Run build_index.py first.")

        with open(bm25_path, "rb") as f:
            artifact = pickle.load(f)
        self.bm25 = artifact["bm25"]
        self.codes = artifact["codes"]

    def retrieve(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Retrieve top-K matching ICD-11 codes for a query string."""
        if not query or not query.strip():
            return []

        clean_q = clean_text(expand_abbreviations(query))
        tokens = tokenize(clean_q)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.codes[idx], score))
        return results


class DenseRetriever:
    """Semantic vector retrieval using dense sentence embeddings and cosine similarity."""

    def __init__(
        self,
        embeddings_path: str = "kb/dense_embeddings.npy",
        codes_path: str = "kb/dense_codes.json",
        model_name_or_path: str = "models/all-MiniLM-L6-v2",
    ):
        if not os.path.exists(embeddings_path) or not os.path.exists(codes_path):
            raise FileNotFoundError("Dense index files not found in kb/. Run build_index.py first.")

        self.embeddings = np.load(embeddings_path)
        with open(codes_path, "r", encoding="utf-8") as f:
            self.codes = json.load(f)

        if not os.path.exists(model_name_or_path):
            model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name_or_path)

    def retrieve(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Retrieve top-K semantic nearest neighbors for a query string."""
        if not query or not query.strip():
            return []

        clean_q = clean_text(expand_abbreviations(query))
        # Embed query vector and normalize
        query_vec = self.model.encode(
            clean_q,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Cosine similarity via dot product (since both are unit-normalized)
        similarities = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            results.append((self.codes[idx], sim))
        return results


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    weights: Optional[List[float]] = None,
    k: int = 60,
    top_k: int = 50,
) -> List[Tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Formula: RRF(d) = Σ w_i / (k + rank_i(d))
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: Dict[str, float] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank_idx, (code, _) in enumerate(ranked_list):
            rank = rank_idx + 1  # 1-based rank
            rrf_score = weight / (k + rank)
            scores[code] = scores.get(code, 0.0) + rrf_score

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


class HybridRetriever:
    """Integrated hybrid retriever that combines BM25 and Dense with RRF."""

    def __init__(
        self,
        bm25_path: str = "kb/bm25.pkl",
        dense_embeddings_path: str = "kb/dense_embeddings.npy",
        dense_codes_path: str = "kb/dense_codes.json",
        model_path: str = "models/all-MiniLM-L6-v2",
    ):
        self.bm25_retriever = BM25Retriever(bm25_path)
        self.dense_retriever = DenseRetriever(dense_embeddings_path, dense_codes_path, model_path)

    def retrieve(
        self,
        primary_query: str,
        fallback_query: str = "",
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """Retrieve candidates by querying BM25 and dense with primary and fallback queries."""
        ranked_lists = []
        weights = []

        # 1. Primary query (extracted diagnostic phrases)
        if primary_query and primary_query.strip():
            bm25_primary = self.bm25_retriever.retrieve(primary_query, top_k=top_k)
            dense_primary = self.dense_retriever.retrieve(primary_query, top_k=top_k)

            ranked_lists.append(bm25_primary)
            weights.append(1.5)

            ranked_lists.append(dense_primary)
            weights.append(1.5)

        # 2. Fallback query (filtered full note text)
        if fallback_query and fallback_query.strip() and fallback_query != primary_query:
            bm25_fb = self.bm25_retriever.retrieve(fallback_query, top_k=top_k)
            dense_fb = self.dense_retriever.retrieve(fallback_query, top_k=top_k)

            ranked_lists.append(bm25_fb)
            weights.append(1.0)

            ranked_lists.append(dense_fb)
            weights.append(1.0)

        # Fuse all ranked lists
        fused = reciprocal_rank_fusion(ranked_lists, weights=weights, k=60, top_k=top_k)
        return fused

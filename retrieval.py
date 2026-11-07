"""Retrieval system combining BM25, Dense embeddings, and Reciprocal Rank Fusion (RRF).
Operates completely offline with precomputed indexes and local sentence-transformers.
Includes robust fallback: if neural parts fail to load, gracefully degrades to BM25-only.
"""

import hashlib
import json
import os
import pickle
import sys

import numpy as np

from text_utils import clean_text, expand_abbreviations, tokenize


class BM25Retriever:
    """Fast sparse lexical retrieval using BM25Okapi."""

    def __init__(self, bm25_path: str = "kb/bm25.pkl", kb_path: str = "kb/enriched_kb.json"):
        loaded = False
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, "rb") as f:
                    artifact = pickle.load(f)
                self.bm25 = artifact["bm25"]
                self.codes = artifact["codes"]
                loaded = True
            except Exception as e:
                sys.stderr.write(f"[WARN] Failed to unpickle {bm25_path}: {e}. Rebuilding BM25 in memory...\n")

        if not loaded:
            if not os.path.exists(kb_path):
                raise FileNotFoundError(f"Neither {bm25_path} nor {kb_path} could be found.")
            with open(kb_path, "r", encoding="utf-8") as f:
                kb = json.load(f)
            corpus = [info.get("search_text", info.get("title", "")) for info in kb.values()]
            self.codes = list(kb.keys())
            tokenized_corpus = [tokenize(doc) for doc in corpus]
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
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


PINNED_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PINNED_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EXPECTED_DIM = 384

DEFAULT_DENSE_MODEL = (
    "models/paraphrase-multilingual-MiniLM-L12-v2"
    if os.path.exists("models/paraphrase-multilingual-MiniLM-L12-v2")
    else PINNED_MODEL_ID
)


class DenseRetriever:
    """Semantic vector retrieval using dense sentence embeddings and cosine similarity."""

    def __init__(
        self,
        embeddings_path: str = "kb/dense_embeddings.npz",
        codes_path: str = "kb/dense_codes.json",
        model_name_or_path: str = DEFAULT_DENSE_MODEL,
    ):
        if not os.path.exists(embeddings_path):
            npy_path = embeddings_path.replace(".npz", ".npy")
            if os.path.exists(npy_path):
                embeddings_path = npy_path
            else:
                npz_alt = "kb/dense_embeddings.npz"
                if os.path.exists(npz_alt):
                    embeddings_path = npz_alt
                else:
                    raise FileNotFoundError(f"Dense embeddings not found at {embeddings_path}")

        if not os.path.exists(codes_path):
            raise FileNotFoundError(f"Dense codes mapping not found at {codes_path}")

        with open(codes_path, "r", encoding="utf-8") as f:
            self.codes = json.load(f)

        expected_checksum = hashlib.sha256("".join(self.codes).encode("utf-8")).hexdigest()

        if embeddings_path.endswith(".npz"):
            npz = np.load(embeddings_path)
            self.embeddings = npz["embeddings"]

            # Consistency guard checks
            meta_model_id = str(npz["model_id"]) if "model_id" in npz else None
            meta_dim = int(npz["embedding_dim"]) if "embedding_dim" in npz else None
            meta_checksum = str(npz["codes_checksum"]) if "codes_checksum" in npz else None

            errors = []
            if meta_model_id is None or meta_model_id != PINNED_MODEL_ID:
                errors.append(f"Model ID mismatch: expected {PINNED_MODEL_ID}, got {meta_model_id}")
            if meta_dim is None or meta_dim != EXPECTED_DIM or self.embeddings.shape[1] != EXPECTED_DIM:
                errors.append(f"Dimension mismatch: expected {EXPECTED_DIM}, got meta={meta_dim}, shape={self.embeddings.shape[1]}")
            if meta_checksum is None or meta_checksum != expected_checksum:
                errors.append(f"Codes checksum mismatch: expected {expected_checksum}, got {meta_checksum}")

            if errors:
                err_msg = "Embedding consistency guard failed: " + "; ".join(errors)
                if os.environ.get("STRICT") == "1":
                    raise RuntimeError(err_msg)
                else:
                    sys.stderr.write(f"[WARN] {err_msg}. Refusing dense embeddings and degrading to BM25.\n")
                    raise ValueError(err_msg)
        else:
            if os.environ.get("STRICT") == "1":
                raise RuntimeError(f"Dense embeddings must be an .npz file with consistency metadata: {embeddings_path}")
            sys.stderr.write(f"[WARN] Dense embeddings at {embeddings_path} lacks metadata. Refusing dense.\n")
            raise ValueError("Missing .npz metadata")

        from sentence_transformers import SentenceTransformer
        if not os.path.exists(model_name_or_path):
            model_name_or_path = PINNED_MODEL_ID
        self.model = SentenceTransformer(model_name_or_path)

    def retrieve(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Retrieve top-K semantic nearest neighbors for a query string."""
        if not query or not query.strip():
            return []

        clean_q = clean_text(expand_abbreviations(query))
        query_vec = self.model.encode(
            clean_q,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        similarities = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            results.append((self.codes[idx], sim))
        return results


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int = 50,
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[str, float] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank_idx, (code, _) in enumerate(ranked_list):
            rank = rank_idx + 1
            rrf_score = weight / (k + rank)
            scores[code] = scores.get(code, 0.0) + rrf_score

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


class HybridRetriever:
    """Integrated hybrid retriever that combines BM25 and Dense with RRF.
    Safely degrades to BM25-only if neural dense retrieval fails.
    """

    def __init__(
        self,
        bm25_path: str = "kb/bm25.pkl",
        dense_embeddings_path: str = "kb/dense_embeddings.npz",
        dense_codes_path: str = "kb/dense_codes.json",
        model_path: str = DEFAULT_DENSE_MODEL,
        kb_path: str = "kb/enriched_kb.json",
    ):
        self.bm25_retriever = BM25Retriever(bm25_path, kb_path=kb_path)
        self.dense_available = False
        self.dense_retriever = None

        try:
            self.dense_retriever = DenseRetriever(dense_embeddings_path, dense_codes_path, model_path)
            self.dense_available = True
        except Exception as e:
            if os.environ.get("STRICT") == "1":
                raise
            sys.stderr.write(f"[WARN] Neural dense retriever failed to initialize: {e}. Degraded to BM25-only.\n")

    def retrieve(
        self,
        primary_query: str,
        fallback_query: str = "",
        raw_query: str = "",
        top_k: int = 50,
        use_bm25: bool = True,
        use_dense: bool = True,
    ) -> list[tuple[str, float]]:
        """Retrieve candidates by querying BM25 and dense with primary, fallback, and raw multilingual queries."""
        ranked_lists = []
        weights = []

        # 1. Primary query
        if primary_query and primary_query.strip():
            if use_bm25:
                bm25_primary = self.bm25_retriever.retrieve(primary_query, top_k=top_k)
                ranked_lists.append(bm25_primary)
                weights.append(1.5)

            if use_dense and self.dense_available and self.dense_retriever:
                try:
                    dense_primary = self.dense_retriever.retrieve(primary_query, top_k=top_k)
                    ranked_lists.append(dense_primary)
                    weights.append(1.5)
                except Exception as e:
                    sys.stderr.write(f"[WARN] Dense retrieval error on primary query: {e}\n")

        # 2. Raw multilingual query (native language embedded by multilingual sentence transformer)
        if raw_query and raw_query.strip() and raw_query != primary_query:
            if use_dense and self.dense_available and self.dense_retriever:
                try:
                    dense_raw = self.dense_retriever.retrieve(raw_query, top_k=top_k)
                    ranked_lists.append(dense_raw)
                    weights.append(1.2)
                except Exception as e:
                    sys.stderr.write(f"[WARN] Dense retrieval error on raw query: {e}\n")

        # 3. Fallback query
        if fallback_query and fallback_query.strip() and fallback_query != primary_query:
            if use_bm25:
                bm25_fb = self.bm25_retriever.retrieve(fallback_query, top_k=top_k)
                ranked_lists.append(bm25_fb)
                weights.append(1.0)

            if use_dense and self.dense_available and self.dense_retriever:
                try:
                    dense_fb = self.dense_retriever.retrieve(fallback_query, top_k=top_k)
                    ranked_lists.append(dense_fb)
                    weights.append(1.0)
                except Exception as e:
                    sys.stderr.write(f"[WARN] Dense retrieval error on fallback query: {e}\n")

        if not ranked_lists:
            if use_bm25:
                return self.bm25_retriever.retrieve(primary_query or fallback_query, top_k=top_k)
            return []

        return reciprocal_rank_fusion(ranked_lists, weights=weights, k=60, top_k=top_k)

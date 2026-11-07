"""Unit test for dense embedding consistency guard.

Verifies:
1. kb/dense_embeddings.npz contains required metadata fields (model_id, revision, embedding_dim, codes_checksum).
2. Metadata matches the pinned model configuration and codes checksum.
3. Re-encoding search text for random codes with SentenceTransformer matches stored vectors with cosine > 0.99.
"""

import hashlib
import json
import os
import random
import unittest
import numpy as np

PINNED_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PINNED_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EXPECTED_DIM = 384


class TestEmbeddingConsistency(unittest.TestCase):
    """Test suite ensuring offline embeddings match the pinned model and code mapping."""

    @classmethod
    def setUpClass(cls):
        cls.npz_path = "kb/dense_embeddings.npz"
        cls.codes_path = "kb/dense_codes.json"
        cls.kb_path = "kb/enriched_kb.json"

        if not os.path.exists(cls.npz_path) or not os.path.exists(cls.codes_path):
            raise unittest.SkipTest("Dense embeddings or codes file not found.")

        cls.npz_data = np.load(cls.npz_path)
        cls.embeddings = cls.npz_data["embeddings"]

        with open(cls.codes_path, "r", encoding="utf-8") as f:
            cls.codes = json.load(f)

    def test_metadata_fields_present_and_valid(self):
        """Metadata keys must be present and match pinned configuration."""
        self.assertIn("model_id", self.npz_data)
        self.assertIn("revision", self.npz_data)
        self.assertIn("embedding_dim", self.npz_data)
        self.assertIn("codes_checksum", self.npz_data)

        self.assertEqual(str(self.npz_data["model_id"]), PINNED_MODEL_ID)
        self.assertEqual(str(self.npz_data["revision"]), PINNED_REVISION)
        self.assertEqual(int(self.npz_data["embedding_dim"]), EXPECTED_DIM)
        self.assertEqual(self.embeddings.shape[1], EXPECTED_DIM)

        expected_checksum = hashlib.sha256("".join(self.codes).encode("utf-8")).hexdigest()
        self.assertEqual(str(self.npz_data["codes_checksum"]), expected_checksum)

    def test_reencode_cosine_similarity(self):
        """Re-encoding 20 random codes' search text must yield cosine similarity > 0.99."""
        if not os.path.exists(self.kb_path):
            raise unittest.SkipTest("Enriched KB not found.")

        with open(self.kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)

        from sentence_transformers import SentenceTransformer
        from text_utils import clean_text, expand_abbreviations

        model_path = (
            "models/paraphrase-multilingual-MiniLM-L12-v2"
            if os.path.exists("models/paraphrase-multilingual-MiniLM-L12-v2")
            else PINNED_MODEL_ID
        )
        model = SentenceTransformer(model_path)

        # Select 20 deterministic pseudo-random codes
        rng = random.Random(42)
        sample_indices = rng.sample(range(len(self.codes)), 20)

        for idx in sample_indices:
            code = self.codes[idx]
            self.assertIn(code, kb, f"Code {code} missing from enriched KB")
            entry = kb[code]
            title = entry.get("title", "")
            search_text = entry.get("search_text", "")
            combined = f"{code} {title}. {search_text}"
            expanded = expand_abbreviations(combined)
            full_text = clean_text(f"{combined} {expanded}")
            self.assertTrue(bool(full_text), f"Full text for {code} is empty")

            vec = model.encode(
                full_text,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)

            stored_vec = self.embeddings[idx]
            # Verify normalized
            stored_norm = float(np.linalg.norm(stored_vec))
            self.assertAlmostEqual(stored_norm, 1.0, places=4)

            cosine_sim = float(np.dot(vec, stored_vec))
            self.assertGreater(
                cosine_sim,
                0.99,
                f"Cosine similarity for code {code} is {cosine_sim:.4f}, expected > 0.99",
            )


if __name__ == "__main__":
    unittest.main()

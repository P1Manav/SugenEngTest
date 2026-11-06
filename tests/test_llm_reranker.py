"""Unit tests for LLMReranker: gating, cost guard, candidate restriction, and fallback.
Verifies Task 4 requirements.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_reranker import LLMReranker


class TestLLMReranker(unittest.TestCase):
    def setUp(self):
        self.valid_codes = {"1A00", "1F40.Z", "CA23.00", "BA00.Z"}
        self.kb = {
            "1A00": {"title": "Cholera", "definition": "Vibrio cholerae"},
            "1F40.Z": {"title": "Malaria", "definition": "Plasmodium"},
            "CA23.00": {"title": "Allergic asthma", "definition": "Asthma"},
            "BA00.Z": {"title": "Hypertension", "definition": "High BP"},
        }

    def test_disabled_by_default(self):
        """When USE_LLM_RERANK is not '1', LLMReranker must be disabled with zero overhead."""
        with patch.dict(os.environ, {"USE_LLM_RERANK": "0"}, clear=False):
            reranker = LLMReranker(self.valid_codes, self.kb)
            self.assertFalse(reranker.is_enabled())
            cands = ["1A00", "1F40.Z", "CA23.00"]
            res = reranker.rerank_note("note-1", "Fever and chills", None, cands)
            self.assertEqual(res, ["1A00", "1F40.Z", "CA23.00"])

    def test_spend_cap_guard(self):
        """When spend reaches or exceeds MAX_SPEND_USD ($4.00), reranker disables itself."""
        reranker = LLMReranker(self.valid_codes, self.kb)
        reranker.provider = "openai"
        reranker.client = MagicMock()
        self.assertTrue(reranker.is_enabled())

        # Exceed budget
        reranker.total_cost_usd = 4.05
        self.assertFalse(reranker.is_enabled())
        # Call should immediately return baseline candidates
        cands = ["1A00", "1F40.Z"]
        res = reranker.rerank_note("note-2", "Text", None, cands)
        self.assertEqual(res, ["1A00", "1F40.Z"])

    def test_candidate_restriction_and_hallucination_filtering(self):
        """Model must not emit codes outside the candidate list or outside valid_codes."""
        reranker = LLMReranker(self.valid_codes, self.kb)
        reranker.provider = "openai"
        mock_client = MagicMock()
        reranker.client = mock_client

        # Mock LLM returning one valid candidate, one hallucinated code, and one not in candidates
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({"codes": ["CA23.00", "FAKE_CODE", "BA00.Z"]})
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_response

        # Candidates provided to reranker (notice BA00.Z is NOT in candidate_codes)
        candidate_codes = ["1A00", "CA23.00", "1F40.Z"]
        res = reranker.rerank_note("note-3", "Asthma symptoms", None, candidate_codes)

        # "CA23.00" is accepted. "FAKE_CODE" rejected (not in valid_codes or candidates).
        # "BA00.Z" rejected (not in candidate_codes).
        # Backfilled with remaining valid candidates: "1A00", "1F40.Z"
        self.assertEqual(res, ["CA23.00", "1A00", "1F40.Z"])
        self.assertEqual(len(res), 3)

    def test_graceful_fallback_on_api_error(self):
        """On API failure (network error, rate limit, timeout), return baseline candidates without error."""
        reranker = LLMReranker(self.valid_codes, self.kb)
        reranker.provider = "openai"
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("OpenAI 500 Internal Server Error")
        reranker.client = mock_client

        candidate_codes = ["1A00", "1F40.Z", "CA23.00"]
        res = reranker.rerank_note("note-4", "Text", None, candidate_codes)
        self.assertEqual(res, ["1A00", "1F40.Z", "CA23.00"])
        self.assertEqual(reranker.fallback_count, 1)


if __name__ == "__main__":
    unittest.main()

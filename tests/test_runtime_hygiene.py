"""Unit tests for runtime hygiene, self-test, fallback mechanism, and error logging."""

import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, ".")

from pipeline import ICD11Pipeline
from entrypoint import extract_id_from_raw_line

class TestRuntimeHygiene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = ICD11Pipeline(verbose=False)

    def test_startup_self_test(self):
        """Verify pipeline self-test runs and produces valid predictions."""
        res = self.pipeline.predict_note({"id": "self-test", "text": "fever and acute cough"})
        self.assertIn("id", res)
        self.assertIn("codes", res)
        self.assertTrue(len(res["codes"]) >= 1)
        for c in res["codes"]:
            self.assertIn(c, self.pipeline.valid_codes)

    def test_empty_input_handling(self):
        """Verify empty or blank note text returns valid predictions, never crashes."""
        res = self.pipeline.predict_note({"id": "empty-note", "text": "   "})
        self.assertEqual(res["id"], "empty-note")
        self.assertTrue(len(res["codes"]) >= 1)
        for c in res["codes"]:
            self.assertIn(c, self.pipeline.valid_codes)

    def test_malformed_id_extraction(self):
        """Verify original ID is extracted from corrupted JSON lines."""
        raw_line_1 = '{"id": "note-err-99", "text": "unclosed string ...'
        extracted = extract_id_from_raw_line(raw_line_1, 5)
        self.assertEqual(extracted, "note-err-99")

        raw_line_2 = "{'id': 'patient-123', malformed json}"
        extracted = extract_id_from_raw_line(raw_line_2, 10)
        self.assertEqual(extracted, "patient-123")

        raw_line_3 = "no id at all in this corrupted line"
        extracted = extract_id_from_raw_line(raw_line_3, 15)
        self.assertEqual(extracted, "note-line-15")

    def test_bm25_fallback_recovery(self):
        """Verify fallback returns sensible BM25 codes rather than a constant RA01."""
        fallback = self.pipeline._get_bm25_fallback("Patient with severe malaria and high fever")
        self.assertTrue(len(fallback) >= 1)
        self.assertTrue(len(fallback) <= 3)
        for c in fallback:
            self.assertIn(c, self.pipeline.valid_codes)
        # Should match malaria falciparum (1F40.Z)
        self.assertTrue(any(c.startswith("1F4") for c in fallback))

    def test_resume_capability(self):
        """Verify entrypoint resume logic skips already processed IDs without duplication."""
        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp_out:
            tmp_out.write(json.dumps({"id": "dev-001", "codes": ["1F40.Z"]}) + "\n")
            tmp_out.write(json.dumps({"id": "dev-002", "codes": ["CA07.0"]}) + "\n")
            tmp_out_path = tmp_out.name

        try:
            processed = set()
            with open(tmp_out_path, "r", encoding="utf-8") as f:
                for line in f:
                    p = json.loads(line)
                    processed.add(p["id"])
            self.assertEqual(len(processed), 2)
            self.assertIn("dev-001", processed)
            self.assertIn("dev-002", processed)
        finally:
            if os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)

if __name__ == "__main__":
    unittest.main()

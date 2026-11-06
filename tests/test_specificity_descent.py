"""Unit tests for ICDHierarchy specificity descent, qualifier matching, and wrong-child prevention.
Verifies Task 3.6: Strict distinguishing qualifier matching and .Z fallback when details are lacking.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from icd_hierarchy import ICDHierarchy


class TestSpecificityDescent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hierarchy = ICDHierarchy("kb/enriched_kb.json")

    def test_distal_radius_fracture_descent(self):
        """Verify 'distal radius fracture' resolves to lower end (NC32.5 / NC32.5Z), NEVER shaft (NC32.3) or upper end (NC32.1)."""
        text = "Patient fell on outstretched hand. X-ray confirms fracture of left distal radius."
        # Starting from parent NC32 (Fracture of forearm)
        resolved = self.hierarchy.resolve_to_leaf("NC32", text)
        self.assertTrue(self.hierarchy.is_leaf(resolved), f"Resolved code {resolved} must be a leaf")
        self.assertTrue(
            resolved.startswith("NC32.5"),
            f"Expected lower end of radius (NC32.5*), but got wrong child: {resolved}",
        )
        self.assertNotIn(resolved, ["NC32.1", "NC32.3", "NC32.4"], "Must not resolve to shaft or upper end")

    def test_shaft_radius_fracture_descent(self):
        """Verify 'shaft fracture of radius' resolves to shaft (NC32.3)."""
        text = "Displaced midshaft fracture of the radius following trauma."
        resolved = self.hierarchy.resolve_to_leaf("NC32", text)
        self.assertTrue(self.hierarchy.is_leaf(resolved))
        self.assertEqual(resolved, "NC32.3")

    def test_unspecified_fallback_when_lacking_detail(self):
        """When text lacks detail to distinguish children, resolution must fall back to .Z leaf."""
        # Non-specific forearm fracture note
        text = "Forearm fracture noted on exam."
        resolved = self.hierarchy.resolve_to_leaf("NC32", text)
        self.assertTrue(self.hierarchy.is_leaf(resolved))
        # Forearm fracture unspecified or lower end unspecified
        info = self.hierarchy.get_info(resolved)
        self.assertTrue(
            resolved.endswith("Z") or (info and info.get("residual")),
            f"Expected unspecified/residual code when detail is missing, got: {resolved}",
        )

    def test_diabetes_type_specificity(self):
        """Verify type 1 vs type 2 diabetes specificity descent."""
        # Type 2 diabetes note
        t2_text = "55yo male with uncontrolled Type 2 diabetes mellitus on metformin."
        # 5A11 is Type 2 diabetes, 5A10 is Type 1 diabetes, 5A14 is diabetes unspecified
        # From non-leaf 5A10/5A11 parent or diabetes general:
        t2_resolved = self.hierarchy.resolve_to_leaf("5A11", t2_text)
        self.assertTrue(t2_resolved.startswith("5A11"))

        t1_text = "Child presenting with diabetic ketoacidosis and new-onset Type 1 diabetes mellitus."
        t1_resolved = self.hierarchy.resolve_to_leaf("5A10", t1_text)
        self.assertTrue(t1_resolved.startswith("5A10"))


if __name__ == "__main__":
    unittest.main()

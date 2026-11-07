"""Unit tests for data-driven clinical concept rules (rules/clinical_concepts.yaml).
Tests concept extraction and qualifier normalization using strictly NON-DEV clinical examples.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from clinical_concepts import extract_clinical_qualifiers, load_concept_rules
from icd_hierarchy import ICDHierarchy


class TestClinicalConceptRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_concept_rules("rules/clinical_concepts.yaml")
        cls.hierarchy = ICDHierarchy("kb/enriched_kb.json")

    def test_rules_schema_and_justifications(self):
        """Verify every concept group has a one-line justification and rules."""
        self.assertTrue(len(self.rules) >= 5, "Must define at least 5 concept categories")
        for concept_group, data in self.rules.items():
            self.assertIn("justification", data, f"Group {concept_group} must have justification")
            self.assertTrue(len(data["justification"]) > 10, f"Group {concept_group} justification too short")
            self.assertIn("rules", data, f"Group {concept_group} must have rules list")
            for r in data["rules"]:
                self.assertIn("concept", r)
                self.assertIn("qualifier_terms", r)

    def test_blood_pressure_phenotypes_non_dev(self):
        """Verify blood pressure phenotype extraction on non-dev synthetic cases."""
        # Case 1: Combined hypertension (e.g. 168/102 mmHg)
        note_1 = "58-year-old postal worker with routine checkup. Office BP measured 168/102 mmHg on right arm."
        q1 = extract_clinical_qualifiers(note_1)
        self.assertIn("combined diastolic and systolic", q1)

        # Case 2: Isolated systolic hypertension in elderly (e.g. 174/78 mmHg)
        note_2 = "76yo retired teacher evaluated for lightheadedness. Seated BP is 174/78 mmHg."
        q2 = extract_clinical_qualifiers(note_2)
        self.assertIn("isolated systolic", q2)

        # Case 3: Isolated diastolic hypertension in young adult (e.g. 128/96 mmHg)
        note_3 = "34yo delivery courier undergoing pre-employment screening. Resting BP 128/96 mmHg."
        q3 = extract_clinical_qualifiers(note_3)
        self.assertIn("isolated diastolic", q3)

    def test_obstetric_parity_non_dev(self):
        """Verify obstetric parity extraction on non-dev synthetic cases."""
        # Non-dev Case 1: Primigravida G1P0
        note_1 = "Initial booking visit at district hospital. Patient is a 24-year-old primigravida at 10 weeks gestation."
        q1 = extract_clinical_qualifiers(note_1)
        self.assertIn("first pregnancy", q1)

        # Non-dev Case 2: G1P0 abbreviation
        note_2 = "Routine antenatal check: 21F G1P0 presenting with mild morning sickness."
        q2 = extract_clinical_qualifiers(note_2)
        self.assertIn("first pregnancy", q2)

        # Non-dev Case 3: Multipara G3P2
        note_3 = "32yo multigravida G3P2 at 34 weeks gestation presenting with dependent leg oedema."
        q3 = extract_clinical_qualifiers(note_3)
        self.assertIn("subsequent pregnancy", q3)

    def test_angina_presentation_non_dev(self):
        """Verify angina stability extraction on non-dev synthetic cases."""
        # Non-dev Case 1: Stable exertional angina
        note_1 = "61-year-old accountant complains of retrosternal tightness on climbing stairs, relieved by rest within 3 minutes."
        q1 = extract_clinical_qualifiers(note_1)
        self.assertIn("stable", q1)

        # Non-dev Case 2: Unstable rest angina
        note_2 = "59-year-old smoker reports severe squeezing chest pain at rest that started 40 minutes ago."
        q2 = extract_clinical_qualifiers(note_2)
        self.assertIn("unstable", q2)

    def test_food_allergy_triggers_non_dev(self):
        """Verify food allergy triggers extraction on non-dev synthetic cases."""
        # Non-dev Case 1: Peanut ingestion
        note_1 = "14-year-old adolescent with sudden facial angioedema after eating peanut butter sandwich."
        q1 = extract_clinical_qualifiers(note_1)
        self.assertIn("food", q1)

        # Non-dev Case 2: Seafood/shellfish ingestion
        note_2 = "40yo diner presenting with acute urticaria and wheezing after consuming seafood paella."
        q2 = extract_clinical_qualifiers(note_2)
        self.assertIn("food", q2)

    def test_contact_sensitizers_non_dev(self):
        """Verify metal contact allergen extraction on non-dev synthetic cases."""
        # Non-dev Case 1: Nickel jewelry/earring
        note_1 = "19F student with pruritic vesicular erythematous patch beneath metal earrings containing nickel."
        q1 = extract_clinical_qualifiers(note_1)
        self.assertIn("metals", q1)

        # Non-dev Case 2: Belt buckle
        note_2 = "27M carpenter presenting with localized periumbilical eczema directly beneath metal belt buckle."
        q2 = extract_clinical_qualifiers(note_2)
        self.assertIn("metals", q2)

    def test_respiratory_exacerbation_non_dev(self):
        """Verify acute exacerbation extraction on non-dev synthetic cases."""
        note_1 = "66M heavy smoker with chronic bronchitis presenting with acute flare, increased breathlessness and purulent sputum."
        q1 = extract_clinical_qualifiers(note_1)
        self.assertIn("exacerbation", q1)

    def test_hierarchy_descent_with_concept_rules_non_dev(self):
        """Verify that hierarchy descent resolves correctly using concept qualifiers without hardcoded codes."""
        # Non-dev hypertension case with combined reading
        note_htn = "52-year-old librarian diagnosed with essential hypertension. Repeated clinic readings 166/104 mmHg."
        resolved_htn = self.hierarchy.resolve_to_leaf("BA00", note_htn)
        self.assertEqual(resolved_htn, "BA00.0", "Should resolve to Combined diastolic and systolic hypertension")

        # Non-dev food anaphylaxis case
        note_ana = "8yo girl with acute systemic anaphylactic reaction after ingesting peanut cookies."
        resolved_ana = self.hierarchy.resolve_to_leaf("4A84", note_ana)
        self.assertEqual(resolved_ana, "4A84.0", "Should resolve to Anaphylaxis due to food")

        # Non-dev first pregnancy antenatal visit
        note_anc = "Healthy 20yo primigravida attending first routine antenatal consultation at 8 weeks."
        resolved_anc = self.hierarchy.resolve_to_leaf("QA42", note_anc)
        self.assertEqual(resolved_anc, "QA42.0", "Should resolve to Supervision of normal first pregnancy")


if __name__ == "__main__":
    unittest.main()


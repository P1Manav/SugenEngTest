"""Unit tests for AssertionFilter and Negation across all three clinical styles:
1. typed_consult (5 cases)
2. short_consult (5 cases)
3. ambient_transcript (5 cases)

Verifies that negated findings, ruled-out differentials, and family history
are strictly excluded while active primary diagnoses are preserved.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from assertion_filter import AssertionFilter
from pipeline import ICD11Pipeline


class TestAssertionStyles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filter = AssertionFilter()
        cls.pipeline = ICD11Pipeline(verbose=False)

    # -------------------------------------------------------------------------
    # 1. Typed Consult Style (5 cases)
    # -------------------------------------------------------------------------
    def test_typed_consult_tb_negation(self):
        """Typed: Sputum AFB negative, rule out TB -> TB must not be primary diagnosis."""
        text = "42M c/o cough x3 weeks. Chest X-ray clear. Sputum AFB negative x3. Rule out pulmonary tuberculosis. Assessment: Acute bronchitis."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("tuberculosis" in r.lower() for r in rejected))
        self.assertTrue(self.filter.is_candidate_negated_in_note("Respiratory tuberculosis", text))
        res = self.pipeline.predict_note({"id": "test-tc-1", "text": text, "style": "typed_consult"})
        self.assertNotIn("1B10.0", res["codes"][:1])
        self.assertNotIn("1B10.Z", res["codes"][:1])

    def test_typed_consult_cardiac_denial(self):
        """Typed: Patient denies chest pain/SOB, confirms peptic ulcer disease."""
        text = "50M. Patient denies chest pain, denies shortness of breath. Complains of epigastric burning pain x1 month, worse fasting. Impression: Peptic ulcer disease."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("chest pain" in r.lower() for r in rejected))
        res = self.pipeline.predict_note({"id": "test-tc-2", "text": text, "style": "typed_consult"})
        self.assertTrue(any(c.startswith("DA6") for c in res["codes"]))

    def test_typed_consult_family_history_diabetes(self):
        """Typed: Mother had diabetes, patient normoglycemic, headache primary."""
        text = "28F. Family history: mother with type 2 diabetes mellitus. Patient fasting glucose 4.9 mmol/L (normal). Today complains of severe tension-type headache x2 days. Dx: Tension headache."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("diabetes" in r.lower() for r in rejected))
        res = self.pipeline.predict_note({"id": "test-tc-3", "text": text, "style": "typed_consult"})
        self.assertFalse(any(c.startswith("5A11") for c in res["codes"]))
        self.assertTrue(any(c.startswith("8A8") for c in res["codes"]))

    def test_typed_consult_malaria_ruled_out(self):
        """Typed: mRDT negative for malaria, confirmed acute cystitis."""
        text = "31F c/o fever, dysuria, and urinary frequency x3 days. Blood smear and mRDT negative for malaria. Urinalysis reveals 3+ leukocytes and nitrites. Assessment: Acute cystitis."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("malaria" in r.lower() for r in rejected))
        self.assertTrue(self.filter.is_candidate_negated_in_note("Malaria", text))
        res = self.pipeline.predict_note({"id": "test-tc-4", "text": text, "style": "typed_consult"})
        self.assertFalse(any(c.startswith("1F4") for c in res["codes"]))

    def test_typed_consult_preeclampsia_excluded(self):
        """Typed: Pre-eclampsia excluded, normal pregnancy antenatal care."""
        text = "26F G2P1 at 28 weeks gestation. BP 112/72 mmHg. Urine protein nil. No headache, no blurred vision. Pre-eclampsia excluded. Routine supervision of pregnancy."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("pre-eclampsia" in r.lower() for r in rejected))
        res = self.pipeline.predict_note({"id": "test-tc-5", "text": text, "style": "typed_consult"})
        self.assertFalse(any(c.startswith("JA24") for c in res["codes"]))

    # -------------------------------------------------------------------------
    # 2. Short Consult Style (5 cases)
    # -------------------------------------------------------------------------
    def test_short_consult_coryza_denied(self):
        """Short: Denies cough or rhinorrhea, positive strep pharyngitis."""
        text = "19F c/o severe sore throat, fever 38.5C. Exudative pharyngitis. Denies cough or coryza. Rapid Strep test positive. Dx: Streptococcal pharyngitis."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("cough" in r.lower() or "coryza" in r.lower() for r in rejected))
        res = self.pipeline.predict_note({"id": "test-sc-1", "text": text, "style": "short_consult"})
        self.assertTrue(any(c.startswith("CA0") or c.startswith("1C1") for c in res["codes"]))

    def test_short_consult_pregnancy_negated(self):
        """Short: UPT negative, acute uncomplicated cystitis."""
        text = "24F with acute burning micturition x2 days. UPT negative. No vaginal discharge. Urine dip positive for nitrite. Dx: Acute cystitis."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("upt negative" in r.lower() or "negative" in r.lower() for r in rejected))
        res = self.pipeline.predict_note({"id": "test-sc-2", "text": text, "style": "short_consult"})
        self.assertFalse(any(c.startswith(("JA", "JB")) for c in res["codes"][:1]))

    def test_short_consult_fracture_ruled_out(self):
        """Short: X-ray reveals no fracture, knee osteoarthritis confirmed."""
        text = "62M with chronic right knee pain. X-ray: no fracture or dislocation; shows medial joint space narrowing. Dx: Knee osteoarthritis."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("fracture" in r.lower() for r in rejected))
        self.assertTrue(self.filter.is_candidate_negated_in_note("Fracture of femur", text))
        res = self.pipeline.predict_note({"id": "test-sc-3", "text": text, "style": "short_consult"})
        self.assertTrue(any(c.startswith("FA01") for c in res["codes"]))
        self.assertFalse(any(c.startswith("NC") for c in res["codes"]))

    def test_short_consult_asthma_denied(self):
        """Short: Denies asthma or food allergies, contact dermatitis to nickel buckle."""
        text = "29M with itchy papules on abdomen. Denies asthma, denies food allergy. Rash directly under nickel belt buckle. Dx: Allergic contact dermatitis due to metals."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("asthma" in r.lower() for r in rejected))
        self.assertTrue(self.filter.is_candidate_negated_in_note("Asthma", text))
        res = self.pipeline.predict_note({"id": "test-sc-4", "text": text, "style": "short_consult"})
        self.assertTrue(any(c.startswith("EK00") for c in res["codes"]))

    def test_short_consult_angina_ruled_out(self):
        """Short: Troponin negative, normal ECG, gastro-oesophageal reflux disease."""
        text = "48M with retrosternal burning. ECG normal, serial troponin negative. No exertional component. Endoscopy shows erosive esophagitis. Dx: GERD."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("troponin" in r.lower() for r in rejected))
        res = self.pipeline.predict_note({"id": "test-sc-5", "text": text, "style": "short_consult"})
        self.assertTrue(any(c.startswith("DA2") for c in res["codes"]))
        self.assertFalse(any(c.startswith("BA4") for c in res["codes"][:1]))

    # -------------------------------------------------------------------------
    # 3. Ambient Transcript Style (5 cases)
    # -------------------------------------------------------------------------
    def test_ambient_transcript_dysentery_negated(self):
        """Ambient: Patient confirms no blood in stool, diagnosed with Giardiasis."""
        text = "Doctor: Have you noticed any blood in your stool? Patient: No, never, no blood at all, just loose watery diarrhoea for three days. Doctor: Stool test confirms Giardia cysts. We will start metronidazole for Giardiasis."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("blood" in r.lower() for r in rejected))
        res = self.pipeline.predict_note({"id": "test-amb-1", "text": text, "style": "ambient_transcript"})
        self.assertTrue(any(c.startswith("1A3") for c in res["codes"]))

    def test_ambient_transcript_exertional_angina_denied(self):
        """Ambient: Patient denies pain on exertion, confirmed acid reflux."""
        text = "Doctor: Does the chest pain get worse when you climb stairs? Patient: No, not at all, climbing stairs doesn't cause it. It only happens when I lie down after eating. Doctor: Your ECG is reassuringly normal. You have gastroesophageal acid reflux."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        res = self.pipeline.predict_note({"id": "test-amb-2", "text": text, "style": "ambient_transcript"})
        self.assertTrue(any(c.startswith("DA2") for c in res["codes"]))

    def test_ambient_transcript_wheeze_denied(self):
        """Ambient: Parent confirms no wheezing/breathing trouble, diagnosed with atopic eczema."""
        text = "Doctor: Has he had any wheezing or shortness of breath? Mother: No, his chest is clear, no wheezing at all. But his skin is extremely dry and itchy behind the knees. Doctor: This is an acute flare of atopic dermatitis."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("wheezing" in r.lower() for r in rejected))
        self.assertTrue(self.filter.is_candidate_negated_in_note("Asthma", text) or any("wheezing" in r.lower() for r in rejected))
        res = self.pipeline.predict_note({"id": "test-amb-3", "text": text, "style": "ambient_transcript"})
        self.assertTrue(any(c.startswith("EA80") for c in res["codes"]))
        self.assertFalse(any(c.startswith("CA23") for c in res["codes"][:1]))

    def test_ambient_transcript_pregnancy_denied(self):
        """Ambient: Patient confirms negative pregnancy test, diagnosed with ovarian cyst."""
        text = "Doctor: Is there any possibility of pregnancy? Patient: No, I took a home pregnancy test yesterday and it was negative, my period was last week. Doctor: Pelvic ultrasound demonstrates a simple follicular ovarian cyst."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("pregnancy" in r.lower() for r in rejected))
        res = self.pipeline.predict_note({"id": "test-amb-4", "text": text, "style": "ambient_transcript"})
        self.assertTrue(any(c.startswith("GA1") for c in res["codes"]))
        self.assertFalse(any(c.startswith(("JA", "JB")) for c in res["codes"][:1]))

    def test_ambient_transcript_respiratory_denied(self):
        """Ambient: Denies cough/fever, confirmed severe iron deficiency anaemia."""
        text = "Doctor: Any cough, fever, or night sweats? Patient: No cough and no fever at all, just constant fatigue and cold hands. Doctor: Blood work confirms severe iron deficiency anaemia with Hb 7.1."
        clean_text, rejected = self.filter.filter_text_for_retrieval(text)
        self.assertTrue(any("cough" in r.lower() for r in rejected))
        res = self.pipeline.predict_note({"id": "test-amb-5", "text": text, "style": "ambient_transcript"})
        self.assertTrue(any(c.startswith("3A00") for c in res["codes"]))


if __name__ == "__main__":
    unittest.main()

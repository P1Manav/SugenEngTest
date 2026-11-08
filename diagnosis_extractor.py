"""Rule-based diagnosis extraction from clinical notes.
Tailored for typed consults, short consults, and ambient transcripts.
"""

import re
from typing import List, Dict, Any, Optional

from text_utils import expand_abbreviations, clean_text
from assertion_filter import split_sentences_and_clauses

# Assessment / Impression / Diagnosis line headers (multilingual)
RE_ASSESSMENT_HEADER = re.compile(
    r"(?:^|\n|\b)(?:A|Imp|Dx|Diagnosis|Assessment|Impression|D|Diagnose|Befund|Beurteilung|Diagnóstico|Diagnostico|Diagnostic|Utambuzi|Biópsia|Biopsia|Biopsie|Endoscopia|Endoscopie|Histology|Histopathology|Histopathologie|निदान|डायग्नोसिस|التشخيص|诊断|中医辨证|辨证诊断)\s*[:\-]\s*(.*?)(?=(?:\n\s*(?:P|Plan|Rx|Mx|Ix|S|O|O/E|R/v|Review|Return|Therapie|Tratamiento|Traitement|Plano|Mpango)\s*[:\-]|\Z))",
    re.IGNORECASE | re.DOTALL,
)

# Inline diagnosis markers often seen in short consults without clean newlines
RE_INLINE_DX = re.compile(
    r"(?:\b(?:Dx|Imp|Impression|Diagnosis|Assessment|Diagnose|Befund|Beurteilung|Diagnóstico|Diagnostico|Diagnostic|Utambuzi|Biópsia|Biopsia|Biopsie|Endoscopia|Endoscopie|Histology|Histopathology|Histopathologie|निदान|डायग्नोसिस|التشخيص|诊断|中医辨证|辨证诊断)\b\s*[:\-]?\s*([^\n.;]+)|\b(?:诊断|中医辨证|辨证诊断)\s*[:：]\s*([^\n.;]+))",
    re.IGNORECASE,
)

# Diagnostic test conclusions (e.g. Biopsy confirms squamous cell carcinoma, X-ray confirms fracture)
RE_DIAGNOSTIC_TESTS = re.compile(
    r"\b(?:biopsy|histology|histopathology|cytology|endoscopy|colonoscopy|bronchoscopy|x-ray|xr|ct|mri|ultrasound|uss|echo|culture|smear|pcr|rapid test|antigen test|swab|biopsia|biópsia|biopsie|endoscopie|endoscopia)\s*(?:shows?|reveals?|confirmed?|confirms?|consistent with|reported as|diagnostic of|positive for|showed|revealed|revela|muestra|confirmé|confirme|confirma|mostró)?\s*[:\-]\s*([^\n.;]+)",
    re.IGNORECASE,
)

# Ambient transcript doctor diagnostic statements
RE_DOCTOR_PATTERNS = [
    re.compile(r"\bthis\s+is\s+(?:what\s+we\s+call\s+)?([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\bso\s+this\s+is\s+([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\bi\s+think\s+this\s+is\s+([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\bwhat\s+you\s+have\s+is\s+([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\bwe\s+call\s+it\s+([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\bthe\s+diagnosis\s+is\s+([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\bthis\s+looks\s+like\s+([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\bit's\s+([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\bit\s+is\s+([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\byou\s+have\s+([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\bvalues\s+show\s+([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\bin\s+the\s+([^\n.;:?!–—]+(?:\s+range|\s+category))", re.IGNORECASE),
    re.compile(r"\b(?:the\s+)?([A-Za-z\s]*heart\s+failure[^\n.;:?!–—]*)", re.IGNORECASE),
    re.compile(r"\b(?:the\s+)?([A-Za-z\s]*kidney\s+failure[^\n.;:?!–—]*)", re.IGNORECASE),
    re.compile(r"\b(?:the\s+)?([A-Za-z\s]*liver\s+failure[^\n.;:?!–—]*)", re.IGNORECASE),
    re.compile(r"\b(?:the\s+)?([A-Za-z\s]*respiratory\s+failure[^\n.;:?!–—]*)", re.IGNORECASE),
    re.compile(r"\bproblem\s+is\s+([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\b(?:le\s+diagnostic\s+est|vous\s+avez|il\s+s'agit\s+d'un(?:e)?)\s+([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\b(?:el\s+diagn[oó]stico\s+es|tiene|se\s+trata\s+de\s+(?:un|una)?)\s+([^\n.;:?!–—]+)", re.IGNORECASE),
    re.compile(r"\b(?:huu\s+ni|utambuzi\s+ni|una\s+ugonjwa\s+wa)\s+([^\n.;:?!–—]+)", re.IGNORECASE),
]

# Next-action or treatment boundary words to trim trailing clinician speech in transcripts
TRANSCRIPT_STOPPERS = [
    r"\bwe'll\b",
    r"\bwe\s+will\b",
    r"\bi'll\b",
    r"\bi\s+will\b",
    r"\bi'm\b",
    r"\bi\s+am\b",
    r"\bgive\b",
    r"\btake\b",
    r"\bstart\b",
    r"\brest\b",
    r"\bcome\s+back\b",
    r"\bif\b",
    r"\bfor\s+now\b",
    r"\bso\b",
]
RE_STOPPERS = re.compile(r"|".join(TRANSCRIPT_STOPPERS), re.IGNORECASE)


class DiagnosisExtractor:
    """Extract diagnostic phrases and clean clinical notes."""

    def __init__(self):
        pass

    def extract_from_typed_or_short(self, text: str) -> List[str]:
        """Extract assessment / impression / diagnosis sections from typed/short consults."""
        phrases = []

        # 1. Look for multiline or single line section header
        matches = RE_ASSESSMENT_HEADER.findall(text)
        for m in matches:
            cleaned = clean_text(m)
            # Take only first couple sentences if it bleeds
            first_clause = split_sentences_and_clauses(cleaned)
            if first_clause:
                phrases.append(first_clause[0])
            if cleaned and cleaned not in phrases:
                phrases.append(cleaned)

        # 2. Look for inline Dx: or Imp: patterns
        inline_matches = RE_INLINE_DX.findall(text)
        for im in inline_matches:
            val = im if isinstance(im, str) else next((g for g in im if g), "")
            im_clean = clean_text(val)
            if im_clean and len(im_clean) > 2 and im_clean not in phrases:
                phrases.append(im_clean)

        # 3. Look for diagnostic test conclusions (e.g. Biopsy confirms squamous cell carcinoma)
        test_matches = RE_DIAGNOSTIC_TESTS.findall(text)
        for tm in test_matches:
            val = tm if isinstance(tm, str) else next((g for g in tm if g), "")
            tm_clean = clean_text(val)
            if tm_clean and len(tm_clean) > 3 and tm_clean not in phrases:
                phrases.append(tm_clean)

        # 4. Fallback for consults without explicit section headers:
        # Check closing declarative sentence if it contains a definitive disease statement
        if not phrases:
            sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]
            if sentences:
                last_sent = sentences[-1]
                words = last_sent.split()
                if 2 <= len(words) <= 15:
                    if any(t in last_sent.lower() for t in ["disorder", "disease", "syndrome", "fracture", "carcinoma", "neoplasm", "infection", "depression", "anxiety", "malaria", "pneumonia", "asthma", "hypertension", "diabetes", "bronchitis", "cataract", "scabies", "arthritis", "colles", "angina"]):
                        phrases.append(clean_text(last_sent))

        return phrases

    def extract_from_transcript(self, text: str) -> List[str]:
        """Extract doctor diagnostic statements from ambient transcripts."""
        phrases = []

        # Only look at lines spoken by Doctor
        lines = text.split("\n")
        doctor_lines = []
        for line in lines:
            if re.match(r"^\s*(?:Doctor|Dr|Médecin|Medecin|Médico|Medico|Daktari|Doktor|医生)\s*:", line, re.IGNORECASE):
                doctor_lines.append(line)

        target_text = " ".join(doctor_lines) if doctor_lines else text

        for pattern in RE_DOCTOR_PATTERNS:
            for match in pattern.finditer(target_text):
                phrase = match.group(1).strip()
                # Stop phrase if it hits a treatment or next-step word
                stop_m = RE_STOPPERS.search(phrase)
                if stop_m:
                    phrase = phrase[:stop_m.start()].strip()

                phrase = clean_text(phrase)
                if phrase and len(phrase) > 2 and len(phrase.split()) <= 15:
                    if phrase not in phrases:
                        phrases.append(phrase)

        return phrases

    def extract_diagnoses(self, text: str, style: Optional[str] = None) -> Dict[str, Any]:
        """Extract diagnostic phrases based on note style and content."""
        expanded_text = expand_abbreviations(text)

        phrases = []
        if style == "ambient_transcript" or "Doctor:" in text:
            phrases = self.extract_from_transcript(expanded_text)
            # Fallback to typed check if nothing found
            if not phrases:
                phrases = self.extract_from_typed_or_short(expanded_text)
        else:
            phrases = self.extract_from_typed_or_short(expanded_text)
            if not phrases and ("Doctor:" in text or "Patient:" in text):
                phrases = self.extract_from_transcript(expanded_text)

        # Primary query: join top extracted phrases, or fallback to expanded text
        primary_query = " ".join(phrases) if phrases else expanded_text

        return {
            "phrases": phrases,
            "primary_query": clean_text(primary_query),
            "expanded_text": expanded_text,
        }

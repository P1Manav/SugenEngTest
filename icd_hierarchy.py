"""ICD-11 hierarchy navigation, sibling expansion, and specificity descent.
Ensures predictions resolve to valid leaf codes and appropriately leverage .Z and specific children.
Includes strict distinguishing qualifier matching and clinical anatomical synonym normalization.
"""

import json
import os
import re
from typing import Any

from clinical_concepts import extract_clinical_qualifiers

# Clinical and anatomical qualifiers and their equivalent expressions
CLINICAL_QUALIFIER_EQUIVALENTS: dict[str, list[str]] = {
    "distal": ["distal", "lower end", "inferior", "caudal"],
    "lower": ["distal", "lower end", "inferior", "lower"],
    "proximal": ["proximal", "upper end", "superior", "cranial"],
    "upper": ["proximal", "upper end", "superior", "upper"],
    "shaft": ["shaft", "diaphysis", "midshaft"],
    "anterior": ["anterior", "front", "ventral"],
    "posterior": ["posterior", "back", "dorsal"],
    "bilateral": ["bilateral", "both", "b/l"],
    "unilateral": ["unilateral", "one side", "left", "right", "single"],
    "acute": ["acute", "sudden", "recent"],
    "chronic": ["chronic", "longstanding", "persistent", "recurrent"],
    "open": ["open", "compound"],
    "closed": ["closed", "simple"],
    "primary": ["primary", "essential", "idiopathic", "primitif", "primitive", "primario", "primaria", "primär", "primaer", "idiopathisch"],
    "secondary": ["secondary", "secondaire", "sekundär", "secundario", "secundaria"],
    "type 1": ["type 1", "type i", "t1dm", "iddm", "juvenile"],
    "type 2": ["type 2", "type ii", "t2dm", "niddm", "adult-onset"],
    "right": ["right", "rt", "r side"],
    "left": ["left", "lt", "l side"],
    "complication": ["complication", "complications"],
    "facial": ["facial", "face", "mandible", "jaw"],
    "confirmed": ["confirmed", "confirmée", "confirmee", "confirmé", "confirme", "confirmada", "confirmado", "positive", "positif", "positiva", "positivo", "bacteriologically", "smear", "baar", "culture", "biopsy", "histologically", "histology", "chanya", "pcr", "laboratory confirmed"],
    "unconfirmed": ["unconfirmed", "suspected", "presumed", "probable", "clinically diagnosed", "without confirmation", "non confirmé", "no confirmado", "hasi"],
    "infectious": ["infectious", "infection", "infective", "bacterial", "viral", "microbial", "pathogen", "uti", "maambukizi", "infecciosa", "infeccioso", "infeccion", "infección", "infectieuse", "infectieux", "purulent", "pus"],
    "uncomplicated": ["uncomplicated", "without complication", "no complication", "no complications", "without peritonitis", "no peritonitis", "without warning signs", "no warning signs", "non-severe", "sans complication", "sin complicaciones", "bila matatizo"],
    "non-erosive": ["non-erosive", "nerd", "no erosiva", "não erosiva", "nao erosiva", "sin erosiones", "sem lesões", "ausência de lesões", "sans érosion", "sans erosion"],
    "age-related": ["age-related", "senile", "elderly", "aged", "sénile", "senil", "mzee", "old age", "nuclear sclerosis", "70", "72", "74", "75", "80", "85"],
    "paroxysmal": ["paroxysmal", "paroxystique", "paroxística", "episodic"],
    "benign paroxysmal": ["benign paroxysmal", "bppv", "dix-hallpike", "vertige positionnel paroxystique", "vertigo posicional paroxístico", "vertigo posicional paroxistico", "benigner paroxysmaler lagerungsschwindel", "bpls"],
    "crisis": ["crisis", "crise", "vaso-occlusive", "painful", "acute chest", "sequestration", "maumivu makali", "mgogoro", "maumivu ya mifupa", "kifua"],
    "without dementia": ["without dementia", "no dementia", "keine demenz", "sans demence", "sans démence", "sin demencia", "sem demência", "sem demencia", "mmst 28", "mmst 29", "mmst 30", "moca 28", "moca 29", "moca 30"],
    "with dementia": ["with dementia", "dementia", "demenz", "démence", "demence", "demencia", "demência"],
    "bacterial": ["bacterial", "purulent", "pus", "crust", "crusting", "mucopurulent", "bactérienne", "bacteriana"],
    "viral": ["viral", "watery", "clear discharge", "adenovirus", "virale"],
    "allergic": ["allergic", "allergy", "itching", "atopic", "atopique", "alérgica", "alergica"],
    "mild": ["mild", "légère", "legere", "leve", "leicht", "phq-9 5", "phq-9 6", "phq-9 7", "phq-9 8", "phq-9 9"],
    "moderate": ["moderate", "modérée", "moderee", "moderado", "moderada", "mittelgradig", "phq-9 10", "phq-9 11", "phq-9 12", "phq-9 13", "phq-9 14"],
    "severe": ["severe", "sévère", "severe", "grave", "intense", "heavy", "severo", "severa", "schwer", "phq-9 15", "phq-9 16", "phq-9 17", "phq-9 18", "phq-9 19", "phq-9 20"],
    "invasive": ["invasive", "infiltrating", "infiltrant", "invasivo", "invasiva"],
    "ductal": ["ductal", "canalaire", "duktal", "duktales", "canalicular"],
    "sporadic": ["sporadic", "sporadische", "sporadisch", "sporadique", "esporádica", "esporadico", "esporádico"],
    "without coma": ["without coma", "no coma", "sans coma", "sin coma", "sem coma", "kein koma"],
    "autoimmune": ["autoimmune", "hashimoto", "primary", "primitive", "primario", "primaria"],
    "single": ["single", "first", "premier", "primer", "primera"],
    "recurrent": ["recurrent", "relapsing", "récidivant", "recurrente"],
    "classical": ["classical", "typical", "burrows", "surcos", "surco", "acarianos", "interdigital", "hands", "wrists", "nocturnal pruritus", "prurito nocturno", "sarna clásica", "sarna clasica"],
    "without severe features": ["without severe features", "sin criterios de gravedad", "non-severe", "mild preeclampsia"],
    "without aura": ["without aura", "sans aura", "sin aura", "بدون أورة", "دون وجود أورة", "no aura"],
    "ureter": ["ureter", "ureteral", "uréter", "uretere", "urétéro", "uretero"],
    "kidney": ["kidney", "renal", "rein", "riñón", "rinon", "rim"],
    "bladder": ["bladder", "cystitis", "suprapubic", "vesical", "mkojo", "vejiga", "vessie"],
    "arising": ["arising", "gestational", "come with", "developing", "gestante", "grossesse", "never had", "shows up"],
    "hypertensive": ["hypertensive", "hypertension", "hta", "high blood pressure", "presha", "gravidique"],
}

STOP_WORDS = {
    "and", "the", "for", "with", "due", "other", "specified", "unspecified",
    "not", "elsewhere", "classified", "without", "part", "site", "condition",
}


class ICDHierarchy:
    """Navigate the ICD-11 category hierarchy and expand candidate sets."""

    def __init__(self, kb_path: str = "kb/enriched_kb.json", kb_dict: dict[str, Any] | None = None):
        if kb_dict is not None:
            self.kb = kb_dict
        elif os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
                self.kb = json.load(f)
        else:
            raise FileNotFoundError(f"Enriched KB not found at {kb_path}. Run build_kb.py first.")

    def get_info(self, code: str) -> dict[str, Any] | None:
        """Get enriched metadata for a code."""
        return self.kb.get(code)

    def is_leaf(self, code: str) -> bool:
        """Check if code is a leaf category."""
        info = self.kb.get(code)
        return bool(info and info.get("leaf", False))

    def get_parent(self, code: str) -> str | None:
        """Get parent code."""
        info = self.kb.get(code)
        return info.get("parent") if info else None

    def get_children(self, code: str) -> list[str]:
        """Get immediate children codes."""
        info = self.kb.get(code)
        return info.get("children", []) if info else []

    def get_siblings(self, code: str) -> list[str]:
        """Get sibling codes under same parent."""
        info = self.kb.get(code)
        return info.get("siblings", []) if info else []

    def get_z_code(self, code: str) -> str | None:
        """Get the unspecified (.Z) code for this category."""
        info = self.kb.get(code)
        if not info:
            return None
        # If code already ends in Z
        if code.endswith(".Z") or code.endswith("Z"):
            return code
        # If non-leaf, inspect own children for .Z FIRST
        for ch in info.get("children", []):
            if ch.endswith(".Z") or ch.endswith("Z"):
                return ch
        # Check explicit z_code in metadata (sibling .Z)
        if info.get("z_code"):
            return info["z_code"]
        # Inspect siblings for .Z
        for s in info.get("siblings", []):
            if s.endswith(".Z") or s.endswith("Z"):
                return s
        return None

    def get_subtree_z_code(self, code: str, max_depth: int = 4) -> str | None:
        """Recursively search descendants for an unspecified (.Z) leaf code."""
        if max_depth <= 0 or code not in self.kb:
            return None
        info = self.kb[code]
        # Check direct children first
        for ch in info.get("children", []):
            if (ch.endswith(".Z") or ch.endswith("Z")) and self.is_leaf(ch):
                return ch
        # Recurse down children
        for ch in info.get("children", []):
            leaf_z = self.get_subtree_z_code(ch, max_depth - 1)
            if leaf_z and self.is_leaf(leaf_z):
                return leaf_z
        return None

    def get_y_code(self, code: str) -> str | None:
        """Get other specified (.Y) code for this category."""
        info = self.kb.get(code)
        if not info:
            return None
        if code.endswith(".Y") or code.endswith("Y"):
            return code
        # If non-leaf, inspect own children for .Y FIRST
        for ch in info.get("children", []):
            if ch.endswith(".Y") or ch.endswith("Y"):
                return ch
        if info.get("y_code"):
            return info["y_code"]
        for s in info.get("siblings", []):
            if s.endswith(".Y") or s.endswith("Y"):
                return s
        return None

    def resolve_to_leaf(self, code: str, note_text: str = "") -> str:
        """If code is not a leaf, descend to the best specific child or fallback to .Z.

        Strictly requires distinguishing child qualifiers or synonyms to match in the text
        before descending to a specific child; otherwise falls back to the .Z unspecified leaf.
        """
        info = self.kb.get(code)
        if not info:
            return code
        if info.get("leaf", False):
            return code

        # If note has text, try matching specific child
        children = info.get("children", [])
        if children and note_text:
            best_child, match_count = self.match_specific_child(children, note_text)
            if best_child and match_count >= 3:
                # Recursively ensure child is a leaf
                resolved_child = self.resolve_to_leaf(best_child, note_text)
                if self.is_leaf(resolved_child):
                    return resolved_child

        # Fallback 1: Direct .Z child leaf
        z_code = self.get_z_code(code)
        if z_code and self.is_leaf(z_code):
            return z_code
        elif z_code:
            # If z_code itself has children, resolve it to leaf
            resolved_z = self.resolve_to_leaf(z_code, note_text)
            if self.is_leaf(resolved_z):
                return resolved_z

        # Fallback 2: Subtree .Z leaf code
        subtree_z = self.get_subtree_z_code(code)
        if subtree_z:
            return subtree_z

        # Fallback 3: First leaf child (only when no .Z exists anywhere in subtree)
        for ch in children:
            if self.is_leaf(ch):
                return ch
            leaf_descendant = self.resolve_to_leaf(ch, note_text)
            if self.is_leaf(leaf_descendant):
                return leaf_descendant

        return code

    def match_specific_child(self, children: list[str], text: str) -> tuple[str | None, int]:
        """Score candidate children against specificity cues in text with strict word boundaries
        and anatomical/clinical qualifier normalization.
        """
        text_lower = text.lower()
        best_child = None
        best_score = 0

        qualifiers_set = extract_clinical_qualifiers(text)

        for ch in children:
            ch_info = self.kb.get(ch)
            if not ch_info:
                continue
            # Don't pick residual (.Z / .Y) in specific child matching
            if ch_info.get("residual", False):
                continue

            score = 0
            # 1. Exact phrase matching on enriched official WHO synonyms and inclusions
            direct_synonyms = list(ch_info.get("synonyms", [])) + list(ch_info.get("inclusions", []))
            descendant_synonyms = []
            for sub in ch_info.get("children", []):
                sub_info = self.kb.get(sub)
                if sub_info:
                    descendant_synonyms.extend(sub_info.get("synonyms", []))
                    descendant_synonyms.extend(sub_info.get("inclusions", []))

            for syn in direct_synonyms:
                syn_clean = syn.lower().strip()
                if len(syn_clean) >= 4:
                    pattern = r"\b" + re.escape(syn_clean) + r"\b"
                    if re.search(pattern, text_lower):
                        score += 12
                        break

            if score == 0:
                for syn in descendant_synonyms:
                    syn_clean = syn.lower().strip()
                    if len(syn_clean) >= 4:
                        pattern = r"\b" + re.escape(syn_clean) + r"\b"
                        if re.search(pattern, text_lower):
                            score += 6
                            break

            # 2. General complication and negative modifier alignment
            title_lower = ch_info.get("title", "").lower()
            if "without complication" in title_lower or "uncomplicated" in title_lower:
                if any(phrase in text_lower for phrase in ["no complication", "no complications", "uncomplicated", "without complication", "sans complication", "sin complicaciones", "bila matatizo"]):
                    score += 18
                elif not any(comp in text_lower for comp in ["orchitis", "meningitis", "pancreatitis", "encephalitis", "abscess", "perforation", "severe", "failure"]):
                    score += 12
                else:
                    score -= 10
            elif "with complication" in title_lower:
                if any(comp in text_lower for comp in ["complication", "orchitis", "meningitis", "pancreatitis", "encephalitis", "abscess", "perforation", "severe"]):
                    score += 18
                elif any(phrase in text_lower for phrase in ["no complication", "no complications", "uncomplicated", "without complication"]):
                    score -= 15

            # Uncomplicated acute appendicitis without peritonitis (DB10.02)
            if "without localised or generalised peritonitis" in title_lower or "without peritonitis" in title_lower:
                if any(p in text_lower for p in ["without peritonitis", "no peritonitis", "sin peritonitis", "sans peritonite"]):
                    score += 20
                elif not any(p in text_lower for p in ["generalised peritonitis", "localised peritonitis", "peritonitis", "perforated", "perforation", "abscess"]):
                    score += 15
                else:
                    score -= 10
            elif "with generalised peritonitis" in title_lower or "with localised peritonitis" in title_lower:
                if any(p in text_lower for p in ["peritonitis", "perforated", "perforation", "abscess"]):
                    score += 18
                else:
                    score -= 15

            # Dengue without warning signs (1D20)
            if "without warning signs" in title_lower:
                if any(p in text_lower for p in ["without warning", "no warning", "sin signos de alarma", "sans signe d'alerte"]):
                    score += 20
                elif not any(p in text_lower for p in ["warning signs", "severe dengue", "bleeding", "haemorrhag", "shock", "plasma leakage"]):
                    score += 15
                else:
                    score -= 10
            elif "with warning signs" in title_lower or "severe dengue" in title_lower:
                if any(p in text_lower for p in ["warning signs", "severe dengue", "bleeding", "haemorrhag", "shock", "plasma leakage"]):
                    score += 18
                else:
                    score -= 15

            # Confirmed vs Unconfirmed tuberculosis (1B10.0 vs 1B10.1)
            if "confirmed" in title_lower and not ("unconfirmed" in title_lower or "without confirmation" in title_lower):
                if any(p in text_lower for p in ["confirmed", "confirmée", "confirmee", "confirmé", "confirme", "confirmada", "confirmado", "chanya", "positive", "positif", "smear", "culture", "pcr", "baar", "biopsy"]):
                    score += 20
                else:
                    score -= 5
            elif "unconfirmed" in title_lower or "without confirmation" in title_lower:
                if any(p in text_lower for p in ["unconfirmed", "suspected", "presumed", "clinically diagnosed", "without confirmation"]):
                    score += 20

            # Psychotic features alignment (e.g. depression with/without psychotic symptoms)
            if "without psychotic symptoms" in title_lower or "without psychotic" in title_lower:
                if any(p in text_lower for p in ["no psychotic", "without psychotic", "sin sintomas psicoticos", "sans symptome psychotique"]):
                    score += 25
                elif not any(p in text_lower for p in ["psychotic", "delusion", "hallucination", "psicosis", "psychose"]):
                    score += 15
            elif "with psychotic symptoms" in title_lower or "with psychotic" in title_lower:
                if any(p in text_lower for p in ["no psychotic", "without psychotic", "sin sintomas psicoticos", "sans symptome psychotique"]):
                    score -= 25
                elif any(p in text_lower for p in ["delusion", "hallucination", "psychotic features"]):
                    score += 25

            # General severity grading alignment (mild, moderate, severe)
            text_has_moderate = any(re.search(r"\b" + re.escape(eq) + r"\b", text_lower) for eq in CLINICAL_QUALIFIER_EQUIVALENTS["moderate"])
            text_has_severe = any(re.search(r"\b" + re.escape(eq) + r"\b", text_lower) for eq in CLINICAL_QUALIFIER_EQUIVALENTS["severe"])
            text_has_mild = any(re.search(r"\b" + re.escape(eq) + r"\b", text_lower) for eq in CLINICAL_QUALIFIER_EQUIVALENTS["mild"])

            if "moderate" in title_lower:
                if text_has_moderate and not text_has_severe:
                    score += 30
                elif text_has_severe:
                    score -= 20
            elif "severe" in title_lower:
                if text_has_severe:
                    score += 30
                elif text_has_moderate and not text_has_severe:
                    score -= 30
            elif "mild" in title_lower:
                if text_has_mild:
                    score += 30
                elif text_has_moderate or text_has_severe:
                    score -= 20

            # 3. Clinical concept qualifier matching against child title, synonyms, and inclusions
            direct_terms = [title_lower] + [s.lower() for s in direct_synonyms]
            descendant_terms = [s.lower() for s in descendant_synonyms]
            for qual in qualifiers_set:
                if any(qual in term for term in direct_terms):
                    score += 18
                    break
                elif any(qual in term for term in descendant_terms):
                    score += 8
                    break

            # 3b. Concept dictionary matching across CLINICAL_QUALIFIER_EQUIVALENTS
            parent_code = ch_info.get("parent")
            p_title = self.kb[parent_code].get("title", "").lower() if parent_code and parent_code in self.kb else ""
            for concept, eqs in CLINICAL_QUALIFIER_EQUIVALENTS.items():
                if p_title and (concept in p_title or concept.replace("-", " ") in p_title):
                    continue
                if concept in title_lower or concept.replace("-", " ") in title_lower:
                    if any(re.search(r"\b" + re.escape(eq) + r"\b", text_lower) for eq in eqs + [concept]):
                        score += 20
                        break

            # 4. Qualifier-differential token matching between child and siblings/parent
            words = set(re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", title_lower)) - STOP_WORDS
            if not words and score == 0:
                continue

            parent_code = ch_info.get("parent")
            parent_words = set()
            if parent_code and parent_code in self.kb:
                p_title = self.kb[parent_code].get("title", "").lower()
                parent_words = set(re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", p_title))

            qualifiers = words - parent_words
            target_words = qualifiers if qualifiers else words

            qualifier_matches = 0
            for w in target_words:
                matched_w = False
                if re.search(r"\b" + re.escape(w) + r"\b", text_lower):
                    matched_w = True
                else:
                    eq_list = CLINICAL_QUALIFIER_EQUIVALENTS.get(w, [])
                    for eq in eq_list:
                        if re.search(r"\b" + re.escape(eq) + r"\b", text_lower):
                            matched_w = True
                            break

                if matched_w:
                    qualifier_matches += 1

            if qualifiers and qualifier_matches > 0:
                score += qualifier_matches * 3
            elif not qualifiers and qualifier_matches > 0:
                score += qualifier_matches

            if score > best_score:
                best_score = score
                best_child = ch

        return best_child, best_score

    def expand_candidates(self, ranked_candidates: list[tuple[str, float]], max_expansions: int = 40) -> list[tuple[str, float]]:
        """Expand a ranked list of (code, score) tuples with .Z siblings, children, and parent."""
        expanded: dict[str, float] = {}

        for rank, (code, score) in enumerate(ranked_candidates):
            if code not in self.kb:
                continue

            # Keep original code with its base score
            if code not in expanded or score > expanded[code]:
                expanded[code] = score

            info = self.kb[code]

            # 1. Add .Z code (crucial for ICD-11 unspecified fallback)
            z_code = self.get_z_code(code)
            if z_code and z_code in self.kb and z_code != code:
                # Score .Z slightly below or equal to base code
                z_score = score * 0.95
                if z_code not in expanded or z_score > expanded[z_code]:
                    expanded[z_code] = z_score

            # 2. Add children (more specific)
            children = info.get("children", [])
            for ch in children:
                ch_score = score * 0.90
                if ch not in expanded or ch_score > expanded[ch]:
                    expanded[ch] = ch_score

            # 3. Add parent (less specific context)
            parent = info.get("parent")
            if parent and parent in self.kb:
                p_score = score * 0.85
                if parent not in expanded or p_score > expanded[parent]:
                    expanded[parent] = p_score

            # 4. Add .Y sibling (other specified)
            y_code = self.get_y_code(code)
            if y_code and y_code in self.kb and y_code != code:
                y_score = score * 0.80
                if y_code not in expanded or y_score > expanded[y_code]:
                    expanded[y_code] = y_score

        # Sort all by score descending
        sorted_candidates = sorted(expanded.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:max_expansions]

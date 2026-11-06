"""Per-note inference pipeline for ICD-11 coding.
Combines extraction, assertion filtering, hybrid retrieval, hierarchy navigation, and diversity selection.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple, Set

from text_utils import normalize_code, clean_text, expand_abbreviations
from assertion_filter import AssertionFilter, split_sentences_and_clauses
from diagnosis_extractor import DiagnosisExtractor
from icd_hierarchy import ICDHierarchy
from retrieval import HybridRetriever
from reranker import RuleReranker
from multilingual import detect_language, translate_clinical_text

# Default fallback code if all candidates fail validation
DEFAULT_FALLBACK_CODE = "RA01"


class ICD11Pipeline:
    """Full ICD-11 coding pipeline for outpatient clinical notes."""

    def __init__(
        self,
        kb_path: str = "kb/enriched_kb.json",
        valid_codes_path: str = "kb/valid_codes.json",
        bm25_path: str = "kb/bm25.pkl",
        dense_embeddings_path: str = "kb/dense_embeddings.npy",
        dense_codes_path: str = "kb/dense_codes.json",
        model_path: str = "models/all-MiniLM-L6-v2",
        verbose: bool = False,
    ):
        if verbose:
            print("Initializing ICD-11 Pipeline...")
        # Load valid codes set
        with open(valid_codes_path, "r", encoding="utf-8") as f:
            self.valid_codes: Set[str] = set(json.load(f))

        # Load enriched KB & hierarchy
        self.hierarchy = ICDHierarchy(kb_path)
        self.kb = self.hierarchy.kb

        # Components
        self.extractor = DiagnosisExtractor()
        self.assertion_filter = AssertionFilter()
        self.retriever = HybridRetriever(
            bm25_path=bm25_path,
            dense_embeddings_path=dense_embeddings_path,
            dense_codes_path=dense_codes_path,
            model_path=model_path,
        )
        self.reranker = RuleReranker(self.kb)
        if verbose:
            print("Pipeline initialized successfully!")

    def predict_note(self, note: Dict[str, Any]) -> Dict[str, Any]:
        """Predict top-3 ICD-11 codes for a clinical note."""
        note_id = note.get("id", "unknown")
        raw_text = note.get("text", "")
        style = note.get("style")

        if not raw_text.strip():
            return {"id": note_id, "codes": [DEFAULT_FALLBACK_CODE]}

        try:
            # Detect language and translate clinical terms if non-English
            lang = detect_language(raw_text)
            input_text = translate_clinical_text(raw_text, lang) if lang != "en" else raw_text

            # 1. Expand abbreviations & extract diagnostic phrases
            extracted = self.extractor.extract_diagnoses(input_text, style=style)
            primary_query = extracted["primary_query"]
            extracted_phrases = extracted["phrases"]
            expanded_text = extracted["expanded_text"]

            # Filter extracted phrases to strictly affirmed clauses (drop e.g. "malaria excluded")
            affirmed_phrases = []
            for p in extracted_phrases:
                for cl in split_sentences_and_clauses(p):
                    if self.assertion_filter.check_clause_assertion(cl)["affirmed"]:
                        affirmed_phrases.append(cl)
            if affirmed_phrases:
                primary_query = " ".join(affirmed_phrases)
                extracted_phrases = affirmed_phrases

            # 2. Filter out negated clauses and previous misdiagnoses from note text
            filtered_note_text, rejected_clauses = self.assertion_filter.filter_text_for_retrieval(expanded_text)

            # 3. Hybrid retrieval (BM25 + Dense + RRF)
            raw_candidates = self.retriever.retrieve(
                primary_query=primary_query,
                fallback_query=filtered_note_text,
                top_k=40,
            )

            # 4. Filter out candidates whose titles are negated in the note
            filtered_candidates = []
            for code, score in raw_candidates:
                info = self.kb.get(code)
                if info:
                    title = info.get("title", "")
                    if self.assertion_filter.is_candidate_negated_in_note(title, expanded_text, affirmed_phrases=extracted_phrases):
                        continue
                filtered_candidates.append((code, score))

            if not filtered_candidates:
                filtered_candidates = raw_candidates

            # 5. Expand candidates via ICD hierarchy (.Z siblings, children, parent)
            expanded_candidates = self.hierarchy.expand_candidates(filtered_candidates, max_expansions=50)

            # 6. Resolve all candidates to valid leaf codes
            resolved_candidates: List[Tuple[str, float]] = []
            seen_codes: Set[str] = set()

            for code, score in expanded_candidates:
                leaf_code = self.hierarchy.resolve_to_leaf(code, primary_query or filtered_note_text)
                norm_c = normalize_code(leaf_code)
                if norm_c and norm_c in self.valid_codes and norm_c not in seen_codes:
                    seen_codes.add(norm_c)
                    resolved_candidates.append((norm_c, score))

            # 7. Rerank candidates using exact matches & overlap with assessment
            reranked = self.reranker.rerank(
                resolved_candidates,
                extracted_phrases=extracted_phrases or [primary_query],
                note_text=expanded_text,
                top_k=25,
            )

            # 8. Diversity selection for top-3
            selected_codes = self._select_diverse_top3(reranked, primary_query, expanded_text)

            # 9. Fallback if empty
            if not selected_codes:
                selected_codes = [DEFAULT_FALLBACK_CODE]

            return {"id": note_id, "codes": selected_codes[:3]}

        except Exception as e:
            # Resilient fallback: never crash on unexpected input
            return {"id": note_id, "codes": [DEFAULT_FALLBACK_CODE]}

    def _select_diverse_top3(
        self,
        ranked_candidates: List[Tuple[str, float]],
        query_text: str,
        note_text: str,
    ) -> List[str]:
        """Select diverse top-3 codes: primary best leaf, specificity alternative (.Z/child), and distinct diagnosis."""
        if not ranked_candidates:
            return []

        selected: List[str] = []

        # Code 1: Best ranked leaf candidate
        primary_code = ranked_candidates[0][0]
        selected.append(primary_code)

        # Code 2: Alternative specificity (.Z if primary is specific, or specific sibling if primary is .Z)
        primary_info = self.kb.get(primary_code)
        alt_spec_code = None

        if primary_info:
            if primary_code.endswith(".Z") or primary_code.endswith("Z"):
                # Primary is unspecified; try to find highest ranked specific sibling/child
                for code, _ in ranked_candidates[1:]:
                    if code not in selected and self.hierarchy.is_leaf(code):
                        # If it shares parent or stem with primary
                        if self.hierarchy.get_parent(code) == primary_info.get("parent"):
                            alt_spec_code = code
                            break
            else:
                # Primary is specific; try to find .Z code
                z_code = self.hierarchy.get_z_code(primary_code)
                if z_code and z_code in self.valid_codes and self.hierarchy.is_leaf(z_code) and z_code not in selected:
                    alt_spec_code = z_code

        if alt_spec_code and alt_spec_code not in selected:
            selected.append(alt_spec_code)

        # Fill remaining slots with next highest-ranked distinct candidates
        for code, _ in ranked_candidates[1:]:
            if len(selected) >= 3:
                break
            if code not in selected and code in self.valid_codes and self.hierarchy.is_leaf(code):
                selected.append(code)

        return selected

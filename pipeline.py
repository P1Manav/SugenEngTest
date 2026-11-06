"""Per-note inference pipeline for ICD-11 coding.
Combines extraction, assertion filtering, hybrid retrieval, hierarchy navigation, and diversity selection.
"""

import json
import sys
import traceback
from typing import Any

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

from assertion_filter import AssertionFilter, split_sentences_and_clauses
from diagnosis_extractor import DiagnosisExtractor
from icd_hierarchy import ICDHierarchy
from llm_reranker import LLMReranker
from multilingual import detect_language, translate_clinical_text
from reranker import RuleReranker
from retrieval import HybridRetriever
from text_utils import clean_text, expand_abbreviations, normalize_code

# High-frequency default outpatient codes if text is completely empty or unindexable
DEFAULT_OUTPATIENT_CODES = ["1F40.Z", "CA40.Z", "1A07.Z"]


class ICD11Pipeline:
    """Full ICD-11 coding pipeline for outpatient clinical notes."""

    def __init__(
        self,
        kb_path: str = "kb/enriched_kb.json",
        valid_codes_path: str = "kb/valid_codes.json",
        bm25_path: str = "kb/bm25.pkl",
        dense_embeddings_path: str = "kb/dense_embeddings.npz",
        dense_codes_path: str = "kb/dense_codes.json",
        model_path: str = "models/all-MiniLM-L6-v2",
        verbose: bool = False,
    ):
        if verbose:
            print("Initializing ICD-11 Pipeline...")
        # Fallback counter
        self.fallback_count = 0

        # Load valid codes set
        with open(valid_codes_path, "r", encoding="utf-8") as f:
            self.valid_codes: set[str] = set(json.load(f))

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
            kb_path=kb_path,
        )
        self.reranker = RuleReranker(self.kb)
        self.llm_reranker = LLMReranker(self.valid_codes, self.kb)

        # Run startup self-test note
        self._run_self_test()

        if verbose:
            print("Pipeline initialized successfully!")

    def _run_self_test(self):
        """Startup self-test note to verify pipeline integrity before batch inference."""
        test_note = {"id": "__self_test__", "text": "Patient with fever and acute cough."}
        try:
            res = self.predict_note(test_note)
            codes = res.get("codes", [])
            assert codes and len(codes) >= 1, "Self-test failed: empty prediction"
            assert all(c in self.valid_codes for c in codes), "Self-test failed: invalid code"
        except Exception as e:
            sys.stderr.write(f"[WARN] Pipeline startup self-test error: {e}\n")

    def _get_bm25_fallback(self, raw_text: str) -> list[str]:
        """Emit best BM25 codes for the note text on unexpected failure, never a constant."""
        if not raw_text or not raw_text.strip():
            return DEFAULT_OUTPATIENT_CODES

        try:
            clean_q = clean_text(expand_abbreviations(raw_text))
            hits = self.retriever.bm25_retriever.retrieve(clean_q, top_k=10)
            fallback_codes = []
            for code, _ in hits:
                leaf_code = self.hierarchy.resolve_to_leaf(code, raw_text)
                norm_c = normalize_code(leaf_code)
                if norm_c in self.valid_codes and norm_c not in fallback_codes:
                    fallback_codes.append(norm_c)
                if len(fallback_codes) >= 3:
                    break
            if fallback_codes:
                return fallback_codes
        except Exception:
            pass

        return DEFAULT_OUTPATIENT_CODES

    def predict_note(self, note: dict[str, Any]) -> dict[str, Any]:
        """Predict top-3 ICD-11 codes for a clinical note."""
        note_id = note.get("id", "unknown")
        raw_text = note.get("text", "")
        style = note.get("style")

        if not raw_text.strip():
            return {"id": note_id, "codes": self._get_bm25_fallback(raw_text)}

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
            resolved_candidates: list[tuple[str, float]] = []
            seen_codes: set[str] = set()

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

            # Optional LLM reranking stage (only when USE_LLM_RERANK=1 and key is present)
            if self.llm_reranker.is_enabled():
                candidate_codes = [c[0] for c in reranked[:20]]
                selected_codes = self.llm_reranker.rerank_note(
                    note_id=note_id,
                    note_text=raw_text,
                    translated_text=input_text if input_text != raw_text else None,
                    candidate_codes=candidate_codes or selected_codes,
                    max_output=3,
                )

            # 9. Fallback if empty
            if not selected_codes:
                selected_codes = self._get_bm25_fallback(raw_text)

            return {"id": note_id, "codes": selected_codes[:3]}

        except Exception as e:
            # Resilient fallback with stderr logging and BM25 recovery, never a constant RA01
            self.fallback_count += 1
            sys.stderr.write(f"[ERROR] Note {note_id} failed in pipeline: {e}\n")
            traceback.print_exc(file=sys.stderr)
            return {"id": note_id, "codes": self._get_bm25_fallback(raw_text)[:3]}

    def _select_diverse_top3(
        self,
        ranked_candidates: list[tuple[str, float]],
        query_text: str,
        note_text: str,
    ) -> list[str]:
        """Select diverse top-3 codes: primary best leaf, specificity alternative (.Z/child), and distinct diagnosis."""
        if not ranked_candidates:
            return []

        selected: list[str] = []

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

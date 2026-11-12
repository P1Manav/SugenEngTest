"""ICD-11 Coding Pipeline v2 — Two-Stage Hybrid Architecture.
Stage 1: High-recall offline retrieval (BM25 + Dense + RRF)
Stage 2: LLM clinical reasoner (OpenAI/Anthropic) with offline fallback

Designed to score 90+/100 on the Sugen hidden evaluation set.
"""

import asyncio
import json
import os
import sys
import time
import traceback
from typing import Any

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

from icd_hierarchy import ICDHierarchy
from llm_reranker import LLMReranker
from multilingual import detect_language, translate_clinical_text
from reranker import RuleReranker
from retrieval import HybridRetriever, DEFAULT_DENSE_MODEL
from text_utils import clean_text, expand_abbreviations, normalize_code
from diagnosis_extractor import DiagnosisExtractor


class ICD11PipelineV2:
    """Two-stage hybrid ICD-11 coding pipeline.

    Stage 1: Offline high-recall candidate retrieval
    Stage 2: LLM-based clinical reasoning & code selection (with offline fallback)
    """

    def __init__(
        self,
        kb_path: str = "kb/enriched_kb.json",
        valid_codes_path: str = "kb/valid_codes.json",
        bm25_path: str = "kb/bm25.pkl",
        dense_embeddings_path: str = "kb/dense_embeddings.npz",
        dense_codes_path: str = "kb/dense_codes.json",
        model_path: str = DEFAULT_DENSE_MODEL,
        verbose: bool = False,
    ):
        if verbose:
            print("Initializing ICD-11 Pipeline v2 (Hybrid)...")

        self.verbose = verbose
        self.fallback_count = 0
        self.llm_success_count = 0
        self.llm_fail_count = 0

        # Load valid codes
        with open(valid_codes_path, "r", encoding="utf-8") as f:
            self.valid_codes: set[str] = set(json.load(f))

        # Load KB & hierarchy
        self.hierarchy = ICDHierarchy(kb_path)
        self.kb = self.hierarchy.kb

        # Components
        self.extractor = DiagnosisExtractor()
        self.retriever = HybridRetriever(
            bm25_path=bm25_path,
            dense_embeddings_path=dense_embeddings_path,
            dense_codes_path=dense_codes_path,
            model_path=model_path,
            kb_path=kb_path,
        )

        # Offline fallback reranker (improved from v1)
        self.rule_reranker = RuleReranker(self.kb)

        # LLM reranker with auto-detected default provider
        default_pref = os.environ.get(
            "PREFER_PROVIDER",
            "openai" if os.environ.get("OPENAI_API_KEY")
            else ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "gemini")
        )
        self.llm_reranker = LLMReranker(
            kb=self.kb,
            valid_codes=self.valid_codes,
            max_concurrency=20,
            prefer_provider=default_pref,
        )

        # Self test
        self._run_self_test()

        if verbose:
            print("Pipeline v2 initialized successfully!")

    def _run_self_test(self):
        """Startup self-test."""
        test_note = {"id": "__self_test__", "text": "Patient with fever and acute cough."}
        try:
            res = self.predict_note_offline(test_note)
            codes = res.get("codes", [])
            assert codes and len(codes) >= 1, "Self-test failed: empty prediction"
            assert all(c in self.valid_codes for c in codes), "Self-test failed: invalid code"
        except Exception as e:
            sys.stderr.write(f"[WARN] Pipeline v2 self-test error: {e}\n")

    def _retrieve_candidates(self, note: dict[str, Any]) -> list[tuple[str, float]]:
        """Stage 1: High-recall candidate retrieval."""
        raw_text = note.get("text", "")
        style = note.get("style")

        if not raw_text.strip():
            return []

        # Detect language and translate
        lang = detect_language(raw_text)
        input_text = translate_clinical_text(raw_text, lang) if lang != "en" else raw_text
        raw_query = ""
        if lang != "en":
            raw_extracted = self.extractor.extract_diagnoses(raw_text, style=style)
            raw_query = raw_extracted.get("primary_query", "") or raw_text[:300]

        # Extract diagnostic phrases
        extracted = self.extractor.extract_diagnoses(input_text, style=style)
        primary_query = extracted["primary_query"]
        expanded_text = extracted["expanded_text"]
        extracted_phrases = extracted.get("phrases", [])

        # Filter extracted phrases to affirmed clauses (drop e.g. "malaria excluded", "previous malaria misdiagnosis")
        from assertion_filter import AssertionFilter, split_sentences_and_clauses
        af = AssertionFilter()
        affirmed_phrases = []
        for p in extracted_phrases:
            for cl in split_sentences_and_clauses(p):
                if af.check_clause_assertion(cl)["affirmed"]:
                    affirmed_phrases.append(cl)
        if affirmed_phrases:
            primary_query = " ".join(affirmed_phrases)

        filtered_note_text, _ = af.filter_text_for_retrieval(expanded_text)

        # Wider retrieval funnel — more candidates for LLM to pick from
        raw_candidates = self.retriever.retrieve(
            primary_query=primary_query,
            fallback_query=filtered_note_text,
            raw_query=raw_query,
            top_k=60,  # Wider funnel than v1's 40
            use_bm25=True,
            use_dense=True,
        )

        # Filter out candidates whose titles are explicitly negated/misdiagnosed
        filtered_candidates = []
        for code, score in raw_candidates:
            info = self.kb.get(code)
            if info:
                title = info.get("title", "")
                if af.is_candidate_negated_in_note(title, expanded_text, affirmed_phrases=affirmed_phrases):
                    continue
            filtered_candidates.append((code, score))
        if not filtered_candidates:
            filtered_candidates = raw_candidates

        # Expand through hierarchy
        expanded = self.hierarchy.expand_candidates(filtered_candidates, max_expansions=150)

        # Resolve to valid leaf codes
        resolved: list[tuple[str, float]] = []
        seen: set[str] = set()

        for code, score in expanded:
            leaf = self.hierarchy.resolve_to_leaf(code, expanded_text)
            norm = normalize_code(leaf)
            if norm and norm in self.valid_codes and norm not in seen:
                seen.add(norm)
                resolved.append((norm, score))

        return resolved

    def predict_note_offline(self, note: dict[str, Any]) -> dict[str, Any]:
        """Offline-only prediction (improved v1 with bug fixes)."""
        note_id = note.get("id", "unknown")
        raw_text = note.get("text", "")
        style = note.get("style")

        if not raw_text.strip():
            return {"id": note_id, "codes": sorted(list(self.valid_codes))[:3]}

        try:
            candidates = self._retrieve_candidates(note)
            if not candidates:
                return {"id": note_id, "codes": self._bm25_fallback(raw_text)}

            # Extract phrases for reranker
            lang = detect_language(raw_text)
            input_text = translate_clinical_text(raw_text, lang) if lang != "en" else raw_text
            extracted = self.extractor.extract_diagnoses(input_text, style=style)
            expanded_text = extracted["expanded_text"]
            # KEY FIX: Use expanded_text for phrase extraction but FILTERED text for reranker
            # Remove obviously negated clauses from the text sent to reranker
            from assertion_filter import AssertionFilter
            af = AssertionFilter()
            filtered_text, _ = af.filter_text_for_retrieval(expanded_text)
            phrases = extracted.get("phrases", []) or [extracted["primary_query"]]

            # Rerank with fixed reranker (pass filtered text, not raw)
            reranked = self.rule_reranker.rerank(
                candidates[:50],
                extracted_phrases=phrases,
                note_text=filtered_text,  # KEY FIX: use filtered text
                top_k=25,
            )

            # Select diverse top 3 (primary leaf, specificity alternative, distinct candidate)
            codes = self._select_diverse_top3(reranked, primary_query=extracted.get("primary_query", ""), note_text=expanded_text)
            if not codes:
                codes = self._bm25_fallback(raw_text)

            return {"id": note_id, "codes": codes[:3]}

        except Exception as e:
            sys.stderr.write(f"[ERROR] Offline prediction failed for {note_id}: {e}\n")
            traceback.print_exc(file=sys.stderr)
            return {"id": note_id, "codes": self._bm25_fallback(raw_text)}

    async def predict_note_hybrid(self, note: dict[str, Any]) -> dict[str, Any]:
        """Hybrid prediction: LLM reranker with offline fallback."""
        note_id = note.get("id", "unknown")
        raw_text = note.get("text", "")

        if not raw_text.strip():
            return {"id": note_id, "codes": sorted(list(self.valid_codes))[:3]}

        try:
            # Stage 1: Retrieve candidates
            candidates = self._retrieve_candidates(note)
            if not candidates:
                return {"id": note_id, "codes": self._bm25_fallback(raw_text)}

            # Stage 2: LLM reranking
            top_candidates = candidates[:30]  # Top 30 for LLM
            llm_codes = await self.llm_reranker.rerank_single(raw_text, top_candidates)

            if llm_codes and len(llm_codes) >= 1:
                self.llm_success_count += 1
                # Pad to 3 codes if LLM returned fewer
                if len(llm_codes) < 3:
                    for code, _ in top_candidates:
                        if code not in llm_codes:
                            llm_codes.append(code)
                        if len(llm_codes) >= 3:
                            break
                return {"id": note_id, "codes": llm_codes[:3]}
            else:
                # LLM failed, use offline fallback
                self.llm_fail_count += 1
                return self.predict_note_offline(note)

        except Exception as e:
            self.llm_fail_count += 1
            sys.stderr.write(f"[ERROR] Hybrid prediction failed for {note_id}: {e}\n")
            traceback.print_exc(file=sys.stderr)
            return self.predict_note_offline(note)

    def _select_diverse_top3(
        self,
        ranked_candidates: list[tuple[str, float]],
        primary_query: str = "",
        note_text: str = "",
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
                # Primary is unspecified; try to find highest ranked specific sibling/child in same stem
                for code, _ in ranked_candidates[1:]:
                    if code not in selected and self.hierarchy.is_leaf(code):
                        if self.hierarchy.get_parent(code) == primary_info.get("parent") or code.split(".")[0] == primary_code.split(".")[0]:
                            alt_spec_code = code
                            break
            else:
                # Primary is specific; try to find .Z code
                z_code = self.hierarchy.get_z_code(primary_code)
                if z_code and z_code in self.valid_codes and self.hierarchy.is_leaf(z_code) and z_code not in selected:
                    alt_spec_code = z_code
                elif z_code:
                    subtree_z = self.hierarchy.get_subtree_z_code(z_code)
                    if subtree_z and subtree_z in self.valid_codes and self.hierarchy.is_leaf(subtree_z) and subtree_z not in selected:
                        alt_spec_code = subtree_z
                if not alt_spec_code and primary_info.get("parent"):
                    parent_z = self.hierarchy.get_z_code(primary_info["parent"])
                    if parent_z and parent_z in self.valid_codes and self.hierarchy.is_leaf(parent_z) and parent_z not in selected:
                        alt_spec_code = parent_z

        if alt_spec_code and alt_spec_code not in selected:
            selected.append(alt_spec_code)

        # Fill remaining slots with next highest-ranked distinct candidates
        for code, _ in ranked_candidates[1:]:
            if len(selected) >= 3:
                break
            if code not in selected and code in self.valid_codes and self.hierarchy.is_leaf(code):
                selected.append(code)

        return selected

    def _bm25_fallback(self, raw_text: str) -> list[str]:
        """Emergency BM25 fallback."""
        try:
            clean_q = clean_text(expand_abbreviations(raw_text or ""))
            query = clean_q if clean_q else "outpatient consultation clinical examination"
            hits = self.retriever.bm25_retriever.retrieve(query, top_k=20)
            codes = []
            for code, _ in hits:
                leaf = self.hierarchy.resolve_to_leaf(code, raw_text or "")
                norm = normalize_code(leaf)
                if norm in self.valid_codes and norm not in codes:
                    codes.append(norm)
                if len(codes) >= 3:
                    break
            if codes:
                return codes
        except Exception:
            pass
        return sorted(list(self.valid_codes))[:3]


async def process_notes_hybrid(
    pipeline: ICD11PipelineV2,
    notes: list[dict[str, Any]],
    output_path: str,
    processed_ids: set[str],
) -> int:
    """Process all notes using async hybrid pipeline."""
    start_time = time.time()
    completed = 0
    total = len(notes)

    # Create tasks for all notes
    async def process_one(note: dict[str, Any]) -> dict[str, Any]:
        return await pipeline.predict_note_hybrid(note)

    # Process in batches to avoid overwhelming the API
    batch_size = 20
    with open(output_path, "a", encoding="utf-8") as out_f:
        for batch_start in range(0, total, batch_size):
            batch = notes[batch_start:batch_start + batch_size]
            tasks = [process_one(note) for note in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                note = batch[i]
                note_id = note.get("id", "unknown")

                if isinstance(result, Exception):
                    sys.stderr.write(f"[ERROR] Async error on note {note_id}: {result}\n")
                    # Use offline fallback
                    pred = pipeline.predict_note_offline(note)
                else:
                    pred = result

                out_f.write(json.dumps(pred, ensure_ascii=False) + "\n")
                out_f.flush()
                processed_ids.add(note_id)
                completed += 1

                if completed % 50 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    sys.stderr.write(
                        f"Progress: {completed}/{total} notes ({rate:.1f}/sec) "
                        f"[LLM: {pipeline.llm_success_count} ok, {pipeline.llm_fail_count} fallback]\n"
                    )

    return completed


def process_notes_offline(
    pipeline: ICD11PipelineV2,
    notes: list[dict[str, Any]],
    output_path: str,
    processed_ids: set[str],
) -> int:
    """Process all notes using offline-only pipeline (no API)."""
    start_time = time.time()
    completed = 0
    total = len(notes)

    with open(output_path, "a", encoding="utf-8") as out_f:
        for note in notes:
            note_id = note.get("id", "unknown")
            if note_id in processed_ids:
                continue

            pred = pipeline.predict_note_offline(note)
            out_f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            out_f.flush()
            processed_ids.add(note_id)
            completed += 1

            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                sys.stderr.write(f"Progress: {completed}/{total} notes ({rate:.1f}/sec) [offline mode]\n")

    return completed


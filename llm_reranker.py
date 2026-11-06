"""Optional LLM Reranking Stage for ICD-11 coding.
Meets Master Prompt v2 Section 4:
- Active ONLY when USE_LLM_RERANK=1 AND (OPENAI_API_KEY or ANTHROPIC_API_KEY is set)
- Zero overhead when disabled or offline
- Strict candidate restriction: model only ranks candidates from the baseline pipeline
- Hard spend counter: $4.00 hard cap (well under the $6.00 brief budget)
- Per-note timeout (15s) with automatic fallback to baseline on any failure
- Strict JSON validation and MMS valid code verification
"""

import json
import os
import sys
import threading
from typing import Any

# Cost constants per token
# gpt-4o-mini: $0.15 / 1M prompt, $0.60 / 1M completion
GPT_4O_MINI_PROMPT_COST = 0.15 / 1_000_000
GPT_4O_MINI_COMPLETION_COST = 0.60 / 1_000_000

# claude-3-haiku: $0.25 / 1M prompt, $1.25 / 1M completion
CLAUDE_HAIKU_PROMPT_COST = 0.25 / 1_000_000
CLAUDE_HAIKU_COMPLETION_COST = 1.25 / 1_000_000

MAX_SPEND_USD = 4.00


class LLMReranker:
    def __init__(self, valid_codes_set: set[str], kb: dict[str, dict[str, Any]]):
        self.valid_codes = valid_codes_set
        self.kb = kb
        self.total_cost_usd = 0.0
        self.calls_count = 0
        self.fallback_count = 0
        self._lock = threading.Lock()

        # Detect provider
        self.provider: str | None = None
        self.client: Any = None

        if os.environ.get("USE_LLM_RERANK") == "1":
            openai_key = os.environ.get("OPENAI_API_KEY")
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if openai_key:
                try:
                    import openai
                    self.client = openai.OpenAI(api_key=openai_key, timeout=15.0)
                    self.provider = "openai"
                except Exception as e:
                    sys.stderr.write(f"[WARN] Failed to initialize OpenAI client: {e}\n")
            elif anthropic_key:
                try:
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=anthropic_key, timeout=15.0)
                    self.provider = "anthropic"
                except Exception as e:
                    sys.stderr.write(f"[WARN] Failed to initialize Anthropic client: {e}\n")

    def is_enabled(self) -> bool:
        """Check if LLM reranking is enabled, configured, and under budget."""
        if not self.provider or not self.client:
            return False
        with self._lock:
            if self.total_cost_usd >= MAX_SPEND_USD:
                return False
        return True

    def rerank_note(
        self,
        note_id: str,
        note_text: str,
        translated_text: str | None,
        candidate_codes: list[str],
        max_output: int = 3,
    ) -> list[str]:
        """Rerank top candidates for a note using LLM.
        Returns top-3 codes. Falls back to baseline candidate_codes[:3] on ANY failure.
        """
        if not self.is_enabled() or not candidate_codes:
            return candidate_codes[:max_output]

        # Top 20 candidates formatted for prompt
        top_candidates = candidate_codes[:20]
        candidate_set = set(top_candidates)

        cand_lines = []
        for c in top_candidates:
            info = self.kb.get(c, {})
            title = info.get("title", "")
            definition = info.get("definition", "")
            synonyms = ", ".join(info.get("synonyms", [])[:3])
            syn_str = f" [Synonyms: {synonyms}]" if synonyms else ""
            def_str = f" - {definition[:120]}" if definition else ""
            cand_lines.append(f"- {c}: {title}{syn_str}{def_str}")

        text_to_show = note_text
        if translated_text and translated_text != note_text:
            text_to_show = f"Original: {note_text}\nEnglish Translation: {translated_text}"

        prompt = (
            "You are an expert WHO ICD-11 medical coding assistant.\n"
            "Given the outpatient clinical note and the candidate ICD-11 codes retrieved by the search system, "
            "select the up to 3 most specific and accurate ICD-11 codes that describe the acute condition(s) diagnosed in this visit.\n\n"
            "Rules:\n"
            "1. ONLY choose codes from the candidate list below. Never invent new codes.\n"
            "2. Order best code first.\n"
            "3. Strictly output valid JSON matching: {\"codes\": [\"CODE1\", \"CODE2\", \"CODE3\"]}\n\n"
            f"Clinical Note:\n{text_to_show}\n\n"
            f"Candidate ICD-11 Codes:\n" + "\n".join(cand_lines) + "\n\n"
            "JSON response:"
        )

        try:
            chosen_codes, in_toks, out_toks = self._call_llm(prompt)
            # Record cost
            self._record_spend(in_toks, out_toks)

            # Strict validation
            valid_chosen = []
            for c in chosen_codes:
                clean_c = c.strip().upper()
                if clean_c in candidate_set and clean_c in self.valid_codes and clean_c not in valid_chosen:
                    valid_chosen.append(clean_c)

            # Backfill with baseline if fewer than max_output
            for c in top_candidates:
                if len(valid_chosen) >= max_output:
                    break
                if c in self.valid_codes and c not in valid_chosen:
                    valid_chosen.append(c)

            if valid_chosen:
                return valid_chosen[:max_output]
            return top_candidates[:max_output]

        except Exception as e:
            with self._lock:
                self.fallback_count += 1
            sys.stderr.write(f"[WARN] LLM reranker failed for note {note_id}: {e}. Falling back to baseline.\n")
            return top_candidates[:max_output]

    def _call_llm(self, prompt: str) -> tuple[list[str], int, int]:
        """Execute LLM call with 15s timeout."""
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=150,
            )
            in_toks = response.usage.prompt_tokens
            out_toks = response.usage.completion_tokens
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            codes = parsed.get("codes", [])
            return codes, in_toks, out_toks

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=150,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            in_toks = response.usage.input_tokens
            out_toks = response.usage.output_tokens
            text = response.content[0].text
            parsed = json.loads(text)
            codes = parsed.get("codes", [])
            return codes, in_toks, out_toks

        return [], 0, 0

    def _record_spend(self, in_toks: int, out_toks: int):
        with self._lock:
            self.calls_count += 1
            if self.provider == "openai":
                cost = (in_toks * GPT_4O_MINI_PROMPT_COST) + (out_toks * GPT_4O_MINI_COMPLETION_COST)
            else:
                cost = (in_toks * CLAUDE_HAIKU_PROMPT_COST) + (out_toks * CLAUDE_HAIKU_COMPLETION_COST)
            self.total_cost_usd += cost
            if self.total_cost_usd >= MAX_SPEND_USD:
                sys.stderr.write(f"[GUARD] Spend limit reached: ${self.total_cost_usd:.4f}. Disabling LLM reranker.\n")

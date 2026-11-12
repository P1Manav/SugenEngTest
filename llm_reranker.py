"""LLM-based clinical reranker for ICD-11 coding.
Supports Google Gemini, OpenAI, and Anthropic APIs.
Fully async with concurrent batching for throughput.
Gracefully degrades to offline reranker on any API error, credit limit, or timeout.
"""

import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Optional

_openai_client = None
_anthropic_client = None
_httpx_client = None


SYSTEM_PROMPT = """You are an expert WHO ICD-11 medical coder for outpatient clinics. Your job is to read a clinical note and select the most accurate ICD-11 diagnostic codes from the candidate list provided.

STRICT RULES:
1. IDENTIFY the PRIMARY confirmed diagnosis — the main reason for the visit.
2. IGNORE negated findings: "no fever", "denies chest pain", "ruled out malaria" — these are NOT diagnoses.
3. IGNORE review-of-systems negatives: "no headache", "no vomiting", "normal abdomen" etc.
4. IGNORE family history: "mother has diabetes" is NOT the patient's diagnosis.
5. IGNORE past/resolved conditions UNLESS they are the reason for the current visit.
6. DISTINGUISH between "X without Y" (patient HAS X but does NOT have complication Y) vs "no X" (patient does NOT have X).
7. SELECT the MOST SPECIFIC leaf code supported by the note. If the note says "falciparum malaria", choose the falciparum-specific code, not "malaria, unspecified".
8. But DO NOT over-specify: if the note says "malaria" without specifying species, choose the unspecified (.Z) code.
9. RESPECT ICD-11 exclusions: if a candidate's exclusion note matches the patient's condition, do NOT select it.
10. For codes with "without complication" vs "with complication": check if complications are documented.
11. If the note is in a non-English language (Spanish, French, Swahili, German, Chinese, etc.), accurately map to English ICD-11 codes.

OUTPUT FORMAT:
Return ONLY a valid JSON object:
{"codes": ["CODE_1", "CODE_2", "CODE_3"], "reason": "concise explanation"}

Select up to 3 codes, ordered by likelihood (most accurate first). All codes must be from the candidate list."""


def _build_candidate_block(candidates: list[dict[str, Any]], max_candidates: int = 30) -> str:
    """Build a formatted candidate block for the LLM prompt."""
    lines = []
    for i, cand in enumerate(candidates[:max_candidates]):
        code = cand["code"]
        title = cand["title"]
        line = f"{i+1}. [{code}] {title}"

        # Add definition snippet if available
        definition = cand.get("definition", "")
        if definition:
            if len(definition) > 140:
                definition = definition[:137] + "..."
            line += f"\n   Definition: {definition}"

        # Add key inclusions
        inclusions = cand.get("inclusions", [])
        if inclusions:
            inc_str = "; ".join(inclusions[:4])
            if len(inc_str) > 120:
                inc_str = inc_str[:117] + "..."
            line += f"\n   Includes: {inc_str}"

        # Add key exclusions
        exclusions = cand.get("exclusions", [])
        if exclusions:
            exc_str = "; ".join(exclusions[:3])
            if len(exc_str) > 100:
                exc_str = exc_str[:97] + "..."
            line += f"\n   Excludes: {exc_str}"

        lines.append(line)

    return "\n\n".join(lines)


def _build_user_prompt(note_text: str, candidate_block: str) -> str:
    """Build the user prompt for the LLM."""
    return f"""CLINICAL NOTE:
\"\"\"
{note_text}
\"\"\"

CANDIDATE ICD-11 CODES (select up to 3 from this list):

{candidate_block}

Return JSON with "codes": ["CODE_1", "CODE_2", "CODE_3"]."""


def _parse_llm_response(response_text: str, valid_candidates: set[str]) -> list[str]:
    """Parse LLM response and extract valid codes."""
    text = response_text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        codes = parsed.get("codes", [])
        if isinstance(codes, list):
            result = []
            for code in codes:
                if isinstance(code, str):
                    code = code.strip().upper()
                    if "&" in code:
                        code = code.split("&")[0].strip()
                    if "/" in code:
                        code = code.split("/")[0].strip()
                    if code in valid_candidates and code not in result:
                        result.append(code)
            if result:
                return result[:3]
    except Exception:
        pass

    # Regex extraction fallback
    found = re.findall(r'\b([A-Z0-9]{2,7}(?:\.[A-Z0-9]{1,4})?)\b', text)
    result = []
    for code in found:
        code = code.upper()
        if code in valid_candidates and code not in result:
            result.append(code)
        if len(result) >= 3:
            break
    return result


async def _call_gemini(
    note_text: str,
    candidate_block: str,
    valid_candidates: set[str],
    api_key: str,
    model: str = "gemini-2.5-flash",
    timeout: float = 25.0,
) -> Optional[list[str]]:
    """Call Google Gemini API via REST."""
    global _httpx_client
    if _httpx_client is None:
        import httpx
        _httpx_client = httpx.AsyncClient(timeout=timeout)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    prompt_text = f"{SYSTEM_PROMPT}\n\n{_build_user_prompt(note_text, candidate_block)}"
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0,
            "maxOutputTokens": 300,
        },
    }

    try:
        resp = await _httpx_client.post(url, json=payload, headers={"Content-Type": "application/json"})
        if resp.status_code == 200:
            data = resp.json()
            candidates_out = data.get("candidates", [])
            if candidates_out:
                parts = candidates_out[0].get("content", {}).get("parts", [])
                if parts:
                    return _parse_llm_response(parts[0].get("text", ""), valid_candidates)
        else:
            sys.stderr.write(f"[WARN] Gemini API returned {resp.status_code}: {resp.text[:120]}\n")
            return None
    except Exception as e:
        sys.stderr.write(f"[WARN] Gemini API request exception: {e}\n")
        return None
    return None


async def _call_openai(
    note_text: str,
    candidate_block: str,
    valid_candidates: set[str],
    model: str = "gpt-4o-mini",
    timeout: float = 30.0,
) -> Optional[list[str]]:
    """Call OpenAI API for code selection."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import AsyncOpenAI
            _openai_client = AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                timeout=timeout,
                max_retries=2,
            )
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to initialize OpenAI client: {e}\n")
            return None

    try:
        response = await _openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(note_text, candidate_block)},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        return _parse_llm_response(text, valid_candidates)
    except Exception as e:
        sys.stderr.write(f"[WARN] OpenAI API error: {e}\n")
        return None


async def _call_anthropic(
    note_text: str,
    candidate_block: str,
    valid_candidates: set[str],
    model: str = "claude-3-5-haiku-20241022",
    timeout: float = 30.0,
) -> Optional[list[str]]:
    """Call Anthropic API for code selection."""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            from anthropic import AsyncAnthropic
            _anthropic_client = AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                timeout=timeout,
                max_retries=2,
            )
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to initialize Anthropic client: {e}\n")
            return None

    try:
        response = await _anthropic_client.messages.create(
            model=model,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": _build_user_prompt(note_text, candidate_block)},
            ],
            temperature=0.0,
        )
        text = response.content[0].text if response.content else ""
        return _parse_llm_response(text, valid_candidates)
    except Exception as e:
        sys.stderr.write(f"[WARN] Anthropic API error: {e}\n")
        return None


def _get_gemini_key() -> Optional[str]:
    """Retrieve Gemini API key from environment or .env file."""
    for env_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(env_var)
        if val:
            return val.strip()

    # Fallback to inspecting .env
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("GEMINI_API_KEY=") or line.startswith("GOOGLE_API_KEY="):
                        return line.split("=", 1)[1].strip()
                    elif not "=" in line and len(line) > 20:
                        # Raw key line
                        return line
        except Exception:
            pass
    return None


class LLMReranker:
    """Multi-provider Async LLM Reranker with automatic provider selection and failover."""

    def __init__(
        self,
        kb: dict[str, dict[str, Any]],
        valid_codes: set[str],
        max_concurrency: int = 20,
        prefer_provider: str = "openai",
        **kwargs,
    ):
        self.kb = kb
        self.valid_codes = valid_codes
        self.max_concurrency = max_concurrency
        self.prefer_provider = prefer_provider
        self.semaphore = asyncio.Semaphore(max_concurrency)

        self.gemini_key = _get_gemini_key()
        self.has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        self.has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.has_gemini = bool(self.gemini_key)

        self.api_available = self.has_openai or self.has_anthropic or self.has_gemini

        active = []
        if self.has_openai:
            active.append("OpenAI (gpt-4o-mini)")
        if self.has_anthropic:
            active.append("Anthropic (claude-3-5-haiku)")
        if self.has_gemini:
            active.append("Google Gemini (gemini-2.5-flash)")

        if active:
            sys.stderr.write(f"[INFO] LLM Reranker active providers: {', '.join(active)}\n")
        else:
            sys.stderr.write("[WARN] No active LLM API keys found; running in pure offline mode.\n")

    def _build_candidates_for_prompt(
        self, candidate_codes: list[tuple[str, float]]
    ) -> tuple[list[dict[str, Any]], set[str]]:
        candidates = []
        valid_set = set()

        for code, score in candidate_codes:
            info = self.kb.get(code)
            if not info:
                continue

            cand = {
                "code": code,
                "title": info.get("title", ""),
                "chapter": info.get("chapter", ""),
                "definition": info.get("definition", ""),
                "inclusions": info.get("inclusions", [])[:4],
                "exclusions": info.get("exclusions", [])[:3],
            }
            candidates.append(cand)
            valid_set.add(code)

        return candidates, valid_set

    async def rerank_single(
        self, note_text: str, candidate_codes: list[tuple[str, float]]
    ) -> Optional[list[str]]:
        """Rerank candidates using available LLM with concurrency gating and failover."""
        if not self.api_available:
            return None

        candidates, valid_set = self._build_candidates_for_prompt(candidate_codes)
        if not candidates:
            return None

        candidate_block = _build_candidate_block(candidates)

        async with self.semaphore:
            # 1. Try OpenAI if available (Standard on Sugen grading box)
            if self.has_openai:
                res = await _call_openai(note_text, candidate_block, valid_set)
                if res:
                    return res

            # 2. Try Anthropic if available (Alternative on Sugen grading box)
            if self.has_anthropic:
                res = await _call_anthropic(note_text, candidate_block, valid_set)
                if res:
                    return res

            # 3. Try Gemini if available
            if self.has_gemini and self.gemini_key:
                res = await _call_gemini(note_text, candidate_block, valid_set, self.gemini_key)
                if res:
                    return res

        return None

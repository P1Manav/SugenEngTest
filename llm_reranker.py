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
_gemini_disabled_until = 0.0
_gemini_429_count = 0
_openai_disabled_until = 0.0
_openai_429_count = 0
_anthropic_disabled_until = 0.0
_anthropic_429_count = 0


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
        lines = [line for line in lines if not line.strip().startswith("```")]
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


def _sync_call_gemini(
    note_text: str,
    candidate_block: str,
    valid_candidates: set[str],
    api_key: str,
    model: str = "gemini-3.5-flash",
    timeout: float = 8.0,
) -> Optional[list[str]]:
    """Synchronous worker calling Gemini via urllib with circuit breaker and model fallback."""
    global _gemini_disabled_until
    if time.time() < _gemini_disabled_until:
        return None

    import urllib.request
    import urllib.error

    models_to_try = [model, "gemini-3.6-flash"]
    prompt_text = f"{SYSTEM_PROMPT}\n\n{_build_user_prompt(note_text, candidate_block)}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0,
            "maxOutputTokens": 300,
        },
    }).encode("utf-8")

    for m in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    _gemini_429_count = 0
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates_out = data.get("candidates", [])
                    if candidates_out:
                        parts = candidates_out[0].get("content", {}).get("parts", [])
                        if parts:
                            parsed = _parse_llm_response(parts[0].get("text", ""), valid_candidates)
                            if parsed:
                                return parsed
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:100]
            if e.code == 429:
                _gemini_429_count += 1
                backoff = 60.0 if _gemini_429_count == 1 else 300.0
                _gemini_disabled_until = time.time() + backoff
                sys.stderr.write(f"[WARN] Gemini 429 quota/rate limit (count={_gemini_429_count}): circuit open for {int(backoff)}s\n")
                return None
            sys.stderr.write(f"[WARN] Gemini ({m}) HTTP {e.code}: {body}\n")
        except Exception as e:
            sys.stderr.write(f"[WARN] Gemini ({m}) error: {e}\n")

    return None


async def _call_gemini(
    note_text: str,
    candidate_block: str,
    valid_candidates: set[str],
    api_key: str,
    model: str = "gemini-3.5-flash",
    timeout: float = 15.0,
) -> Optional[list[str]]:
    """Call Google Gemini API asynchronously in thread pool with automatic model fallback."""
    return await asyncio.to_thread(
        _sync_call_gemini,
        note_text,
        candidate_block,
        valid_candidates,
        api_key,
        model,
        timeout,
    )


async def _call_openai(
    note_text: str,
    candidate_block: str,
    valid_candidates: set[str],
    model: str = "gpt-4o-mini",
    timeout: float = 20.0,
) -> Optional[list[str]]:
    """Call OpenAI API for code selection."""
    global _openai_client, _openai_disabled_until
    if time.time() < _openai_disabled_until:
        return None

    if _openai_client is None:
        try:
            from openai import AsyncOpenAI
            _openai_client = AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                timeout=timeout,
                max_retries=1,
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
        err_str = str(e).lower()
        if "rate_limit" in err_str or "quota" in err_str or "429" in err_str:
            _openai_disabled_until = time.time() + 30.0
            sys.stderr.write("[WARN] OpenAI rate limit/quota: circuit open for 30s\n")
        else:
            sys.stderr.write(f"[WARN] OpenAI API error: {e}\n")
        return None


async def _call_anthropic(
    note_text: str,
    candidate_block: str,
    valid_candidates: set[str],
    model: str = "claude-3-5-haiku-20241022",
    timeout: float = 20.0,
) -> Optional[list[str]]:
    """Call Anthropic API for code selection."""
    global _anthropic_client, _anthropic_disabled_until
    if time.time() < _anthropic_disabled_until:
        return None

    if _anthropic_client is None:
        try:
            from anthropic import AsyncAnthropic
            _anthropic_client = AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                timeout=timeout,
                max_retries=1,
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
        err_str = str(e).lower()
        if "rate_limit" in err_str or "quota" in err_str or "429" in err_str:
            _anthropic_disabled_until = time.time() + 30.0
            sys.stderr.write("[WARN] Anthropic rate limit/quota: circuit open for 30s\n")
        else:
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
                    elif "=" not in line and len(line) > 20:
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
            active.append("Google Gemini (gemini-3.5-flash)")

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
            # Determine provider order based on preference and available keys
            providers = []
            if self.prefer_provider == "gemini":
                if self.has_gemini:
                    providers.append("gemini")
                if self.has_openai:
                    providers.append("openai")
                if self.has_anthropic:
                    providers.append("anthropic")
            else:
                if self.has_openai:
                    providers.append("openai")
                if self.has_anthropic:
                    providers.append("anthropic")
                if self.has_gemini:
                    providers.append("gemini")

            for p in providers:
                if p == "openai":
                    res = await _call_openai(note_text, candidate_block, valid_set)
                    if res:
                        return res
                elif p == "anthropic":
                    res = await _call_anthropic(note_text, candidate_block, valid_set)
                    if res:
                        return res
                elif p == "gemini" and self.gemini_key:
                    res = await _call_gemini(note_text, candidate_block, valid_set, self.gemini_key)
                    if res:
                        return res

        return None

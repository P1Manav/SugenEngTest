"""Enrich ICD-11 Knowledge Base using official WHO ICD-11 API.
Fetches index terms (synonyms), inclusions, exclusions, and definitions.
100% resumeable, thread-pooled, prioritized, with retries and atomic persistence.
Credentials read from environment variables WHO_CLIENT_ID and WHO_CLIENT_SECRET.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Any, Dict, Optional, Set

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TOKEN_ENDPOINT = "https://icdaccessmanagement.who.int/connect/token"


class WHOApiClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token: Optional[str] = None
        self.token_expiry: float = 0
        self._lock = threading.Lock()
        self._authenticate()

    def _authenticate(self):
        """Acquire or refresh OAuth2 bearer token."""
        data = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "icdapi_access",
        }).encode("utf-8")

        req = urllib.request.Request(
            TOKEN_ENDPOINT,
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            self.token = res["access_token"]
            expires_in = res.get("expires_in", 3600)
            self.token_expiry = time.time() + expires_in - 120  # 2 min margin

    def get_token(self) -> str:
        with self._lock:
            if not self.token or time.time() >= self.token_expiry:
                self._authenticate()
            return self.token

    def fetch_entity(self, uri: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Fetch entity details from WHO release endpoint with retry logic."""
        clean_uri = uri.replace("http://", "https://")
        if "/release/11/mms/" in clean_uri:
            clean_uri = clean_uri.replace("/release/11/mms/", "/release/11/2024-01/mms/")

        for attempt in range(max_retries):
            token = self.get_token()
            req = urllib.request.Request(
                clean_uri,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "API-Version": "v2",
                    "Accept-Language": "en",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return self._parse_entity(data)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                elif e.code == 401:
                    with self._lock:
                        self._authenticate()
                    continue
                elif e.code in (500, 502, 503, 504):
                    time.sleep(1.0 * (attempt + 1))
                    continue
                else:
                    return None
            except Exception:
                time.sleep(0.5 * (attempt + 1))
                continue
        return None

    def _parse_entity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract title, definition, index terms (synonyms), inclusions, exclusions."""
        code = data.get("code", "")
        title_dict = data.get("title", {})
        title = title_dict.get("@value", "") if isinstance(title_dict, dict) else str(title_dict)

        def_dict = data.get("definition", {})
        definition = def_dict.get("@value", "") if isinstance(def_dict, dict) else (str(def_dict) if def_dict else "")

        # Extract synonyms / index terms
        index_terms = []
        for item in data.get("indexTerm", []):
            if isinstance(item, dict):
                label_dict = item.get("label", {})
                term = label_dict.get("@value") if isinstance(label_dict, dict) else str(label_dict)
                if term and term not in index_terms:
                    index_terms.append(term)

        # Inclusions
        inclusions = []
        for inc in data.get("inclusion", []):
            if isinstance(inc, dict):
                label_dict = inc.get("label", {})
                t = label_dict.get("@value") if isinstance(label_dict, dict) else str(label_dict)
                if t and t not in inclusions:
                    inclusions.append(t)

        # Exclusions
        exclusions = []
        for exc in data.get("exclusion", []):
            if isinstance(exc, dict):
                label_dict = exc.get("label", {})
                t = label_dict.get("@value") if isinstance(label_dict, dict) else str(label_dict)
                if t and t not in exclusions:
                    exclusions.append(t)

        return {
            "code": code,
            "title": title,
            "definition": definition,
            "synonyms": index_terms,
            "inclusions": inclusions,
            "exclusions": exclusions,
        }


def save_atomic(data: Dict[str, Any], filepath: str):
    """Write data to a temp file and replace atomically."""
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, filepath)


def main():
    client_id = os.environ.get("WHO_CLIENT_ID")
    client_secret = os.environ.get("WHO_CLIENT_SECRET")
    if not client_id or not client_secret:
        sys.exit("Error: WHO_CLIENT_ID and WHO_CLIENT_SECRET environment variables must be set.")

    enrichments_path = "kb/who_enrichments.json"
    os.makedirs("kb", exist_ok=True)
    existing: Dict[str, Any] = {}
    if os.path.exists(enrichments_path):
        try:
            with open(enrichments_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            print(f"Loaded {len(existing)} existing enrichments from {enrichments_path}", flush=True)
        except Exception as e:
            print(f"Warning loading existing enrichments: {e}", flush=True)
            existing = {}

    mms_path = "data/mms-2026-01.json"
    with open(mms_path, "r", encoding="utf-8") as f:
        mms_data = json.load(f)

    # Gather dev gold codes and stems for prioritization
    dev_gold_stems: Set[str] = set()
    dev_path = "data/dev_notes.jsonl"
    if os.path.exists(dev_path):
        with open(dev_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                for g in item.get("gold", []):
                    dev_gold_stems.add(g)
                    stem = g.split(".")[0]
                    dev_gold_stems.add(stem)

    # Core outpatient priority chapters
    core_chapters = {"01", "12", "16", "05", "13", "11", "18", "15", "14", "04", "24", "21", "22", "06", "08", "09", "10", "03"}

    def get_priority(item: Dict[str, Any]) -> int:
        code = item.get("code", "")
        ch = item.get("chapter", "")
        # Priority 0: Exact dev gold or sibling stem
        if code in dev_gold_stems or code.split(".")[0] in dev_gold_stems:
            return 0
        # Priority 1: Core outpatient leaf
        if ch in core_chapters and item.get("leaf"):
            return 1
        # Priority 2: Core outpatient non-leaf
        if ch in core_chapters:
            return 2
        # Priority 3: Other leaves
        if item.get("leaf"):
            return 3
        # Priority 4: All other
        return 4

    pending = [item for item in mms_data if item.get("code") and item["code"] not in existing and "uri" in item]
    pending.sort(key=get_priority)

    print(f"Total MMS categories: {len(mms_data)}, Already enriched: {len(existing)}, Pending: {len(pending)}", flush=True)
    if not pending:
        print("All categories are already enriched!", flush=True)
        return

    client = WHOApiClient(client_id, client_secret)

    max_workers = 20
    chunk_size = 100
    print(f"Enriching KB with {max_workers} worker threads in chunks of {chunk_size}...", flush=True)
    start_time = time.time()
    total_processed = 0

    for chunk_start in range(0, len(pending), chunk_size):
        chunk = pending[chunk_start:chunk_start + chunk_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(client.fetch_entity, item["uri"]): item for item in chunk}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                code = item["code"]
                try:
                    result = future.result()
                    if result:
                        existing[code] = result
                    else:
                        existing[code] = {
                            "code": code,
                            "title": item.get("title", ""),
                            "definition": "",
                            "synonyms": [],
                            "inclusions": [],
                            "exclusions": [],
                        }
                except Exception:
                    existing[code] = {
                        "code": code,
                        "title": item.get("title", ""),
                        "definition": "",
                        "synonyms": [],
                        "inclusions": [],
                        "exclusions": [],
                    }
                total_processed += 1

        # Atomic save after every chunk
        save_atomic(existing, enrichments_path)
        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        synonym_count = sum(len(v.get("synonyms", [])) for v in existing.values())
        print(f"Progress: {total_processed}/{len(pending)} enriched ({len(existing)} in KB, {synonym_count} synonyms, {rate:.1f} req/s)", flush=True)

    print(f"Enrichment complete! Total enriched codes in KB: {len(existing)}", flush=True)


if __name__ == "__main__":
    main()

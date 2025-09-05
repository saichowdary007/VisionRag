from __future__ import annotations

import json
from typing import List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _get_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.2,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=retry, pool_block=False)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"Connection": "keep-alive"})
    return s


def expand_queries_lmstudio(
    url: str,
    model: str,
    query: str,
    n: int = 3,
    temperature: float = 0.2,
    timeout_s: int = 30,
) -> List[str]:
    """Generate N concise, diverse sub-queries for RAG-Fusion via LM Studio-compatible API.

    Returns a list of strings. Falls back to [query] on failure.
    """
    system = (
        "You are a query reformulation assistant for document retrieval. "
        "Generate concise, diverse search queries that capture exact keywords, part numbers, error codes, and synonyms. "
        "No preamble. Output ONLY a JSON array of strings."
    )
    user = (
        f"Create {max(1, int(n))} diverse queries for retrieving steps from technical manuals.\n"
        f"Original: {query}\n"
        "Constraints: each <= 12 words; include exact model/part numbers if present; avoid quotes and punctuation beyond hyphens and slashes."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": float(temperature),
        "max_tokens": 256,
        "stream": False,
    }
    try:
        sess = _get_session()
        resp = sess.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Expect a JSON array; fallback to line-split if needed
        queries: List[str] = []
        try:
            arr = json.loads(content)
            if isinstance(arr, list):
                queries = [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            # split lines
            for line in content.splitlines():
                line = line.strip().lstrip("-*").strip()
                if not line:
                    continue
                # remove numbering like "1. "
                if len(line) > 2 and line[0].isdigit() and line[1] in ".)":
                    line = line[2:].strip()
                queries.append(line)
        # Ensure the original is included
        base = query.strip()
        if not queries:
            return [base]
        if base not in queries:
            queries = [base] + queries
        else:
            # ensure original is first
            queries = [base] + [q for q in queries if q != base]
        # Deduplicate while preserving order, cap length
        seen = set()
        out: List[str] = []
        for q in queries:
            if q in seen:
                continue
            seen.add(q)
            out.append(q)
            if len(out) >= (n + 1):  # plus original
                break
        return out
    except Exception:
        return [query]


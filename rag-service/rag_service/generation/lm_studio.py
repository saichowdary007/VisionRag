from __future__ import annotations

import json
from typing import List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..types import SearchResult


def _format_context(chunks: List[SearchResult], max_chars: int = 2500) -> str:
    lines: List[str] = []
    total = 0
    for i, ch in enumerate(chunks, 1):
        header = f"[{i}] {ch.doc_id} — p.{ch.page}"
        text = ch.text.strip().replace("\n", " ")
        block = f"{header}\n{text}\n"
        if total + len(block) > max_chars:
            break
        lines.append(block)
        total += len(block)
    return "\n".join(lines)


_session: requests.Session | None = None
_async_client: httpx.AsyncClient | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=3,  # Increased retries
            backoff_factor=0.1,  # Faster backoff
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        # Optimized connection pooling for better performance
        adapter = HTTPAdapter(
            pool_connections=10,  # Increased connection pool
            pool_maxsize=20,      # Increased max connections
            max_retries=retry,
            pool_block=False      # Don't block when pool is full
        )
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        # Keep connections alive longer
        s.headers.update({'Connection': 'keep-alive'})
        _session = s
    return _session


def _get_async_client(timeout_s: int = 120) -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        _async_client = httpx.AsyncClient(timeout=timeout_s, limits=limits)
    return _async_client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.2, min=0.2, max=2.0), reraise=True)
def generate_with_lm_studio(
    url: str,
    model: str,
    user_query: str,
    chunks: List[SearchResult],
    temperature: float = 0.1,
    max_tokens: int = 1024,
    system_prompt: str | None = None,
    timeout_s: int = 120,  # Increased from 15s to 120s for large models
) -> Tuple[str, dict]:
    if not system_prompt:
        system_prompt = (
            "You are a retrieval-augmented assistant that answers using ONLY the provided context. "
            "Documents can be from any domain. For each factual claim, add citations as (Doc, p.X) or (Doc, Section) when page is unknown. "
            "If the context is insufficient, reply exactly: 'Information not found in provided documents.' "
            "Prefer concise, stepwise instructions when the user asks for procedures; preserve units, warnings, and exact terminology. Avoid speculation."
        )

    context = _format_context(chunks)
    prompt = (
        "Use ONLY the provided context. For every claim, add citations.\n\n"
        f"Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer with citations:"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }

    session = _get_session()
    resp = session.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content.strip(), data


# Async version for better performance
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.2, min=0.2, max=2.0), reraise=True)
async def generate_with_lm_studio_async(
    url: str,
    model: str,
    user_query: str,
    chunks: List[SearchResult],
    temperature: float = 0.1,
    max_tokens: int = 1024,
    system_prompt: str | None = None,
    timeout_s: int = 120,
) -> Tuple[str, dict]:
    """Async version of generate_with_lm_studio for better performance"""
    if not system_prompt:
        system_prompt = (
            "You are a retrieval-augmented assistant that answers using ONLY the provided context. "
            "Documents can be from any domain. For each factual claim, add citations as (Doc, p.X) or (Doc, Section) when page is unknown. "
            "If the context is insufficient, reply exactly: 'Information not found in provided documents.' "
            "Prefer concise, stepwise instructions when the user asks for procedures; preserve units, warnings, and exact terminology. Avoid speculation."
        )

    context = _format_context(chunks)
    prompt = (
        "Use ONLY the provided context. For every claim, add citations.\n\n"
        f"Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer with citations:"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }

    client = _get_async_client(timeout_s)
    resp = await client.post(url, headers={"Content-Type": "application/json"}, json=payload)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content.strip(), data


async def generate_with_lm_studio_stream(
    url: str,
    model: str,
    user_query: str,
    chunks: List[SearchResult],
    temperature: float = 0.1,
    max_tokens: int = 1024,
    system_prompt: str | None = None,
    timeout_s: int = 120,
):
    """Async generator yielding streamed tokens from LM Studio compatible API."""
    if not system_prompt:
        system_prompt = (
            "You are a retrieval-augmented assistant that answers using ONLY the provided context. "
            "Documents can be from any domain. For each factual claim, add citations as (Doc, p.X) or (Doc, Section) when page is unknown. "
            "If the context is insufficient, reply exactly: 'Information not found in provided documents.' "
            "Prefer concise, stepwise instructions when the user asks for procedures; preserve units, warnings, and exact terminology. Avoid speculation."
        )

    context = _format_context(chunks)
    prompt = (
        "Use ONLY the provided context. For every claim, add citations.\n\n"
        f"Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer with citations:"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": True,
    }

    client = _get_async_client(timeout_s)
    async with client.stream("POST", url, headers={"Content-Type": "application/json"}, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line.replace("data: ", ""))
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    yield delta
            except Exception:
                continue

from __future__ import annotations

import json
from typing import Any, Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dep
    redis = None  # type: ignore


def get_client(url: str):
    if not url or redis is None:
        return None
    try:
        return redis.Redis.from_url(url, decode_responses=True)
    except Exception:
        return None


def get_json(r, key: str) -> Optional[Any]:
    if r is None:
        return None
    try:
        val = r.get(key)
        if val is None:
            return None
        return json.loads(val)
    except Exception:
        return None


def set_json(r, key: str, value: Any, ttl_seconds: int = 3600) -> None:
    if r is None:
        return
    try:
        r.setex(key, ttl_seconds, json.dumps(value))
    except Exception:
        pass


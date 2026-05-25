from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from threading import Lock
from typing import Any

from ..config import get_settings


_STATE_LOCK = Lock()
_CACHE_STATE: dict[str, Any] = {
    "last_error": None,
    "get_count": 0,
    "hit_count": 0,
    "miss_count": 0,
    "set_count": 0,
    "error_count": 0,
}


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, set):
        normalized = [_safe_json(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_safe_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _mark_error(message: str) -> None:
    with _STATE_LOCK:
        _CACHE_STATE["last_error"] = message
        _CACHE_STATE["error_count"] = int(_CACHE_STATE["error_count"]) + 1


def _mark_counter(name: str) -> None:
    with _STATE_LOCK:
        _CACHE_STATE[name] = int(_CACHE_STATE.get(name, 0)) + 1


def cache_enabled() -> bool:
    settings = get_settings()
    return bool(settings.redis_enabled and settings.redis_url)


@lru_cache(maxsize=1)
def _load_redis_client():
    settings = get_settings()
    if not settings.redis_enabled:
        return None
    if not settings.redis_url:
        raise RuntimeError("Redis caching is enabled but PICKMEAL_REDIS_URL is not configured.")
    try:
        import redis
    except Exception as exc:
        raise RuntimeError("redis is not installed in the current environment.") from exc
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def initialize_cache() -> bool:
    if not cache_enabled():
        return False
    try:
        client = _load_redis_client()
        if client is None:
            return False
        client.ping()
        return True
    except Exception as exc:
        _mark_error(str(exc))
        return False


def _fingerprint(payload: Any) -> str:
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cache_identity(payload: Any, key_context: Any | None = None) -> Any:
    if key_context is None:
        return payload
    return {
        "payload": payload,
        "key_context": key_context,
    }


def _cache_key(category: str, payload: Any, key_context: Any | None = None) -> str:
    settings = get_settings()
    return f"{settings.redis_cache_namespace}:{category}:{_fingerprint(_cache_identity(payload, key_context))}"


def get_cached_payload(category: str, payload: Any, key_context: Any | None = None) -> dict[str, Any] | None:
    _mark_counter("get_count")
    if not cache_enabled():
        return None
    try:
        client = _load_redis_client()
        if client is None:
            return None
        raw = client.get(_cache_key(category, payload, key_context))
        if raw is None:
            _mark_counter("miss_count")
            return None
        _mark_counter("hit_count")
        return json.loads(raw)
    except Exception as exc:
        _mark_error(str(exc))
        return None


def set_cached_payload(
    category: str,
    payload: Any,
    response_payload: Any,
    ttl_seconds: int,
    key_context: Any | None = None,
) -> bool:
    if not cache_enabled() or ttl_seconds <= 0:
        return False
    try:
        client = _load_redis_client()
        if client is None:
            return False
        client.setex(
            _cache_key(category, payload, key_context),
            ttl_seconds,
            _canonical_json(response_payload),
        )
        _mark_counter("set_count")
        return True
    except Exception as exc:
        _mark_error(str(exc))
        return False


def _namespace_counts() -> dict[str, int]:
    if not cache_enabled():
        return {}
    try:
        client = _load_redis_client()
        if client is None:
            return {}
        settings = get_settings()
        prefix = f"{settings.redis_cache_namespace}:"
        counts: dict[str, int] = {}
        for key in client.scan_iter(match=f"{prefix}*"):
            tail = str(key)[len(prefix) :]
            category = tail.split(":", 1)[0] or "unknown"
            counts[category] = counts.get(category, 0) + 1
        return counts
    except Exception as exc:
        _mark_error(str(exc))
        return {}


def load_cache_stats() -> dict[str, Any]:
    settings = get_settings()
    available = False
    server_info: dict[str, Any] = {}
    if cache_enabled():
        try:
            client = _load_redis_client()
            if client is not None:
                available = bool(client.ping())
                info = client.info()
                server_info = {
                    "redis_version": info.get("redis_version"),
                    "redis_mode": info.get("redis_mode"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "uptime_in_seconds": info.get("uptime_in_seconds"),
                }
        except Exception as exc:
            _mark_error(str(exc))
    with _STATE_LOCK:
        counters = {
            "get_count": int(_CACHE_STATE["get_count"]),
            "hit_count": int(_CACHE_STATE["hit_count"]),
            "miss_count": int(_CACHE_STATE["miss_count"]),
            "set_count": int(_CACHE_STATE["set_count"]),
            "error_count": int(_CACHE_STATE["error_count"]),
        }
        last_error = _CACHE_STATE["last_error"]
    return {
        "enabled": bool(settings.redis_enabled),
        "configured": bool(settings.redis_url),
        "available": available,
        "redis_url_present": bool(settings.redis_url),
        "namespace": settings.redis_cache_namespace,
        "recommendation_ttl_seconds": settings.recommendation_cache_ttl_seconds,
        "llm_ttl_seconds": settings.llm_cache_ttl_seconds,
        "get_count": counters["get_count"],
        "hit_count": counters["hit_count"],
        "miss_count": counters["miss_count"],
        "set_count": counters["set_count"],
        "error_count": counters["error_count"],
        "local_counters": counters,
        "namespace_key_counts": _namespace_counts() if available else {},
        "server_info": server_info,
        "last_error": last_error,
    }

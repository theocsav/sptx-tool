from time import time
from typing import Any, Dict, Optional


_CACHE: Dict[str, Dict[str, Any]] = {}


def get_cached_join_result(key: str) -> Optional[Dict[str, Any]]:
    now = time()
    entry = _CACHE.get(key)
    if not entry:
        return None
    if entry["expires_at"] <= now:
        _CACHE.pop(key, None)
        return None
    return entry["value"]


def set_cached_join_result(key: str, value: Dict[str, Any], ttl_seconds: int) -> None:
    expires_at = time() + ttl_seconds
    _CACHE[key] = {"expires_at": expires_at, "value": value}


def build_cache_key(dataset_id: str, manifest_hash: str, preset_id: Optional[str]) -> str:
    preset = preset_id or ""
    return f"{dataset_id}:{manifest_hash}:{preset}"

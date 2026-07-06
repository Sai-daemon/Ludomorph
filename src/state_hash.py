"""
State Hashing & Caching — Phase 2.5

Provides a deterministic, salted SHA‑256 hash of the ``GameState`` for
cache-based LLM decision reuse, and a ``StateCache`` with a configurable
TTL (default 0.3 s) and 200-entry limit.

The time‑rotating salt (regenerated every whole second) ensures that
identical states in different seconds produce different hashes, forcing a
fresh LLM decision after at most one second.  The 0.3 s TTL covers rapid
frame‑duplicate scenarios within the same second.

Usage::

    from src.game_state import GameState
    from src.state_hash import state_hash, StateCache

    cache = StateCache(ttl=0.3)
    ...
    h = state_hash(game_state)
    cached = cache.get(h)
    if cached is not None:
        # reuse cached action
        ...
    else:
        action = await llm.decide(game_state)
        cache.set(h, action)
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from typing import Any

# ---------------------------------------------------------------------------
# Time‑rotating salt
# ---------------------------------------------------------------------------

_salt: str = ""
_salt_timestamp: int = 0

# Sentinel types to exclude from serialisation
_SKIP_TYPES = (bytes, bytearray, memoryview)

# Keys whose values should always be skipped (e.g. raw images)
_SKIP_KEY_SUFFIXES = ("_raw_bar", "_raw_image")


def _get_salt() -> str:
    """Return the current salt, regenerated once per whole second."""
    global _salt, _salt_timestamp
    now = int(time.time())  # whole seconds
    if now != _salt_timestamp:
        _salt = secrets.token_hex(16)  # 32 random hex chars
        _salt_timestamp = now
    return _salt


def _should_skip_key(key: str) -> bool:
    """Return ``True`` if *key* should be excluded from hashing."""
    return any(key.endswith(suffix) for suffix in _SKIP_KEY_SUFFIXES)


def _serialisable_value(value: Any) -> bool:
    """Return ``False`` if *value* is a type we cannot / should not serialise."""
    if isinstance(value, _SKIP_TYPES):
        return False
    # numpy arrays (if numpy is installed)
    if hasattr(value, "dtype"):
        return False
    return True


# ---------------------------------------------------------------------------
# Public: state_hash
# ---------------------------------------------------------------------------


def state_hash(state: "GameState") -> str:  # noqa: F821
    """
    Produce a salted SHA‑256 hash of the schema‑relevant fields of *state*.

    The function:
    * Uses only keys present in the schema (``state.schema_typed_dict()``).
    * Sorts lists and dict keys for determinism.
    * Skips binary / numpy‑array values (e.g. ``*_raw_bar``, raw images).
    * Appends a time‑rotating salt that changes every second.

    Within the same second, identical states produce the same hash.
    """
    # 1. Get schema‑typed fields only
    try:
        raw = state.schema_typed_dict()
    except AttributeError:
        # Graceful fallback if the object doesn't have schema_typed_dict
        try:
            raw = state.to_dict()
        except AttributeError:
            raw = {}

    # 2. Build a deterministic, filtered dict
    data: dict[str, Any] = {}
    for key, value in raw.items():
        if _should_skip_key(key):
            continue
        if not _serialisable_value(value):
            continue
        if isinstance(value, (list, tuple, set)):
            try:
                value = sorted(
                    [str(v) for v in value if _serialisable_value(v)]
                )
            except TypeError:
                value = str(value)
        data[key] = value

    # 3. Deterministic JSON serialisation + salt → SHA‑256
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)
    serialized += _get_salt()
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# StateCache
# ---------------------------------------------------------------------------

class StateCache:
    """Time‑based cache for LLM actions keyed by state hash.

    Each entry is stored as ``(monotonic_timestamp, action)`` and is valid
    for *ttl* seconds after insertion.  On overflow (>200 entries) the
    oldest entry is evicted.

    Thread‑safe for use within a single asyncio event loop (no locks are
    needed when used from the same thread).
    """

    def __init__(self, ttl: float = 0.3) -> None:
        self._ttl = ttl
        self._store: dict[str, tuple[float, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, hash_val: str, now: float | None = None) -> Any | None:
        """Return the cached action for *hash_val* if within TTL, else ``None``.

        Expired entries are automatically evicted on access.
        """
        if now is None:
            now = time.monotonic()
        entry = self._store.get(hash_val)
        if entry is None:
            return None
        ts, action = entry
        if now - ts < self._ttl:
            return action
        # Expired — evict
        del self._store[hash_val]
        return None

    def set(self, hash_val: str, action: Any) -> None:
        """Store *action* keyed by *hash_val* with a timestamp of now."""
        self._store[hash_val] = (time.monotonic(), action)
        # Prevent unbounded growth
        if len(self._store) > 200:
            oldest = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest]

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, hash_val: str) -> bool:
        return hash_val in self._store


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = ["state_hash", "StateCache"]
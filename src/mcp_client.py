"""
MCP Memory Client — Phase 3.2

Async HTTP client for the bundled MCP memory server with local LRU
cache (TTL 5 s), semantic search, and event storage methods.

Spec references
---------------
* ``Implementation_Phases.md`` §3.2 — phase definition
* ``architecture.md`` §6.3 — ``MCPMemoryClient`` skeleton with LRU cache
* ``MCP_research.md`` lines 300‑552 — corrected API endpoints:
  * ``POST /api/search`` (semantic search, JSON body ``{"query":…, "n_results":…, "tags":…}``)
  * ``POST /api/memories`` (store, JSON body ``{"content":…, "memory_type":…, "tags":…}``)
  * ``DELETE /api/memories/{content_hash}`` (deletion)
  * Fallback: ``GET /api/memories`` with pagination + client‑side tag filtering
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL: str = "http://localhost:8000"
_DEFAULT_CACHE_TTL: float = 5.0
_DEFAULT_REQUEST_TIMEOUT: float = 5.0
_MAX_CACHE_ENTRIES: int = 100
_FALLBACK_PAGE_SIZE: int = 20
_FALLBACK_MAX_PAGES: int = 5


# ---------------------------------------------------------------------------
# MCPMemoryClient
# ---------------------------------------------------------------------------


class MCPMemoryClient:
    """Async HTTP client for the MCP memory server with local LRU cache.

    The cache uses a plain ``dict`` with TTL‑aware expiry and size
    limiting (oldest entries evicted first when the cap is breached).

    Every ``search_memories`` call first checks the local cache;
    on a miss the semantic‑search endpoint is called, and the result
    is cached.  ``store_memory`` is never cached (append‑only).

    Attributes:
        base_url: Base URL of the MCP memory server (default
            ``http://localhost:8000``).
        cache_ttl: Time‑to‑live for local cache entries in seconds
            (default 5.0).
    """

    __slots__ = (
        "base_url",
        "cache_ttl",
        "_client",
        "_cache",
        "_cache_order",
    )

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        cache_ttl: float = _DEFAULT_CACHE_TTL,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_ttl = cache_ttl
        self._client = httpx.AsyncClient(timeout=_DEFAULT_REQUEST_TIMEOUT)
        # _cache: key → (monotonic_timestamp, result)
        self._cache: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]]]] = {}
        self._cache_order: list[tuple[Any, ...]] = []  # insertion order for LRU eviction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_memories(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memories semantically and return the most relevant results.

        Checks the local LRU cache first; on a cache miss calls
        ``POST /api/search`` on the MCP server.  Falls back to
        paginated tag‑based listing if the search endpoint fails.

        Args:
            query: Semantic search query string (must be non‑empty).
            tags: Optional list of tags to filter by.
            limit: Maximum number of results to return.

        Returns:
            A list of memory dicts, each augmented with
            ``similarity_score`` and ``relevance_reason`` keys when
            returned via the primary search path.  Returns an empty
            list on total failure.
        """
        cache_key = (query, tuple(tags or []), limit)

        # -- Local cache check --
        if cache_key in self._cache:
            ts, result = self._cache[cache_key]
            if time.monotonic() - ts < self.cache_ttl:
                logger.debug(
                    "MCP cache hit: query={!r} tags={} limit={}",
                    query,
                    tags,
                    limit,
                )
                return result
            # TTL expired — purge the stale entry
            del self._cache[cache_key]
            if cache_key in self._cache_order:
                self._cache_order.remove(cache_key)

        # -- Cache miss — call MCP server --
        logger.debug(
            "MCP cache miss — searching: query={!r} tags={} limit={}",
            query,
            tags,
            limit,
        )

        try:
            result = await self._search_remote(query, tags, limit)
        except Exception as exc:
            logger.warning(
                "MCP search failed ({}) — falling back to paginated listing.",
                exc,
            )
            result = await self._fallback_search(tags, limit)

        # Store in local cache (with LRU eviction)
        self._cache[cache_key] = (time.monotonic(), result)
        self._cache_order.append(cache_key)
        self._maybe_evict_cache()

        return result

    async def store_memory(
        self,
        content: str,
        memory_type: str = "short_term",
        tags: list[str] | None = None,
    ) -> None:
        """Store a new memory event on the MCP server (append‑only).

        This method is intentionally NOT cached — every call results
        in a fresh ``POST /api/memories`` request.

        Args:
            content: The memory content string (can be JSON).
            memory_type: Memory tier tag (e.g. ``"short_term"``,
                ``"medium_term"``, ``"long_term"``).
            tags: Optional list of string tags for filtering.
        """
        # Map our tier names to valid MCP server memory types.
        # The server only accepts: observation, decision, learning,
        # error, pattern, session, note.  We use "observation" as the
        # canonical type and encode the tier in the tags list.
        _TIER_MAP = {
            "short_term": "observation",
            "medium_term": "observation",
            "long_term": "observation",
            "summary": "observation",
        }
        canonical_type = _TIER_MAP.get(memory_type, memory_type)
        payload: dict[str, Any] = {
            "content": content,
            "memory_type": canonical_type,
            "tags": tags or [],
        }
        url = f"{self.base_url}/api/memories"
        logger.debug(
            "Storing memory: type={!r} tags={} content_len={}",
            memory_type,
            tags,
            len(content),
        )
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("MCP store_memory failed: {}", exc)

    async def list_memories_by_tag(
        self,
        tag: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch memories filtered by a single tag via paginated listing.

        Calls ``GET /api/memories?tag=...&page_size=...`` on the MCP server.
        This bypasses the semantic search endpoint (which does not support
        tag filtering) and retrieves memories directly.

        Args:
            tag: A single tag string to filter by.
            limit: Maximum total memories to return across all pages.

        Returns:
            A list of memory dicts sorted most‑recent first.
        """
        all_memories: list[dict[str, Any]] = []
        page = 1

        while len(all_memories) < limit:
            url = f"{self.base_url}/api/memories"
            response = await self._client.get(
                url,
                params={
                    "page": page,
                    "page_size": min(limit, 50),
                    "tag": tag,
                },
            )
            response.raise_for_status()
            data = response.json()

            memories: list[dict[str, Any]] = data.get("memories", [])
            if not memories:
                break

            all_memories.extend(memories)

            if not data.get("has_more", False):
                break

            page += 1
            if page > 10:
                break

        logger.debug(
            "list_memories_by_tag: tag={} returned {} results",
            tag,
            len(all_memories),
        )
        return all_memories[:limit]

    async def delete_memory(self, content_hash: str) -> None:
        """Delete a memory by its ``content_hash``.

        Calls ``DELETE /api/memories/{content_hash}`` on the MCP server.
        Used by the ``MemorySummariser`` (Phase 3.4) to remove raw
        short‑term events after they have been compressed into a summary.

        Args:
            content_hash: The unique content hash of the memory to delete.

        Raises:
            httpx.HTTPError: If the server returns a non‑2xx status.
        """
        url = f"{self.base_url}/api/memories/{content_hash}"
        logger.debug("Deleting memory: hash={}", content_hash)
        response = await self._client.delete(url)
        response.raise_for_status()
        logger.debug("Memory deleted: hash={}", content_hash)

    async def close(self) -> None:
        """Close the underlying ``httpx.AsyncClient``.

        Safe to call multiple times; subsequent operations will raise
        ``httpx.HTTPError`` because the client is closed.
        """
        await self._client.aclose()
        self._cache.clear()
        self._cache_order.clear()
        logger.debug("MCPMemoryClient closed.")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_memory_query(state: Any) -> str:
        """Build a concise query string from the current game state.

        Used to search MCP memories for relevant past events.  This
        mirrors the richer format described in ``MCP_research.md``
        lines 275‑287.

        Args:
            state: A ``GameState`` object (with ``to_dict()``) or
                plain dict.

        Returns:
            A comma‑separated summary suitable for semantic search.
        """
        if hasattr(state, "to_dict"):
            d = state.to_dict()
        elif isinstance(state, dict):
            d = state
        else:
            return "game state"

        parts: list[str] = []
        for k, v in d.items():
            if k.startswith("_") or k.endswith("_raw_bar"):
                continue
            parts.append(f"{k}: {v}")
            if len(parts) >= 8:  # richer than the 5-field stub in decision_loop
                break

        return ", ".join(parts) if parts else "game state"

    # ------------------------------------------------------------------
    # Internal — remote search
    # ------------------------------------------------------------------

    async def _search_remote(
        self,
        query: str,
        tags: list[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Call ``POST /api/search`` and unpack ``SearchResponse``."""
        url = f"{self.base_url}/api/search"
        payload: dict[str, Any] = {"query": query, "n_results": limit}
        if tags:
            payload["tags"] = tags

        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        memories: list[dict[str, Any]] = []
        for r in results:
            mem = r.get("memory", {})
            mem["similarity_score"] = r.get("similarity_score")
            mem["relevance_reason"] = r.get("relevance_reason")
            memories.append(mem)

        logger.debug("MCP search returned {} result(s).", len(memories))
        return memories

    # ------------------------------------------------------------------
    # Internal — fallback (paginated listing + client‑side tag filter)
    # ------------------------------------------------------------------

    async def _fallback_search(
        self,
        tags: list[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch recent memories via paginated ``GET /api/memories`` and
        filter by *tags* on the client side.

        This is only reached when ``POST /api/search`` fails (e.g.
        server overload, unexpected error).  It provides graceful
        degradation rather than returning an empty list immediately.
        """
        all_memories: list[dict[str, Any]] = []
        page = 1

        while len(all_memories) < limit * 2:
            url = f"{self.base_url}/api/memories"
            response = await self._client.get(
                url,
                params={"page": page, "page_size": _FALLBACK_PAGE_SIZE},
            )
            response.raise_for_status()
            data = response.json()

            memories: list[dict[str, Any]] = data.get("memories", [])
            if not memories:
                break

            all_memories.extend(memories)

            if not data.get("has_more", False):
                break

            page += 1
            if page > _FALLBACK_MAX_PAGES:
                break

        # Client‑side tag filtering
        if tags:
            tag_set = set(tags)
            filtered = [
                m
                for m in all_memories
                if tag_set.intersection(m.get("tags", []))
            ]
        else:
            filtered = all_memories

        # Most‑recent first (server already orders by creation)
        return filtered[:limit]

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _maybe_evict_cache(self) -> None:
        """Evict oldest entries when the cache exceeds ``_MAX_CACHE_ENTRIES``."""
        while len(self._cache_order) > _MAX_CACHE_ENTRIES:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "MCPMemoryClient",
]
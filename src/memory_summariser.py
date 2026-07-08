"""
Memory Summariser — Phase 3.4

Periodic background ``asyncio.Task`` that compresses short‑term memory
events into medium‑term summaries using a secondary LLM call, then
deletes the raw events from the MCP memory server.

Trigger
-------
* Every 100 new short‑term events **or** every 5 minutes, whichever
  comes first.
* Concurrency protected by an ``asyncio.Lock`` — overlapping runs are
  impossible.

Pipeline (inside :meth:`_summarise`)
-------------------------------------
1. Search for ``short_term``‑tagged memories via the MCP client.
2. Build a summarisation prompt from those events.
3. Call the secondary LLM (Ollama ``/api/chat``) with the prompt.
4. Store the returned summary as a ``medium_term`` memory.
5. Delete the raw short‑term events by ``content_hash``.
6. Reset the event counter and timestamp.

Spec references
---------------
* ``Implementation_Phases.md`` §3.4 — phase definition
* ``MCP_research.md`` lines 552‑870 — ``MemorySummariser`` pseudocode
  with corrected API endpoints
* ``architecture.md`` §6.4 — memory tiers (short‑term → medium‑term)
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import httpx

from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.mcp_client import MCPMemoryClient

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_EVENT_THRESHOLD: int = 100
"""Number of short‑term events before a summarisation run is triggered."""

_DEFAULT_TIME_INTERVAL: float = 300.0
"""Seconds between time‑based summarisation triggers (5 minutes)."""

_DEFAULT_OLLAMA_TIMEOUT: float = 10.0
"""Timeout in seconds for the secondary LLM summarisation call."""

_POLL_INTERVAL: float = 1.0
"""How often the background loop wakes up to check trigger conditions."""


# ---------------------------------------------------------------------------
# MemorySummariser
# ---------------------------------------------------------------------------


class MemorySummariser:
    """Periodic background summariser for MCP memory tiers.

    Runs an ``asyncio.Task`` that wakes every second and checks whether
    enough events have accumulated or enough time has passed to trigger
    a summarisation.  The summarisation itself is serialised behind an
    ``asyncio.Lock`` so that only one run can be in‑flight at a time.

    Typical usage::

        summariser = MemorySummariser(mcp_client, ollama_url, model)
        await summariser.start()

        # In the decision loop, after each store_memory call:
        await summariser.record_new_event()

        # On shutdown:
        await summariser.stop()

    Attributes:
        client: MCP memory client instance.
        ollama_url: Base URL for Ollama's native ``/api/chat`` endpoint
            (e.g. ``"http://localhost:11434"`` — the ``/v1`` suffix is
            stripped internally).
        model: Ollama model name used for summarisation.
    """

    __slots__ = (
        "client",
        "ollama_url",
        "model",
        "threshold",
        "interval",
        "_event_count",
        "_last_summary_time",
        "_lock",
        "_task",
    )

    def __init__(
        self,
        client: MCPMemoryClient,
        ollama_url: str,
        model: str,
        event_threshold: int = _DEFAULT_EVENT_THRESHOLD,
        time_interval: float = _DEFAULT_TIME_INTERVAL,
    ) -> None:
        self.client = client
        # Strip /v1 suffix from Ollama URL so /api/chat resolves correctly
        self.ollama_url = _strip_openai_suffix(ollama_url)
        self.model = model
        self.threshold = event_threshold
        self.interval = time_interval

        self._event_count: int = 0
        self._last_summary_time: float = time.monotonic()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the background summarisation loop.

        Safe to call multiple times — subsequent calls are no‑ops if
        the task is already running.
        """
        if self._task is not None and not self._task.done():
            logger.debug("MemorySummariser already running.")
            return
        self._task = asyncio.create_task(self._run())
        logger.info(
            "MemorySummariser started (threshold={} events, interval={:.0f}s, model={!r}).",
            self.threshold,
            self.interval,
            self.model,
        )

    async def stop(self) -> None:
        """Cancel the background task and wait for graceful teardown.

        Safe to call multiple times / when the task was never started.
        """
        if self._task is None:
            return
        if not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("MemorySummariser stopped.")

    # ------------------------------------------------------------------
    # Public hook for the decision loop
    # ------------------------------------------------------------------

    async def record_new_event(self) -> None:
        """Increment the short‑term event counter.

        Call this once for every short‑term memory stored by the
        decision loop (Phase 3.3).  The summariser wakes once per second
        and will trigger a run when the counter crosses the threshold.
        """
        self._event_count += 1

    # ------------------------------------------------------------------
    # Internal — background loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Poll loop: sleeps *POLL_INTERVAL* s, then checks trigger."""
        try:
            while True:
                await asyncio.sleep(_POLL_INTERVAL)
                if self._should_summarise():
                    await self._summarise()
        except asyncio.CancelledError:
            logger.debug("MemorySummariser background task cancelled.")
            raise

    def _should_summarise(self) -> bool:
        """Return ``True`` when either the event count **or** the
        time‑since‑last‑summary thresholds are met."""
        enough_events = self._event_count >= self.threshold
        enough_time = (time.monotonic() - self._last_summary_time) >= self.interval
        return enough_events or enough_time

    # ------------------------------------------------------------------
    # Internal — summarisation run
    # ------------------------------------------------------------------

    async def _summarise(self) -> None:
        """Perform a single summarisation cycle.

        The *asyncio.Lock* ensures at most one cycle runs at a time.
        Failures at any stage are logged and the cycle is retried on the
        next poll tick.
        """
        if self._lock.locked():
            logger.debug("Summarisation already in progress; skipping this tick.")
            return

        async with self._lock:
            logger.info(
                "Starting summarisation cycle (events={}, elapsed={:.0f}s).",
                self._event_count,
                time.monotonic() - self._last_summary_time,
            )

            # 1) Fetch recent short‑term memories (use tag listing — the
            #    semantic search endpoint does not support tag filtering)
            try:
                events = await self.client.list_memories_by_tag(
                    tag="short_term",
                    limit=self.threshold * 2,
                )
            except Exception as exc:
                logger.warning("Failed to fetch short-term memories: {}", exc)
                return

            if not events:
                logger.debug("No short‑term events to summarise.")
                self._reset_counters()
                return

            logger.info("Fetched {} short-term event(s) for summarisation.", len(events))

            # 2) Summarise via secondary LLM
            prompt = self._build_summary_prompt(events)
            summary = await self._call_secondary_llm(prompt)
            if not summary:
                logger.warning("Secondary LLM returned empty summary — aborting cycle.")
                return

            # 3) Store summary as medium‑term memory
            try:
                await self.client.store_memory(
                    content=summary,
                    memory_type="medium_term",
                    tags=["summary", "medium_term"],
                )
                logger.info("Medium-term summary stored ({} chars).", len(summary))
            except Exception as exc:
                logger.warning("Failed to store medium-term summary: {}", exc)
                return

            # 4) Delete raw short‑term events
            deleted_count = 0
            for ev in events:
                mem_hash = ev.get("content_hash") or ev.get("memory", {}).get("content_hash")
                if mem_hash:
                    try:
                        await self.client.delete_memory(mem_hash)
                        deleted_count += 1
                    except Exception as exc:
                        logger.debug(
                            "Failed to delete memory {}: {}",
                            mem_hash,
                            exc,
                        )
                else:
                    logger.debug(
                        "Event {!r} has no content_hash; skipping deletion.",
                        ev.get("id", "unknown"),
                    )

            logger.info(
                "Summarisation cycle complete — deleted {}/{} raw events.",
                deleted_count,
                len(events),
            )

            # 5) Reset counters
            self._reset_counters()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary_prompt(events: list[dict[str, Any]]) -> str:
        """Build the secondary‑LLM prompt from a list of event dicts.

        Each event dict is expected to have a ``"content"`` key
        containing the stored text (may be a JSON string from Phase 3.3).
        """
        event_lines: list[str] = []
        for ev in events:
            content = ev.get("content", "")
            # Truncate extremely long content blobs to avoid blowing the
            # context window of the secondary model.
            event_lines.append(f"- {content[:300]}")
        event_list = "\n".join(event_lines)

        return (
            "You are a game AI archivist. Summarise the following series "
            "of in‑game events into a single concise paragraph that captures "
            "the overall situation, key actions taken, and any notable changes.\n\n"
            f"Events:\n{event_list}\n\nSummary:"
        )

    # ------------------------------------------------------------------
    # Secondary LLM call
    # ------------------------------------------------------------------

    async def _call_secondary_llm(self, prompt: str) -> str | None:
        """Call Ollama's ``/api/chat`` with the summarisation prompt.

        Returns the trimmed response text, or ``None`` on any error.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": 0.3,
                "num_predict": 200,
            },
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_OLLAMA_TIMEOUT) as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"].strip()
        except httpx.HTTPError as exc:
            logger.warning("Secondary LLM call failed: {}", exc)
            return None
        except (KeyError, TypeError) as exc:
            logger.warning("Unexpected secondary LLM response format: {}", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_counters(self) -> None:
        """Reset the event counter and update the last‑summary timestamp."""
        self._event_count = 0
        self._last_summary_time = time.monotonic()


# ---------------------------------------------------------------------------
# URL helper
# ---------------------------------------------------------------------------


def _strip_openai_suffix(url: str) -> str:
    """Remove trailing ``/v1`` (or ``/v1/``) from an Ollama URL to obtain
    the base URL where native API endpoints like ``/api/chat`` live.

    Mirrors :func:`src.llm_decision._strip_openai_suffix`.
    """
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path == "/v1":
        parsed = parsed._replace(path="")
    return urlunparse(parsed)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "MemorySummariser",
]
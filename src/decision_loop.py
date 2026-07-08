"""
Decision Loop — Phase 2.9

The main async event loop that ties together capture, frame differencing,
state processing, LLM decision-making, and macro execution.

Pipeline
--------
::

    Capture Producer (background task)
           │
           ├── high_prio_queue (maxsize=2)
           └── normal_queue   (maxsize=2)
                   │
                   ▼
    ┌──────────────────────────────────────────────┐
    │  Decision Loop (main consumer)               │
    │                                              │
    │  1. Dequeue frame (high-prio first)          │
    │  2. Adaptive frame skip (pixel diff)         │
    │  3. Throttle-driven OCR skip                 │
    │  4. StateProcessor.process() ≤ 250 ms        │
    │  5. State cache lookup (Phase 2.5)           │
    │  6. MCP memory query (real, Phase 3.3)      │
    │  7. build_llm_prompt()                       │
    │  8. call_llm_decision() ≤ 200 ms             │
    │  9. MacroExecutor.submit() ≤ 500 ms          │
    │ 10. Store event (real, Phase 3.3)            │
    │ 11. Latency monitoring + adaptive throttling │
    └──────────────────────────────────────────────┘

Spec references
---------------
* ``Implementation_Phases.md`` §2.9 — phase definition
* ``Additional_problems.md`` §Problem 2 — detailed pseudocode
* ``Extra_research04.md`` §15 — state cache integration
* ``Extra_research05.md`` §5 — adaptive frame skipping
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import cv2
import numpy as np

from src.frame_differ import (
    AdaptiveFrameSkipper,
    DiffConfig,
    compute_pixel_diff,
)
from src.llm_decision import call_llm_decision
from src.llm_prompt_builder import build_llm_prompt
from src.logging_config import get_logger
from src.macro_executor import MacroExecutor, MacroPriority, MacroRequest

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_FALLBACK_ACTION: str = "WAIT"
"""Fallback action when LLM fails and no previous action is available."""

_LARGE_DIFF_THRESHOLD: float = 50.0
"""Mean pixel diff above which a frame is considered high-priority."""

_LOW_HEALTH_THRESHOLD: float = 20.0
"""Health percentage below which frames are routed to the priority queue."""

_MAX_LATENCY_SAMPLES: int = 30
"""Number of recent cycle‑latency values kept in the ring buffer."""

_LATENCY_HIGH_WATERMARK_MS: float = 500.0
"""Per‑cycle latency (ms) above which the counter increments."""

_LATENCY_SPIKE_COUNT: int = 3
"""Consecutive high‑latency cycles before throttling engages."""

# ---------------------------------------------------------------------------
# ThrottleState
# ---------------------------------------------------------------------------


class ThrottleState:
    """Adaptive throttling flags adjusted by latency monitoring.

    When the decision loop consistently exceeds 500 ms per cycle,
    throttling engages: OCR alternation, reduced LLM context, and
    longer capture intervals lower CPU/GPU pressure.

    Attributes:
        active: ``True`` when throttling is engaged.
        skip_ocr_alternate: When ``True``, OCR is skipped on every
            other frame.
        reduce_llm_context: When ``True``, the LLM prompt budget is
            halved (800 → 400 tokens) and context size is reduced.
        capture_interval: Seconds between capture frames.  Normal is
            50 ms; throttled is 100 ms.
    """

    __slots__ = (
        "active",
        "skip_ocr_alternate",
        "reduce_llm_context",
        "capture_interval",
    )

    def __init__(self) -> None:
        self.active: bool = False
        self.skip_ocr_alternate: bool = False
        self.reduce_llm_context: bool = False
        self.capture_interval: float = 0.05  # 50 ms normal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_macro_by_name(
    name: str,
    profile_macros: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the macro definition dict whose ``"name"`` matches *name*.

    Returns ``None`` if no macro with that name is found.
    """
    for m in profile_macros:
        if m.get("name") == name:
            return m
    return None


def build_memory_query(state: Any) -> str:
    """Build a concise query string from the current game state.

    Delegates to ``MCPMemoryClient.build_memory_query`` — a richer
    8‑field version with comma‑separated format suitable for semantic
    search (Phase 3.2).  Falls back to a plain "game state" string if
    the MCP client module is not importable.
    """
    try:
        from src.mcp_client import MCPMemoryClient

        return MCPMemoryClient.build_memory_query(state)
    except ImportError:
        pass

    # Fallback when MCP client is not available (should not happen
    # in practice, but keeps the function self‑contained).
    if hasattr(state, "to_dict"):
        state_dict = state.to_dict()
    elif isinstance(state, dict):
        state_dict = state
    else:
        return "game state"

    parts: list[str] = []
    for k, v in state_dict.items():
        if k.startswith("_") or k.endswith("_raw_bar"):
            continue
        parts.append(f"{k}: {v}")
        if len(parts) >= 5:
            break

    return " ".join(parts) if parts else "game state"


def check_high_priority_event(
    last_state: Any,
    diff_score: float | None,
) -> bool:
    """Heuristic for routing frames to the high‑priority queue.

    Returns ``True`` when:
    * ``last_state`` contains ``health`` below 20 %, **or**
    * *diff_score* exceeds the large‑diff threshold (sudden scene change).

    This is called inside :func:`capture_producer` and is intentionally
    cheap — it must not become a bottleneck.
    """
    # Health check from last known state
    if last_state is not None:
        health = last_state.get("health") if hasattr(last_state, "get") else None
        if health is not None and isinstance(health, (int, float)) and health < _LOW_HEALTH_THRESHOLD:
            return True

    # Large frame‑change check
    if diff_score is not None and diff_score > _LARGE_DIFF_THRESHOLD:
        return True

    return False


def _diff_config_from_global(config: dict[str, Any]) -> DiffConfig:
    """Extract a flat ``DiffConfig`` from the nested ``"diff"`` section of
    the global *config* dict.

    Handles the mapping from the nested JSON structure to the flat
    field names expected by :class:`DiffConfig`.
    """
    diff = config.get("diff", {})
    adaptive = diff.get("adaptive", {})
    hash_fb = diff.get("perceptual_hash_fallback", {})

    flat: dict[str, Any] = {
        "downsample_width": diff.get("downsample_width", 640),
        "downsample_height": diff.get("downsample_height", 360),
        "method": diff.get("method", "absdiff_mean"),
        "adaptive_enabled": adaptive.get("enabled", True),
        "adaptive_window": adaptive.get("window_size", 30),
        "static_percentile": adaptive.get("static_percentile", 70),
        "high_motion_percentile": adaptive.get("high_motion_percentile", 85),
        "hash_fallback_enabled": hash_fb.get("enabled", False),
        "hash_method": hash_fb.get("method", "opencv_dhash"),
        "hash_threshold": hash_fb.get("hamming_threshold", 6),
    }
    return DiffConfig.from_dict(flat)


# ---------------------------------------------------------------------------
# Capture Producer
# ---------------------------------------------------------------------------


async def capture_producer(
    capture_obj: Any,
    normal_queue: asyncio.Queue,
    high_prio_queue: asyncio.Queue,
    throttle: ThrottleState,
    last_state_ref: list[Any],
) -> None:
    """Continuously capture frames and distribute to dual priority queues.

    Runs as a background ``asyncio.Task``, decoupling the capture rate
    from the processing rate.  Frames are routed to
    ``high_prio_queue`` when :func:`check_high_priority_event` returns
    ``True``; otherwise they go to ``normal_queue``.

    Backpressure is applied when a target queue exceeds 80 % capacity,
    slowing the capture rate to avoid unbounded memory growth.

    Args:
        capture_obj: An object with an async ``capture()`` method that
            returns a ``np.ndarray`` (or ``None`` on failure) and a
            ``get_last_frame()`` fallback.
        normal_queue: ``asyncio.Queue`` for standard‑priority frames.
        high_prio_queue: ``asyncio.Queue`` for high‑priority frames.
        throttle: Shared ``ThrottleState`` — *capture_interval* controls
            the sleep between captures.
        last_state_ref: Single‑element list whose ``[0]`` is updated by
            the decision‑loop consumer with the latest ``GameState``.
    """
    prev_gray: np.ndarray | None = None

    while True:
        # --- Capture ---
        frame = await capture_obj.capture()
        if frame is None:
            frame = capture_obj.get_last_frame()
            if frame is None:
                await asyncio.sleep(0.05)
                continue

        # --- Compute pixel diff against previous frame ---
        if prev_gray is not None:
            gray = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3
                else frame
            )
            diff_score: float | None = compute_pixel_diff(prev_gray, gray)
        else:
            diff_score = None

        # --- Priority routing ---
        last_state = last_state_ref[0] if last_state_ref else None
        is_high_prio = check_high_priority_event(last_state, diff_score)
        queue = high_prio_queue if is_high_prio else normal_queue

        # Update prev_gray for the next iteration
        if frame.ndim == 3:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = frame.copy()

        # --- Backpressure ---
        while queue.qsize() > queue.maxsize * 0.8:
            await asyncio.sleep(0.02)

        await queue.put(frame)
        await asyncio.sleep(throttle.capture_interval)


# ---------------------------------------------------------------------------
# Decision Loop
# ---------------------------------------------------------------------------


async def decision_loop(
    state_processor: Any,
    macro_executor: MacroExecutor,
    profile_macros: list[dict[str, Any]],
    config: dict[str, Any],
    capture_obj: Any,
    mcp: Any = None,
    summariser: Any = None,
) -> None:
    """Run the main perception → decision → action loop.

    This is the core of Phase 2.9.  It wires together every subsystem
    built in Phases 1–2 into a single continuous async pipeline.

    The loop runs indefinitely until cancelled (e.g. by Ctrl + C).

    Args:
        state_processor: ``StateProcessor`` instance (Phase 2.4).
        macro_executor: ``MacroExecutor`` that has already been
            ``start()``-ed (Phase 1.5).
        profile_macros: List of macro definitions from the active
            profile.  Each dict must have at least ``"name"`` and
            ``"actions"`` keys.
        config: Global configuration dict (Phase 1.1).
        capture_obj: ``ScreenCapture`` instance (Phase 1.2).
        mcp: Optional ``MCPMemoryClient`` instance (Phase 3.2).  If
            ``None``, memory query and storage are silently skipped
            (memory tier disabled for this session).
        summariser: Optional ``MemorySummariser`` instance (Phase 3.4).
            If ``None``, automatic summarisation is disabled.
    """
    _mcp_unavailable = mcp is None
    if _mcp_unavailable:
        logger.warning(
            "MCP client unavailable — memory tier disabled for this session."
        )

    # --- Adaptive frame skipper (Phase 2.8) ---
    diff_config = _diff_config_from_global(config)
    frame_skipper = AdaptiveFrameSkipper(diff_config)

    # --- Dual capture queues + throttle ---
    throttle = ThrottleState()
    high_prio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=2)
    normal_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=2)

    # Shared mutable reference so the capture producer can read the
    # latest GameState for low‑health priority routing.
    last_state_ref: list[Any] = [None]

    # Launch capture producer as a background task
    producer_task = asyncio.create_task(
        capture_producer(
            capture_obj,
            normal_queue,
            high_prio_queue,
            throttle,
            last_state_ref,
        )
    )

    # --- Loop state ---
    last_state = None
    last_action: str | None = None
    prev_gray: np.ndarray | None = None
    high_latency_count: int = 0
    frame_counter: int = 0
    latency_ring: list[float] = []  # ring buffer for cycle latencies

    cycle_start = time.monotonic()

    logger.info("Decision loop started. Press Ctrl+C to stop.")

    try:
        while True:
            # ----------------------------------------------------------
            # 1. Get next frame  (high‑priority first)
            # ----------------------------------------------------------
            if not high_prio_queue.empty():
                frame = await high_prio_queue.get()
            else:
                try:
                    frame = await asyncio.wait_for(
                        normal_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

            # ----------------------------------------------------------
            # 2. Adaptive frame skip via pixel diff
            # ----------------------------------------------------------
            gray = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3
                else frame
            )
            if prev_gray is not None:
                diff_score = compute_pixel_diff(prev_gray, gray)
                should_process = frame_skipper.should_process(
                    diff_score, gray
                )
            else:
                should_process = True
            prev_gray = gray

            if (
                not should_process
                and last_state is not None
                and last_action is not None
            ):
                # Scene is static — reuse previous action
                continue

            # ----------------------------------------------------------
            # 3. Throttle‑driven OCR skip
            # ----------------------------------------------------------
            skip_ocr = (
                throttle.active
                and throttle.skip_ocr_alternate
                and (frame_counter % 2 != 0)
            )

            # ----------------------------------------------------------
            # 4. Extract state  (≤ 250 ms timeout)
            # ----------------------------------------------------------
            try:
                state = await asyncio.wait_for(
                    state_processor.process(frame, skip_ocr=skip_ocr),
                    timeout=0.250,
                )
            except asyncio.TimeoutError:
                logger.debug(
                    "State processing timed out; reusing last state."
                )
                if last_state is None:
                    continue  # nothing to reuse on the very first cycle
                state = last_state

            # Update shared reference for capture producer
            last_state_ref[0] = state

            # ----------------------------------------------------------
            # 5. State cache lookup  (Phase 2.5)
            # ----------------------------------------------------------
            cached_action = state_processor.get_cached_action(state)
            if cached_action is not None:
                logger.debug(
                    "Cache hit — reusing action {!r}", cached_action
                )
                action = cached_action
            else:
                # ------------------------------------------------------
                # 6. Query MCP memories  (Phase 3.3 — real)
                # ------------------------------------------------------
                if not _mcp_unavailable:
                    try:
                        memories = await mcp.search_memories(
                            query=build_memory_query(state),
                            tags=["game_event"],
                            limit=5,
                        )
                    except Exception as exc:
                        logger.warning(
                            "MCP search_memories failed ({}) — "
                            "continuing without memories.",
                            exc,
                        )
                        memories = []
                else:
                    memories = []

                # ------------------------------------------------------
                # 7. Build LLM prompt
                # ------------------------------------------------------
                context_reduction = (
                    throttle.active and throttle.reduce_llm_context
                )
                max_tokens = 800 if not context_reduction else 400
                messages = build_llm_prompt(
                    state=state,
                    available_macros=profile_macros,
                    memories=memories,
                    max_tokens=max_tokens,
                )

                # ------------------------------------------------------
                # 8. LLM decision  (≤ 200 ms timeout)
                # ------------------------------------------------------
                try:
                    action = await call_llm_decision(
                        messages=messages,
                        profile_macros=profile_macros,
                        config=config,
                        last_action=last_action,
                        timeout=config.get("llm_timeout_ms", 200) / 1000.0,
                    )
                except Exception as exc:
                    logger.error(
                        "LLM decision call raised: {}", exc
                    )
                    action = last_action or _DEFAULT_FALLBACK_ACTION

                # Cache the result for future identical states
                state_processor.cache_action(state, action)

            # ----------------------------------------------------------
            # 9. Execute macro
            # ----------------------------------------------------------
            macro_def = find_macro_by_name(action, profile_macros)
            if macro_def:
                steps = macro_def.get(
                    "actions", macro_def.get("steps", [])
                )
                if steps:
                    request = MacroRequest(
                        name=action,
                        actions=steps,
                        priority=MacroPriority.NORMAL,
                    )
                    try:
                        # Submit + wait up to 500 ms for completion.
                        # If it takes longer the macro continues in the
                        # background — we do NOT block the decision loop.
                        await asyncio.wait_for(
                            macro_executor.submit(request), timeout=0.5
                        )
                    except asyncio.TimeoutError:
                        logger.debug(
                            "Macro {!r} still executing; continuing loop.",
                            action,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Macro {!r} submission failed: {}",
                            action,
                            exc,
                        )
                else:
                    logger.debug(
                        "Macro {!r} has no action steps; treating as no-op.",
                        action,
                    )
            else:
                if action != _DEFAULT_FALLBACK_ACTION:
                    logger.warning(
                        "LLM chose unrecognised macro {!r}; no action taken.",
                        action,
                    )
                action = _DEFAULT_FALLBACK_ACTION

            # ----------------------------------------------------------
            # 10. Store event  (fire‑and‑forget, Phase 3.3)
            # ----------------------------------------------------------
            if not _mcp_unavailable:

                async def _store_and_log() -> None:
                    try:
                        await mcp.store_memory(
                            content=json.dumps(
                                {
                                    "state": (
                                        state.to_dict()
                                        if hasattr(state, "to_dict")
                                        else {}
                                    ),
                                    "action": action,
                                }
                            ),
                            memory_type="short_term",
                            tags=["game_event", "auto"],
                        )
                        # Notify the summariser of a new event (Phase 3.4)
                        if summariser is not None:
                            await summariser.record_new_event()
                    except Exception as exc:
                        logger.warning(
                            "MCP store_memory failed — event not persisted: {}",
                            exc,
                        )

                asyncio.create_task(_store_and_log())

            # ----------------------------------------------------------
            # 11. Latency monitoring & adaptive throttling
            # ----------------------------------------------------------
            cycle_elapsed_ms = (
                time.monotonic() - cycle_start
            ) * 1000
            latency_ring.append(cycle_elapsed_ms)
            if len(latency_ring) > _MAX_LATENCY_SAMPLES:
                latency_ring.pop(0)

            if cycle_elapsed_ms > _LATENCY_HIGH_WATERMARK_MS:
                high_latency_count += 1
            else:
                high_latency_count = 0

            if high_latency_count >= _LATENCY_SPIKE_COUNT:
                if not throttle.active:
                    logger.warning(
                    "High latency detected ({:.0f}ms avg over {} cycles) — "
                    "engaging adaptive throttling.",
                        sum(latency_ring[-_LATENCY_SPIKE_COUNT:])
                        / _LATENCY_SPIKE_COUNT,
                        _LATENCY_SPIKE_COUNT,
                    )
                throttle.active = True
                throttle.skip_ocr_alternate = True
                throttle.reduce_llm_context = True
                throttle.capture_interval = 0.100
            else:
                if throttle.active:
                    logger.info("Latency recovered — disengaging throttling.")
                throttle.active = False
                throttle.skip_ocr_alternate = False
                throttle.reduce_llm_context = False
                throttle.capture_interval = 0.05

            # --- Advance loop state ---
            last_state = state
            last_action = action
            frame_counter += 1
            cycle_start = time.monotonic()

    except asyncio.CancelledError:
        logger.info("Decision loop cancelled — shutting down.")
    finally:
        # Tear down the capture producer
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass
        logger.info(
            "Decision loop stopped. Total frames processed: {}",
            frame_counter,
        )


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "decision_loop",
    "capture_producer",
    "ThrottleState",
    "find_macro_by_name",
    "build_memory_query",
    "check_high_priority_event",
]

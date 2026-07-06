"""
Macro Executor ― Step 1.5

Static macro playback with asyncio.PriorityQueue-based scheduling,
cooperative cancellation, accurate key-hold timing, timeouts, and
single-consumer serialisation.

Reads named macros from macros.json (per‑profile format defined in
gameai_profile_format_research.md) and delegates individual actions
to InputController.

Spec references:
- Implementation_Phases.md §1.5
- Extra_research04.md §1–4 (cancellation, timeout, accurate holds, concurrency)
- Extra_research04.md §14 (priority queue & cancel_all)
- gameai_profile_format_research.md (macros.json schema)
- architecture.md §4.3 (InputController interface)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Priority levels
# ============================================================================

class MacroPriority:
    """Integer priority values for macro requests (lower = more urgent)."""

    CRITICAL: int = 0   # emergency quit, pause, panic stop
    HIGH: int = 10       # boss fight, health < 20%
    NORMAL: int = 20     # default LLM actions
    LOW: int = 30        # idle animations, background macros


# ============================================================================
# Custom exceptions
# ============================================================================

class MacroError(Exception):
    """Base exception for macro execution failures."""


class MacroCancelledError(MacroError):
    """Raised inside a running macro when cancellation is requested."""


class MacroRejectedError(MacroError):
    """Raised when a macro cannot be accepted (queue full, timeout, etc.)."""


# ============================================================================
# Cancellation token
# ============================================================================

class CancellationToken:
    """
    Cooperative cancellation signal.

    The executing macro checks ``cancelled`` at every step boundary
    (at least every 50–100 ms) and raises MacroCancelledError when set.
    """

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        """Signal cancellation."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


# ============================================================================
# Macro request DTO
# ============================================================================

@dataclass
class MacroRequest:
    """
    A macro queued for execution.

    Parameters
    ----------
    name : str
        Human‑readable macro name (must match a key in macros.json).
    actions : list[dict]
        Action steps in the per‑profile format (``hold_ms``, ``delay``, etc.).
    priority : int
        Scheduling priority (lower = more urgent).  See MacroPriority.
    id : str
        Unique identifier (auto‑generated UUID hex).
    future : asyncio.Future
        Completes when the macro finishes (or fails / is cancelled).
    timeout : float or None
        Per‑macro maximum runtime in seconds.  None = no limit.
    """

    name: str
    actions: list[dict[str, Any]]
    priority: int = MacroPriority.NORMAL
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    future: asyncio.Future[None] = field(default_factory=asyncio.Future)
    timeout: Optional[float] = None

    def __lt__(self, other: "MacroRequest") -> bool:
        """Compare by priority first, then by id for deterministic ordering.

        Required by ``asyncio.PriorityQueue`` which uses ``heapq.heappop``
        (calls ``__lt__`` when two items have the same priority).
        """
        return (self.priority, self.id) < (other.priority, other.id)


# ============================================================================
# Accurate key‑hold timer
# ============================================================================

async def accurate_hold(duration_seconds: float, token: Optional[CancellationToken] = None) -> None:
    """
    Hold for an exact duration using a hybrid active‑wait / sleep approach.

    For holds < 20 ms: pure busy‑wait for sub‑millisecond precision.
    For holds ≥ 20 ms: sleep for ``duration - 5ms`` then busy‑wait the
    remainder, reducing CPU usage while keeping accuracy within ~3 ms.

    Checks *token*.cancelled periodically so a long hold can be aborted.

    Parameters
    ----------
    duration_seconds : float
        Desired hold time in seconds.
    token : CancellationToken or None
        Optional cancellation signal.

    Raises
    ------
    MacroCancelledError
        If *token*.cancelled becomes True during the hold.
    """
    if duration_seconds <= 0.0:
        return

    if duration_seconds < 0.020:  # pure busy‑wait for very short holds
        start = time.perf_counter()
        while (time.perf_counter() - start) < duration_seconds:
            if token and token.cancelled:
                raise MacroCancelledError()
            # Yield occasionally to avoid starving the event loop
            if (time.perf_counter() - start) > duration_seconds * 0.5:
                await asyncio.sleep(0)
        return

    # Sleep the bulk, then fine‑tune
    coarse = max(duration_seconds - 0.005, 0.0)
    if coarse > 0.0:
        await asyncio.sleep(coarse)
        if token and token.cancelled:
            raise MacroCancelledError()

    start = time.perf_counter()
    remaining = max(duration_seconds - (time.perf_counter() - start), 0.0)
    while remaining > 0.0:
        if token and token.cancelled:
            raise MacroCancelledError()
        remaining = max(duration_seconds - (time.perf_counter() - start), 0.0)
        if remaining < 0.001:
            break
        await asyncio.sleep(0)


# ============================================================================
# MacroExecutor
# ============================================================================

class MacroExecutor:
    """
    Consumes MacroRequests from a bounded asyncio.PriorityQueue and
    executes them one at a time (single‑consumer serialisation).

    Usage::

        executor = MacroExecutor(input_ctrl)
        await executor.start()

        future = await executor.submit(
            MacroRequest(name="attack", actions=[...], priority=MacroPriority.NORMAL)
        )
        await future  # optional: wait for completion

        await executor.stop()
    """

    # ------------------------------------------------------------------
    def __init__(self, input_controller: Any, max_queue_size: int = 32) -> None:
        """
        Parameters
        ----------
        input_controller : InputController
            The cross‑platform input abstraction (src.input_controller).
        max_queue_size : int
            Maximum number of pending macro requests.  Surplus submissions
            will raise MacroRejectedError.
        """
        self._input = input_controller
        self._queue: asyncio.PriorityQueue[tuple[int, MacroRequest]] = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        self._current_task: Optional[asyncio.Task[None]] = None
        self._consumer_task: Optional[asyncio.Task[None]] = None
        self._pending: dict[str, MacroRequest] = {}  # queued, not yet executing
        self._running = False

        # Track keys that are currently pressed so they can be released on cancel.
        self._held_keys: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the consumer coroutine."""
        if self._running:
            return
        self._running = True
        self._consumer_task = asyncio.create_task(self._consumer())
        logger.info("MacroExecutor started (max_queue=%d)", self._queue.maxsize)

    async def stop(self) -> None:
        """Cancel all work and shut down the consumer."""
        if not self._running:
            return
        self._running = False
        if self._consumer_task:
            self._consumer_task.cancel()
            self._consumer_task = None
        await self.cancel_all()
        logger.info("MacroExecutor stopped")

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit(self, macro: MacroRequest) -> asyncio.Future[None]:
        """
        Enqueue *macro* and return a Future that completes when the macro
        finishes, fails, or is cancelled.

        Raises MacroRejectedError if the queue is full.
        """
        try:
            self._queue.put_nowait((macro.priority, macro))
        except asyncio.QueueFull:
            raise MacroRejectedError(
                f"Macro queue full (max={self._queue.maxsize}). "
                f"Macro '{macro.name}' rejected."
            ) from None
        self._pending[macro.id] = macro
        logger.debug("Macro '%s' enqueued (priority=%d, id=%s)", macro.name, macro.priority, macro.id)
        return macro.future

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def cancel_macro(self, macro_id: str) -> bool:
        """
        Cancel a specific queued macro (not yet executing).

        Returns True if the macro was found and cancelled, False otherwise.
        """
        macro = self._pending.pop(macro_id, None)
        if macro is None:
            return False
        if not macro.future.done():
            macro.future.cancel()
        # Rebuild queue without the cancelled item
        new_q: asyncio.PriorityQueue[tuple[int, MacroRequest]] = asyncio.PriorityQueue(
            maxsize=self._queue.maxsize
        )
        while not self._queue.empty():
            _pri, item = self._queue.get_nowait()
            if item.id != macro_id:
                new_q.put_nowait((item.priority, item))
        self._queue = new_q
        logger.debug("Cancelled queued macro '%s' (id=%s)", macro.name, macro_id)
        return True

    async def cancel_all(self) -> None:
        """Cancel every queued macro AND the currently executing macro."""
        # Cancel all pending futures
        for mid in list(self._pending.keys()):
            macro = self._pending.pop(mid)
            if not macro.future.done():
                macro.future.cancel()
        # Drain the queue
        while not self._queue.empty():
            self._queue.get_nowait()
        # Cancel the currently running macro
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        logger.debug("All macros cancelled")

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    async def _consumer(self) -> None:
        """Pull from the priority queue and execute macros sequentially."""
        while self._running:
            try:
                priority, macro = await self._queue.get()
            except asyncio.CancelledError:
                # Flush remaining futures before exiting
                for m in self._pending.values():
                    if not m.future.done():
                        m.future.cancel()
                self._pending.clear()
                raise

            self._pending.pop(macro.id, None)  # now executing, no longer pending
            logger.debug("Executing macro '%s' (priority=%d)", macro.name, priority)

            try:
                self._current_task = asyncio.create_task(self._execute_macro(macro))
                await self._current_task
                if not macro.future.done():
                    macro.future.set_result(None)
            except asyncio.CancelledError:
                # The consumer itself was cancelled (stop / cancel_all).
                for m in self._pending.values():
                    if not m.future.done():
                        m.future.cancel()
                self._pending.clear()
                raise
            except Exception as exc:
                if not macro.future.done():
                    macro.future.set_exception(exc)
                logger.error("Macro '%s' failed: %s", macro.name, exc)
            finally:
                self._current_task = None
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Macro execution
    # ------------------------------------------------------------------

    async def _execute_macro(self, macro: MacroRequest) -> None:
        """
        Play every step of *macro*.

        Cooperative cancellation is checked before each step.  If the
        macro is cancelled (or times out) every key currently held is
        released in the ``finally`` block — the *key‑up guarantee*.
        """
        token = CancellationToken()
        held: set[str] = set()

        coro = self._execute_steps(macro.actions, token, held)

        try:
            if macro.timeout is not None:
                await asyncio.wait_for(coro, timeout=macro.timeout)
            else:
                await coro
        except asyncio.TimeoutError:
            token.cancel()
            logger.warning(
                "Macro '%s' timed out after %.1fs", macro.name, macro.timeout
            )
            raise MacroCancelledError(
                f"Macro '{macro.name}' timed out ({macro.timeout}s)"
            )
        finally:
            # Key‑up guarantee: release every key we pressed
            await self._release_keys(held)

    async def _execute_steps(
        self,
        actions: list[dict[str, Any]],
        token: CancellationToken,
        held: set[str],
    ) -> None:
        """Iterate through *actions*, checking *token*.cancelled at each step."""
        for i, step in enumerate(actions):
            if token.cancelled:
                raise MacroCancelledError()

            step_type = step.get("type")
            try:
                if step_type == "key":
                    key = step["key"]
                    hold_ms = int(step.get("hold_ms", 50))
                    duration = hold_ms / 1000.0

                    await self._input.key_down(key)
                    held.add(key)
                    await accurate_hold(duration, token=token)
                    await self._input.key_up(key)
                    held.discard(key)

                elif step_type == "delay":
                    ms = int(step.get("ms", 100))
                    await accurate_hold(ms / 1000.0, token=token)

                elif step_type == "mouse_move":
                    await self._input.move_mouse(
                        x=int(step["x"]),
                        y=int(step["y"]),
                        relative=bool(step.get("relative", False)),
                    )

                elif step_type == "click":
                    await self._input.click(button=step.get("button", "left"))

                elif step_type == "type_string":
                    await self._input.type_string(str(step.get("text", "")))

                else:
                    raise MacroError(f"Unknown macro step type at index {i}: {step_type!r}")

            except MacroCancelledError:
                raise
            except KeyError as exc:
                raise MacroError(
                    f"Macro step at index {i} missing required key: {exc}"
                ) from exc

    async def _release_keys(self, held: set[str]) -> None:
        """Release all keys in *held* (idempotent — safe to call repeatedly)."""
        for key in list(held):
            try:
                await self._input.key_up(key)
            except Exception as exc:
                logger.warning("Failed to release key '%s' during cleanup: %s", key, exc)
        held.clear()

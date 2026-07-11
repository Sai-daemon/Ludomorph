"""
State Processor — Phase 2.4

Orchestrates the colour bar detector and OCR module to populate a
``GameState`` object each frame. Applies per‑slot fallback priorities
as defined in ``state_schema.json`` (``color_first``, ``ocr_first``,
``color_only``, ``ocr_only``).

All colour bar work is dispatched to a thread pool via
``asyncio.to_thread``; OCR calls are already async. The outer
processing call is wrapped in a configurable timeout to keep the
decision loop responsive.

Usage::

    from src.region_profile import RegionProfile
    from src.game_state import GameState, StateSchema
    from src.ocr_module import OCRModule, OCRConfig
    from src.bar_detector import ColourBarDetector, ColourBarCalibration
    from src.state_processor import StateProcessor

    schema = StateSchema.from_dict(...)
    profile = RegionProfile.from_dict(...)
    ocr = OCRModule(OCRConfig(...))

    processor = StateProcessor(profile=profile, ocr_module=ocr, schema=schema)

    # Per frame:
    state: GameState = await processor.process(frame)
    print(state.to_dict())
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.bar_detector import ColourBarCalibration, ColourBarDetector
from src.game_state import GameState, StateSchema, StateSlotDefinition
from src.logging_config import get_logger
from src.ocr_module import OCRModule, OCRResult
from src.state_hash import StateCache, state_hash

if TYPE_CHECKING:
    from src.region_profile import RegionConfig, RegionProfile

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Dedicated Vision thread pool (§7.7C)
# ---------------------------------------------------------------------------

VisionExecutor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
"""Dedicated thread pool for Vision (YOLO) inference.

Must **not** be shared with the TesseractExecutor or GeneralExecutor to
prevent YOLO inference from blocking input events or file writes.
"""


def shutdown_vision_executor(wait: bool = True) -> None:
    """Gracefully shut down the dedicated vision thread pool."""
    VisionExecutor.shutdown(wait=wait)


# ---------------------------------------------------------------------------
# Minimum confidence thresholds for candidate validity
# ---------------------------------------------------------------------------

_COLOR_BAR_MIN_CONFIDENCE: float = 0.5
"""Minimum confidence for a colour bar reading to be considered valid."""

_OCR_MIN_CONFIDENCE: float = 0.6
"""Minimum confidence for an OCR reading to be considered valid."""

_DEFAULT_OCR_TIMEOUT: float = 0.250
"""Per‑frame aggregate OCR timeout (seconds)."""

# ---------------------------------------------------------------------------
# Vision-OCR contention thresholds (§7.7A)
# ---------------------------------------------------------------------------

_OCR_LATENCY_GATE_MS: float = 150.0
"""If average OCR latency exceeds this (ms), Vision is skipped for the frame."""

_VISION_TIMEOUT_S: float = 0.1
"""Per‑frame Vision timeout (seconds) — must fit within remaining budget after OCR."""

# ---------------------------------------------------------------------------
# PipelineMetrics — telemetry for vision-OCR scheduling (§7.7E)
# ---------------------------------------------------------------------------


@dataclass
class PipelineMetrics:
    """Lightweight in‑memory telemetry for the decision pipeline.

    Tracks per‑stage latencies and vision‑specific contention counters
    required by §7.7E of the architecture spec.  Used by both
    ``StateProcessor`` and the decision loop's adaptive throttling.

    Attributes:
        ocr_latencies: Ring buffer of recent OCR batch latencies (ms).
        vision_frames_skipped_due_to_ocr: Counter incremented when vision
            is skipped because OCR is already slow.
        vision_timeouts: Counter incremented when vision exceeds the
            per‑frame timeout.
        vision_contention_events: Counter for frames where both OCR and
            Vision attempted to run simultaneously.
        vision_detection_interval_current: Current vision stagger interval
            (frames) — may be increased during adaptive throttling.
        vision_disabled_by_throttle: ``True`` when adaptive throttling has
            automatically disabled vision.
    """

    ocr_latencies: list[float] = field(default_factory=list)
    vision_frames_skipped_due_to_ocr: int = 0
    vision_timeouts: int = 0
    vision_contention_events: int = 0
    vision_detection_interval_current: int = 2
    vision_disabled_by_throttle: bool = False

    _MAX_LATENCY_SAMPLES: int = field(default=30, repr=False)

    def record_ocr_latency(self, ms: float) -> None:
        """Append an OCR batch latency sample (ms)."""
        self.ocr_latencies.append(ms)
        if len(self.ocr_latencies) > self._MAX_LATENCY_SAMPLES:
            self.ocr_latencies.pop(0)

    def get_avg_ocr_latency(self) -> float:
        """Return the mean OCR latency over the recent window, or 0 if empty."""
        if not self.ocr_latencies:
            return 0.0
        return sum(self.ocr_latencies) / len(self.ocr_latencies)

    def record_vision_skip(self) -> None:
        """Increment the *vision skipped due to OCR* counter."""
        self.vision_frames_skipped_due_to_ocr += 1

    def record_vision_timeout(self) -> None:
        """Increment the vision timeout counter."""
        self.vision_timeouts += 1

    def record_contention(self) -> None:
        """Increment the contention event counter."""
        self.vision_contention_events += 1

    def reset_counters(self) -> None:
        """Reset all counters (but not latency ring buffer)."""
        self.vision_frames_skipped_due_to_ocr = 0
        self.vision_timeouts = 0
        self.vision_contention_events = 0


# ---------------------------------------------------------------------------
# StateProcessor
# ---------------------------------------------------------------------------


class StateProcessor:
    """Combines bar detector and OCR per region, populating a ``GameState``.

    On construction the processor creates one ``ColourBarDetector`` per
    ``color_bar`` region (using its calibration dict) and stores
    references to the shared ``OCRModule`` and ``StateSchema``.

    The optional *vision_processor* is wired during Phase 4 startup
    (see ``decision_loop.py``).  When provided it must expose::

        is_enabled: bool
        async process_frame(frame) -> SpatialContext
    """

    def __init__(
        self,
        profile: "RegionProfile",
        ocr_module: OCRModule,
        schema: StateSchema,
        vision_processor: Any = None,
        cache_ttl: float = 0.3,
        *,
        metrics: PipelineMetrics | None = None,
        vision_interval: int = 2,
    ) -> None:
        self._profile = profile
        self._ocr = ocr_module
        self._schema = schema
        self._cache = StateCache(ttl=cache_ttl)
        self._metrics = metrics or PipelineMetrics()
        self._vision_interval = vision_interval

        # Build one detector per colour-bar region
        self._colour_detectors: dict[str, ColourBarDetector] = {}
        self._region_lookup: dict[str, "RegionConfig"] = {}
        for region in profile.regions:
            self._region_lookup[region.name] = region
            if region.type == "color_bar":
                try:
                    calib = ColourBarCalibration.from_dict(region.calibration)
                except Exception as exc:
                    logger.error(
                        f"Invalid calibration for region '{region.name}': {exc}. "
                        f"Bar detection will be disabled."
                    )
                    calib = ColourBarCalibration(enabled=False)
                self._colour_detectors[region.name] = ColourBarDetector(calib)

        self._vision = vision_processor
        self.frame_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(
        self,
        frame: np.ndarray,
        *,
        skip_ocr: bool = False,
        ocr_timeout: float = _DEFAULT_OCR_TIMEOUT,
    ) -> GameState:
        """Extract all state slots from *frame* and return a populated ``GameState``.

        Args:
            frame: BGR numpy array (full screen capture).
            skip_ocr: If ``True``, OCR regions are skipped entirely
                (used during high‑latency throttle periods).
            ocr_timeout: Maximum seconds for the aggregate OCR batch.

        Returns:
            A ``GameState`` keyed by schema slot names.  Slots that could
            not be resolved are simply absent from the result.
        """
        state = GameState(self._schema)

        # 1. Group regions by role
        role_to_regions = self._profile.by_role()

        # 2. Colour bar detection (always, concurrent)
        colour_results = await self._run_colour_bars(frame)

        # 3. OCR (concurrent, timeout‑protected)
        if skip_ocr:
            ocr_results: dict[str, OCRResult] = {}
        else:
            ocr_results = await self._run_ocr_batch(frame, ocr_timeout)

        # 4. Index all raw results by region name
        all_results: dict[str, dict[str, Any]] = {}
        for region in self._profile.colour_bar_regions():
            result = colour_results.get(region.name)
            if result is not None:
                pct, conf, success = result
                all_results[region.name] = {
                    "value": pct,
                    "type": "numeric",
                    "confidence": conf,
                    "success": success,
                }
            else:
                all_results[region.name] = {
                    "value": 0.0,
                    "type": "numeric",
                    "confidence": 0.0,
                    "success": False,
                }

        for region in self._profile.ocr_regions():
            result = ocr_results.get(region.name)
            if result is not None:
                all_results[region.name] = {
                    "value": result.text,
                    "type": "text",
                    "confidence": result.confidence,
                    "success": result.success,
                }
            else:
                all_results[region.name] = {
                    "value": "",
                    "type": "text",
                    "confidence": 0.0,
                    "success": False,
                }

        # 5. For each schema slot, choose the best value via priority rules
        for slot_name, slot_def in self._schema.slots.items():
            regions = role_to_regions.get(slot_name, [])
            if not regions:
                continue

            resolved = self._resolve_slot(
                slot_name=slot_name,
                slot_def=slot_def,
                regions=regions,
                all_results=all_results,
                frame=frame,
            )
            if resolved is None:
                continue
            if isinstance(resolved, _RawBarSentinel):
                state.set(f"{slot_name}_raw_bar", resolved.image)
            else:
                state.set(slot_name, resolved)

        # 6. Vision-OCR scheduling (§7.7 — Phase 4.2)
        #
        # OCR always takes absolute priority.  Vision runs concurrently
        # only when:
        #   a. A VisionProcessor is wired AND enabled,
        #   b. The frame counter aligns with the stagger interval,
        #   c. Average OCR latency is below the 150 ms gate,
        #   d. Adaptive throttling has not disabled vision.
        #
        # Vision runs with a 100 ms timeout; on timeout/cancel the
        # spatial data is simply omitted for this frame.
        vision_enabled = (
            self._vision is not None
            and getattr(self._vision, "is_enabled", False)
            and not self._metrics.vision_disabled_by_throttle
        )
        vision_interval = self._metrics.vision_detection_interval_current

        vision_task: asyncio.Task[Any] | None = None
        vision_started: bool = False

        if vision_enabled and (self.frame_counter % vision_interval == 0):
            avg_ocr = self._metrics.get_avg_ocr_latency()
            if avg_ocr < _OCR_LATENCY_GATE_MS:
                # VisionProcessor.process_frame is already async and
                # offloads heavy ONNX inference to a thread pool internally.
                vision_task = asyncio.create_task(
                    self._vision.process_frame(frame)
                )
                vision_started = True
            else:
                self._metrics.record_vision_skip()
                logger.debug(
                    f"Vision skipped — avg OCR latency {avg_ocr:.0f} ms > "
                    f"{_OCR_LATENCY_GATE_MS:.0f} ms gate"
                )

        # Record contention if both OCR *and* Vision attempted this frame.
        # "Attempted" means OCR was not skipped (skip_ocr=False) and
        # a vision task was created.
        if not skip_ocr and vision_started:
            self._metrics.record_contention()

        # If Vision was started, merge results (with timeout).
        # IMPORTANT: we do this AFTER the existing bar/OCR work so that
        # Vision never blocks OCR — the task has been running in the
        # background via the dedicated VisionExecutor thread pool.
        if vision_task is not None:
            try:
                spatial = await asyncio.wait_for(
                    vision_task, timeout=_VISION_TIMEOUT_S
                )
                if isinstance(spatial, Exception):
                    logger.debug(
                        "Vision processing raised: {}", spatial
                    )
                    self._metrics.record_vision_timeout()
                elif spatial is not None:
                    # Merge spatial results into GameState
                    if hasattr(spatial, "context_text"):
                        state.set("spatial_context", spatial.context_text)
                    if hasattr(spatial, "detections"):
                        state.set("detections", spatial.detections)
            except asyncio.TimeoutError:
                vision_task.cancel()
                self._metrics.record_vision_timeout()
                logger.debug("Vision processing timed out after {:.0f} ms".format(
                    _VISION_TIMEOUT_S * 1000
                ))
            except Exception:
                vision_task.cancel()
                self._metrics.record_vision_timeout()
                logger.debug("Vision processing failed", exc_info=True)

        # 7. Phase 2.5 — compute state hash and attach to state
        h = state_hash(state)
        state.set("_state_hash", h)

        self.frame_counter += 1
        return state

    # ------------------------------------------------------------------
    # Slot resolution (priority dispatch)
    # ------------------------------------------------------------------

    def _resolve_slot(
        self,
        slot_name: str,
        slot_def: StateSlotDefinition,
        regions: list["RegionConfig"],
        all_results: dict[str, dict[str, Any]],
        frame: np.ndarray,
    ) -> Any:
        """Apply the schema priority to choose a value for one slot.

        Returns the resolved value, or ``None`` if no valid candidate
        was found (the slot will be left unset).
        """
        priority = slot_def.priority

        color_candidates: list[tuple[float, float]] = []  # (value, confidence)
        ocr_candidates: list[tuple[str, float]] = []
        raw_bar_fallback: np.ndarray | None = None

        for region in regions:
            v = all_results.get(region.name)
            if v is None:
                continue

            if region.type == "color_bar":
                if v["success"] and v["confidence"] >= _COLOR_BAR_MIN_CONFIDENCE:
                    color_candidates.append((v["value"], v["confidence"]))
                elif not v["success"]:
                    # Save raw ROI as fallback for potential LLM consumption
                    roi = self._crop_roi(frame, region)
                    if roi is not None and roi.size > 0:
                        raw_bar_fallback = roi

            elif region.type == "ocr":
                if v["success"] and v["confidence"] >= _OCR_MIN_CONFIDENCE:
                    ocr_candidates.append((v["value"], v["confidence"]))

        # --- Priority dispatch ---
        if priority == "color_only":
            if color_candidates:
                return max(color_candidates, key=lambda x: x[1])[0]

        elif priority == "ocr_only":
            if ocr_candidates:
                return max(ocr_candidates, key=lambda x: x[1])[0]

        elif priority == "color_first":
            if color_candidates:
                return max(color_candidates, key=lambda x: x[1])[0]
            if ocr_candidates:
                return max(ocr_candidates, key=lambda x: x[1])[0]

        elif priority == "ocr_first":
            if ocr_candidates:
                return max(ocr_candidates, key=lambda x: x[1])[0]
            if color_candidates:
                return max(color_candidates, key=lambda x: x[1])[0]

        # --- No valid candidate — store raw bar fallback if available ---
        if raw_bar_fallback is not None:
            # The raw image is stored as an "extra" key (not in the schema)
            # on the GameState by the caller? No — we can't set it here because
            # we'd need to return a tuple. Instead, the caller checks.
            # We return a sentinel and the caller sets extras.
            # Actually, let the caller handle extras. We'll use a sentinel.
            return _RawBarSentinel(slot_name, raw_bar_fallback)

        return None

    # ------------------------------------------------------------------
    # Concurrent colour bar detection
    # ------------------------------------------------------------------

    async def _run_colour_bars(
        self, frame: np.ndarray
    ) -> dict[str, tuple[float, float, bool]]:
        """Run all colour bar detectors concurrently via ``asyncio.to_thread``.

        Returns ``{region_name: (percentage, confidence, success)}``.
        """
        colour_regions = self._profile.colour_bar_regions()
        if not colour_regions:
            return {}

        tasks = [
            asyncio.to_thread(self._process_colour_bar, region, frame)
            for region in colour_regions
        ]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, tuple[float, float, bool]] = {}
        for region, result in zip(colour_regions, results_list):
            if isinstance(result, Exception):
                logger.warning(
                    f"Colour bar detection failed for '{region.name}': {result}"
                )
                output[region.name] = (0.0, 0.0, False)
            else:
                output[region.name] = result

        return output

    def _process_colour_bar(
        self, region: "RegionConfig", frame: np.ndarray
    ) -> tuple[float, float, bool]:
        """Synchronous colour bar detection (runs in thread pool).

        Normalises BGRA → BGR when the screen capture backend returns
        4‑channel frames (e.g. ``mss`` on Linux).
        """
        roi = self._crop_roi(frame, region)
        if roi is None or roi.size == 0:
            return (0.0, 0.0, False)
        # Channel normalisation: bar detector expects 3-channel BGR.
        # The capture backend may return grayscale (1-ch) or BGRA (4-ch).
        if roi.ndim == 2:
            # Grayscale → BGR
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        elif roi.ndim == 3 and roi.shape[2] == 4:
            # BGRA → BGR
            roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
        elif roi.ndim == 3 and roi.shape[2] == 1:
            # Single-channel 3D → BGR
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        detector = self._colour_detectors[region.name]
        return detector.process(roi)

    # ------------------------------------------------------------------
    # Concurrent OCR
    # ------------------------------------------------------------------

    async def _run_ocr_batch(
        self,
        frame: np.ndarray,
        timeout: float,
    ) -> dict[str, OCRResult]:
        """Run all OCR recognitions concurrently with an aggregate timeout.

        Records the batch latency to ``self._metrics`` for the vision-OCR
        scheduling gate (§7.7A).

        If the batch times out, all regions receive ``OCRResult.empty(name)``.
        """
        ocr_regions = self._profile.ocr_regions()
        if not ocr_regions:
            return {}

        tasks = [
            self._ocr.recognize_region(frame, region)
            for region in ocr_regions
        ]

        _t0 = time.monotonic()
        try:
            results_list = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            _elapsed_ms = timeout * 1000
            self._metrics.record_ocr_latency(_elapsed_ms)
            logger.warning(
                f"OCR batch timed out after {_elapsed_ms:.0f} ms; "
                f"all OCR regions will be empty"
            )
            return {r.name: OCRResult.empty(r.name) for r in ocr_regions}
        else:
            _elapsed_ms = (time.monotonic() - _t0) * 1000
            self._metrics.record_ocr_latency(_elapsed_ms)

        output: dict[str, OCRResult] = {}
        for region, result in zip(ocr_regions, results_list):
            if isinstance(result, Exception):
                logger.warning(f"OCR failed for '{region.name}': {result}")
                output[region.name] = OCRResult.empty(region.name)
            elif isinstance(result, OCRResult):
                output[region.name] = result
            else:
                output[region.name] = OCRResult.empty(region.name)

        return output

    # ------------------------------------------------------------------
    # Phase 2.5 — cache integration
    # ------------------------------------------------------------------

    def get_cached_action(self, state: GameState) -> Any | None:
        """Return a previously cached LLM action for *state*, if any.

        The caller (decision loop) should call this **after** ``process()``.
        A non‑``None`` return value means the LLM call can be skipped and
        the cached action used directly.
        """
        hash_val = state.get("_state_hash")
        if hash_val is None:
            return None
        return self._cache.get(hash_val)

    def cache_action(self, state: GameState, action: Any) -> None:
        """Store *action* in the cache keyed by the state hash.

        The caller (decision loop) should call this **after** a successful
        LLM decision so that future identical states within the TTL window
        can reuse the result.
        """
        hash_val = state.get("_state_hash")
        if hash_val is None:
            logger.warning(
                "cache_action() called on a GameState without _state_hash; "
                "did you forget to call process() first?"
            )
            return
        self._cache.set(hash_val, action)
        logger.debug(f"Cached action for state hash {hash_val[:8]}...")

    @property
    def cache(self) -> StateCache:
        """Expose the underlying StateCache for inspection / testing."""
        return self._cache

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_roi(frame: np.ndarray, region: "RegionConfig") -> np.ndarray | None:
        """Safely crop *region.bounds* from *frame*.

        Returns ``None`` if the bounds are entirely out‑of‑range.
        """
        x1, y1, x2, y2 = region.bounds
        h, w = frame.shape[:2]

        # Clamp to frame dimensions
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Sentinel for raw-bar fallback
# ---------------------------------------------------------------------------


class _RawBarSentinel:
    """Internal sentinel returned when no valid bar/OCR reading exists
    but a raw ROI image was captured.  The caller should inspect the
    result and attach the image as ``{slot}_raw_bar`` on the GameState.
    """

    __slots__ = ("slot_name", "image")

    def __init__(self, slot_name: str, image: np.ndarray) -> None:
        self.slot_name = slot_name
        self.image = image

    def __repr__(self) -> str:
        return f"<RawBarSentinel slot={self.slot_name!r}>"


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "StateProcessor",
    "PipelineMetrics",
    "VisionExecutor",
    "shutdown_vision_executor",
]

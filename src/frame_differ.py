"""
Adaptive Frame Skipping ― Phase 2.8

Lightweight pixel-diff detector that decides whether a new frame warrants
full processing (state extraction → OCR → LLM) or can be skipped in favour
of reusing the previous action.

Architecture
------------
::

    Capture → Downsample + Grayscale → compute_pixel_diff() → AdaptiveFrameSkipper
                                                                        │
                                              ┌─────────────────────────┘
                                              ▼
                              True  → full pipeline (process)
                              False → skip, reuse previous action

Algorithm
---------
Primary (recommended by spec research):
  ``cv2.absdiff`` + ``cv2.mean`` ― mean absolute pixel difference.
  Sub‑0.5 ms on 640×360 grayscale frames.

Secondary (optional, disabled by default):
  Perceptual dHash fallback via OpenCV ``img_hash`` for exact scene‑cut
  detection when pixel diff is ambiguous.

Usage::

    from src.frame_differ import compute_pixel_diff, AdaptiveFrameSkipper, DiffConfig

    diff_config = DiffConfig()
    skipper = AdaptiveFrameSkipper(diff_config)
    prev_gray = None

    while True:
        frame = await capture()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            score = compute_pixel_diff(prev_gray, gray)
            if not skipper.should_process(score) and last_state:
                # Skip ― reuse last_action
                continue

        prev_gray = gray
        # … run full pipeline …
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DiffConfig:
    """Configuration for the adaptive frame‑skipping subsystem.

    All fields mirror the defaults recommended in the spec
    (Extra_research05 §5.3) and the architecture document (§7.4).
    """

    # -- Primary pixel-diff parameters --
    downsample_width: int = 640
    """Target width for downsampling before differencing."""

    downsample_height: int = 360
    """Target height for downsampling before differencing."""

    method: str = "absdiff_mean"
    """Diff aggregation method.  Supported values:
    ``"absdiff_mean"`` (default) or ``"norm_l1"``.
    """

    # -- Adaptive skipper --
    adaptive_enabled: bool = True
    """When False, ``AdaptiveFrameSkipper.should_process()`` always
    returns ``True`` (i.e. processing is never skipped)."""

    adaptive_window: int = 30
    """Number of recent diff scores used for percentile calculation."""

    static_percentile: int = 70
    """Percentile below which a frame is considered static (→ skip)."""

    high_motion_percentile: int = 85
    """Percentile above which a frame is considered high‑motion
    (→ always process)."""

    # -- Perceptual hash fallback (secondary, disabled by default) --
    hash_fallback_enabled: bool = False
    """When True, a perceptual dHash is computed when the pixel diff
    falls into the ambiguous middle band, providing a more precise
    scene‑cut signal."""

    hash_method: str = "opencv_dhash"
    """Hash implementation: ``"opencv_dhash"`` (fast, C++ backend) or
    ``"imagehash_dhash"`` (portable pure‑Python fallback)."""

    hash_threshold: int = 6
    """Maximum Hamming distance under which two hashes are considered
    the same scene.  Lower = more sensitive to changes."""

    @classmethod
    def from_dict(cls, data: dict) -> DiffConfig:
        """Populate a ``DiffConfig`` from a configuration dictionary.

        Unknown keys are silently ignored so that the global config
        can carry extra keys without causing errors.
        """
        valid_keys = {
            "downsample_width",
            "downsample_height",
            "method",
            "adaptive_enabled",
            "adaptive_window",
            "static_percentile",
            "high_motion_percentile",
            "hash_fallback_enabled",
            "hash_method",
            "hash_threshold",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Core pixel-diff function
# ---------------------------------------------------------------------------


def compute_pixel_diff(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    method: str = "absdiff_mean",
    downsample_size: tuple[int, int] | None = (640, 360),
) -> float:
    """Compute a single scalar diff score between two frames.

    Both inputs are expected to be grayscale ``uint8`` arrays.  If they
    are not already grayscale they will be converted automatically but
    callers should pre‑convert once per capture cycle for performance.

    Args:
        prev_frame: Previous grayscale frame.
        curr_frame: Current grayscale frame.
        method: ``"absdiff_mean"`` or ``"norm_l1"``.
        downsample_size: Optional ``(width, height)`` to resize to before
            differencing.  Pass ``None`` to use the frames as‑is.

    Returns:
        Mean absolute pixel difference (range ~0–255).  Lower values
        indicate a more static scene.
    """
    # Ensure grayscale
    if prev_frame.ndim == 3:
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if curr_frame.ndim == 3:
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Optionally downsample
    if downsample_size is not None:
        prev_frame = cv2.resize(
            prev_frame, downsample_size, interpolation=cv2.INTER_NEAREST
        )
        curr_frame = cv2.resize(
            curr_frame, downsample_size, interpolation=cv2.INTER_NEAREST
        )

    if method == "norm_l1":
        total = cv2.norm(prev_frame, curr_frame, cv2.NORM_L1)
        pixels = prev_frame.shape[0] * prev_frame.shape[1]
        return float(total) / float(pixels)
    else:
        # Default: absdiff_mean
        diff = cv2.absdiff(prev_frame, curr_frame)
        return float(cv2.mean(diff)[0])


# ---------------------------------------------------------------------------
# Perceptual hash (secondary, optional)
# ---------------------------------------------------------------------------


def _compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Compute a 64‑bit difference hash for *image* (grayscale).

    Uses OpenCV's ``img_hash`` module when available; falls back to a
    pure‑NumPy implementation otherwise.

    Hash size ``8`` produces a 64‑bit hash (``8×8`` grid of differences
    on a ``8×9`` resized image).
    """
    # OpenCV img_hash path (C++ backend, fastest)
    try:
        if not hasattr(cv2, "img_hash"):
            raise ImportError("cv2.img_hash not available")
        resized = cv2.resize(image, (hash_size + 1, hash_size))
        hash_obj = cv2.img_hash.BlockMeanHash_create(mode=1)
        hash_arr = hash_obj.compute(resized)
        # hash_arr is a single-row uint8 array; pack into int
        return int.from_bytes(hash_arr.tobytes(), "little")
    except Exception:
        pass

    # Pure-NumPy fallback
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    return sum(bit << i for i, bit in enumerate(bits))


def _hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two 64‑bit integers."""
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------------
# Adaptive Frame Skipper
# ---------------------------------------------------------------------------


class AdaptiveFrameSkipper:
    """Adaptive threshold‑based frame skipper.

    Maintains a rolling window of recent diff scores and uses percentile
    thresholds to decide whether a frame should be fully processed or
    can be skipped.  The thresholds self‑calibrate to the game's typical
    motion level over time.

    Decision logic
    --------------
    * **Warm‑up** (history < window/2): always process.
    * **Static** (score < ``static_percentile``): skip.
    * **High‑motion** (score > ``high_motion_percentile``): always process.
    * **Middle band**: probabilistic ― chance proportional to position
      between the two thresholds.

    Adapted from the reference implementation in Extra_research05 §2.1.
    """

    def __init__(self, config: DiffConfig | None = None) -> None:
        self._config = config or DiffConfig()
        self._diff_history: list[float] = []
        self._last_hash: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_process(
        self,
        diff_score: float,
        current_frame: np.ndarray | None = None,
    ) -> bool:
        """Decide whether *diff_score* warrants full pipeline processing.

        Args:
            diff_score: Mean absolute pixel difference (from
                :func:`compute_pixel_diff`).
            current_frame: Optional current frame; only needed when
                perceptual‑hash fallback is enabled.

        Returns:
            ``True`` if the frame should be processed; ``False`` if it
            can be skipped.
        """
        # Bypass ― adaptive skipping disabled
        if not self._config.adaptive_enabled:
            return True

        # Record score
        self._diff_history.append(diff_score)
        window = self._config.adaptive_window
        if len(self._diff_history) > window:
            self._diff_history.pop(0)

        # Warm‑up phase ― not enough data yet
        if len(self._diff_history) < window // 2:
            return True

        low = np.percentile(self._diff_history, self._config.static_percentile)
        high = np.percentile(self._diff_history, self._config.high_motion_percentile)

        if diff_score < low:
            return False  # static → skip

        if diff_score > high:
            return True  # big change → process

        # Middle band ― probabilistic
        if high == low:
            return diff_score >= low

        prob = (diff_score - low) / (high - low)

        # Hash fallback (optional): refine the probability with a
        # more sensitive scene‑cut detector.
        if (
            self._config.hash_fallback_enabled
            and current_frame is not None
        ):
            prob = self._apply_hash_fallback(current_frame, prob)

        return _random.random() < prob

    def reset(self) -> None:
        """Clear diff history (e.g. after a scene change is detected)."""
        self._diff_history.clear()
        self._last_hash = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_hash_fallback(
        self, frame: np.ndarray, base_prob: float
    ) -> float:
        """Refine the processing probability using a perceptual hash.

        If the dHash differs significantly from the last known hash,
        boost the probability toward 1.0 to force a processing cycle.
        """
        try:
            current_hash = _compute_dhash(frame)
        except Exception:
            logger.debug("dHash fallback failed; using base probability.", exc_info=True)
            return base_prob

        if self._last_hash is not None:
            distance = _hamming_distance(self._last_hash, current_hash)
            if distance >= self._config.hash_threshold:
                # Significant scene change → force process
                self._last_hash = current_hash
                return 1.0

        self._last_hash = current_hash
        return base_prob


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "compute_pixel_diff",
    "AdaptiveFrameSkipper",
    "DiffConfig",
]
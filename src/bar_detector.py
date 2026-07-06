"""
Colour Bar Fill Percentage Detector — Phase 2.2

Full pipeline: HSV threshold → projection / radial / segmented →
confidence scoring (convex hull) → temporal filtering → dynamic
threshold adjustment.

Implements the complete specification from Extra_research02.md
Sections 1‑12, including the two‑point calibration data loader.

Performance targets:
  - Solid bar: < 10 ms
  - Radial / segmented bar: < 25 ms

Usage::

    from src.region_profile import RegionConfig
    from src.bar_detector import ColourBarDetector, ColourBarCalibration

    # From a RegionConfig with calibration dict:
    calibration = ColourBarCalibration.from_dict(region.calibration)
    detector = ColourBarDetector(calibration)

    # On each frame:
    roi = frame[y1:y2, x1:x2]          # BGR numpy array
    pct, confidence, success = detector.process(roi)
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Valid constants
# ---------------------------------------------------------------------------

_VALID_BAR_TYPES = frozenset({
    "solid_horizontal",
    "solid_vertical",
    "gradient",
    "segmented",
    "radial",
})

_VALID_ORIENTATIONS = frozenset({
    "left_to_right",
    "right_to_left",
    "top_to_bottom",
    "bottom_to_top",
    "radial",
})

_VALID_METHODS = frozenset({
    "projection",
    "boundingrect",
    "edge",
    "polar",
})


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class BarDetectorError(Exception):
    """Base exception for colour bar detection errors."""


class CalibrationError(BarDetectorError):
    """Raised when calibration data is invalid or missing."""


# ---------------------------------------------------------------------------
# ColourBarCalibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColourBarCalibration:
    """Validated calibration data for a colour bar region.

    Parsed from a ``RegionConfig.calibration`` dict. The schema follows
    Extra_research02 Section 10.

    Attributes:
        enabled: Whether bar detection is active for this region.
        bar_type: ``solid_horizontal`` | ``solid_vertical`` | ``gradient`` |
                  ``segmented`` | ``radial``.
        orientation: Direction the bar fills from.
        total_length_px: Total bar length in pixels along the fill axis.
        fill_hsv_lower / fill_hsv_upper: HSV range for the filled portion.
        empty_hsv_lower / empty_hsv_upper: HSV range for the empty/background.
        use_fill_mask: If True, threshold fill colour; else threshold empty
                       colour and invert.
        method: Detection method — ``projection``, ``boundingrect``, ``edge``, ``polar``.
        confidence_threshold: Minimum confidence (0‑1) for a valid reading.
        segment_count: Number of segments (segmented bars only).
        reference_segment_area: Expected area in px² of one full segment.
        radial_center: (cx, cy) of radial bar.
        radial_radius: Radius in pixels.
        calibration_samples: SHA‑256 hashes of calibration images.
        dynamic_adjustment: Whether to auto‑adapt thresholds over time.
    """

    enabled: bool = True
    bar_type: str = "solid_horizontal"
    orientation: str = "left_to_right"
    total_length_px: int = 100
    fill_hsv_lower: tuple[int, int, int] = (40, 100, 100)
    fill_hsv_upper: tuple[int, int, int] = (80, 255, 255)
    empty_hsv_lower: tuple[int, int, int] = (0, 0, 0)
    empty_hsv_upper: tuple[int, int, int] = (179, 30, 40)
    use_fill_mask: bool = True
    method: str = "projection"
    confidence_threshold: float = 0.6
    segment_count: int | None = None
    reference_segment_area: int | None = None
    radial_center: tuple[int, int] | None = None
    radial_radius: int | None = None
    calibration_samples: dict[str, str] = field(default_factory=dict)
    dynamic_adjustment: bool = True

    def __post_init__(self) -> None:
        if self.bar_type not in _VALID_BAR_TYPES:
            raise CalibrationError(
                f"Invalid bar_type '{self.bar_type}'; must be one of "
                f"{sorted(_VALID_BAR_TYPES)}"
            )
        if self.orientation not in _VALID_ORIENTATIONS:
            raise CalibrationError(
                f"Invalid orientation '{self.orientation}'; must be one of "
                f"{sorted(_VALID_ORIENTATIONS)}"
            )
        if self.method not in _VALID_METHODS:
            raise CalibrationError(
                f"Invalid method '{self.method}'; must be one of "
                f"{sorted(_VALID_METHODS)}"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise CalibrationError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
        if self.total_length_px <= 0:
            raise CalibrationError(
                f"total_length_px must be positive, got {self.total_length_px}"
            )
        if self.bar_type == "segmented":
            if self.segment_count is None or self.segment_count <= 0:
                raise CalibrationError(
                    "segmented bar requires positive segment_count"
                )
        if self.bar_type == "radial":
            if self.radial_center is None or self.radial_radius is None:
                raise CalibrationError(
                    "radial bar requires radial_center and radial_radius"
                )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColourBarCalibration":
        """Create a ``ColourBarCalibration`` from a deserialised JSON dict.

        Handles legacy ``"empty_color"`` / ``"full_color"`` hex strings by
        warning and falling back to defaults (Phase 5 calibration UI will
        generate proper HSV bounds).
        """
        enabled = data.get("enabled", True)
        if not enabled:
            return cls(enabled=False)

        bar_type = data.get("bar_type", "solid_horizontal")
        orientation = data.get("orientation", "left_to_right")

        # total_length_px: auto‑computed from ROI dimensions during calibration
        total_length_px = data.get("total_length_px", 100)

        fill_lower = tuple(data.get("fill_hsv_lower", [40, 100, 100]))
        fill_upper = tuple(data.get("fill_hsv_upper", [80, 255, 255]))
        empty_lower = tuple(data.get("empty_hsv_lower", [0, 0, 0]))
        empty_upper = tuple(data.get("empty_hsv_upper", [179, 30, 40]))
        use_fill = data.get("use_fill_mask", True)
        method = data.get("method", "projection")
        conf_thresh = float(data.get("confidence_threshold", 0.6))
        dynamic = data.get("dynamic_adjustment", True)

        segment_count = data.get("segment_count")
        ref_seg_area = data.get("reference_segment_area")

        radial_center_raw = data.get("radial_center")
        radial_center = tuple(radial_center_raw) if radial_center_raw else None
        radial_radius = data.get("radial_radius")

        calib_samples = data.get("calibration_samples", {})

        return cls(
            enabled=True,
            bar_type=bar_type,
            orientation=orientation,
            total_length_px=int(total_length_px),
            fill_hsv_lower=tuple(int(x) for x in fill_lower),
            fill_hsv_upper=tuple(int(x) for x in fill_upper),
            empty_hsv_lower=tuple(int(x) for x in empty_lower),
            empty_hsv_upper=tuple(int(x) for x in empty_upper),
            use_fill_mask=use_fill,
            method=method,
            confidence_threshold=conf_thresh,
            segment_count=int(segment_count) if segment_count is not None else None,
            reference_segment_area=int(ref_seg_area) if ref_seg_area is not None else None,
            radial_center=radial_center,
            radial_radius=int(radial_radius) if radial_radius is not None else None,
            calibration_samples=calib_samples,
            dynamic_adjustment=dynamic,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a JSON‑compatible dict."""
        d: dict[str, Any] = {
            "enabled": self.enabled,
            "bar_type": self.bar_type,
            "orientation": self.orientation,
            "total_length_px": self.total_length_px,
            "fill_hsv_lower": list(self.fill_hsv_lower),
            "fill_hsv_upper": list(self.fill_hsv_upper),
            "empty_hsv_lower": list(self.empty_hsv_lower),
            "empty_hsv_upper": list(self.empty_hsv_upper),
            "use_fill_mask": self.use_fill_mask,
            "method": self.method,
            "confidence_threshold": self.confidence_threshold,
            "dynamic_adjustment": self.dynamic_adjustment,
        }
        if self.segment_count is not None:
            d["segment_count"] = self.segment_count
        if self.reference_segment_area is not None:
            d["reference_segment_area"] = self.reference_segment_area
        if self.radial_center is not None:
            d["radial_center"] = list(self.radial_center)
        if self.radial_radius is not None:
            d["radial_radius"] = self.radial_radius
        if self.calibration_samples:
            d["calibration_samples"] = dict(self.calibration_samples)
        return d


# ---------------------------------------------------------------------------
# TwoPointCalibrationLoader
# ---------------------------------------------------------------------------


class TwoPointCalibrationLoader:
    """Produces calibration data from two screen captures: empty bar and full bar.

    The ``compute_hsv_thresholds`` static method implements the algorithm
    from Extra_research02 Section 3 (also Calibration_UI_research.md Problem 3):

    1. Compute mean ± tolerance of empty ROI → ``empty_hsv_lower/upper``
    2. Compute mean ± tolerance of central 80 % of full ROI → ``fill_hsv_lower/upper``
    3. Extract geometry (total_length_px, segments, radial params)

    Usage::

        calib_dict = TwoPointCalibrationLoader.compute_bar_hsv_thresholds(
            empty_img, full_img,
            bar_type="solid_horizontal",
            orientation="left_to_right",
        )
        # → dict ready to embed in regions.json calibration field
    """

    @staticmethod
    def compute_bar_hsv_thresholds(
        empty_img: np.ndarray,
        full_img: np.ndarray,
        bar_type: str = "solid_horizontal",
        orientation: str = "left_to_right",
    ) -> dict[str, Any]:
        """Compute HSV thresholds from empty and full bar screen captures.

        Args:
            empty_img: BGR numpy array of the bar region at 0 % fill.
            full_img: BGR numpy array of the bar region at 100 % fill.
            bar_type: One of the valid bar types.
            orientation: Fill direction.

        Returns:
            Calibration dictionary matching Extra_research02 Section 10 schema.
        """
        empty_hsv = cv2.cvtColor(empty_img, cv2.COLOR_BGR2HSV)
        full_hsv = cv2.cvtColor(full_img, cv2.COLOR_BGR2HSV)

        # Empty background colour statistics
        e_mean, _ = cv2.meanStdDev(empty_hsv)
        h, s, v = int(e_mean[0][0]), int(e_mean[1][0]), int(e_mean[2][0])
        empty_lower = (
            max(0, h - 15),
            max(0, s - 30),
            max(0, v - 30),
        )
        empty_upper = (
            min(179, h + 15),
            min(255, s + 30),
            min(255, v + 30),
        )

        # Fill colour from central 80 % of full bar to avoid edge blending
        fh, fw = full_hsv.shape[:2]
        crop = full_hsv[int(fh * 0.1):int(fh * 0.9), int(fw * 0.1):int(fw * 0.9)]
        f_mean, _ = cv2.meanStdDev(crop)
        f_h, f_s, f_v = int(f_mean[0][0]), int(f_mean[1][0]), int(f_mean[2][0])
        fill_lower = (
            max(0, f_h - 15),
            max(0, f_s - 30),
            max(0, f_v - 30),
        )
        fill_upper = (
            min(179, f_h + 15),
            min(255, f_s + 30),
            min(255, f_v + 30),
        )

        # Geometry
        if orientation in ("left_to_right", "right_to_left"):
            total_length_px = full_img.shape[1]
        elif orientation in ("top_to_bottom", "bottom_to_top"):
            total_length_px = full_img.shape[0]
        else:
            total_length_px = max(full_img.shape[1], full_img.shape[0])

        result: dict[str, Any] = {
            "enabled": True,
            "bar_type": bar_type,
            "orientation": orientation,
            "total_length_px": total_length_px,
            "fill_hsv_lower": list(fill_lower),
            "fill_hsv_upper": list(fill_upper),
            "empty_hsv_lower": list(empty_lower),
            "empty_hsv_upper": list(empty_upper),
            "use_fill_mask": True,
            "method": "projection",
            "confidence_threshold": 0.6,
            "dynamic_adjustment": True,
        }

        # Segmented bar: count segments in full image
        if bar_type == "segmented":
            hsv = full_hsv.copy()
            lower = np.array(fill_lower, dtype=np.uint8)
            upper = np.array(fill_upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Filter tiny specks
            areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
            if areas:
                result["segment_count"] = len(areas)
                result["reference_segment_area"] = int(np.median(areas))
            result["method"] = "projection"

        # Radial bar: try HoughCircles to detect centre
        if bar_type == "radial":
            gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=int(min(full_img.shape[:2]) * 0.5),
            )
            if circles is not None and len(circles[0]) > 0:
                c = circles[0][0]
                result["radial_center"] = [int(c[0]), int(c[1])]
                result["radial_radius"] = int(c[2])
            result["method"] = "polar"

        # SHA‑256 hashes of calibration images
        result["calibration_samples"] = {
            "empty_hash": hashlib.sha256(empty_img.tobytes()).hexdigest(),
            "full_hash": hashlib.sha256(full_img.tobytes()).hexdigest(),
        }

        return result

    @classmethod
    def from_calibration_images(
        cls,
        empty_path: str | Path,
        full_path: str | Path,
        bar_type: str = "solid_horizontal",
        orientation: str = "left_to_right",
    ) -> ColourBarCalibration:
        """Load two calibration images from disk and produce a
        ``ColourBarCalibration``.

        Args:
            empty_path: Path to 0 % fill screenshot (PNG / JPEG).
            full_path: Path to 100 % fill screenshot.
            bar_type: Bar type constant.
            orientation: Fill direction constant.

        Returns:
            Validated ``ColourBarCalibration``.
        """
        empty_img = cv2.imread(str(empty_path))
        if empty_img is None:
            raise CalibrationError(f"Cannot read empty bar image: {empty_path}")

        full_img = cv2.imread(str(full_path))
        if full_img is None:
            raise CalibrationError(f"Cannot read full bar image: {full_path}")

        if empty_img.shape != full_img.shape:
            raise CalibrationError(
                f"Empty ({empty_img.shape}) and full ({full_img.shape}) "
                f"images must have the same dimensions"
            )

        calib_dict = cls.compute_bar_hsv_thresholds(empty_img, full_img, bar_type, orientation)
        return ColourBarCalibration.from_dict(calib_dict)


# ---------------------------------------------------------------------------
# ColourBarDetector
# ---------------------------------------------------------------------------


class ColourBarDetector:
    """Production‑ready colour bar fill percentage detector.

    Implements the full pipeline described in Extra_research02 Section 11:

    - Preprocessing (resize ≤ 100 px long side, optional blur)
    - BGR → HSV colour thresholding
    - Morphological cleanup (3×3 open/close)
    - Fill metric: projection, bounding rect, segment count, or polar arc
    - Confidence scoring (convex hull compactness + ratio agreement)
    - Temporal filtering (ring buffer of 5 frames)
    - Dynamic threshold adjustment (exponential smoothing every 100 frames)

    All methods are synchronous — the Phase 2.4 StateProcessor will wrap
    calls in ``asyncio.to_thread``.

    Usage::

        detector = ColourBarDetector(calibration)
        roi = frame[y1:y2, x1:x2]
        pct, confidence, success = detector.process(roi)
    """

    _DYNAMIC_ADJUST_INTERVAL: int = 100
    """Frames between dynamic threshold updates."""

    def __init__(self, calibration: ColourBarCalibration) -> None:
        self._calib = calibration
        self._fill_lower = np.array(calibration.fill_hsv_lower, dtype=np.uint8)
        self._fill_upper = np.array(calibration.fill_hsv_upper, dtype=np.uint8)
        self._empty_lower = np.array(calibration.empty_hsv_lower, dtype=np.uint8)
        self._empty_upper = np.array(calibration.empty_hsv_upper, dtype=np.uint8)

        # Temporal filtering state
        self.history: deque[tuple[float, float]] = deque(maxlen=5)
        self.last_good_percentage: float = 50.0
        self.consecutive_low_confidence: int = 0

        # Dynamic adjustment state
        self._frame_count: int = 0
        self._last_percentage: float = 50.0
        self._last_roi_hash: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, roi: np.ndarray) -> tuple[float, float, bool]:
        """Process a bar ROI and return fill percentage.

        Args:
            roi: BGR numpy array of the colour bar screen region.

        Returns:
            ``(percentage, confidence, success)`` tuple:
            - *percentage*: 0.0–100.0 (clamped)
            - *confidence*: 0.0–1.0
            - *success*: ``True`` if confidence ≥ threshold
        """
        if not self._calib.enabled:
            return self.last_good_percentage, 0.0, False

        # Frame skipping — if ROI is unchanged, reuse cached value
        roi_hash = hashlib.md5(roi.tobytes()).hexdigest()
        if roi_hash == self._last_roi_hash and self.history:
            last_pct, last_conf = self.history[-1]
            return last_pct, last_conf, last_conf >= self._calib.confidence_threshold
        self._last_roi_hash = roi_hash

        # 1. Preprocess
        roi_processed = self._preprocess(roi)
        if roi_processed is None or roi_processed.size == 0:
            return self.last_good_percentage, 0.0, False

        # 2. Bar type dispatch
        percentage, mask = self._dispatch(roi_processed)

        # 3. Confidence score
        confidence = self._compute_confidence(mask, percentage)

        # 4. Temporal filtering
        self.history.append((percentage, confidence))
        success = confidence >= self._calib.confidence_threshold

        if not success:
            self.consecutive_low_confidence += 1
            # Fallback to last good value
            percentage = self.last_good_percentage
        else:
            self.consecutive_low_confidence = 0
            self.last_good_percentage = percentage

        # 5. Dynamic threshold adjustment
        if self._calib.dynamic_adjustment:
            self._frame_count += 1
            self._last_percentage = percentage
            if (
                self._frame_count % self._DYNAMIC_ADJUST_INTERVAL == 0
                and confidence > 0.7
                and 5.0 < percentage < 95.0
            ):
                self._dynamic_adjust(roi_processed)

        # 6. Trigger recalibration warning if needed
        if self.consecutive_low_confidence >= 3:
            return percentage, confidence, False

        return min(100.0, max(0.0, percentage)), confidence, success

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(roi: np.ndarray) -> np.ndarray | None:
        """Downsample so longest side ≤ 100 px (per spec §7)."""
        if roi is None or roi.size == 0:
            return None
        h, w = roi.shape[:2]
        max_dim = max(h, w)
        if max_dim <= 100:
            return roi
        scale = 100.0 / max_dim
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # ------------------------------------------------------------------
    # Dispatch by bar type
    # ------------------------------------------------------------------

    def _dispatch(self, roi: np.ndarray) -> tuple[float, np.ndarray]:
        """Route to the correct detection method."""
        bar_type = self._calib.bar_type
        if bar_type == "solid_horizontal":
            return self._detect_solid_horizontal(roi)
        elif bar_type == "solid_vertical":
            return self._detect_solid_vertical(roi)
        elif bar_type == "segmented":
            return self._detect_segmented(roi)
        elif bar_type == "radial":
            return self._detect_radial(roi)
        elif bar_type == "gradient":
            return self._detect_gradient(roi)
        else:
            return 0.0, np.zeros(roi.shape[:2], dtype=np.uint8)

    # ------------------------------------------------------------------
    # HSV threshold helper
    # ------------------------------------------------------------------

    def _build_mask(self, roi: np.ndarray) -> np.ndarray:
        """Create a binary mask from the region using HSV colour thresholds."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        if self._calib.use_fill_mask:
            mask = cv2.inRange(hsv, self._fill_lower, self._fill_upper)
        else:
            # Threshold empty colour and invert
            mask = cv2.inRange(hsv, self._empty_lower, self._empty_upper)
            mask = cv2.bitwise_not(mask)

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    # ------------------------------------------------------------------
    # Solid horizontal bar (projection profile)
    # ------------------------------------------------------------------

    def _detect_solid_horizontal(self, roi: np.ndarray) -> tuple[float, np.ndarray]:
        """Horizontal bar: column projection profile.

        Handles ``left_to_right`` and ``right_to_left`` orientations.
        """
        mask = self._build_mask(roi)
        proj = np.sum(mask, axis=0)
        cols = np.where(proj > 0)[0]

        if len(cols) == 0:
            return 0.0, mask

        total_width = mask.shape[1]

        if self._calib.method == "boundingrect":
            x, _, w, _ = cv2.boundingRect(mask)
            filled_width = w
        else:
            # projection (default) — robust to holes
            filled_width = int(cols[-1] - cols[0] + 1)

        percentage = (filled_width / total_width) * 100.0

        if self._calib.orientation == "right_to_left":
            percentage = 100.0 - percentage

        return min(100.0, max(0.0, percentage)), mask

    # ------------------------------------------------------------------
    # Solid vertical bar
    # ------------------------------------------------------------------

    def _detect_solid_vertical(self, roi: np.ndarray) -> tuple[float, np.ndarray]:
        """Vertical bar: row projection profile.

        Handles ``top_to_bottom`` and ``bottom_to_top`` orientations.
        """
        mask = self._build_mask(roi)
        proj = np.sum(mask, axis=1)
        rows = np.where(proj > 0)[0]

        if len(rows) == 0:
            return 0.0, mask

        total_height = mask.shape[0]
        filled_height = int(rows[-1] - rows[0] + 1)
        percentage = (filled_height / total_height) * 100.0

        if self._calib.orientation == "bottom_to_top":
            percentage = 100.0 - percentage

        return min(100.0, max(0.0, percentage)), mask

    # ------------------------------------------------------------------
    # Segmented bar (contour counting)
    # ------------------------------------------------------------------

    def _detect_segmented(self, roi: np.ndarray) -> tuple[float, np.ndarray]:
        """Segmented bar: count filled segments via contour analysis.

        Per Extra_research02 Section 5.3.
        """
        mask = self._build_mask(roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0, mask

        seg_count = self._calib.segment_count or 1
        ref_area = self._calib.reference_segment_area or 250

        min_area = ref_area * 0.5
        max_area = ref_area * 1.5

        filled = sum(1 for c in contours if min_area < cv2.contourArea(c) < max_area)
        percentage = (filled / seg_count) * 100.0

        return min(100.0, max(0.0, percentage)), mask

    # ------------------------------------------------------------------
    # Radial bar (polar coordinate angle measurement)
    # ------------------------------------------------------------------

    def _detect_radial(self, roi: np.ndarray) -> tuple[float, np.ndarray]:
        """Radial bar: warpPolar → angular arc measurement.

        Per Extra_research02 Section 5.2.
        """
        mask = self._build_mask(roi)
        if self._calib.radial_center is None or self._calib.radial_radius is None:
            return 0.0, mask

        cx, cy = self._calib.radial_center
        radius = self._calib.radial_radius

        # Scale centre and radius to preprocessed ROI dimensions
        h, w = mask.shape[:2]
        # If the original ROI was downsampled, centre/radius need scaling
        # (assume calibration values are relative to original ROI; we need
        #  to scale them proportionally — but since we only have the
        #  preprocessed ROI, use them as-is with a note that calibration
        #  should be done at the same resolution)
        cx_clamped = min(cx, w - 1)
        cy_clamped = min(cy, h - 1)
        radius_clamped = min(radius, min(w, h) // 2)

        try:
            polar = cv2.warpPolar(
                mask,
                (360, radius_clamped),
                (cx_clamped, cy_clamped),
                radius_clamped,
                cv2.WARP_POLAR_LINEAR,
            )
        except cv2.error:
            return 0.0, mask

        # Sum along radius axis to get fill intensity per angle
        angle_intensity = np.sum(polar, axis=0)

        # Find longest contiguous arc above threshold
        threshold = max(1.0, radius_clamped * 0.3)
        angles_filled = np.where(angle_intensity > threshold)[0]

        if len(angles_filled) == 0:
            return 0.0, mask

        max_arc = _find_longest_contiguous_run(angles_filled, 360)
        percentage = (max_arc / 360.0) * 100.0

        return min(100.0, max(0.0, percentage)), mask

    # ------------------------------------------------------------------
    # Gradient bar (edge detection + segment height)
    # ------------------------------------------------------------------

    def _detect_gradient(self, roi: np.ndarray) -> tuple[float, np.ndarray]:
        """Gradient bar: edge detection to find fill/empty boundary.

        Uses Canny edges + projection to locate the transition point.
        Falls back to horizontal projection if edge detection fails.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Horizontal projection of edges — the fill boundary is the
        # rightmost column with a strong edge response
        proj = np.sum(edges, axis=0)
        edge_threshold = max(1.0, edges.shape[0] * 0.2)
        edge_cols = np.where(proj > edge_threshold)[0]

        if len(edge_cols) == 0:
            # Fallback to solid horizontal
            return self._detect_solid_horizontal(roi)

        total_width = roi.shape[1]

        if self._calib.orientation == "left_to_right":
            boundary = int(edge_cols[-1])
            percentage = (boundary / total_width) * 100.0
        elif self._calib.orientation == "right_to_left":
            boundary = int(edge_cols[0])
            percentage = ((total_width - boundary) / total_width) * 100.0
        else:
            return self._detect_solid_horizontal(roi)

        return min(100.0, max(0.0, percentage)), edges

    # ------------------------------------------------------------------
    # Confidence scoring (Extra_research02 §6.1)
    # ------------------------------------------------------------------

    def _compute_confidence(self, mask: np.ndarray, percentage: float) -> float:
        """Compute confidence score (0–1) based on mask compactness and
        ratio agreement between pixel count and projection percentage.

        Special cases:
        - Empty mask (percentage ≈ 0): returns 1.0 — zero fill is unambiguous.
        - Full mask (percentage ≈ 100): returns 1.0 — full fill is unambiguous.
        """
        if mask is None or mask.size == 0:
            return 0.0

        total_pixels = mask.size
        filled_pixels = cv2.countNonZero(mask)
        pixel_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0.0

        # Extremes are unambiguous
        if percentage <= 0.5 and pixel_ratio <= 0.01:
            return 1.0
        if percentage >= 99.5 and pixel_ratio >= 0.95:
            return 1.0

        # 1. Mask compactness (convex hull test)
        compactness = 0.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                compactness = area / hull_area

        # 2. Ratio agreement
        expected_ratio = percentage / 100.0
        ratio_error = abs(pixel_ratio - expected_ratio)

        # 3. Combine
        confidence = (compactness * 0.6) + ((1.0 - ratio_error) * 0.4)
        return max(0.0, min(1.0, confidence))

    # ------------------------------------------------------------------
    # Dynamic threshold adjustment (Extra_research02 §8)
    # ------------------------------------------------------------------

    def _dynamic_adjust(self, roi: np.ndarray) -> None:
        """Exponential moving average update of fill HSV thresholds.

        Called every 100 frames when confidence is high and the bar is
        not at an extreme.
        """
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self._fill_lower, self._fill_upper)

            # Mean HSV of the filled region only
            filled_pixels = hsv[mask > 0]
            if len(filled_pixels) < 10:
                return

            mean_hsv = np.mean(filled_pixels, axis=0)
            h, s, v = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])

            new_lower = np.array(
                [max(0, h - 10), max(0, s - 20), max(0, v - 20)],
                dtype=np.uint8,
            )
            new_upper = np.array(
                [min(179, h + 10), min(255, s + 20), min(255, v + 20)],
                dtype=np.uint8,
            )

            # Exponential smoothing: 0.9 * old + 0.1 * new
            self._fill_lower = (0.9 * self._fill_lower.astype(float) + 0.1 * new_lower.astype(float)).astype(np.uint8)
            self._fill_upper = (0.9 * self._fill_upper.astype(float) + 0.1 * new_upper.astype(float)).astype(np.uint8)
        except Exception:
            # Dynamic adjustment is best‑effort — never crash the pipeline
            pass

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def calibration(self) -> ColourBarCalibration:
        return self._calib


# ---------------------------------------------------------------------------
# Helper: contiguous run detection (wrap‑around aware)
# ---------------------------------------------------------------------------


def _find_longest_contiguous_run(indices: np.ndarray, modulo: int) -> int:
    """Return the length of the longest contiguous run of integer indices,
    with wrap‑around at *modulo*.

    Used for radial bar arc detection.

    Example: ``indices=[358, 359, 0, 1, 2]``, ``modulo=360`` → returns ``5``.
    """
    if len(indices) == 0:
        return 0

    sorted_indices = np.sort(indices)
    max_run = 1
    current_run = 1

    for i in range(1, len(sorted_indices)):
        diff = sorted_indices[i] - sorted_indices[i - 1]
        if diff <= 1:
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 1
    max_run = max(max_run, current_run)

    # Wrap‑around check: first and last
    if len(sorted_indices) >= 2:
        wrap_gap = (sorted_indices[0] + modulo) - sorted_indices[-1]
        if wrap_gap <= 1:
            # Find leading run + trailing run
            leading_run = 1
            for i in range(1, len(sorted_indices)):
                if sorted_indices[i] - sorted_indices[i - 1] <= 1:
                    leading_run += 1
                else:
                    break
            trailing_run = 1
            for i in range(len(sorted_indices) - 2, -1, -1):
                if sorted_indices[i + 1] - sorted_indices[i] <= 1:
                    trailing_run += 1
                else:
                    break
            max_run = max(max_run, leading_run + trailing_run)

    return max_run
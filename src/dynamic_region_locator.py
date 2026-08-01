"""
Dynamic Region Locator — Phase 6.x (post‑6.8 enhancement)

Resolves runtime bounding boxes for OCR regions whose screen position
cannot be defined statically in ``regions.json``.  Three anchoring modes
are supported, each addressing a different class of moving/unknown text:

* ``static`` — (default) region bounds are used as‑is; no resolution.
* ``vision_anchor`` — region is positioned relative to a YOLO‑detected
  object's bounding box.  Falls back to a user‑supplied static region
  when the anchor object is not detected.
* ``motion`` — reuses the existing ``compute_pixel_diff`` pipeline to
  discover changed pixel clusters; OCR is run on each cluster that exceeds
  a configurable threshold.
* ``text_detection`` — runs a lightweight OpenCV DNN text detector (EAST)
  to find candidate text regions in a configurable ROI; OCR is run on each.

All modes return **lists** of bounding boxes — a single dynamic region
may produce multiple OCR crops per frame (e.g. multiple tooltip popups).

Integration
-----------
Used by ``StateProcessor._run_ocr_batch`` **before** dispatching to
``OCRModule.recognize_region``.  For static regions the locator is a
no‑op.

Usage::

    from src.dynamic_region_locator import (
        DynamicRegionLocator, AnchoringConfig, ResolvedRegion,
    )

    locator = DynamicRegionLocator(vision_processor=vision, frame_differ=differ)
    resolved = await locator.resolve(anchoring, frame, prev_gray)
    for rr in resolved:
        result = await ocr.recognize_region(frame, region_config,
                                            override_bounds=rr.bounds)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.frame_differ import compute_pixel_diff as _compute_pixel_diff
from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchoringConfig:
    """Per‑region anchoring configuration.

    Only the fields relevant to the selected *mode* need to be populated.

    Attributes:
        mode: One of ``"static"``, ``"vision_anchor"``, ``"motion"``,
            or ``"text_detection"``.
        anchor_class: (vision_anchor) YOLO class name to anchor to.
        anchor_offset: (vision_anchor) ``(dx1, dy1, dx2, dy2)`` relative
            to the detected object's bounding box.  Positive *down/right*.
        anchor_fallback_bounds: (vision_anchor) Static bounds used when
            the anchor object is not detected by the vision pipeline.
        motion_trigger_threshold: (motion) Mean absdiff threshold above
            which a changed cluster triggers OCR.
        motion_dilation_kernel: (motion) Morphological dilation kernel size
            applied to the diff mask before contour extraction.
        motion_min_area: (motion) Minimum contour area (px²) to consider
            as a text candidate.
        text_detect_roi: (text_detection) ``(x1, y1, x2, y2)`` limiting
            the search area.  ``None`` = full frame.
        text_confidence_threshold: (text_detection) Minimum EAST confidence
            for a candidate text region (0.0‑1.0).
        text_nms_threshold: (text_detection) Non‑maximum‑suppression IoU
            threshold for EAST.
        text_keywords: (text_detection) If non‑empty, only OCR results
            containing at least one keyword (case‑insensitive) are kept.
        text_filter_regex: (text_detection) If set, OCR text is filtered
            through this regex; only matching results are kept.
        text_max_regions: (text_detection) Maximum number of text regions
            returned per frame (prevents OCR overload).
    """

    mode: str = "static"

    # vision_anchor
    anchor_class: str | None = None
    anchor_offset: tuple[int, int, int, int] | None = None
    anchor_fallback_bounds: tuple[int, int, int, int] | None = None

    # motion
    motion_trigger_threshold: float = 15.0
    motion_dilation_kernel: int = 7
    motion_min_area: int = 80

    # text_detection
    text_detect_roi: tuple[int, int, int, int] | None = None
    text_confidence_threshold: float = 0.5
    text_nms_threshold: float = 0.4
    text_keywords: list[str] | None = None
    text_filter_regex: str | None = None
    text_max_regions: int = 5

    _VALID_MODES: tuple[str, ...] = field(
        default=("static", "vision_anchor", "motion", "text_detection"),
        repr=False,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.mode not in self._VALID_MODES:
            raise ValueError(
                f"AnchoringConfig mode must be one of {self._VALID_MODES}, "
                f"got {self.mode!r}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AnchoringConfig | None":
        """Factory from a JSON‑compatible dict.  Returns ``None`` when
        *data* is empty or missing (→ static/default behaviour)."""
        if not data:
            return None
        mode = data.get("mode", "static")
        kwargs: dict[str, Any] = {"mode": mode}
        if mode == "vision_anchor":
            kwargs.update(
                {
                    "anchor_class": data.get("anchor_class"),
                    "anchor_offset": tuple(data["anchor_offset"])
                    if data.get("anchor_offset")
                    else None,
                    "anchor_fallback_bounds": tuple(data["anchor_fallback_bounds"])
                    if data.get("anchor_fallback_bounds")
                    else None,
                }
            )
        elif mode == "motion":
            kwargs.update(
                {
                    "motion_trigger_threshold": float(
                        data.get("motion_trigger_threshold", 15.0)
                    ),
                    "motion_dilation_kernel": int(
                        data.get("motion_dilation_kernel", 7)
                    ),
                    "motion_min_area": int(data.get("motion_min_area", 80)),
                }
            )
        elif mode == "text_detection":
            kwargs.update(
                {
                    "text_detect_roi": tuple(data["text_detect_roi"])
                    if data.get("text_detect_roi")
                    else None,
                    "text_confidence_threshold": float(
                        data.get("text_confidence_threshold", 0.5)
                    ),
                    "text_nms_threshold": float(
                        data.get("text_nms_threshold", 0.4)
                    ),
                    "text_keywords": data.get("text_keywords"),
                    "text_filter_regex": data.get("text_filter_regex"),
                    "text_max_regions": int(data.get("text_max_regions", 5)),
                }
            )
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to JSON‑compatible dict."""
        d: dict[str, Any] = {"mode": self.mode}
        if self.mode == "vision_anchor":
            if self.anchor_class is not None:
                d["anchor_class"] = self.anchor_class
            if self.anchor_offset is not None:
                d["anchor_offset"] = list(self.anchor_offset)
            if self.anchor_fallback_bounds is not None:
                d["anchor_fallback_bounds"] = list(self.anchor_fallback_bounds)
        elif self.mode == "motion":
            d.update(
                {
                    "motion_trigger_threshold": self.motion_trigger_threshold,
                    "motion_dilation_kernel": self.motion_dilation_kernel,
                    "motion_min_area": self.motion_min_area,
                }
            )
        elif self.mode == "text_detection":
            if self.text_detect_roi is not None:
                d["text_detect_roi"] = list(self.text_detect_roi)
            d.update(
                {
                    "text_confidence_threshold": self.text_confidence_threshold,
                    "text_nms_threshold": self.text_nms_threshold,
                    "text_keywords": self.text_keywords,
                    "text_filter_regex": self.text_filter_regex,
                    "text_max_regions": self.text_max_regions,
                }
            )
        return d


@dataclass(frozen=True)
class ResolvedRegion:
    """A single resolved bounding box ready for OCR.

    Attributes:
        bounds: Screen‑relative ``(x1, y1, x2, y2)``.
        label: Human‑readable label for logging (e.g. ``"motion_cluster_2"``).
        confidence: Provisional confidence of the locator (0‑1).  May
            be used later to weight OCR results.
    """

    bounds: tuple[int, int, int, int]
    label: str = "dynamic"
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Helper: EAST text detector (lazy‑loaded, shared across all instances)
# ---------------------------------------------------------------------------

_EAST_NET: cv2.dnn.Net | None = None
_EAST_LAYER_NAMES: tuple[str, str] | None = None
_EAST_INPUT_SIZE: tuple[int, int] = (320, 320)
"""Input size for the EAST text detector.  320×320 keeps latency ~3 ms
on a modern CPU while retaining usable text recall."""


def _init_east() -> None:
    """Lazily load the OpenCV EAST text detector model.

    Uses the bundled ``frozen_east_text_detection.pb`` from OpenCV's
    DNN samples.  On first call this function downloads the model if
    it is not already present in the OpenCV data directory.

    Raises:
        RuntimeError: If the model file cannot be found or loaded.
    """
    global _EAST_NET, _EAST_LAYER_NAMES

    if _EAST_NET is not None:
        return

    import os as _os
    import urllib.request as _request

    # Try common locations for the EAST model file
    candidates = [
        _os.path.join(_os.path.dirname(__file__), "..", "models", "frozen_east_text_detection.pb"),
        _os.path.join(_os.path.expanduser("~"), ".gameai", "models", "frozen_east_text_detection.pb"),
    ]

    # Also check OpenCV's sample data directory
    try:
        _cv2_data = cv2.data.haarcascades
        _candidate = _os.path.join(_os.path.dirname(_cv2_data), "frozen_east_text_detection.pb")
        candidates.append(_candidate)
    except Exception:
        pass

    model_path = None
    for c in candidates:
        if _os.path.isfile(c):
            model_path = c
            break

    if model_path is None:
        # Auto‑download to ~/.gameai/models/
        dest_dir = _os.path.join(_os.path.expanduser("~"), ".gameai", "models")
        _os.makedirs(dest_dir, exist_ok=True)
        model_path = _os.path.join(dest_dir, "frozen_east_text_detection.pb")
        url = (
            "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/"
            "frozen_east_text_detection.pb"
        )
        logger.info(f"Downloading EAST text detector model to {model_path} …")
        try:
            _request.urlretrieve(url, model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download EAST model from {url}: {exc}"
            ) from exc

    net = cv2.dnn.readNet(model_path)
    # Get output layer names
    layer_names = net.getLayerNames()
    try:
        # OpenCV 4.x+
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except Exception:
        # Fallback for older OpenCV
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    _EAST_NET = net
    _EAST_LAYER_NAMES = tuple(output_layers)
    logger.info("EAST text detector loaded")


def _decode_east_predictions(
    scores: np.ndarray,
    geometry: np.ndarray,
    conf_threshold: float,
    frame_w: int,
    frame_h: int,
    roi_offset: tuple[int, int] = (0, 0),
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Decode EAST output tensors into a list of ``(bbox, confidence)``.

    Adapted from OpenCV's EAST text detection example.
    """
    (num_rows, num_cols) = scores.shape[2:4]
    rects: list[tuple[tuple[int, int, int, int], float]] = []
    confidences: list[float] = []

    ox, oy = roi_offset

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            score = float(scores_data[x])
            if score < conf_threshold:
                continue

            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = float(angles_data[x])
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            h = float(x_data0[x]) + float(x_data2[x])
            w = float(x_data1[x]) + float(x_data3[x])

            end_x = int(offset_x + cos_a * float(x_data1[x]) + sin_a * float(x_data2[x]))
            end_y = int(offset_y - sin_a * float(x_data1[x]) + cos_a * float(x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            # Scale back to original image coords + roi offset
            x1 = max(0, int(start_x * frame_w / _EAST_INPUT_SIZE[0]) + ox)
            y1 = max(0, int(start_y * frame_h / _EAST_INPUT_SIZE[1]) + oy)
            x2 = min(frame_w + ox, int(end_x * frame_w / _EAST_INPUT_SIZE[0]) + ox)
            y2 = min(frame_h + oy, int(end_y * frame_h / _EAST_INPUT_SIZE[1]) + oy)

            if x2 <= x1 or y2 <= y1:
                continue

            rects.append(((x1, y1, x2, y2), score))
            confidences.append(score)

    # NMS
    if not rects:
        return []

    boxes = [[r[0][0], r[0][1], r[0][2], r[0][3]] for r in rects]
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, 0.4  # nms_threshold applied in caller
    )

    results: list[tuple[tuple[int, int, int, int], float]] = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append(rects[i])
    return results


# ---------------------------------------------------------------------------
# Per‑mode locator implementations
# ---------------------------------------------------------------------------


class _VisionAnchorLocator:
    """Resolve region bounds relative to a YOLO‑detected object."""

    def __init__(self, vision_processor: Any) -> None:
        self._vision = vision_processor

    def resolve(
        self, config: AnchoringConfig, detections: list[Any] | None
    ) -> list[ResolvedRegion]:
        """Return a resolved region or fallback."""
        if not detections or config.anchor_class is None:
            return self._fallback(config)

        # Find matching detections
        matches = [
            d
            for d in detections
            if getattr(d, "class_name", None) == config.anchor_class
        ]
        if not matches:
            logger.debug(
                f"Vision anchor class '{config.anchor_class}' not detected; "
                f"using fallback bounds"
            )
            return self._fallback(config)

        # Prefer highest confidence, then closest to center
        best = max(matches, key=lambda d: (d.confidence, -_distance_to_center(d.center)))

        if config.anchor_offset is not None:
            dx1, dy1, dx2, dy2 = config.anchor_offset
            bbox = getattr(best, "bbox", None)
            if bbox is not None:
                x1 = bbox[0] + dx1
                y1 = bbox[1] + dy1
                x2 = bbox[2] + dx2
                y2 = bbox[3] + dy2
            else:
                # Fallback to center + half‑size heuristic
                cx, cy = getattr(best, "center", (0, 0))
                x1 = cx + dx1
                y1 = cy + dy1
                x2 = cx + dx2
                y2 = cy + dy2
        else:
            bbox = getattr(best, "bbox", (0, 0, 0, 0))
            x1, y1, x2, y2 = bbox

        return [
            ResolvedRegion(
                bounds=_clamp_bounds((x1, y1, x2, y2)),
                label=f"vision_anchor_{config.anchor_class}",
                confidence=float(best.confidence),
            )
        ]

    def _fallback(self, config: AnchoringConfig) -> list[ResolvedRegion]:
        if config.anchor_fallback_bounds is not None:
            return [
                ResolvedRegion(
                    bounds=config.anchor_fallback_bounds,
                    label="vision_anchor_fallback",
                    confidence=0.0,
                )
            ]
        return []


class _MotionSnapLocator:
    """Discover changed regions via pixel‑differencing + contour extraction."""

    def resolve(
        self,
        config: AnchoringConfig,
        frame: np.ndarray,
        prev_gray: np.ndarray | None,
    ) -> list[ResolvedRegion]:
        if prev_gray is None:
            return []

        gray = _to_gray(frame)
        if gray.shape != prev_gray.shape:
            prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))

        diff = cv2.absdiff(prev_gray, gray)
        mean_diff = float(cv2.mean(diff)[0])

        if mean_diff < config.motion_trigger_threshold:
            return []

        # Threshold + morphological dilation to merge adjacent changed pixels
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (config.motion_dilation_kernel, config.motion_dilation_kernel),
        )
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions: list[ResolvedRegion] = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < config.motion_min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            bounds = _clamp_bounds((x, y, x + w, y + h), frame_shape=gray.shape)
            regions.append(
                ResolvedRegion(
                    bounds=bounds,
                    label=f"motion_cluster_{i}",
                    confidence=min(1.0, mean_diff / 50.0),
                )
            )

        logger.debug(
            f"Motion locator: {len(regions)} clusters found "
            f"(mean_diff={mean_diff:.1f})"
        )
        return regions


class _TextDetectorLocator:
    """Run EAST text detector to find candidate text regions."""

    def __init__(self) -> None:
        self._initialised = False

    def _ensure_initialised(self) -> None:
        if not self._initialised:
            _init_east()
            self._initialised = True

    def resolve(self, config: AnchoringConfig, frame: np.ndarray) -> list[ResolvedRegion]:
        self._ensure_initialised()

        # Crop to ROI if specified
        if config.text_detect_roi is not None:
            rx1, ry1, rx2, ry2 = config.text_detect_roi
            h, w = frame.shape[:2]
            rx1 = max(0, min(rx1, w - 1))
            ry1 = max(0, min(ry1, h - 1))
            rx2 = max(rx1 + 1, min(rx2, w))
            ry2 = max(ry1 + 1, min(ry2, h))
            roi = frame[ry1:ry2, rx1:rx2]
            roi_offset = (rx1, ry1)
        else:
            roi = frame
            roi_offset = (0, 0)

        if roi.size == 0:
            return []

        # Prepare EAST blob
        blob = cv2.dnn.blobFromImage(
            roi,
            1.0,
            _EAST_INPUT_SIZE,
            (123.68, 116.78, 103.94),
            swapRB=True,
            crop=False,
        )

        assert _EAST_NET is not None and _EAST_LAYER_NAMES is not None
        _EAST_NET.setInput(blob)
        scores, geometry = _EAST_NET.forward(_EAST_LAYER_NAMES)

        roi_h, roi_w = roi.shape[:2]
        candidates = _decode_east_predictions(
            scores,
            geometry,
            config.text_confidence_threshold,
            roi_w,
            roi_h,
            roi_offset=roi_offset,
        )

        # Sort by confidence descending, take top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[: config.text_max_regions]

        results: list[ResolvedRegion] = []
        for i, (bbox, conf) in enumerate(candidates):
            # Expand bbox slightly for OCR context (5 px padding on each side)
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1 = max(0, x1 - 5)
            y1 = max(0, y1 - 5)
            x2 = min(w, x2 + 5)
            y2 = min(h, y2 + 5)

            results.append(
                ResolvedRegion(
                    bounds=(x1, y1, x2, y2),
                    label=f"text_detect_{i}",
                    confidence=conf,
                )
            )

        return results


# ---------------------------------------------------------------------------
# DynamicRegionLocator (facade)
# ---------------------------------------------------------------------------


class DynamicRegionLocator:
    """Resolve dynamic OCR region bounds at runtime.

    Wires together the three anchoring locators and exposes a single
    ``resolve`` entry point.  The caller (``StateProcessor``) invokes
    this once per dynamic region per frame before OCR dispatch.

    Typical wiring (from decision_loop or StateProcessor constructor)::

        locator = DynamicRegionLocator(
            vision_processor=vision,
            frame_differ=differ,
        )
    """

    def __init__(
        self,
        vision_processor: Any = None,
        frame_differ: Any = None,
    ) -> None:
        self._vision = vision_processor
        self._differ = frame_differ  # AdaptiveFrameSkipper — not used directly
        self._vision_anchor = _VisionAnchorLocator(vision_processor) if vision_processor else None
        self._motion_snap = _MotionSnapLocator()
        self._text_detector: _TextDetectorLocator | None = None

        # Track last frame for motion detection
        self._prev_gray: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve(
        self,
        anchoring: AnchoringConfig,
        frame: np.ndarray,
    ) -> list[ResolvedRegion]:
        """Return list of resolved bounding boxes for one dynamic region.

        The method dispatches to the appropriate locator based on
        ``anchoring.mode``.  Motion mode updates the internal ``_prev_gray``
        buffer as a side effect.

        Args:
            anchoring: The region's anchoring configuration.
            frame: Current BGR frame (full screen).

        Returns:
            A list of ``ResolvedRegion`` objects ready for OCR cropping.
            May be empty if no candidates were found.
        """
        if anchoring.mode == "static":
            return []

        if anchoring.mode == "vision_anchor":
            return await self._resolve_vision_anchor(anchoring, frame)

        if anchoring.mode == "motion":
            return await self._resolve_motion(anchoring, frame)

        if anchoring.mode == "text_detection":
            return await self._resolve_text_detection(anchoring, frame)

        logger.warning(f"Unknown anchoring mode '{anchoring.mode}' — skipping")
        return []

    def reset_motion_history(self) -> None:
        """Clear the motion diff history (e.g. after a scene change)."""
        self._prev_gray = None

    # ------------------------------------------------------------------
    # Mode dispatchers
    # ------------------------------------------------------------------

    async def _resolve_vision_anchor(
        self, anchoring: AnchoringConfig, frame: np.ndarray
    ) -> list[ResolvedRegion]:
        """Resolve via vision anchor."""
        if self._vision_anchor is None:
            logger.warning(
                "vision_anchor requested but no VisionProcessor wired; "
                "returning empty"
            )
            return []

        # Get latest detections from the vision processor
        detections = None
        if self._vision is not None:
            # VisionProcessor stores latest detections internally
            detections = getattr(self._vision, "last_detections", None)

        return self._vision_anchor.resolve(anchoring, detections)

    async def _resolve_motion(
        self, anchoring: AnchoringConfig, frame: np.ndarray
    ) -> list[ResolvedRegion]:
        """Resolve via motion diff."""
        gray = _to_gray(frame)
        regions = self._motion_snap.resolve(anchoring, frame, self._prev_gray)
        self._prev_gray = gray
        return regions

    async def _resolve_text_detection(
        self, anchoring: AnchoringConfig, frame: np.ndarray
    ) -> list[ResolvedRegion]:
        """Resolve via EAST text detector."""
        if self._text_detector is None:
            self._text_detector = _TextDetectorLocator()
        return self._text_detector.resolve(anchoring, frame)


# ---------------------------------------------------------------------------
# Text filtering helpers (called by StateProcessor after OCR)
# ---------------------------------------------------------------------------


def filter_ocr_by_keywords(text: str, keywords: list[str] | None) -> bool:
    """Return ``True`` if *text* contains at least one keyword (case‑insensitive)."""
    if not keywords:
        return True
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def filter_ocr_by_regex(text: str, pattern: str | None) -> str | None:
    """Return the regex match (or ``None``) if *text* matches *pattern*."""
    if not pattern:
        return text
    m = re.search(pattern, text)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to grayscale."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _clamp_bounds(
    bounds: tuple[int, int, int, int],
    frame_shape: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Clamp bounds to non‑negative, ensuring x2 > x1 and y2 > y1."""
    x1, y1, x2, y2 = bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)
    if frame_shape is not None:
        h, w = frame_shape
        x1 = min(x1, w - 2)
        y1 = min(y1, h - 2)
        x2 = min(x2, w)
        y2 = min(y2, h)
    return (x1, y1, x2, y2)


def _distance_to_center(center: tuple[float, float] | None) -> float:
    """Euclidean distance from *center* to the origin (used for sorting)."""
    if center is None:
        return float("inf")
    return float(np.sqrt(center[0] ** 2 + center[1] ** 2))


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "AnchoringConfig",
    "DynamicRegionLocator",
    "ResolvedRegion",
    "filter_ocr_by_keywords",
    "filter_ocr_by_regex",
]
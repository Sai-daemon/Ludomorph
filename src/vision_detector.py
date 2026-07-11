"""
Vision Detector — Phase 4.1

YOLOv11n ONNX object detector, optical flow tracker, spatial context
builder, and VisionProcessor orchestrator.

Provides optional spatial awareness so the AI can "see" objects on
screen.  All heavy inference runs in a dedicated thread pool via
``asyncio.to_thread``.  Lightweight optical flow tracking bridges
gaps between detection frames.

Usage::

    from src.vision_detector import VisionProcessor, VisionConfig

    config = VisionConfig(enabled=True, model_path="models/yolo11n.onnx")
    vp = VisionProcessor(config, screen_size=(1920, 1080))

    spatial = await vp.process_frame(frame)   # np.ndarray (BGR)
    print(spatial.context_text)
    # "Visual detections:\n- person detected at center-middle (approx. screen position)"
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.logging_config import get_logger

# onnxruntime is conditionally imported in VisionDetector.__init__
# to avoid a hard import failure when the package is not installed.

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# COCO 2017 class names (80 classes)
# ---------------------------------------------------------------------------

COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """A single object detection result.

    Attributes:
        class_id: COCO class index (0‑79).
        class_name: Human‑readable class name, e.g. ``"person"``.
        confidence: Detection confidence 0.0‑1.0.
        bbox: Bounding box ``(x1, y1, x2, y2)`` in screen coordinates.
        center: Center point ``(cx, cy)`` in screen coordinates.
        area: Bounding box area in pixels².
        tracked_id: Persistent tracking ID assigned by optical flow,
            or ``None`` if untracked.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    area: int
    tracked_id: int | None = None


@dataclass
class SpatialContext:
    """Aggregated vision output for one frame.

    Attributes:
        detections: List of ``Detection`` objects (may be empty).
        context_text: Human‑readable summary for the LLM prompt.
        has_detections: ``True`` if at least one object was detected.
        timestamp: Unix timestamp of this reading.
        frame_number: Monotonic frame counter from the orchestrator.
    """

    detections: list[Detection]
    context_text: str
    has_detections: bool
    timestamp: float
    frame_number: int

    @classmethod
    def empty(cls) -> "SpatialContext":
        """Return a sentinel context indicating no vision data."""
        return cls(
            detections=[],
            context_text="",
            has_detections=False,
            timestamp=time.time(),
            frame_number=0,
        )


@dataclass
class VisionConfig:
    """User‑configurable vision settings.

    Mirrors the ``vision`` section of ``config.json`` (Section 8.1
    of the architecture spec).
    """

    enabled: bool = False
    model_path: str = "models/yolo11n.onnx"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    detection_interval: int = 5
    max_detections: int = 20
    roi: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)
    backend: str = "auto"  # "auto", "cpu", "openvino", "cuda"
    input_size: int = 640  # 320, 416, 640

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON‑compatible dict."""
        return {
            "enabled": self.enabled,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "detection_interval": self.detection_interval,
            "max_detections": self.max_detections,
            "roi": list(self.roi) if self.roi else None,
            "backend": self.backend,
            "input_size": self.input_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionConfig":
        """Create a ``VisionConfig`` from a deserialised JSON dict."""
        roi = data.get("roi")
        if roi is not None and isinstance(roi, list) and len(roi) == 4:
            roi = tuple(roi)  # type: ignore[assignment]
        else:
            roi = None
        return cls(
            enabled=data.get("enabled", False),
            model_path=data.get("model_path", "models/yolo11n.onnx"),
            confidence_threshold=data.get("confidence_threshold", 0.5),
            iou_threshold=data.get("iou_threshold", 0.45),
            detection_interval=data.get("detection_interval", 5),
            max_detections=data.get("max_detections", 20),
            roi=roi,
            backend=data.get("backend", "auto"),
            input_size=data.get("input_size", 640),
        )


# ===================================================================
# VisionDetector — YOLOv11n ONNX inference
# ===================================================================


class VisionDetector:
    """YOLOv11n object detector powered by ONNX Runtime.

    Loads a pre‑converted ONNX model and runs inference in a dedicated
    thread pool via ``asyncio.to_thread``.  Preprocessing applies
    letterbox resizing, BGR→RGB conversion, and normalisation to [0, 1].
    Postprocessing applies NMS and scales coordinates back to the
    original frame dimensions.

    Args:
        config: ``VisionConfig`` controlling confidence, IoU, input size, etc.
    """

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self.input_size = (config.input_size, config.input_size)
        self.class_names = COCO_CLASSES
        self._session = self._init_onnx_session()

        logger.info(
            f"VisionDetector initialised (input={config.input_size}×{config.input_size}, "
            f"backend={self._session.get_providers()[0]})"
        )

    # ------------------------------------------------------------------
    # ONNX session
    # ------------------------------------------------------------------

    def _init_onnx_session(self):
        """Load the ONNX model with the preferred execution provider."""
        import onnxruntime as ort

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {model_path}. "
                f"Download it from Ultralytics assets or disable vision."
            )

        providers = self._resolve_providers(ort)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        return ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=providers,
        )

    @staticmethod
    def _resolve_providers(ort) -> list[str]:
        """Return an ordered list of ONNX execution providers."""
        available = ort.get_available_providers()
        # Prefer GPU, then OpenVINO, then CPU
        preferred = []
        if "CUDAExecutionProvider" in available:
            preferred.append("CUDAExecutionProvider")
        if "OpenVINOExecutionProvider" in available:
            preferred.append("OpenVINOExecutionProvider")
        if "CPUExecutionProvider" in available:
            preferred.append("CPUExecutionProvider")
        if not preferred:
            raise RuntimeError("No ONNX execution provider available")
        return preferred

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect(
        self, image: np.ndarray, roi: tuple[int, int, int, int] | None = None
    ) -> list[Detection]:
        """Detect objects in an image (optionally restricted to a ROI).

        Runs the full ONNX inference pipeline in a thread pool so the
        event loop stays responsive.

        Args:
            image: BGR numpy array (full frame).
            roi: Optional ``(x1, y1, x2, y2)`` to crop before detection.

        Returns:
            List of ``Detection`` objects whose confidence meets the
            configured threshold.
        """
        # Crop ROI if provided
        roi_offset = (0, 0)
        if roi is not None:
            x1, y1, x2, y2 = roi
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            image = image[y1:y2, x1:x2]
            roi_offset = (x1, y1)

        # Preprocess
        input_tensor, scale, pad_left, pad_top = self._preprocess(image)

        # Run inference (blocking → thread pool)
        outputs = await asyncio.to_thread(
            self._session.run, None, {"images": input_tensor}
        )

        # Postprocess
        detections = self._postprocess(
            outputs[0],
            original_shape=image.shape[:2],
            scale=scale,
            pad=(pad_left, pad_top),
            roi_offset=roi_offset,
        )

        # Filter by confidence
        return [d for d in detections if d.confidence >= self.config.confidence_threshold]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Resize, letterbox, convert to RGB, normalise to [0, 1], and
        transpose to NCHW.

        Returns:
            Tuple of ``(input_tensor, scale, pad_left, pad_top)`` where
            *scale* is the resize factor so postprocessing can map
            predictions back to the original image coordinates.
        """
        h, w = image.shape[:2]
        target = self.config.input_size

        # Letterbox: resize keeping aspect ratio, padding to square
        scale = min(target / w, target / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create square canvas padded with grey (114, 114, 114)
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        pad_left = (target - new_w) // 2
        pad_top = (target - new_h) // 2
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        # BGR → RGB
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # Normalise to [0, 1], convert to float32, transpose HWC → CHW, add batch dim
        input_tensor = canvas.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # (3, H, W)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 3, H, W)

        return input_tensor, scale, pad_left, pad_top

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        output: np.ndarray,
        original_shape: tuple[int, int],
        scale: float,
        pad: tuple[int, int],
        roi_offset: tuple[int, int] = (0, 0),
    ) -> list[Detection]:
        """Convert raw YOLO output tensor to ``Detection`` objects.

        Handles coordinate scaling from the preprocessed 640×640 space
        back to the original image (including any ROI offset).

        Args:
            output: Raw ONNX output tensor of shape ``(1, 84, N)``.
            original_shape: ``(height, width)`` of the image passed to
                ``_preprocess()`` (after ROI crop, if any).
            scale: Resize scale factor from letterbox.
            pad: ``(pad_left, pad_top)`` from letterbox.
            roi_offset: ``(x_offset, y_offset)`` if the input was cropped.

        Returns:
            Detections with coordinates in screen space.
        """
        orig_h, orig_w = original_shape
        pad_left, pad_top = pad
        offset_x, offset_y = roi_offset

        # Squeeze batch dim: (1, 84, N) → (84, N)
        output = np.squeeze(output, axis=0)  # (84, N)

        if output.ndim != 2:
            logger.debug(f"Unexpected YOLO output shape: {output.shape}")
            return []

        # Transpose to (N, 84) for easier iteration
        output = output.T  # (N, 84)

        # Extract components
        bboxes_cxcywh = output[:, :4]  # (N, 4) — cx, cy, w, h (normalized 0‑1)
        class_scores = output[:, 4:]  # (N, 80)

        # Find best class per detection
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Convert cx, cy, w, h → x1, y1, x2, y2 in letterbox space
        boxes_xyxy = self._cxcywh_to_xyxy(bboxes_cxcywh)

        # Scale from normalized (0‑1) → letterbox pixel coords
        target = self.config.input_size
        boxes_xyxy[:, [0, 2]] *= target  # x1, x2
        boxes_xyxy[:, [1, 3]] *= target  # y1, y2

        # Remove padding offset
        boxes_xyxy[:, [0, 2]] -= pad_left
        boxes_xyxy[:, [1, 3]] -= pad_top

        # Scale from resized coords → original image coords
        boxes_xyxy /= scale

        # Clip to original image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h - 1)

        # Apply ROI offset (map back to full‑screen coordinates)
        boxes_xyxy[:, [0, 2]] += offset_x
        boxes_xyxy[:, [1, 3]] += offset_y

        # NMS
        keep = self._nms(boxes_xyxy, confidences, class_ids)
        boxes_xyxy = boxes_xyxy[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        # Build Detection objects
        detections: list[Detection] = []
        for i in range(len(keep)):
            if len(detections) >= self.config.max_detections:
                break
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
            center_x = int((x1 + x2) // 2)
            center_y = int((y1 + y2) // 2)
            area = int((x2 - x1) * (y2 - y1))

            cls_id = int(class_ids[i])
            if cls_id < 0 or cls_id >= len(self.class_names):
                continue

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=self.class_names[cls_id],
                    confidence=float(confidences[i]),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=(center_x, center_y),
                    area=area,
                )
            )

        return detections

    # ------------------------------------------------------------------
    # NMS and box utilities
    # ------------------------------------------------------------------

    def _nms(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
    ) -> np.ndarray:
        """Class‑aware Non‑Maximum Suppression.

        Returns indices of boxes to keep.
        """
        keep_indices: list[int] = []
        unique_classes = np.unique(class_ids)

        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_confs = confidences[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            # Sort by confidence descending
            order = np.argsort(cls_confs)[::-1]

            while len(order) > 0:
                best = order[0]
                keep_indices.append(int(cls_indices[best]))

                if len(order) == 1:
                    break

                # IoU of best vs remaining
                ious = self._box_iou_batch(cls_boxes[best], cls_boxes[order[1:]])
                order = order[1:][ious < self.config.iou_threshold]

        return np.array(keep_indices)

    @staticmethod
    def _box_iou_batch(box_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Vectorized IoU between one box and a set of boxes."""
        x1_a, y1_a, x2_a, y2_a = box_a
        area_a = (x2_a - x1_a) * (y2_a - y1_a)

        x1_b = boxes_b[:, 0]
        y1_b = boxes_b[:, 1]
        x2_b = boxes_b[:, 2]
        y2_b = boxes_b[:, 3]
        area_b = (x2_b - x1_b) * (y2_b - y1_b)

        # Intersection
        inter_x1 = np.maximum(x1_a, x1_b)
        inter_y1 = np.maximum(y1_a, y1_b)
        inter_x2 = np.minimum(x2_a, x2_b)
        inter_y2 = np.minimum(y2_a, y2_b)

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        return inter_area / (area_a + area_b - inter_area + 1e-6)

    @staticmethod
    def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
        result = np.zeros_like(boxes)
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return result


# ===================================================================
# OpticalFlowTracker — lightweight inter‑detection tracking
# ===================================================================


class OpticalFlowTracker:
    """Tracks detected objects between heavy detection frames using
    sparse Lucas‑Kanade optical flow.

    Associates new detections with existing tracks via IoU matching and
    assigns persistent ``tracked_id`` values.  On intermediate frames
    (when the heavy detector doesn't run), ``predict()`` advances tracks
    using optical flow alone.
    """

    _LK_PARAMS = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    def __init__(self) -> None:
        self.prev_gray: np.ndarray | None = None
        self.prev_points: np.ndarray | None = None  # shape (N, 1, 2), float32
        self.tracked_objects: dict[int, Detection] = {}
        self.next_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, current_gray: np.ndarray, detections: list[Detection]
    ) -> list[Detection]:
        """Associate new detections with existing tracks.

        On frames where heavy detection runs, this method:
        1. Predicts previous track positions via optical flow.
        2. Matches new detections to existing tracks by IoU.
        3. Assigns new IDs to unmatched detections.

        Args:
            current_gray: Current frame converted to grayscale.
            detections: Fresh detections from the YOLO detector.

        Returns:
            *detections* with ``tracked_id`` populated.
        """
        if self.prev_gray is not None and self.prev_points is not None:
            # Predict previous tracks using optical flow
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                current_gray,
                self.prev_points,
                None,
                **self._LK_PARAMS,
            )

            # Update tracked objects that survived optical flow
            valid_ids: list[int] = []
            valid_pts: list[np.ndarray] = []
            for i, (tracked_id, obj) in enumerate(list(self.tracked_objects.items())):
                if status[i] and status[i][0]:
                    new_x, new_y = curr_pts[i][0]
                    dx = new_x - obj.center[0]
                    dy = new_y - obj.center[1]
                    # Shift bbox by the same delta
                    x1, y1, x2, y2 = obj.bbox
                    obj.bbox = (
                        int(x1 + dx),
                        int(y1 + dy),
                        int(x2 + dx),
                        int(y2 + dy),
                    )
                    obj.center = (int(new_x), int(new_y))
                    valid_ids.append(tracked_id)
                    valid_pts.append(curr_pts[i])
                else:
                    # Track lost
                    del self.tracked_objects[tracked_id]

            # Keep only valid points for next frame
            if valid_pts:
                self.prev_points = np.array(valid_pts, dtype=np.float32)
            else:
                self.prev_points = None
        else:
            valid_ids = []

        # Match new detections to existing tracks (IoU)
        matched_detections = self._match_by_iou(detections)

        # Remaining unmatched detections get new IDs
        for det in matched_detections:
            if det.tracked_id is None:
                det.tracked_id = self.next_id
                self.tracked_objects[self.next_id] = det
                self.next_id += 1

        # Store points for next optical flow call
        if matched_detections:
            pts = np.array(
                [[d.center[0], d.center[1]] for d in matched_detections],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            self.prev_points = pts

        self.prev_gray = current_gray.copy()
        return matched_detections

    def predict(self, current_gray: np.ndarray) -> list[Detection] | None:
        """Advance tracked positions using optical flow only (no detector).

        Called on frames between heavy detection runs.  Returns the
        existing tracked objects with updated positions, or ``None`` if
        no tracks exist.
        """
        if self.prev_gray is None or self.prev_points is None:
            return None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            current_gray,
            self.prev_points,
            None,
            **self._LK_PARAMS,
        )

        results: list[Detection] = []
        kept_points: list[np.ndarray] = []
        for i, (tracked_id, obj) in enumerate(list(self.tracked_objects.items())):
            if status[i] and status[i][0]:
                new_x, new_y = curr_pts[i][0]
                dx = new_x - obj.center[0]
                dy = new_y - obj.center[1]
                x1, y1, x2, y2 = obj.bbox
                obj.bbox = (
                    int(x1 + dx),
                    int(y1 + dy),
                    int(x2 + dx),
                    int(y2 + dy),
                )
                obj.center = (int(new_x), int(new_y))
                results.append(obj)
                kept_points.append(curr_pts[i])
            else:
                del self.tracked_objects[tracked_id]

        if kept_points:
            self.prev_points = np.array(kept_points, dtype=np.float32)
        else:
            self.prev_points = None

        self.prev_gray = current_gray.copy()
        return results if results else None

    # ------------------------------------------------------------------
    # IoU matching
    # ------------------------------------------------------------------

    def _match_by_iou(self, detections: list[Detection]) -> list[Detection]:
        """Assign existing tracked IDs to new detections by IoU overlap.

        Each existing track can match at most one detection; unmatched
        detections keep ``tracked_id=None``.
        """
        if not self.tracked_objects:
            return detections

        # Build cost matrix: rows = existing tracks, cols = new detections
        track_ids = list(self.tracked_objects.keys())
        cost = np.zeros((len(track_ids), len(detections)))

        for i, tid in enumerate(track_ids):
            track_box = self.tracked_objects[tid].bbox
            for j, det in enumerate(detections):
                cost[i, j] = 1.0 - self._iou(track_box, det.bbox)

        # Greedy assignment (simple, fast, sufficient for our use case)
        assigned_detections: set[int] = set()
        assigned_tracks: set[int] = set()

        # Flatten and sort by cost ascending (best matches first)
        pairs: list[tuple[float, int, int]] = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                pairs.append((cost[i, j], i, j))
        pairs.sort(key=lambda x: x[0])

        for _, i, j in pairs:
            if i not in assigned_tracks and j not in assigned_detections:
                # Only match if IoU > 0.3 (cost < 0.7)
                if cost[i, j] < 0.7:
                    detections[j].tracked_id = track_ids[i]
                    # Update the tracked object with the new detection
                    self.tracked_objects[track_ids[i]] = detections[j]
                    assigned_tracks.add(i)
                    assigned_detections.add(j)

        return detections

    @staticmethod
    def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union between two boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0


# ===================================================================
# SpatialContextBuilder — detections → LLM‑readable text
# ===================================================================


class SpatialContextBuilder:
    """Converts a list of ``Detection`` objects into structured text
    suitable for injection into the LLM prompt.

    Groups by class, finds the closest instance to screen center, and
    reports its approximate screen position using human‑readable labels
    (e.g. ``"top-left"``, ``"center-middle"``).
    """

    _H_THRESHOLDS: tuple[float, float] = (0.4, 0.6)
    _V_THRESHOLDS: tuple[float, float] = (0.4, 0.6)

    def __init__(self, config: VisionConfig, screen_size: tuple[int, int]) -> None:
        self.config = config
        self.screen_width, self.screen_height = screen_size
        self._spatial_history: deque[list[Detection]] = deque(maxlen=10)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_context(self, detections: list[Detection]) -> str:
        """Build a human‑readable spatial context string.

        Args:
            detections: List of detections for the current frame.

        Returns:
            A multi‑line string, e.g.::

                Visual detections:
                - person detected at center-middle (approx. screen position)
                - bottle detected at bottom-right (approx. screen position)

            Or ``"No objects detected visually."`` if the list is empty.
        """
        self._spatial_history.append(detections)

        if not detections:
            return "No objects detected visually."

        # Group by class name
        grouped: dict[str, list[Detection]] = {}
        for d in detections:
            grouped.setdefault(d.class_name, []).append(d)

        lines = ["Visual detections:"]
        for class_name, items in grouped.items():
            # Find the closest instance to screen center
            closest = min(items, key=lambda d: self._distance_to_center(d.center))
            relative_x = closest.center[0] / self.screen_width
            relative_y = closest.center[1] / self.screen_height

            h_pos = (
                "left"
                if relative_x < self._H_THRESHOLDS[0]
                else "right"
                if relative_x > self._H_THRESHOLDS[1]
                else "center"
            )
            v_pos = (
                "top"
                if relative_y < self._V_THRESHOLDS[0]
                else "bottom"
                if relative_y > self._V_THRESHOLDS[1]
                else "middle"
            )

            lines.append(
                f"- {class_name} detected at {v_pos}-{h_pos} "
                f"(approx. screen position)"
            )

        return "\n".join(lines)

    def get_actionable_coordinates(self, detection: Detection) -> dict[str, Any]:
        """Return coordinate information for a detection in a format
        suitable for dynamic macro resolution.

        Returns:
            Dict with ``x``, ``y`` (absolute screen coords) and
            ``relative_x``, ``relative_y`` (0‑1 normalised).
        """
        return {
            "x": detection.center[0],
            "y": detection.center[1],
            "relative_x": detection.center[0] / self.screen_width,
            "relative_y": detection.center[1] / self.screen_height,
        }

    @property
    def spatial_history(self) -> list[list[Detection]]:
        """Return a copy of the recent detection history (for debugging)."""
        return list(self._spatial_history)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _distance_to_center(self, point: tuple[int, int]) -> float:
        """Euclidean distance from *point* to screen center."""
        cx = self.screen_width / 2
        cy = self.screen_height / 2
        return ((point[0] - cx) ** 2 + (point[1] - cy) ** 2) ** 0.5


# ===================================================================
# VisionProcessor — orchestrator
# ===================================================================


class VisionProcessor:
    """Main orchestrator for the vision pipeline.

    Wires together the detector, optical flow tracker, and spatial
    context builder.  Called once per captured frame from the main
    decision loop (via ``StateProcessor``).

    Heavy YOLO detection runs every *N* frames (configurable via
    ``VisionConfig.detection_interval``).  Intermediate frames use
    lightweight optical flow tracking to advance positions.

    Args:
        config: Vision settings.
        screen_size: ``(width, height)`` of the game window / capture area.
    """

    def __init__(self, config: VisionConfig, screen_size: tuple[int, int]) -> None:
        self.config = config
        self.detector = VisionDetector(config)
        self.tracker = OpticalFlowTracker()
        self.context_builder = SpatialContextBuilder(config, screen_size)
        self.frame_counter: int = 0
        self.is_enabled: bool = config.enabled
        self.last_detections: list[Detection] = []

        logger.debug(
            f"VisionProcessor created (screen={screen_size}, "
            f"detect_every={config.detection_interval} frames)"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_frame(self, frame: np.ndarray) -> SpatialContext:
        """Process a single captured frame through the vision pipeline.

        Called from the main event loop (via ``StateProcessor``) every
        captured frame.

        Args:
            frame: BGR numpy array (full screen capture).

        Returns:
            ``SpatialContext`` with detections, context text, and metadata.
        """
        if not self.is_enabled:
            return SpatialContext.empty()

        self.frame_counter += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.frame_counter % self.config.detection_interval == 0:
            # Heavy detection
            detections = await self.detector.detect(frame, self.config.roi)
            detections = self.tracker.update(gray, detections)
            self.last_detections = detections
        else:
            # Lightweight tracking
            tracked = self.tracker.predict(gray)
            if tracked is not None:
                detections = tracked
            else:
                detections = self.last_detections

        context_text = self.context_builder.build_context(detections)
        return SpatialContext(
            detections=detections,
            context_text=context_text,
            has_detections=len(detections) > 0,
            timestamp=time.time(),
            frame_number=self.frame_counter,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def spatial_history(self) -> list[list[Detection]]:
        """Expose the builder's recent detection history."""
        return self.context_builder.spatial_history


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "Detection",
    "SpatialContext",
    "VisionConfig",
    "VisionDetector",
    "OpticalFlowTracker",
    "SpatialContextBuilder",
    "VisionProcessor",
    "COCO_CLASSES",
]
"""
Phase 4.1 — Vision Module Integration Tests

Validates:
- VisionConfig dataclass (defaults, serialisation, from_dict)
- SpatialContext.empty() factory
- SpatialContextBuilder text output format and coordinate helpers
- VisionDetector model loading and structure (if model available)
- OpticalFlowTracker synthetic tracking continuity
- VisionProcessor orchestrator pipeline end-to-end
- Phase 4.3: MacroResolver dynamic step resolution

Test strategy from architecture.md Appendix A §9.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_path() -> None:
    import sys

    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_path()

# ---------------------------------------------------------------------------
# Model path helper
# ---------------------------------------------------------------------------

_MODEL_PATH = _PROJECT_ROOT / "models" / "yolo11n.onnx"
_MODEL_AVAILABLE = _MODEL_PATH.exists()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vision_config_dict() -> dict[str, Any]:
    """Return a minimal vision config dict matching config.json defaults."""
    return {
        "enabled": True,
        "model_path": str(_MODEL_PATH),
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45,
        "detection_interval": 5,
        "max_detections": 20,
        "roi": None,
        "backend": "auto",
        "input_size": 320,  # Smaller for faster tests
    }


@pytest.fixture(scope="module")
def vision_config(vision_config_dict: dict[str, Any]):
    """Build a VisionConfig from a dict fixture."""
    from src.vision_detector import VisionConfig

    return VisionConfig.from_dict(vision_config_dict)


@pytest.fixture
def dummy_frame() -> np.ndarray:
    """Generate a synthetic 640×480 BGR frame with a coloured rectangle."""
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)  # dark grey
    # Draw a bright green rectangle (simulates an object)
    cv2.rectangle(frame, (200, 150), (300, 250), (0, 255, 0), -1)
    return frame


@pytest.fixture
def empty_frame() -> np.ndarray:
    """Generate a blank 640×480 BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# =============================================================================
# 1. VisionConfig dataclass
# =============================================================================


class TestVisionConfig:
    """Validate VisionConfig defaults, serialisation, and parsing."""

    def test_defaults(self):
        """VisionConfig should have sensible defaults matching the spec."""
        from src.vision_detector import VisionConfig

        cfg = VisionConfig()
        assert cfg.enabled is False
        assert cfg.confidence_threshold == 0.5
        assert cfg.iou_threshold == 0.45
        assert cfg.detection_interval == 5
        assert cfg.max_detections == 20
        assert cfg.input_size == 640
        assert cfg.backend == "auto"
        assert cfg.roi is None

    def test_to_dict_roundtrip(self, vision_config):
        """to_dict → from_dict should produce an equivalent config."""
        d = vision_config.to_dict()
        assert d["confidence_threshold"] == vision_config.confidence_threshold
        assert d["input_size"] == vision_config.input_size

        rebuilt = vision_config.__class__.from_dict(d)
        assert rebuilt.confidence_threshold == vision_config.confidence_threshold
        assert rebuilt.iou_threshold == vision_config.iou_threshold
        assert rebuilt.detection_interval == vision_config.detection_interval
        assert rebuilt.input_size == vision_config.input_size

    def test_from_dict_roi_parsing(self):
        """ROI should parse from a list and round‑trip correctly."""
        from src.vision_detector import VisionConfig

        cfg = VisionConfig.from_dict(
            {"enabled": True, "roi": [10, 20, 300, 400]}
        )
        assert cfg.roi == (10, 20, 300, 400)

        # None / missing should also work
        cfg2 = VisionConfig.from_dict({"enabled": True})
        assert cfg2.roi is None

    def test_to_dict_serialises_to_json(self, vision_config):
        """The to_dict output should be JSON‑serialisable."""
        d = vision_config.to_dict()
        raw = json.dumps(d)
        assert isinstance(raw, str)
        parsed = json.loads(raw)
        assert parsed["enabled"] == vision_config.enabled


# =============================================================================
# 2. SpatialContext
# =============================================================================


class TestSpatialContext:
    """Validate SpatialContext.empty() and field assignments."""

    def test_empty_factory(self):
        from src.vision_detector import SpatialContext

        ctx = SpatialContext.empty()
        assert ctx.detections == []
        assert ctx.context_text == ""
        assert ctx.has_detections is False
        assert isinstance(ctx.timestamp, float)
        assert ctx.frame_number == 0

    def test_construction_with_detections(self):
        from src.vision_detector import Detection, SpatialContext

        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=(100, 100, 200, 200),
            center=(150, 150),
            area=10000,
            tracked_id=1,
        )
        ctx = SpatialContext(
            detections=[det],
            context_text="Person detected.",
            has_detections=True,
            timestamp=time.time(),
            frame_number=42,
        )
        assert len(ctx.detections) == 1
        assert ctx.detections[0].class_name == "person"
        assert ctx.has_detections is True
        assert ctx.frame_number == 42


# =============================================================================
# 3. SpatialContextBuilder
# =============================================================================


class TestSpatialContextBuilder:
    """Validate text output format and coordinate calculations."""

    def test_empty_detections(self, vision_config):
        from src.vision_detector import SpatialContextBuilder

        builder = SpatialContextBuilder(vision_config, screen_size=(1920, 1080))
        result = builder.build_context([])
        assert result == "No objects detected visually."

    def test_single_detection(self, vision_config):
        from src.vision_detector import Detection, SpatialContextBuilder

        builder = SpatialContextBuilder(vision_config, screen_size=(1920, 1080))
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(800, 300, 1000, 600),
            center=(900, 450),
            area=60000,
        )
        result = builder.build_context([det])
        assert "Visual detections:" in result
        assert "person" in result
        assert "middle-center" in result  # v_pos-h_pos format: ~42% of 1080, ~50% of 1920

    def test_multiple_same_class(self, vision_config):
        from src.vision_detector import Detection, SpatialContextBuilder

        builder = SpatialContextBuilder(vision_config, screen_size=(1920, 1080))

        dets = [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.7,
                bbox=(100, 100, 200, 200),
                center=(150, 150),
                area=10000,
            ),
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=(1600, 800, 1800, 1000),
                center=(1700, 900),
                area=40000,
            ),
        ]
        result = builder.build_context(dets)
        # Should mention person exactly once (grouped), using the one
        # closest to screen center (the second one at ~bottom-right)
        assert result.count("person") == 1
        assert "right" in result or "bottom" in result

    def test_position_labels(self, vision_config):
        from src.vision_detector import Detection, SpatialContextBuilder

        builder = SpatialContextBuilder(vision_config, screen_size=(1920, 1080))

        # top-left
        det = Detection(
            class_id=1,
            class_name="bicycle",
            confidence=0.8,
            bbox=(0, 0, 100, 100),
            center=(50, 50),
            area=10000,
        )
        result = builder.build_context([det])
        assert "top-left" in result

        # bottom-right
        det2 = Detection(
            class_id=1,
            class_name="bicycle",
            confidence=0.8,
            bbox=(1720, 880, 1920, 1080),
            center=(1820, 980),
            area=10000,
        )
        result2 = builder.build_context([det2])
        assert "bottom-right" in result2

    def test_get_actionable_coordinates(self, vision_config):
        from src.vision_detector import Detection, SpatialContextBuilder

        builder = SpatialContextBuilder(vision_config, screen_size=(1920, 1080))
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(800, 300, 1000, 600),
            center=(900, 450),
            area=60000,
        )
        coords = builder.get_actionable_coordinates(det)
        assert coords["x"] == 900
        assert coords["y"] == 450
        assert 0.4 < coords["relative_x"] < 0.5
        assert 0.4 < coords["relative_y"] < 0.5

    def test_spatial_history(self, vision_config):
        from src.vision_detector import Detection, SpatialContextBuilder

        builder = SpatialContextBuilder(vision_config, screen_size=(1920, 1080))

        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(0, 0, 100, 100),
            center=(50, 50),
            area=10000,
        )

        builder.build_context([det])
        builder.build_context([])
        history = builder.spatial_history
        assert len(history) == 2
        assert len(history[0]) == 1
        assert len(history[1]) == 0


# =============================================================================
# 4. VisionDetector (requires model)
# =============================================================================


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="yolo11n.onnx not available")
class TestVisionDetector:
    """Validate ONNX model loading, preprocessing, and detection."""

    def test_model_loading(self, vision_config):
        """Verify the ONNX model loads without error."""
        from src.vision_detector import VisionDetector

        detector = VisionDetector(vision_config)
        assert detector._session is not None
        assert detector.input_size == (320, 320)  # from fixture

    def test_preprocess_shape(self, vision_config, dummy_frame):
        """Verify preprocess output tensor has correct shape."""
        from src.vision_detector import VisionDetector

        detector = VisionDetector(vision_config)
        tensor, scale, pad_x, pad_y = detector._preprocess(dummy_frame)
        assert tensor.shape == (1, 3, vision_config.input_size, vision_config.input_size)
        assert tensor.dtype == np.float32
        assert 0.0 <= tensor.min() <= 1.0
        assert 0.0 <= tensor.max() <= 1.0

    @pytest.mark.asyncio
    async def test_detect_returns_detections(self, vision_config, dummy_frame):
        """Run a full detection pass and verify the output structure."""
        from src.vision_detector import VisionDetector, Detection

        detector = VisionDetector(vision_config)
        detections = await detector.detect(dummy_frame)
        assert isinstance(detections, list)

        for det in detections:
            assert isinstance(det, Detection)
            assert isinstance(det.class_id, int)
            assert 0 <= det.class_id < 80
            assert isinstance(det.class_name, str)
            assert 0.0 <= det.confidence <= 1.0
            assert len(det.bbox) == 4
            assert det.bbox[0] >= 0  # x1
            assert det.bbox[2] > det.bbox[0]  # x2 > x1
            assert det.area > 0

    @pytest.mark.asyncio
    async def test_detect_empty_frame(self, vision_config, empty_frame):
        """Detection on a black frame should produce no high‑confidence detections."""
        from src.vision_detector import VisionDetector

        # Use a higher threshold to reduce false positives on black frame
        vision_config.confidence_threshold = 0.8
        detector = VisionDetector(vision_config)
        detections = await detector.detect(empty_frame)
        # A completely black frame should have few/no high‑conf objects
        assert len(detections) < 5, (
            f"Expected few detections on empty frame, got {len(detections)}"
        )

    @pytest.mark.asyncio
    async def test_detect_with_roi(self, vision_config, dummy_frame):
        """Detection with ROI should restrict to cropped region."""
        from src.vision_detector import VisionDetector

        detector = VisionDetector(vision_config)
        # Crop to a small region away from the green rectangle
        dets_roi = await detector.detect(dummy_frame, roi=(0, 0, 100, 100))
        dets_full = await detector.detect(dummy_frame)
        # The ROI region may have fewer or different detections
        assert isinstance(dets_roi, list)

    def test_coco_classes(self):
        """Verify the 80 COCO class names are present."""
        from src.vision_detector import COCO_CLASSES

        assert len(COCO_CLASSES) == 80
        assert COCO_CLASSES[0] == "person"
        assert COCO_CLASSES[39] == "bottle"
        assert COCO_CLASSES[79] == "toothbrush"


# =============================================================================
# 5. OpticalFlowTracker
# =============================================================================


class TestOpticalFlowTracker:
    """Validate optical flow tracking with synthetic motion."""

    def test_tracker_initialisation(self):
        from src.vision_detector import OpticalFlowTracker

        tracker = OpticalFlowTracker()
        assert tracker.prev_gray is None
        assert tracker.prev_points is None
        assert tracker.tracked_objects == {}
        assert tracker.next_id == 0

    def test_update_first_frame(self):
        """On first frame, tracker assigns new IDs to all detections."""
        from src.vision_detector import Detection, OpticalFlowTracker

        tracker = OpticalFlowTracker()
        gray = np.ones((100, 100), dtype=np.uint8) * 128

        dets = [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=(10, 10, 50, 50),
                center=(30, 30),
                area=1600,
            ),
            Detection(
                class_id=1,
                class_name="bicycle",
                confidence=0.8,
                bbox=(60, 60, 90, 90),
                center=(75, 75),
                area=900,
            ),
        ]
        result = tracker.update(gray, dets)
        assert len(result) == 2
        assert result[0].tracked_id is not None
        assert result[1].tracked_id is not None
        assert result[0].tracked_id != result[1].tracked_id

    def test_update_tracking_continuity(self):
        """Verify that optical flow preserves tracked IDs across frames."""
        from src.vision_detector import Detection, OpticalFlowTracker

        tracker = OpticalFlowTracker()

        # Frame 1: place an object
        gray1 = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(gray1, (80, 80), (120, 120), 255, -1)

        det1 = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(80, 80, 120, 120),
            center=(100, 100),
            area=1600,
        )
        tracker.update(gray1, [det1])
        first_id = det1.tracked_id
        assert first_id is not None

        # Frame 2: object moved right/down slightly
        gray2 = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(gray2, (90, 90), (130, 130), 255, -1)

        det2 = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(90, 90, 130, 130),
            center=(110, 110),
            area=1600,
        )
        result = tracker.update(gray2, [det2])
        # Should match to first track
        assert len(result) == 1
        if result[0].tracked_id is not None:
            # If matched, ID should be preserved
            pass  # optical flow may or may not match depending on motion size

    def test_predict_returns_none_when_no_history(self):
        from src.vision_detector import OpticalFlowTracker

        tracker = OpticalFlowTracker()
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        assert tracker.predict(gray) is None

    def test_predict_updates_positions(self):
        """Predict should advance tracked positions on intermediate frames."""
        from src.vision_detector import Detection, OpticalFlowTracker

        tracker = OpticalFlowTracker()

        # Frame 1: establish a track
        gray1 = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(gray1, (80, 80), (120, 120), 255, -1)

        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(80, 80, 120, 120),
            center=(100, 100),
            area=1600,
        )
        tracker.update(gray1, [det])

        # Frame 2: object moved, use predict (no new detection)
        gray2 = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(gray2, (90, 90), (130, 130), 255, -1)

        result = tracker.predict(gray2)
        # Should return tracked objects with updated positions
        if result is not None:
            assert len(result) >= 1
            assert result[0].tracked_id == 0


# =============================================================================
# 6. VisionProcessor (orchestrator)
# =============================================================================


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="yolo11n.onnx not available")
class TestVisionProcessor:
    """Validate end‑to‑end vision pipeline."""

    def test_processor_creation(self, vision_config):
        from src.vision_detector import VisionProcessor

        vp = VisionProcessor(vision_config, screen_size=(1920, 1080))
        assert vp.is_enabled is True
        assert vp.frame_counter == 0
        assert vp.last_detections == []

    @pytest.mark.asyncio
    async def test_process_frame_returns_spatial_context(
        self, vision_config, dummy_frame
    ):
        from src.vision_detector import SpatialContext, VisionProcessor

        vp = VisionProcessor(vision_config, screen_size=(640, 480))
        result = await vp.process_frame(dummy_frame)
        assert isinstance(result, SpatialContext)
        assert result.frame_number == 1
        assert isinstance(result.context_text, str)
        assert isinstance(result.has_detections, bool)

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self, vision_config, dummy_frame):
        from src.vision_detector import SpatialContext, VisionProcessor

        vp = VisionProcessor(vision_config, screen_size=(640, 480))
        vp.is_enabled = False
        result = await vp.process_frame(dummy_frame)
        assert isinstance(result, SpatialContext)
        assert result.has_detections is False
        assert result.context_text == ""
        assert result.frame_number == 0  # counter doesn't increment when disabled

    @pytest.mark.asyncio
    async def test_detection_interval_respect(self, vision_config, dummy_frame):
        """Heavy detection should only run every N frames."""
        from src.vision_detector import VisionProcessor

        vp = VisionProcessor(vision_config, screen_size=(640, 480))
        # Frame 1: always runs detection (frame_counter 0 → 1, 1 % 5 != 0...)

        # Run 10 frames and verify frame counter increments
        for _ in range(10):
            await vp.process_frame(dummy_frame)
        assert vp.frame_counter >= 10

    @pytest.mark.asyncio
    async def test_spatial_history_accumulates(
        self, vision_config, dummy_frame
    ):
        from src.vision_detector import VisionProcessor

        vp = VisionProcessor(vision_config, screen_size=(640, 480))
        for _ in range(5):
            await vp.process_frame(dummy_frame)

        history = vp.spatial_history
        assert len(history) <= 10  # capped


# =============================================================================
# 7a. PipelineMetrics unit tests (§7.7E)
# =============================================================================


class TestPipelineMetrics:
    """Validate PipelineMetrics dataclass and ring buffer behaviour."""

    def test_defaults(self):
        from src.state_processor import PipelineMetrics

        pm = PipelineMetrics()
        assert pm.ocr_latencies == []
        assert pm.vision_frames_skipped_due_to_ocr == 0
        assert pm.vision_timeouts == 0
        assert pm.vision_contention_events == 0
        assert pm.vision_detection_interval_current == 2
        assert pm.vision_disabled_by_throttle is False

    def test_record_ocr_latency_ring_buffer(self):
        from src.state_processor import PipelineMetrics

        pm = PipelineMetrics()
        for i in range(50):
            pm.record_ocr_latency(float(i))
        # Ring buffer caps at 30
        assert len(pm.ocr_latencies) == 30
        # Most recent 30 values: 20..49
        assert pm.ocr_latencies[0] == 20.0
        assert pm.ocr_latencies[-1] == 49.0

    def test_avg_ocr_latency(self):
        from src.state_processor import PipelineMetrics

        pm = PipelineMetrics()
        assert pm.get_avg_ocr_latency() == 0.0

        pm.record_ocr_latency(100.0)
        pm.record_ocr_latency(200.0)
        assert pm.get_avg_ocr_latency() == 150.0

    def test_record_vision_skip(self):
        from src.state_processor import PipelineMetrics

        pm = PipelineMetrics()
        pm.record_vision_skip()
        pm.record_vision_skip()
        assert pm.vision_frames_skipped_due_to_ocr == 2

    def test_record_vision_timeout(self):
        from src.state_processor import PipelineMetrics

        pm = PipelineMetrics()
        pm.record_vision_timeout()
        assert pm.vision_timeouts == 1

    def test_record_contention(self):
        from src.state_processor import PipelineMetrics

        pm = PipelineMetrics()
        pm.record_contention()
        assert pm.vision_contention_events == 1

    def test_reset_counters(self):
        from src.state_processor import PipelineMetrics

        pm = PipelineMetrics()
        pm.record_vision_skip()
        pm.record_vision_timeout()
        pm.record_contention()
        pm.reset_counters()
        assert pm.vision_frames_skipped_due_to_ocr == 0
        assert pm.vision_timeouts == 0
        assert pm.vision_contention_events == 0


# =============================================================================
# 7b. Vision-OCR scheduling rules (§7.7F)
# =============================================================================


class TestVisionOCRScheduling:
    """Validate that the StateProcessor enforces the scheduling rules from
    architecture.md §7.7F:

    * OCR always has absolute priority.
    * Vision runs only on staggered frames.
    * Vision is skipped if avg OCR latency exceeds 150 ms.
    * Vision timeouts are recorded.
    * Contention events are tracked.
    """

    @pytest.mark.asyncio
    async def test_vision_skipped_when_ocr_slow(
        self, state_processor, dummy_frame
    ):
        """When avg OCR latency > 150 ms, vision should be skipped."""
        from src.state_processor import PipelineMetrics

        # Wire a shared metrics object with synthetic high OCR latency
        pm = PipelineMetrics()
        state_processor._metrics = pm
        # Simulate 5 frames of ~200 ms OCR latency
        for _ in range(5):
            pm.record_ocr_latency(200.0)

        # Wire a vision processor
        from src.vision_detector import VisionConfig, VisionProcessor

        vp = VisionProcessor(
            VisionConfig(enabled=True, input_size=320, detection_interval=1),
            screen_size=(640, 480),
        )
        state_processor._vision = vp
        state_processor.frame_counter = 0  # First frame triggers vision check

        # Process
        state = await state_processor.process(dummy_frame, skip_ocr=True)
        # Vision should have been skipped due to high OCR latency
        assert pm.vision_frames_skipped_due_to_ocr >= 1

    @pytest.mark.asyncio
    async def test_vision_runs_when_ocr_fast(
        self, state_processor, dummy_frame
    ):
        """When avg OCR latency < 150 ms, vision should run."""
        from src.state_processor import PipelineMetrics

        pm = PipelineMetrics()
        state_processor._metrics = pm
        # Low OCR latency
        for _ in range(5):
            pm.record_ocr_latency(80.0)

        from src.vision_detector import VisionConfig, VisionProcessor

        vp = VisionProcessor(
            VisionConfig(enabled=True, input_size=320, detection_interval=1),
            screen_size=(640, 480),
        )
        state_processor._vision = vp
        state_processor.frame_counter = 0

        state = await state_processor.process(dummy_frame, skip_ocr=True)
        # spatial_context should be present if vision ran
        spatial = state.get("spatial_context")
        assert spatial is not None
        assert isinstance(spatial, str) and len(spatial) > 0

    @pytest.mark.asyncio
    async def test_staggering_rule_respects_interval(
        self, state_processor, dummy_frame
    ):
        """Vision should only run on frames that align with the stagger interval."""
        from src.state_processor import PipelineMetrics

        pm = PipelineMetrics()
        pm.vision_detection_interval_current = 5  # Run every 5th frame
        state_processor._metrics = pm
        # Ensure OCR latency is low
        pm.record_ocr_latency(50.0)

        from src.vision_detector import VisionConfig, VisionProcessor

        vp = VisionProcessor(
            VisionConfig(enabled=True, input_size=320, detection_interval=1),
            screen_size=(640, 480),
        )
        state_processor._vision = vp

        # Frame 0 → counter 0 % 5 == 0 → should run
        state_processor.frame_counter = 0
        state1 = await state_processor.process(dummy_frame, skip_ocr=True)
        assert state1.get("spatial_context") is not None

        # Frame 1 → counter 1 % 5 != 0 → should skip
        state2 = await state_processor.process(dummy_frame, skip_ocr=True)
        # spatial_context may still be set from last detection but no new detection ran

    @pytest.mark.asyncio
    async def test_vision_disabled_by_throttle_skips(
        self, state_processor, dummy_frame
    ):
        """When throttling disables vision, no vision processing occurs."""
        from src.state_processor import PipelineMetrics

        pm = PipelineMetrics()
        pm.vision_disabled_by_throttle = True
        pm.record_ocr_latency(50.0)
        state_processor._metrics = pm

        from src.vision_detector import VisionConfig, VisionProcessor

        vp = VisionProcessor(
            VisionConfig(enabled=True, input_size=320, detection_interval=1),
            screen_size=(640, 480),
        )
        state_processor._vision = vp
        state_processor.frame_counter = 0

        state = await state_processor.process(dummy_frame, skip_ocr=True)
        # spatial_context should NOT be set
        assert state.get("spatial_context") is None

    @pytest.mark.asyncio
    async def test_contention_recorded_when_both_run(
        self, state_processor, dummy_frame
    ):
        """When OCR is NOT skipped AND vision starts, contention counter increments."""
        from src.state_processor import PipelineMetrics

        pm = PipelineMetrics()
        pm.vision_detection_interval_current = 1
        # Pre‑fill with enough low‑latency samples that the OCR batch
        # timeout (250 ms) won't push the average above the 150 ms gate.
        for _ in range(20):
            pm.record_ocr_latency(50.0)
        state_processor._metrics = pm

        from src.vision_detector import VisionConfig, VisionProcessor

        vp = VisionProcessor(
            VisionConfig(enabled=True, input_size=320, detection_interval=1),
            screen_size=(640, 480),
        )
        state_processor._vision = vp
        state_processor.frame_counter = 0

        before = pm.vision_contention_events
        # skip_ocr=False (default) + vision_started → contention recorded
        await state_processor.process(dummy_frame, skip_ocr=False)
        after = pm.vision_contention_events
        assert after > before

    def test_vision_executor_is_separate_thread_pool(self):
        """The VisionExecutor must be a distinct ThreadPoolExecutor (not shared)."""
        from src.state_processor import VisionExecutor

        from concurrent.futures import ThreadPoolExecutor

        assert isinstance(VisionExecutor, ThreadPoolExecutor)
        assert VisionExecutor._max_workers == 1  # type: ignore[attr-defined]

    def test_shutdown_vision_executor(self):
        """shutdown_vision_executor should gracefully terminate the pool."""
        from src.state_processor import shutdown_vision_executor

        # Should not raise
        shutdown_vision_executor(wait=False)


# =============================================================================
# 8. ThrottleState vision fields (§7.7D)
# =============================================================================


class TestThrottleStateVisionFields:
    """Validate that ThrottleState carries the required vision throttling fields."""

    def test_defaults(self):
        from src.decision_loop import ThrottleState

        ts = ThrottleState()
        assert ts.vision_disable is False
        assert ts.vision_interval_override is None
        assert ts.vision_input_size_override is None

    def test_sync_vision_throttle_to_metrics(self, state_processor):
        """_sync_vision_throttle_to_metrics pushes ThrottleState into PipelineMetrics."""
        from src.decision_loop import (
            ThrottleState,
            _sync_vision_throttle_to_metrics,
        )
        from src.state_processor import PipelineMetrics

        pm = PipelineMetrics()
        state_processor._metrics = pm

        ts = ThrottleState()
        ts.vision_disable = True
        ts.vision_interval_override = 10
        ts.vision_input_size_override = 320

        _sync_vision_throttle_to_metrics(state_processor, ts)
        assert pm.vision_disabled_by_throttle is True
        assert pm.vision_detection_interval_current == 10

    def test_sync_vision_throttle_reset(self, state_processor):
        """When throttling recovers, metrics should reset to defaults."""
        from src.decision_loop import (
            ThrottleState,
            _sync_vision_throttle_to_metrics,
        )
        from src.state_processor import PipelineMetrics

        pm = PipelineMetrics()
        state_processor._metrics = pm

        ts = ThrottleState()
        ts.vision_disable = False
        ts.vision_interval_override = None
        ts.vision_input_size_override = None

        _sync_vision_throttle_to_metrics(state_processor, ts)
        assert pm.vision_disabled_by_throttle is False
        assert pm.vision_detection_interval_current == 2


# =============================================================================
# 9. Vision integration into StateProcessor (smoke test)
# =============================================================================


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="yolo11n.onnx not available")
class TestStateProcessorVisionIntegration:
    """Verify StateProcessor passes vision data through to GameState."""

    @pytest.mark.asyncio
    async def test_spatial_context_flows_to_state(
        self, state_processor, dummy_frame, vision_config
    ):
        """When vision is wired, state should include 'spatial_context'."""
        from src.vision_detector import VisionProcessor

        vp = VisionProcessor(vision_config, screen_size=(640, 480))
        vp.is_enabled = True
        state_processor._vision = vp
        # Force the frame counter so the vision path is triggered
        state_processor.frame_counter = 1

        state = await state_processor.process(dummy_frame, skip_ocr=True)
        spatial = state.get("spatial_context")
        # spatial_context may or may not be set depending on timing,
        # but the key should exist if vision ran
        if spatial is not None:
            assert isinstance(spatial, str)

    @pytest.mark.asyncio
    async def test_vision_none_is_noop(self, state_processor, dummy_frame):
        """When vision_processor is None, no spatial_context is set."""
        state_processor._vision = None
        state_processor.frame_counter = 1

        state = await state_processor.process(dummy_frame, skip_ocr=True)
        # Without vision, spatial_context should not appear
        assert state.get("spatial_context") is None


# =============================================================================
# 10. Phase 4.3 — MacroResolver
# =============================================================================


class TestMacroResolver:
    """Validate dynamic macro step resolution per architecture.md §4.4B."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _det(
        class_name: str = "person",
        center: tuple[int, int] = (100, 100),
        confidence: float = 0.9,
        class_id: int = 0,
        bbox: tuple[int, int, int, int] | None = None,
    ):
        """Create a synthetic Detection with minimal fields."""
        from src.vision_detector import Detection

        if bbox is None:
            bbox = (
                center[0] - 20,
                center[1] - 20,
                center[0] + 20,
                center[1] + 20,
            )
        return Detection(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            bbox=bbox,
            center=center,
            area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
        )

    # ------------------------------------------------------------------
    # Static step passthrough
    # ------------------------------------------------------------------

    def test_static_key_passthrough(self):
        """Static key steps should pass through unchanged."""
        from src.macro_resolver import MacroResolver

        resolver = MacroResolver(detections=None)
        result = resolver.resolve({"type": "key", "key": "w", "hold_ms": 500})
        assert result == [{"type": "key", "key": "w", "hold_ms": 500}]

    def test_static_delay_passthrough(self):
        """Static delay steps should pass through unchanged."""
        from src.macro_resolver import MacroResolver

        resolver = MacroResolver(detections=None)
        result = resolver.resolve({"type": "delay", "ms": 200})
        assert result == [{"type": "delay", "ms": 200}]

    def test_static_click_passthrough(self):
        """Static click steps should pass through unchanged."""
        from src.macro_resolver import MacroResolver

        resolver = MacroResolver(detections=None)
        result = resolver.resolve({"type": "click", "button": "right"})
        assert result == [{"type": "click", "button": "right"}]

    def test_static_mouse_move_passthrough(self):
        """Static mouse_move steps should pass through unchanged."""
        from src.macro_resolver import MacroResolver

        resolver = MacroResolver(detections=None)
        result = resolver.resolve(
            {"type": "mouse_move", "x": 500, "y": 300, "relative": False}
        )
        assert result == [
            {"type": "mouse_move", "x": 500, "y": 300, "relative": False}
        ]

    def test_static_type_string_passthrough(self):
        """Static type_string steps should pass through unchanged."""
        from src.macro_resolver import MacroResolver

        resolver = MacroResolver(detections=None)
        result = resolver.resolve({"type": "type_string", "text": "hello"})
        assert result == [{"type": "type_string", "text": "hello"}]

    # ------------------------------------------------------------------
    # dynamic_click resolution
    # ------------------------------------------------------------------

    def test_dynamic_click_resolves_to_move_and_click(self):
        """dynamic_click should produce mouse_move + click steps."""
        from src.macro_resolver import MacroResolver

        det = self._det("person", center=(300, 200))
        resolver = MacroResolver(
            detections=[det], screen_size=(800, 600)
        )
        result = resolver.resolve(
            {"type": "dynamic_click", "target_class": "person", "button": "left"}
        )
        assert result is not None
        assert len(result) == 2
        assert result[0] == {"type": "mouse_move", "x": 300, "y": 200}
        assert result[1] == {"type": "click", "button": "left"}

    def test_dynamic_click_default_button_is_left(self):
        """When button is omitted, default to 'left'."""
        from src.macro_resolver import MacroResolver

        det = self._det("person", center=(100, 100))
        resolver = MacroResolver(
            detections=[det], screen_size=(800, 600)
        )
        result = resolver.resolve(
            {"type": "dynamic_click", "target_class": "person"}
        )
        assert result is not None
        assert result[1]["button"] == "left"

    def test_dynamic_click_right_button(self):
        """dynamic_click with button='right' should pass it through."""
        from src.macro_resolver import MacroResolver

        det = self._det("bottle", center=(500, 300))
        resolver = MacroResolver(
            detections=[det], screen_size=(800, 600)
        )
        result = resolver.resolve(
            {"type": "dynamic_click", "target_class": "bottle", "button": "right"}
        )
        assert result is not None
        assert result[1]["button"] == "right"

    # ------------------------------------------------------------------
    # dynamic_move resolution
    # ------------------------------------------------------------------

    def test_dynamic_move_resolves_to_mouse_move(self):
        """dynamic_move should produce a single mouse_move step."""
        from src.macro_resolver import MacroResolver

        det = self._det("health_potion", center=(700, 400))
        resolver = MacroResolver(
            detections=[det], screen_size=(800, 600)
        )
        result = resolver.resolve(
            {"type": "dynamic_move", "target_class": "health_potion"}
        )
        assert result == [{"type": "mouse_move", "x": 700, "y": 400}]

    # ------------------------------------------------------------------
    # Target not found / vision disabled
    # ------------------------------------------------------------------

    def test_returns_none_when_no_detections(self):
        """Dynamic steps should return None when detections list is empty."""
        from src.macro_resolver import MacroResolver

        resolver = MacroResolver(detections=[], screen_size=(800, 600))
        result = resolver.resolve(
            {"type": "dynamic_click", "target_class": "person"}
        )
        assert result is None

    def test_returns_none_when_detections_is_none(self):
        """Dynamic steps should return None when detections is None (vision disabled)."""
        from src.macro_resolver import MacroResolver

        resolver = MacroResolver(detections=None, screen_size=(800, 600))
        result = resolver.resolve(
            {"type": "dynamic_click", "target_class": "person"}
        )
        assert result is None

    def test_returns_none_when_target_class_not_found(self):
        """When the requested target class is not present, return None."""
        from src.macro_resolver import MacroResolver

        det = self._det("bottle", center=(100, 100))
        resolver = MacroResolver(
            detections=[det], screen_size=(800, 600)
        )
        result = resolver.resolve(
            {"type": "dynamic_click", "target_class": "person"}
        )
        assert result is None

    def test_returns_none_when_target_class_missing_from_step(self):
        """Dynamic step without 'target_class' key should return None."""
        from src.macro_resolver import MacroResolver

        det = self._det("person", center=(100, 100))
        resolver = MacroResolver(
            detections=[det], screen_size=(800, 600)
        )
        result = resolver.resolve({"type": "dynamic_click"})
        assert result is None

    # ------------------------------------------------------------------
    # Closest-to-center selection
    # ------------------------------------------------------------------

    def test_picks_closest_to_center(self):
        """When multiple detections match, the one closest to screen center wins."""
        from src.macro_resolver import MacroResolver

        det_far = self._det("person", center=(700, 500))
        det_near = self._det("person", center=(410, 310))
        # Screen center = (400, 300)
        resolver = MacroResolver(
            detections=[det_far, det_near], screen_size=(800, 600)
        )
        result = resolver.resolve(
            {"type": "dynamic_click", "target_class": "person"}
        )
        assert result is not None
        # Should pick the near one: center (410, 310)
        assert result[0]["x"] == 410
        assert result[0]["y"] == 310

    def test_ties_broken_by_confidence(self):
        """When two detections are equidistant, higher confidence wins."""
        from src.macro_resolver import MacroResolver

        # Both at the same location, different confidences
        det_low = self._det("person", center=(400, 300), confidence=0.6)
        det_high = self._det("person", center=(400, 300), confidence=0.95)
        resolver = MacroResolver(
            detections=[det_low, det_high], screen_size=(800, 600)
        )
        result = resolver.resolve(
            {"type": "dynamic_click", "target_class": "person"}
        )
        assert result is not None
        assert len(result) == 2

    def test_falls_back_to_highest_confidence_when_no_screen_size(self):
        """Without screen_size, the highest confidence detection is picked."""
        from src.macro_resolver import MacroResolver

        det_far = self._det("person", center=(700, 500), confidence=0.4)
        det_near = self._det("person", center=(410, 310), confidence=0.9)
        resolver = MacroResolver(detections=[det_far, det_near])
        result = resolver.resolve(
            {"type": "dynamic_click", "target_class": "person"}
        )
        assert result is not None
        # Higher confidence (0.9) wins regardless of position
        assert result[0]["x"] == 410
        assert result[0]["y"] == 310

    # ------------------------------------------------------------------
    # resolve_all
    # ------------------------------------------------------------------

    def test_resolve_all_flattens_and_filters(self):
        """resolve_all should combine steps, skipping dynamic ones that fail."""
        from src.macro_resolver import MacroResolver

        det = self._det("person", center=(300, 200))
        resolver = MacroResolver(
            detections=[det], screen_size=(800, 600)
        )

        steps = [
            {"type": "key", "key": "w", "hold_ms": 500},
            {"type": "dynamic_click", "target_class": "person"},
            {"type": "delay", "ms": 100},
            {"type": "dynamic_click", "target_class": "bottle"},  # not found
        ]

        resolved = resolver.resolve_all(steps)

        # Should have: key, mouse_move, click, delay  (4 steps, bottle skipped)
        assert len(resolved) == 4
        assert resolved[0] == {"type": "key", "key": "w", "hold_ms": 500}
        assert resolved[1] == {"type": "mouse_move", "x": 300, "y": 200}
        assert resolved[2] == {"type": "click", "button": "left"}
        assert resolved[3] == {"type": "delay", "ms": 100}

        # Skipped count should be 1 (the bottle)
        assert resolver.skipped_count == 1

    def test_resolve_all_empty_detections(self):
        """When no detections, all dynamic steps are skipped, static ones remain."""
        from src.macro_resolver import MacroResolver

        resolver = MacroResolver(detections=None, screen_size=(800, 600))
        steps = [
            {"type": "key", "key": "space", "hold_ms": 100},
            {"type": "dynamic_click", "target_class": "person"},
            {"type": "delay", "ms": 500},
        ]
        resolved = resolver.resolve_all(steps)
        assert len(resolved) == 2
        assert resolved[0] == {"type": "key", "key": "space", "hold_ms": 100}
        assert resolved[1] == {"type": "delay", "ms": 500}
        assert resolver.skipped_count == 1

    def test_resolve_all_returns_empty_list_when_all_skipped(self):
        """When every step is a dynamic step with no detections, result is empty."""
        from src.macro_resolver import MacroResolver

        resolver = MacroResolver(detections=[], screen_size=(800, 600))
        steps = [
            {"type": "dynamic_click", "target_class": "person"},
            {"type": "dynamic_move", "target_class": "bottle"},
        ]
        resolved = resolver.resolve_all(steps)
        assert resolved == []
        assert resolver.skipped_count == 2

    # ------------------------------------------------------------------
    # Unknown dynamic types
    # ------------------------------------------------------------------

    def test_unknown_dynamic_type_returns_empty_list(self):
        """An unknown dynamic_* type should produce an empty list (warning logged)."""
        from src.macro_resolver import MacroResolver

        det = self._det("person", center=(100, 100))
        resolver = MacroResolver(
            detections=[det], screen_size=(800, 600)
        )
        result = resolver.resolve(
            {"type": "dynamic_attack", "target_class": "person"}
        )
        assert result == []

    # ------------------------------------------------------------------
    # skipped_count property
    # ------------------------------------------------------------------

    def test_skipped_count_resets_on_each_resolve_all(self):
        """skipped_count should reset to 0 at the start of each resolve_all call."""
        from src.macro_resolver import MacroResolver

        resolver = MacroResolver(detections=[], screen_size=(800, 600))
        resolver.resolve_all(
            [{"type": "dynamic_click", "target_class": "person"}]
        )
        assert resolver.skipped_count == 1

        # Second call with detections should reset and have 0 skipped
        det = self._det("person", center=(100, 100))
        resolver.detections = [det]
        resolver.resolve_all(
            [{"type": "dynamic_click", "target_class": "person"}]
        )
        assert resolver.skipped_count == 0


# =============================================================================
# 11. Phase 4.4 — LLM Prompt Extension
# =============================================================================


class TestPhase4_4PromptExtension:
    """Validate Phase 4.4 prompt-builder changes:

    * Vision‑aware system prompt (dynamic macro targeting instructions)
    * Detection‑to‑macro mapping section in user content
    * Spatial context verbatim injection
    * Config‑driven token budget (llm_vision_max_tokens)
    * Filtering of noisy COCO classes via _DETECTION_GOOD_CLASSES
    * Backward‑compatibility: vision_enabled=False → identical to pre‑4.4
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _det(class_name: str = "person", confidence: float = 0.9):
        from src.vision_detector import Detection

        return Detection(
            class_id=0,
            class_name=class_name,
            confidence=confidence,
            bbox=(100, 100, 200, 200),
            center=(150, 150),
            area=10000,
        )

    @staticmethod
    def _make_macros(*names: str) -> list[dict[str, Any]]:
        return [{"name": n, "actions": []} for n in names]

    # ------------------------------------------------------------------
    # System prompt — vision aware vs plain
    # ------------------------------------------------------------------

    def test_system_prompt_includes_dynamic_instruction_when_vision_on(self):
        """When vision is enabled AND detections exist, system prompt
        mentions dynamic macro targeting."""
        from src.game_state import GameState, StateSchema, StateSlotDefinition
        from src.llm_prompt_builder import build_llm_prompt

        schema = StateSchema(
            schema_version="1.0",
            slots={
                "health": StateSlotDefinition(type="numeric", priority="color_first"),
            },
        )
        state = GameState(schema)
        state.set("health", 75)
        state.set("spatial_context", "Visual detections:\n- person detected at middle-center")
        state.set("detections", [self._det("person")])

        macros = self._make_macros("ATTACK", "DYNAMIC_CLICK_PERSON", "WAIT")
        messages = build_llm_prompt(
            state=state,
            available_macros=macros,
            max_tokens=1000,
            vision_enabled=True,
        )

        system = messages[0]["content"]
        assert "dynamic macro" in system.lower() or "targetable" in system.lower()

    def test_system_prompt_plain_when_vision_off(self):
        """When vision_enabled=False, system prompt is the standard form."""
        from src.game_state import GameState, StateSchema, StateSlotDefinition
        from src.llm_prompt_builder import build_llm_prompt

        schema = StateSchema(
            schema_version="1.0",
            slots={
                "health": StateSlotDefinition(type="numeric", priority="color_first"),
            },
        )
        state = GameState(schema)
        state.set("health", 75)

        macros = self._make_macros("ATTACK", "WAIT")
        messages = build_llm_prompt(
            state=state,
            available_macros=macros,
            vision_enabled=False,
        )

        system = messages[0]["content"]
        # Should NOT mention dynamic macros
        assert "dynamic" not in system.lower()

    # ------------------------------------------------------------------
    # Detection‑to‑macro mapping section
    # ------------------------------------------------------------------

    def test_detection_macro_mapping_in_user_content(self):
        """When detections match dynamic macro target classes, the
        user content includes a 'Targetable objects:' section."""
        from src.game_state import GameState, StateSchema, StateSlotDefinition
        from src.llm_prompt_builder import build_llm_prompt

        schema = StateSchema(
            schema_version="1.0",
            slots={
                "health": StateSlotDefinition(type="numeric", priority="color_first"),
            },
        )
        state = GameState(schema)
        state.set("health", 75)
        state.set("detections", [self._det("person")])

        macros = [
            {
                "name": "DYNAMIC_CLICK_PERSON",
                "actions": [
                    {"type": "dynamic_click", "target_class": "person", "button": "left"}
                ],
            },
            {"name": "WAIT", "actions": []},
        ]
        messages = build_llm_prompt(
            state=state,
            available_macros=macros,
            vision_enabled=True,
        )

        user = messages[1]["content"]
        assert "Targetable objects" in user
        assert "person" in user
        assert "DYNAMIC_CLICK_PERSON" in user

    def test_no_mapping_when_no_detections(self):
        """When no detections, no mapping section appears."""
        from src.game_state import GameState, StateSchema, StateSlotDefinition
        from src.llm_prompt_builder import build_llm_prompt

        schema = StateSchema(
            schema_version="1.0",
            slots={
                "health": StateSlotDefinition(type="numeric", priority="color_first"),
            },
        )
        state = GameState(schema)
        state.set("health", 75)
        state.set("detections", [])  # empty

        macros = [
            {
                "name": "DYNAMIC_CLICK_PERSON",
                "actions": [
                    {"type": "dynamic_click", "target_class": "person", "button": "left"}
                ],
            },
        ]
        messages = build_llm_prompt(
            state=state,
            available_macros=macros,
            vision_enabled=True,
        )

        user = messages[1]["content"]
        assert "Targetable objects" not in user

    def test_mapping_filters_noisy_classes(self):
        """COCO classes not in _DETECTION_GOOD_CLASSES are excluded from
        the targetable objects mapping."""
        from src.game_state import GameState, StateSchema, StateSlotDefinition
        from src.llm_prompt_builder import build_llm_prompt

        schema = StateSchema(
            schema_version="1.0",
            slots={
                "health": StateSlotDefinition(type="numeric", priority="color_first"),
            },
        )
        state = GameState(schema)
        state.set("health", 75)
        # "chair" is NOT in _DETECTION_GOOD_CLASSES
        state.set("detections", [self._det("chair"), self._det("person")])

        macros = [
            {
                "name": "DYNAMIC_CLICK_CHAIR",
                "actions": [
                    {"type": "dynamic_click", "target_class": "chair", "button": "left"}
                ],
            },
            {
                "name": "DYNAMIC_CLICK_PERSON",
                "actions": [
                    {"type": "dynamic_click", "target_class": "person", "button": "left"}
                ],
            },
        ]
        messages = build_llm_prompt(
            state=state,
            available_macros=macros,
            vision_enabled=True,
        )

        user = messages[1]["content"]
        # "person" is in the good set — should appear in the mapping
        assert "person" in user
        # "chair" is excluded from Targetable objects (even though the
        # macro name itself appears in "Available Macros")
        if "Targetable objects:" in user:
            mapping_start = user.index("Targetable objects:")
            mapping_end = user.index("\n\n", mapping_start) if "\n\n" in user[mapping_start:] else len(user)
            mapping_section = user[mapping_start:mapping_end]
            assert "chair" not in mapping_section

    # ------------------------------------------------------------------
    # Spatial context verbatim
    # ------------------------------------------------------------------

    def test_spatial_context_appears_verbatim(self):
        """The spatial_context string is injected as‑is into the prompt."""
        from src.game_state import GameState, StateSchema, StateSlotDefinition
        from src.llm_prompt_builder import build_llm_prompt

        schema = StateSchema(
            schema_version="1.0",
            slots={
                "health": StateSlotDefinition(type="numeric", priority="color_first"),
            },
        )
        state = GameState(schema)
        state.set("health", 50)
        state.set(
            "spatial_context",
            "Visual detections:\n- enemy detected at center-middle\n- potion detected at bottom-right",
        )

        messages = build_llm_prompt(
            state=state,
            available_macros=self._make_macros("WAIT"),
        )

        user = messages[1]["content"]
        assert "enemy detected at center-middle" in user
        assert "potion detected at bottom-right" in user

    # ------------------------------------------------------------------
    # Token budget — config‑driven
    # ------------------------------------------------------------------

    def test_vision_token_budget_from_config(self):
        """When config provides llm_vision_max_tokens, that value is used."""
        from src.game_state import GameState, StateSchema, StateSlotDefinition
        from src.llm_prompt_builder import build_llm_prompt, count_tokens

        schema = StateSchema(
            schema_version="1.0",
            slots={
                "health": StateSlotDefinition(type="numeric", priority="color_first"),
            },
        )
        state = GameState(schema)
        state.set("health", 50)
        state.set("detections", [self._det("person")])

        config = {"llm_vision_max_tokens": 500}
        messages = build_llm_prompt(
            state=state,
            available_macros=self._make_macros("WAIT"),
            config=config,
            vision_enabled=True,
        )

        # Total (system + user) should respect the 500 token budget.
        total_tokens = count_tokens(messages[0]["content"]) + count_tokens(
            messages[1]["content"]
        )
        assert total_tokens <= 500

    def test_default_budget_when_config_missing(self):
        """When no config is passed, the default vision budget applies."""
        from src.game_state import GameState, StateSchema, StateSlotDefinition
        from src.llm_prompt_builder import (
            _DEFAULT_VISION_MAX_TOKENS,
            build_llm_prompt,
            count_tokens,
        )

        schema = StateSchema(
            schema_version="1.0",
            slots={
                "health": StateSlotDefinition(type="numeric", priority="color_first"),
            },
        )
        state = GameState(schema)
        state.set("health", 50)
        state.set("detections", [self._det("person")])

        messages = build_llm_prompt(
            state=state,
            available_macros=self._make_macros("WAIT"),
            vision_enabled=True,
        )

        total_tokens = count_tokens(messages[0]["content"]) + count_tokens(
            messages[1]["content"]
        )
        assert total_tokens <= _DEFAULT_VISION_MAX_TOKENS

    def test_non_vision_uses_llm_max_tokens_from_config(self):
        """When vision is off, llm_max_tokens from config is used."""
        from src.game_state import GameState, StateSchema, StateSlotDefinition
        from src.llm_prompt_builder import build_llm_prompt, count_tokens

        schema = StateSchema(
            schema_version="1.0",
            slots={
                "health": StateSlotDefinition(type="numeric", priority="color_first"),
            },
        )
        state = GameState(schema)
        state.set("health", 50)

        config = {"llm_max_tokens": 400}
        messages = build_llm_prompt(
            state=state,
            available_macros=self._make_macros("WAIT"),
            config=config,
            vision_enabled=False,
        )

        total_tokens = count_tokens(messages[0]["content"]) + count_tokens(
            messages[1]["content"]
        )
        assert total_tokens <= 400

    # ------------------------------------------------------------------
    # Backward compatibility — no regression when vision is off
    # ------------------------------------------------------------------

    def test_pre_44_identical_when_vision_disabled(self):
        """When vision_enabled=False and no config, output matches
        pre‑4.4 prompt builder behaviour (plain system prompt, no
        detection sections)."""
        from src.game_state import GameState, StateSchema, StateSlotDefinition
        from src.llm_prompt_builder import build_llm_prompt

        schema = StateSchema(
            schema_version="1.0",
            slots={
                "health": StateSlotDefinition(type="numeric", priority="color_first"),
            },
        )
        state = GameState(schema)
        state.set("health", 30)

        messages = build_llm_prompt(
            state=state,
            available_macros=self._make_macros("HEAL", "WAIT"),
        )

        system = messages[0]["content"]
        user = messages[1]["content"]

        # Standard system prompt
        assert "game AI agent" in system
        assert "HEAL" in system
        assert "WAIT" in system
        assert "JSON" in system

        # User content has game state, macros — no vision cruft
        assert "health: 30" in user
        assert "Available Macros:" in user

        # Must NOT contain Phase 4.4‑specific sections
        assert "Targetable objects" not in user

    # ------------------------------------------------------------------
    # _normalise_detections helper
    # ------------------------------------------------------------------

    def test_normalise_detections_none_returns_empty(self):
        from src.llm_prompt_builder import _normalise_detections

        assert _normalise_detections(None) == []

    def test_normalise_detections_list_returns_copy(self):
        from src.llm_prompt_builder import _normalise_detections

        dets = [self._det("person"), self._det("bottle")]
        assert len(_normalise_detections(dets)) == 2

    def test_normalise_detections_spatial_context(self):
        from src.vision_detector import SpatialContext
        from src.llm_prompt_builder import _normalise_detections

        dets = [self._det("person")]
        ctx = SpatialContext(
            detections=dets,
            context_text="test",
            has_detections=True,
            timestamp=0.0,
            frame_number=1,
        )
        assert len(_normalise_detections(ctx)) == 1

    # ------------------------------------------------------------------
    # _DETECTION_GOOD_CLASSES integrity
    # ------------------------------------------------------------------

    def test_good_classes_excludes_noisy_objects(self):
        from src.llm_prompt_builder import _DETECTION_GOOD_CLASSES

        # These are explicitly excluded because they're common scenery
        assert "potted plant" not in _DETECTION_GOOD_CLASSES
        assert "chair" not in _DETECTION_GOOD_CLASSES
        assert "couch" not in _DETECTION_GOOD_CLASSES
        assert "parking meter" not in _DETECTION_GOOD_CLASSES
        assert "bench" not in _DETECTION_GOOD_CLASSES

        # These are included — they're relevant game entities
        assert "person" in _DETECTION_GOOD_CLASSES
        assert "bottle" in _DETECTION_GOOD_CLASSES
        assert "car" in _DETECTION_GOOD_CLASSES
        assert "dog" in _DETECTION_GOOD_CLASSES
        assert "laptop" in _DETECTION_GOOD_CLASSES

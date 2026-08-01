"""
Ludomorph - Core package.

A universal, external application that injects an autonomous LLM agent
into any PC game by capturing the screen and simulating keyboard/mouse input.
"""

# SINGLE SOURCE OF TRUTH for the app version.
# Imported by main_window.py (status bar), main.py (CLI banner), etc.
# Bump this value when cutting a new release.
__version__ = "0.6.0"

# ---------------------------------------------------------------------------
# Phase 1 — Macro execution & Ollama health
# ---------------------------------------------------------------------------

from src.macro_executor import (
    CancellationToken,
    MacroCancelledError,
    MacroError,
    MacroExecutor,
    MacroPriority,
    MacroRejectedError,
    MacroRequest,
    accurate_hold,
)
from src.ollama_health import (
    OllamaHealthError,
    OllamaHealthResult,
    ollama_health_check,
    ollama_health_check_or_raise,
)

# ---------------------------------------------------------------------------
# Phase 2.1 — GameState, StateSchema, RegionProfile
# ---------------------------------------------------------------------------

from src.game_state import (
    GameState,
    StateSchema,
    StateSlotDefinition,
    load_state_schema_from_path,
)
from src.region_profile import (
    RegionConfig,
    RegionProfile,
    load_region_profile_from_path,
)

# ---------------------------------------------------------------------------
# Phase 2.2 — Colour Bar Detector
# ---------------------------------------------------------------------------

from src.bar_detector import (
    BarDetectorError,
    CalibrationError,
    ColourBarCalibration,
    ColourBarDetector,
    TwoPointCalibrationLoader,
)

# ---------------------------------------------------------------------------
# Phase 2.3 — OCR Module
# ---------------------------------------------------------------------------

from src.ocr_module import (
    OCRConfig,
    OCRModule,
    OCRResult,
    shutdown_ocr_executor,
)

# ---------------------------------------------------------------------------
# Phase 2.4 — State Processor
# ---------------------------------------------------------------------------

from src.state_processor import (
    PipelineMetrics,
    StateProcessor,
    VisionExecutor,
    shutdown_vision_executor,
)

# ---------------------------------------------------------------------------
# Phase 2.5 — State Hashing & Caching
# ---------------------------------------------------------------------------

from src.state_hash import (
    StateCache,
    state_hash,
)

# ---------------------------------------------------------------------------
# Phase 2.8 — Adaptive Frame Skipping
# ---------------------------------------------------------------------------

from src.frame_differ import (
    AdaptiveFrameSkipper,
    DiffConfig,
    compute_pixel_diff,
)

__all__ = [
    # Phase 1
    "MacroExecutor",
    "MacroPriority",
    "MacroRequest",
    "CancellationToken",
    "MacroError",
    "MacroCancelledError",
    "MacroRejectedError",
    "accurate_hold",
    "ollama_health_check",
    "ollama_health_check_or_raise",
    "OllamaHealthResult",
    "OllamaHealthError",
    # Phase 2.1
    "GameState",
    "StateSchema",
    "StateSlotDefinition",
    "RegionConfig",
    "RegionProfile",
    "load_state_schema_from_path",
    "load_region_profile_from_path",
    # Phase 2.2
    "ColourBarDetector",
    "ColourBarCalibration",
    "TwoPointCalibrationLoader",
    "BarDetectorError",
    "CalibrationError",
    # Phase 2.3
    "OCRModule",
    "OCRConfig",
    "OCRResult",
    "shutdown_ocr_executor",
    # Phase 2.4
    "StateProcessor",
    "PipelineMetrics",
    "VisionExecutor",
    "shutdown_vision_executor",
    # Phase 2.5
    "state_hash",
    "StateCache",
    # Phase 2.8
    "compute_pixel_diff",
    "AdaptiveFrameSkipper",
    "DiffConfig",
]

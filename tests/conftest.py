"""
Shared pytest fixtures for AI Game Master integration tests.

Provides pre‑built config dicts, schemas, region profiles, macros, and
OCR module stubs so that individual test modules can focus on assertions
rather than boilerplate setup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
_TEST_PROFILE_DIR = _TESTS_DIR / "test_profile"
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _ensure_in_sys_path() -> None:
    """Make sure the project root is on sys.path for ``src`` imports."""
    import sys

    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_in_sys_path()

# ---------------------------------------------------------------------------
# Global config fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def global_config() -> dict[str, Any]:
    """Return a representative global config dict (mirrors config/config.json)."""
    return {
        "ollama_url": "http://localhost:11434/v1",
        "ollama_model": "phi3.5:3.8b-mini-instruct-q4_K_M",
        "mcp_url": "http://localhost:8000",
        "memory_max_events": 10000,
        "enable_summarization": True,
        "action_cooldown_ms": 150,
        "diff": {
            "downsample_width": 640,
            "downsample_height": 360,
            "method": "absdiff_mean",
            "adaptive": {
                "enabled": True,
                "window_size": 30,
                "static_percentile": 70,
                "high_motion_percentile": 85,
            },
            "perceptual_hash_fallback": {
                "enabled": False,
                "method": "opencv_dhash",
                "hamming_threshold": 6,
            },
        },
        "state_cache_ttl_seconds": 0.3,
        "ocr_cache_ttl_seconds": 2.0,
        "auto_focus_window": True,
        "log_level": "INFO",
        "input_backend": "auto",
    }


# ---------------------------------------------------------------------------
# State schema fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def state_schema_data() -> dict[str, Any]:
    """Load state_schema.json from the test profile directory."""
    path = _TEST_PROFILE_DIR / "state_schema.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    # Fallback to bundled config
    fallback = _CONFIG_DIR / "state_schema.json"
    return json.loads(fallback.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def state_schema(state_schema_data: dict[str, Any]):
    """Build a validated StateSchema from the test profile data."""
    from src.game_state import StateSchema

    return StateSchema.from_dict(state_schema_data)


# ---------------------------------------------------------------------------
# Region profile fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def region_profile_data() -> dict[str, Any]:
    """Load regions.json from the test profile directory."""
    path = _TEST_PROFILE_DIR / "regions.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    fallback = _CONFIG_DIR / "regions.json"
    return json.loads(fallback.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def region_profile(region_profile_data: dict[str, Any]):
    """Build a validated RegionProfile from the test profile data."""
    from src.region_profile import RegionProfile

    return RegionProfile.from_dict(region_profile_data)


# ---------------------------------------------------------------------------
# Profile macros fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def profile_macros() -> list[dict[str, Any]]:
    """Load macros from the test profile directory."""
    path = _TEST_PROFILE_DIR / "macros.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw.get("macros", [])
        if isinstance(raw, list):
            return raw
    # Fallback
    fallback = _CONFIG_DIR / "macros.json"
    raw = json.loads(fallback.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw.get("macros", [])
    if isinstance(raw, list):
        return raw
    return []


# ---------------------------------------------------------------------------
# OCR module fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def ocr_module() -> Any:
    """Return a configured OCRModule (uses real Tesseract if available)."""
    from src.ocr_module import OCRConfig, OCRModule

    config = OCRConfig(cache_ttl_seconds=2.0)
    return OCRModule(config)


# ---------------------------------------------------------------------------
# StateProcessor fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def state_processor(
    region_profile: Any,
    ocr_module: Any,
    state_schema: Any,
    global_config: dict[str, Any],
) -> Any:
    """Build a fully wired StateProcessor with the test profile."""
    from src.state_processor import StateProcessor

    return StateProcessor(
        profile=region_profile,
        ocr_module=ocr_module,
        schema=state_schema,
        vision_processor=None,
        cache_ttl=global_config.get("state_cache_ttl_seconds", 0.3),
    )


# ---------------------------------------------------------------------------
# Simulated frame helper (re‑exports frame_generator)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def frame_generator():
    """Return the frame_generator module for convenience."""
    from tests.frame_generator import create_simulated_frame

    return create_simulated_frame
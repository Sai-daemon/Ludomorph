"""
Configuration Manager - Atomic JSON read/write and profile load/save.

Handles:
- Global config at ~/.gameai/config.json
- Per-game profiles under ~/.gameai/profiles/<name>/
  - settings.json (per-profile overrides of global config)
  - macros.json (macro definitions)
  - state_schema.json (Phase 2.1 — state slot definitions)
  - regions.json (Phase 2.1 — screen region definitions with role mapping)
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import appdirs

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default schema for config.json (matching spec Section 9.1)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
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

DEFAULT_MACROS: list[dict[str, Any]] = []

# ---------------------------------------------------------------------------
# Defaults for Phase 2.1 profile files
# ---------------------------------------------------------------------------

DEFAULT_STATE_SCHEMA: dict[str, Any] = {
    "schema_version": "1.0.0",
    "slots": {
        "health": {"type": "numeric", "priority": "color_first"},
        "health_text": {"type": "text", "priority": "ocr_first"},
        "mana": {"type": "numeric", "priority": "color_first"},
        "mana_text": {"type": "text", "priority": "ocr_first"},
        "location": {"type": "text", "priority": "ocr_first"},
        "inventory": {"type": "text", "priority": "ocr_first"},
        "objective": {"type": "text", "priority": "ocr_first"},
        "enemy_present": {"type": "boolean", "priority": "ocr_first"},
    },
}

DEFAULT_REGIONS: dict[str, Any] = {
    "version": "1.0.0",
    "regions": [],
}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

CONFIG_DIR = Path(appdirs.user_config_dir("gameai", appauthor=False))
PROFILES_DIR = CONFIG_DIR / "profiles"


def _ensure_config_dir() -> None:
    """Create ~/.gameai and ~/.gameai/profiles if they don't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def global_config_path() -> Path:
    return CONFIG_DIR / "config.json"


def profile_dir(name: str) -> Path:
    return PROFILES_DIR / name


def profile_settings_path(name: str) -> Path:
    return profile_dir(name) / "settings.json"


def profile_macros_path(name: str) -> Path:
    return profile_dir(name) / "macros.json"


def profile_state_schema_path(name: str) -> Path:
    return profile_dir(name) / "state_schema.json"


def profile_regions_path(name: str) -> Path:
    return profile_dir(name) / "regions.json"


# ---------------------------------------------------------------------------
# Atomic JSON helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Any) -> None:
    """
    Write data atomically to a JSON file.

    Writes to a temp file in the same directory, then renames on top of
    the target. This avoids partial writes / corruption on power loss.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_json(path: Path, default: Any = None) -> Any:
    """
    Safely load a JSON file, returning *default* if the file is missing
    or unparseable.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.debug(f"JSON file not found (using default): {path}")
        return default
    except json.JSONDecodeError as exc:
        logger.warning(f"Corrupt JSON in {path}: {exc}. Using default.")
        return default


# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

def load_global_config() -> dict[str, Any]:
    """
    Load global configuration from ~/.gameai/config.json.

    On first run the file won't exist - we return DEFAULT_CONFIG and
    atomically save it so the user has a starting point to edit.
    """
    _ensure_config_dir()
    path = global_config_path()
    data = load_json(path)

    if data is None:
        logger.info("No config.json found; writing defaults.")
        _atomic_write_json(path, DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)

    # Merge with defaults so new keys added in future versions are present
    merged = dict(DEFAULT_CONFIG)
    merged.update(data)
    return merged


def save_global_config(config: dict[str, Any]) -> None:
    """Atomically persist global config."""
    _ensure_config_dir()
    _atomic_write_json(global_config_path(), config)
    logger.info("Global config saved.")


# ---------------------------------------------------------------------------
# Profile management
# ---------------------------------------------------------------------------

def list_profiles() -> list[str]:
    """
    Return a sorted list of available profile names (directory names
    inside ~/.gameai/profiles).
    """
    _ensure_config_dir()
    if not PROFILES_DIR.exists():
        return []
    return sorted(
        d.name
        for d in PROFILES_DIR.iterdir()
        if d.is_dir() and (d / "settings.json").exists()
    )


def load_profile(name: str) -> dict[str, Any] | None:
    """
    Load a per-game profile's settings.json.

    Returns None if the profile directory or file doesn't exist.
    """
    path = profile_settings_path(name)
    return load_json(path)


def save_profile(name: str, settings: dict[str, Any]) -> None:
    """Atomically save per-game profile settings."""
    profile_dir(name).mkdir(parents=True, exist_ok=True)
    _atomic_write_json(profile_settings_path(name), settings)
    logger.info(f"Profile '{name}' settings saved.")


def delete_profile(name: str) -> None:
    """Delete a profile directory and all its contents."""
    import shutil

    pdir = profile_dir(name)
    if pdir.exists():
        shutil.rmtree(pdir)
        logger.info(f"Profile '{name}' deleted.")


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------

def load_macros(name: str) -> list[dict[str, Any]]:
    """
    Load macros for a profile.

    Returns DEFAULT_MACROS (empty list) if the file doesn't exist.
    """
    path = profile_macros_path(name)
    data = load_json(path, DEFAULT_MACROS)
    if data is None:
        data = DEFAULT_MACROS
    if not isinstance(data, list):
        logger.warning(f"macros.json for profile '{name}' is not a list; resetting.")
        data = DEFAULT_MACROS
    return data


def save_macros(name: str, macros: list[dict[str, Any]]) -> None:
    """Atomically save macros for a profile."""
    profile_dir(name).mkdir(parents=True, exist_ok=True)
    _atomic_write_json(profile_macros_path(name), macros)
    logger.info(f"Profile '{name}' macros saved.")


# ---------------------------------------------------------------------------
# State Schema (Phase 2.1)
# ---------------------------------------------------------------------------

def load_state_schema(name: str) -> dict[str, Any]:
    """
    Load the state schema for a profile.

    Returns DEFAULT_STATE_SCHEMA if the file doesn't exist or is unparseable.
    """
    path = profile_state_schema_path(name)
    data = load_json(path)
    if data is None:
        logger.info(f"No state_schema.json for profile '{name}'; using defaults.")
        return dict(DEFAULT_STATE_SCHEMA)
    if not isinstance(data, dict):
        logger.warning(
            f"state_schema.json for profile '{name}' is not a dict; using defaults."
        )
        return dict(DEFAULT_STATE_SCHEMA)
    return data


def save_state_schema(name: str, schema: dict[str, Any]) -> None:
    """Atomically save the state schema for a profile."""
    profile_dir(name).mkdir(parents=True, exist_ok=True)
    _atomic_write_json(profile_state_schema_path(name), schema)
    logger.info(f"Profile '{name}' state schema saved.")


# ---------------------------------------------------------------------------
# Regions (Phase 2.1)
# ---------------------------------------------------------------------------

def load_regions(name: str) -> dict[str, Any]:
    """
    Load the region definitions for a profile.

    Returns DEFAULT_REGIONS (empty region list) if the file doesn't exist
    or is unparseable.
    """
    path = profile_regions_path(name)
    data = load_json(path)
    if data is None:
        logger.info(f"No regions.json for profile '{name}'; using defaults.")
        return dict(DEFAULT_REGIONS)
    if not isinstance(data, dict):
        logger.warning(
            f"regions.json for profile '{name}' is not a dict; using defaults."
        )
        return dict(DEFAULT_REGIONS)
    return data


def save_regions(name: str, regions: dict[str, Any]) -> None:
    """Atomically save the region definitions for a profile."""
    profile_dir(name).mkdir(parents=True, exist_ok=True)
    _atomic_write_json(profile_regions_path(name), regions)
    logger.info(f"Profile '{name}' regions saved.")
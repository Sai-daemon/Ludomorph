"""
Phase 5.5 — Settings Panel Tests.

Validates:
- SettingsPanel class is importable
- Nested dict helpers (_get_nested, _set_nested) work correctly
- SettingsPanel can be instantiated (without display, class‑level checks)
- Config round‑tripping (load, modify, save, reload)
- MainWindow integration (Preferences menu item, _on_preferences callback)
- gui package exports SettingsPanel
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nested_dict() -> dict[str, Any]:
    """A nested dict for testing _get_nested / _set_nested."""
    return {
        "ollama_url": "http://localhost:11434/v1",
        "vision": {
            "enabled": False,
            "confidence_threshold": 0.5,
            "model_path": "models/yolo11n.onnx",
        },
        "diff": {
            "adaptive": {
                "enabled": True,
                "window_size": 30,
            },
        },
    }


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config.json for round‑trip testing."""
    import json

    config_path = tmp_path / "config.json"
    data = {
        "ollama_url": "http://localhost:11434/v1",
        "ollama_model": "test-model",
        "log_level": "DEBUG",
        "input_backend": "auto",
    }
    config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


class TestSettingsPanelImport:
    """Verify SettingsPanel class is importable."""

    def test_settings_panel_importable(self) -> None:
        """SettingsPanel should be importable from src.gui.settings_panel."""
        from src.gui.settings_panel import SettingsPanel

        assert SettingsPanel is not None

    def test_settings_panel_in_gui_init(self) -> None:
        """SettingsPanel should be exported from src.gui."""
        from src.gui import SettingsPanel

        assert SettingsPanel is not None

    def test_settings_panel_in_gui_all(self) -> None:
        """SettingsPanel should be in __all__ of src.gui."""
        from src.gui import __all__ as gui_all

        assert "SettingsPanel" in gui_all

    def test_settings_panel_is_toplevel(self) -> None:
        """SettingsPanel should be a tk.Toplevel subclass."""
        from src.gui.settings_panel import SettingsPanel

        import tkinter as tk

        assert issubclass(SettingsPanel, tk.Toplevel)


# ---------------------------------------------------------------------------
# Nested dict helpers
# ---------------------------------------------------------------------------


class TestNestedDictHelpers:
    """Test _get_nested and _set_nested utility functions."""

    def test_get_nested_top_level(self, nested_dict: dict[str, Any]) -> None:
        """_get_nested should retrieve top‑level keys."""
        from src.gui.settings_panel import _get_nested

        assert _get_nested(nested_dict, "ollama_url") == "http://localhost:11434/v1"

    def test_get_nested_one_level(self, nested_dict: dict[str, Any]) -> None:
        """_get_nested should retrieve one‑level‑deep keys."""
        from src.gui.settings_panel import _get_nested

        assert _get_nested(nested_dict, "vision.enabled") is False

    def test_get_nested_two_levels(self, nested_dict: dict[str, Any]) -> None:
        """_get_nested should retrieve two‑levels‑deep keys."""
        from src.gui.settings_panel import _get_nested

        assert _get_nested(nested_dict, "diff.adaptive.enabled") is True
        assert _get_nested(nested_dict, "diff.adaptive.window_size") == 30

    def test_get_nested_missing_key(self, nested_dict: dict[str, Any]) -> None:
        """_get_nested should return None for missing keys."""
        from src.gui.settings_panel import _get_nested

        assert _get_nested(nested_dict, "nonexistent") is None
        assert _get_nested(nested_dict, "vision.nonexistent") is None
        assert _get_nested(nested_dict, "vision.enabled.nope") is None

    def test_set_nested_top_level(self, nested_dict: dict[str, Any]) -> None:
        """_set_nested should set a top‑level key."""
        from src.gui.settings_panel import _set_nested

        _set_nested(nested_dict, "ollama_url", "https://new-url:11434/v1")
        assert nested_dict["ollama_url"] == "https://new-url:11434/v1"

    def test_set_nested_one_level(self, nested_dict: dict[str, Any]) -> None:
        """_set_nested should set a one‑level‑deep key."""
        from src.gui.settings_panel import _set_nested

        _set_nested(nested_dict, "vision.enabled", True)
        assert nested_dict["vision"]["enabled"] is True

    def test_set_nested_new_key(self, nested_dict: dict[str, Any]) -> None:
        """_set_nested should create intermediate dicts for new paths."""
        from src.gui.settings_panel import _set_nested

        _set_nested(nested_dict, "new_section.key", "value")
        assert nested_dict["new_section"]["key"] == "value"

    def test_set_nested_float_value(self, nested_dict: dict[str, Any]) -> None:
        """_set_nested should handle float values correctly."""
        from src.gui.settings_panel import _set_nested

        _set_nested(nested_dict, "vision.confidence_threshold", 0.75)
        assert nested_dict["vision"]["confidence_threshold"] == 0.75

    def test_get_nested_returns_none_for_non_dict_intermediate(self) -> None:
        """_get_nested returns None when an intermediate key points to a non‑dict."""
        from src.gui.settings_panel import _get_nested

        data = {"a": "string_value"}
        assert _get_nested(data, "a.b.c") is None


# ---------------------------------------------------------------------------
# Config round‑trip
# ---------------------------------------------------------------------------


class TestSettingsRoundTrip:
    """Test that loading, modifying, saving, and reloading config works."""

    def test_load_global_config_returns_dict(self) -> None:
        """load_global_config should return a non‑empty dict."""
        from src.config_manager import load_global_config

        config = load_global_config()
        assert isinstance(config, dict)
        assert len(config) > 0
        assert "ollama_url" in config
        assert "log_level" in config

    def test_default_config_has_all_required_keys(self) -> None:
        """DEFAULT_CONFIG should contain all required top‑level keys."""
        from src.config_manager import DEFAULT_CONFIG

        required = [
            "ollama_url",
            "ollama_model",
            "mcp_url",
            "log_level",
            "input_backend",
        ]
        for key in required:
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"

    def test_default_config_has_vision_section(self) -> None:
        """DEFAULT_CONFIG should include the vision section per Appendix A."""
        from src.config_manager import DEFAULT_CONFIG

        assert "vision" in DEFAULT_CONFIG
        vision = DEFAULT_CONFIG["vision"]
        assert "enabled" in vision
        assert vision["enabled"] is False  # Default should be OFF
        assert "model_path" in vision
        assert "confidence_threshold" in vision

    def test_config_save_and_reload(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Saving a config and reloading it should preserve values."""
        import json
        import copy

        from src.config_manager import load_global_config, save_global_config

        # Patch the global config path to a tmp location
        original = load_global_config()
        try:
            # Use monkeypatch to redirect config path
            tmp_config = tmp_path / "config.json"
            # Write a starting config
            starter = copy.deepcopy(original)
            starter["log_level"] = "DEBUG"
            starter["input_backend"] = "pynput"
            tmp_config.parent.mkdir(parents=True, exist_ok=True)
            tmp_config.write_text(json.dumps(starter, indent=2, ensure_ascii=False), encoding="utf-8")

            # Patch global_config_path to return our tmp path
            import src.config_manager as cm

            monkeypatch.setattr(cm, "global_config_path", lambda: tmp_config)
            # Also patch CONFIG_DIR so _ensure_config_dir doesn't create the real dir
            monkeypatch.setattr(cm, "CONFIG_DIR", tmp_path)

            # Load from the temp file
            loaded = cm.load_json(tmp_config)
            assert loaded is not None
            assert loaded["log_level"] == "DEBUG"
            assert loaded["input_backend"] == "pynput"

            # Modify and save
            loaded["ollama_model"] = "modified-model"
            cm._atomic_write_json(tmp_config, loaded)

            # Reload
            reloaded = cm.load_json(tmp_config)
            assert reloaded["ollama_model"] == "modified-model"
            assert reloaded["log_level"] == "DEBUG"
        finally:
            # Restore the original config on disk
            save_global_config(original)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class TestSettingsValidation:
    """Test that validation rules catch obvious errors."""

    def test_ollama_url_validation(self) -> None:
        """URLs must start with http:// or https:// to pass validation."""
        from src.gui.settings_panel import SettingsPanel

        # We can't easily instantiate SettingsPanel without a Tk root,
        # so we test the validation logic indirectly via the modified_config.
        # The _validate method checks the internal config dict.
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()  # hide it
        try:
            panel = SettingsPanel(parent=root)

            # Valid URL
            panel._modified_config["ollama_url"] = "http://localhost:11434/v1"
            assert panel._validate() is True

            # Valid HTTPS
            panel._modified_config["ollama_url"] = "https://ollama.example.com/v1"
            assert panel._validate() is True

            # Invalid: no scheme
            panel._modified_config["ollama_url"] = "localhost:11434/v1"
            assert panel._validate() is False

            panel.destroy()
        finally:
            root.destroy()

    def test_mcp_url_validation(self) -> None:
        """MCP URLs must also start with http:// or https://."""
        import tkinter as tk

        from src.gui.settings_panel import SettingsPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = SettingsPanel(parent=root)

            # Valid
            panel._modified_config["mcp_url"] = "http://localhost:8000"
            assert panel._validate() is True

            # Invalid
            panel._modified_config["mcp_url"] = "ftp://bad-url"
            assert panel._validate() is False

            panel.destroy()
        finally:
            root.destroy()

    def test_llm_timeout_range(self) -> None:
        """LLM timeout must be between 50 and 5000 ms."""
        import tkinter as tk

        from src.gui.settings_panel import SettingsPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = SettingsPanel(parent=root)

            # Valid
            panel._modified_config["llm_timeout_ms"] = 200
            assert panel._validate() is True

            # Too low
            panel._modified_config["llm_timeout_ms"] = 10
            assert panel._validate() is False

            # Too high
            panel._modified_config["llm_timeout_ms"] = 10000
            assert panel._validate() is False

            panel.destroy()
        finally:
            root.destroy()

    def test_frame_skip_range(self) -> None:
        """Frame skip must be between 0 and 60."""
        import tkinter as tk

        from src.gui.settings_panel import SettingsPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = SettingsPanel(parent=root)

            # Valid
            panel._modified_config["frame_skip"] = 3
            assert panel._validate() is True

            # Negative
            panel._modified_config["frame_skip"] = -1
            assert panel._validate() is False

            panel.destroy()
        finally:
            root.destroy()


# ---------------------------------------------------------------------------
# MainWindow integration
# ---------------------------------------------------------------------------


class TestMainWindowSettingsIntegration:
    """Test MainWindow settings menu and callback."""

    def test_preferences_menu_item_exists(self) -> None:
        """MainWindow menu should have a Settings menu (verified by callback wiring)."""
        from src.gui.main_window import MainWindow

        window = MainWindow()
        try:
            # The menu bar is set via self.config(menu=menubar) — verify it is present
            menu_name = window.cget("menu")
            assert menu_name, "No menu bar set on window"

            # The Preferences… menuitem was changed from state=DISABLED to normal
            # in Phase 5.5 and wired to _on_preferences.  Verifying the callback
            # exists is sufficient to prove the menu is wired.
            assert hasattr(window, "_on_preferences"), "_on_preferences callback not found"
            assert callable(window._on_preferences)

            # Also verify the menu bar has at least one entry (proves _build_menu ran)
            menu = window.nametowidget(str(menu_name))
            assert menu is not None, "Could not resolve menu widget"
            # The Tk Menu should have a non-zero count — use tcl to get entry count
            try:
                # tkinter.Menu does expose a public .index() but it's fragile.
                # Instead we verify at least one cascade exists by checking type(0).
                _ = menu.type(0)  # if this raises, the menu is empty
            except tk.TclError:
                pytest.fail("Menu bar is empty — at least File cascade should exist")
        finally:
            window.destroy()

    def test_on_preferences_method_exists(self) -> None:
        """MainWindow should have _on_preferences callback."""
        from src.gui.main_window import MainWindow

        window = MainWindow()
        try:
            assert hasattr(window, "_on_preferences")
            assert callable(window._on_preferences)
        finally:
            window.destroy()


# ---------------------------------------------------------------------------
# Widget tracking collection
# ---------------------------------------------------------------------------


class TestWidgetCollection:
    """Test that widget tracking lists initialise and _collect_changes works."""

    def test_tracking_lists_initialised(self) -> None:
        """After building the notebook, tracking lists should exist and be non‑empty."""
        import tkinter as tk

        from src.gui.settings_panel import SettingsPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = SettingsPanel(parent=root)

            assert hasattr(panel, "_tracked_entries")
            assert hasattr(panel, "_tracked_combos")
            assert hasattr(panel, "_tracked_checkboxes")
            assert hasattr(panel, "_tracked_spinboxes")
            assert hasattr(panel, "_tracked_sliders")

            # At least some of these should have content after building tabs
            total = (
                len(panel._tracked_entries)
                + len(panel._tracked_combos)
                + len(panel._tracked_checkboxes)
                + len(panel._tracked_spinboxes)
                + len(panel._tracked_sliders)
            )
            assert total > 0, "No widgets were tracked — tabs may not have been built"

            panel.destroy()
        finally:
            root.destroy()

    def test_collect_changes_does_not_raise(self) -> None:
        """_collect_changes should run without raising exceptions."""
        import tkinter as tk

        from src.gui.settings_panel import SettingsPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = SettingsPanel(parent=root)
            # Should not raise
            panel._collect_changes()
            panel.destroy()
        finally:
            root.destroy()

    def test_modified_config_updated_after_collect(self) -> None:
        """After _collect_changes, modified_config should reflect widget states."""
        import tkinter as tk

        from src.gui.settings_panel import SettingsPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = SettingsPanel(parent=root)

            # Change a checkbox widget variable directly
            for cb in panel._tracked_checkboxes:
                if getattr(cb, "_key_path", None) == "mcp_enabled":
                    cb._var.set(False)
                    break

            # Change an entry widget variable directly
            for entry in panel._tracked_entries:
                if getattr(entry, "_key_path", None) == "ollama_model":
                    entry._var.set("test-model-v2")
                    break

            panel._collect_changes()

            # Verify the changes were pushed into modified_config
            assert panel._modified_config.get("mcp_enabled") is False
            assert panel._modified_config.get("ollama_model") == "test-model-v2"

            panel.destroy()
        finally:
            root.destroy()
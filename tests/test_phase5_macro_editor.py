"""
Phase 5.4 — Macro JSON Editor Tests.

Validates:
- MacroEditor class is importable
- Syntax highlighting tags are configured correctly
- _validate_macro_json helper catches malformed JSON
- MacroEditor can be instantiated (without display, class‑level checks)
- JSON round‑tripping preserves macro count
- MainWindow toolbar has the Edit Macros button
- gui package exports MacroEditor
"""

from __future__ import annotations

import json
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
def sample_macro() -> dict[str, Any]:
    """A valid sample macro object."""
    return {
        "name": "test_macro",
        "description": "A test macro for validation",
        "actions": [
            {"type": "key", "key": "w", "hold_ms": 500},
            {"type": "delay", "ms": 200},
            {"type": "mouse_move", "x": 100, "y": 200, "relative": False},
            {"type": "click", "button": "left"},
            {"type": "type_string", "text": "hello"},
        ],
    }


@pytest.fixture
def temp_macros_file(tmp_path: Path) -> Path:
    """Create a temporary macros.json for round‑trip testing."""
    macros_path = tmp_path / "macros.json"
    data = {
        "version": "1.0.0",
        "macros": [
            {
                "name": "move_forward",
                "description": "Press W for half a second",
                "actions": [{"type": "key", "key": "w", "hold_ms": 500}],
            },
            {
                "name": "jump",
                "description": "Quick spacebar tap",
                "actions": [{"type": "key", "key": "space", "hold_ms": 100}],
            },
        ],
    }
    macros_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return macros_path


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


class TestMacroEditorImport:
    """Verify MacroEditor class is importable."""

    def test_macro_editor_importable(self) -> None:
        """MacroEditor should be importable from src.gui.macro_editor."""
        from src.gui.macro_editor import MacroEditor

        assert MacroEditor is not None

    def test_macro_editor_in_gui_init(self) -> None:
        """MacroEditor should be exported from src.gui."""
        from src.gui import MacroEditor

        assert MacroEditor is not None

    def test_macro_editor_in_gui_all(self) -> None:
        """MacroEditor should be in __all__ of src.gui."""
        from src.gui import __all__ as gui_all

        assert "MacroEditor" in gui_all

    def test_macro_editor_is_toplevel(self) -> None:
        """MacroEditor should be a tk.Toplevel subclass."""
        from src.gui.macro_editor import MacroEditor

        import tkinter as tk
        assert issubclass(MacroEditor, tk.Toplevel)


# ---------------------------------------------------------------------------
# Syntax highlighting
# ---------------------------------------------------------------------------


class TestSyntaxHighlighting:
    """Test syntax highlighting tag configuration."""

    def test_highlight_tags_configured(self) -> None:
        """All expected highlight tags should be defined as constants."""
        from src.gui import macro_editor as me

        expected_tags = {"key", "string", "number", "bool_null", "bracket"}
        # These are tag names used via tag_configure
        assert hasattr(me, "_TAG_KEY")
        assert hasattr(me, "_TAG_STRING")
        assert hasattr(me, "_TAG_NUMBER")
        assert hasattr(me, "_TAG_BOOL_NULL")
        assert hasattr(me, "_TAG_BRACKET")

    def test_regex_compiled(self) -> None:
        """All syntax regexes should be compiled patterns."""
        from src.gui import macro_editor as me

        import re
        assert isinstance(me._RE_KEY, re.Pattern)
        assert isinstance(me._RE_STRING, re.Pattern)
        assert isinstance(me._RE_NUMBER, re.Pattern)
        assert isinstance(me._RE_BOOL_NULL, re.Pattern)
        assert isinstance(me._RE_BRACKET, re.Pattern)

    def test_number_regex_matches(self) -> None:
        """Number regex should match integers and floats."""
        from src.gui.macro_editor import _RE_NUMBER

        assert _RE_NUMBER.search("42") is not None
        assert _RE_NUMBER.search("3.14") is not None
        assert _RE_NUMBER.search("-100") is not None
        assert _RE_NUMBER.search("1e10") is not None
        assert _RE_NUMBER.search("hello") is None

    def test_bool_null_regex_matches(self) -> None:
        """Bool/null regex should match true, false, null."""
        from src.gui.macro_editor import _RE_BOOL_NULL

        assert _RE_BOOL_NULL.search("true") is not None
        assert _RE_BOOL_NULL.search("false") is not None
        assert _RE_BOOL_NULL.search("null") is not None
        assert _RE_BOOL_NULL.search("True") is None   # uppercase not valid JSON

    def test_string_regex_matches(self) -> None:
        """String regex should match quoted strings including escapes."""
        from src.gui.macro_editor import _RE_STRING

        assert _RE_STRING.search('"hello"') is not None
        assert _RE_STRING.search('"escaped \\" quote"') is not None
        assert _RE_STRING.search("not quoted") is None

    def test_key_regex_matches(self) -> None:
        """Key regex should match quoted strings followed by colon (group 1 captures key)."""
        from src.gui.macro_editor import _RE_KEY

        # Realistic indented JSON as produced by json.dumps(..., indent=2)
        text = '{\n  "name": "value",\n  "age": 42\n}'
        matches = _RE_KEY.findall(text)
        assert len(matches) == 2
        assert '"name"' in matches, f"matches: {matches}"
        assert '"age"' in matches, f"matches: {matches}"


# ---------------------------------------------------------------------------
# JSON validation
# ---------------------------------------------------------------------------


class TestValidateMacroJson:
    """Test the _validate_macro_json helper."""

    def test_valid_macro_passes(self, sample_macro: dict[str, Any]) -> None:
        """A valid macro should pass validation."""
        from src.gui.macro_editor import _validate_macro_json

        text = json.dumps(sample_macro)
        ok, err = _validate_macro_json(text)
        assert ok, f"Expected valid, got: {err}"
        assert err == ""

    def test_missing_name_fails(self) -> None:
        """Macro without 'name' should fail."""
        from src.gui.macro_editor import _validate_macro_json

        obj = {"actions": [{"type": "key", "key": "a", "hold_ms": 100}]}
        ok, err = _validate_macro_json(json.dumps(obj))
        assert not ok
        assert "name" in err.lower()

    def test_missing_actions_fails(self) -> None:
        """Macro without 'actions' should fail."""
        from src.gui.macro_editor import _validate_macro_json

        obj = {"name": "test"}
        ok, err = _validate_macro_json(json.dumps(obj))
        assert not ok
        assert "actions" in err.lower()

    def test_actions_not_list_fails(self) -> None:
        """Macro with 'actions' as non‑list should fail."""
        from src.gui.macro_editor import _validate_macro_json

        obj = {"name": "test", "actions": "not_a_list"}
        ok, err = _validate_macro_json(json.dumps(obj))
        assert not ok
        assert "actions" in err.lower()

    def test_action_missing_type_fails(self) -> None:
        """An action without 'type' should fail."""
        from src.gui.macro_editor import _validate_macro_json

        obj = {"name": "test", "actions": [{"key": "w"}]}
        ok, err = _validate_macro_json(json.dumps(obj))
        assert not ok
        assert "Action" in err

    def test_unknown_action_type_fails(self) -> None:
        """An action with unknown type should fail."""
        from src.gui.macro_editor import _validate_macro_json

        obj = {"name": "test", "actions": [{"type": "banana_split", "key": "w"}]}
        ok, err = _validate_macro_json(json.dumps(obj))
        assert not ok
        assert "unknown type" in err.lower()

    def test_invalid_json_fails(self) -> None:
        """Completely invalid JSON should fail."""
        from src.gui.macro_editor import _validate_macro_json

        ok, err = _validate_macro_json("{not valid json")
        assert not ok
        assert "Invalid JSON" in err

    def test_not_a_dict_fails(self) -> None:
        """Top‑level JSON that isn't an object should fail."""
        from src.gui.macro_editor import _validate_macro_json

        ok, err = _validate_macro_json("[1, 2, 3]")
        assert not ok
        assert "object" in err.lower() or "dictionary" in err.lower()

    def test_all_valid_action_types_pass(self) -> None:
        """All spec‑defined action types should pass validation."""
        from src.gui.macro_editor import _validate_macro_json

        valid_types = [
            "key", "delay", "mouse_move", "click", "type_string",
            "dynamic_click", "dynamic_move", "dynamic_attack",
        ]
        actions = [{"type": t, "key": "a", "hold_ms": 100} for t in valid_types]
        obj = {"name": "test", "description": "all types", "actions": actions}
        ok, err = _validate_macro_json(json.dumps(obj))
        assert ok, f"Expected valid for all action types, got: {err}"

    def test_macro_without_description_passes(self) -> None:
        """Description is optional — macro should pass without it."""
        from src.gui.macro_editor import _validate_macro_json

        obj = {"name": "minimal", "actions": [{"type": "delay", "ms": 100}]}
        ok, err = _validate_macro_json(json.dumps(obj))
        assert ok, f"Expected valid minimal macro, got: {err}"


# ---------------------------------------------------------------------------
# MainWindow integration
# ---------------------------------------------------------------------------


class TestMainWindowMacroIntegration:
    """Test MainWindow macro editor button and callback."""

    def test_edit_macros_button_exists(self) -> None:
        """MainWindow toolbar should have an Edit Macros button."""
        from src.gui.main_window import MainWindow

        window = MainWindow()
        try:
            assert hasattr(window, "_btn_macros"), "Missing Edit Macros button"
            btn = window._btn_macros
            assert btn is not None
            # Verify button text contains "Macros"
            text = btn.cget("text")
            assert "Macros" in text, f"Button text {text!r} should contain 'Macros'"
        finally:
            window.destroy()

    def test_on_edit_macros_method_exists(self) -> None:
        """MainWindow should have _on_edit_macros callback."""
        from src.gui.main_window import MainWindow

        window = MainWindow()
        try:
            assert hasattr(window, "_on_edit_macros")
            assert callable(window._on_edit_macros)
        finally:
            window.destroy()


# ---------------------------------------------------------------------------
# File I/O round‑trip
# ---------------------------------------------------------------------------


class TestMacroFileRoundTrip:
    """Test JSON load/save round‑tripping."""

    def test_load_macros_from_disk(self, temp_macros_file: Path) -> None:
        """Should load 2 macros from the temp file."""
        data = json.loads(temp_macros_file.read_text(encoding="utf-8"))
        macros = data.get("macros", [])
        assert len(macros) == 2
        names = {m["name"] for m in macros}
        assert names == {"move_forward", "jump"}

    def test_save_then_reload_preserves_count(self, tmp_path: Path) -> None:
        """Saving and reloading should preserve macro count."""
        file_path = tmp_path / "macros.json"

        macros = [
            {"name": "a", "description": "first", "actions": [{"type": "delay", "ms": 100}]},
            {"name": "b", "description": "second", "actions": [{"type": "key", "key": "x", "hold_ms": 50}]},
            {"name": "c", "description": "third", "actions": [{"type": "click", "button": "left"}]},
        ]
        data = {"version": "1.0.0", "macros": macros}
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        # Reload
        loaded = json.loads(file_path.read_text(encoding="utf-8"))
        assert len(loaded["macros"]) == 3

        # Modify and save again
        loaded["macros"].append(
            {"name": "d", "description": "fourth", "actions": [{"type": "delay", "ms": 500}]}
        )
        file_path.write_text(json.dumps(loaded, indent=2, ensure_ascii=False), encoding="utf-8")

        # Reload again
        reloaded = json.loads(file_path.read_text(encoding="utf-8"))
        assert len(reloaded["macros"]) == 4

    def test_all_bundled_macros_valid(self) -> None:
        """Every macro in config/macros.json should pass validation."""
        from src.gui.macro_editor import _validate_macro_json

        config_path = _PROJECT_ROOT / "config" / "macros.json"
        assert config_path.exists(), f"Bundled macros not found at {config_path}"

        data = json.loads(config_path.read_text(encoding="utf-8"))
        macros = data.get("macros", [])
        assert len(macros) > 0, "Expected at least 1 macro in config/macros.json"

        for macro in macros:
            text = json.dumps(macro, indent=2, ensure_ascii=False)
            ok, err = _validate_macro_json(text)
            name = macro.get("name", "?")
            assert ok, f"Macro '{name}' failed validation: {err}"
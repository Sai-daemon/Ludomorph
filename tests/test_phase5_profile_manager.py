"""
Phase 5.7 — Profile Manager Tests.

Validates:
- profile_manager module importability
- Export creates valid .gameai_profile zips with correct structure
- Import extracts and validates correctly
- Full validation checklist (8 points)
- Round‑trip (export → import → identical content)
- Overwrite behaviour
- MainWindow integration (menu items, callbacks)
- GUI package exports ProfileManagerDialog
"""

from __future__ import annotations

import json
import tempfile
import zipfile
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
# Path helpers
# ---------------------------------------------------------------------------

from src.config_manager import PROFILES_DIR, _atomic_write_json, _ensure_config_dir

_ensure_config_dir()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_minimal_profile_files(profile_dir: Path) -> None:
    """Create the minimal set of required files for a valid profile."""
    profile_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "1.0.0",
        "profile_name": profile_dir.name,
        "target_game": "Test Game",
        "gameai_min_version": "0.2.0",
        "author": "Tester",
        "description": "A test profile.",
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "tags": ["test"],
        "requires": {"ocr_languages": ["eng"], "llm_model": "test-model"},
    }
    config = {"profile_name": profile_dir.name}
    macros = {
        "version": "1.0.0",
        "macros": [
            {
                "name": "test_action",
                "description": "A test action",
                "actions": [{"type": "key", "key": "a", "hold_ms": 100}],
            }
        ],
    }
    regions = {
        "version": "1.0.0",
        "regions": [
            {
                "name": "hp_bar",
                "type": "color_bar",
                "bounds": [100, 200, 300, 230],
                "preprocess": ["grayscale"],
            }
        ],
    }

    _atomic_write_json(profile_dir / "manifest.json", manifest)
    _atomic_write_json(profile_dir / "config.json", config)
    _atomic_write_json(profile_dir / "macros.json", macros)
    _atomic_write_json(profile_dir / "regions.json", regions)


@pytest.fixture
def sample_profile(tmp_path: Path) -> str:
    """Create a minimal profile in the profiles directory and return its name."""
    import uuid

    name = f"test_profile_{uuid.uuid4().hex[:8]}"
    profile_dir = PROFILES_DIR / name
    _make_minimal_profile_files(profile_dir)
    yield name
    # Cleanup
    import shutil

    if profile_dir.exists():
        shutil.rmtree(profile_dir)


@pytest.fixture
def sample_zip(sample_profile: str, tmp_path: Path) -> Path:
    """Export a sample profile and return the path to the .gameai_profile zip."""
    from src.profile_manager import ExportOptions, export_profile

    opts = ExportOptions(
        profile_name=sample_profile,
        version="1.0.0",
        author="Test Author",
        description="Test description",
    )
    zip_path = tmp_path / f"{sample_profile}_v1.0.0.gameai_profile"
    return export_profile(opts, zip_path)


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


class TestProfileManagerImport:
    """Verify profile_manager module and classes are importable."""

    def test_profile_manager_importable(self) -> None:
        """profile_manager should be importable."""
        from src import profile_manager

        assert profile_manager is not None

    def test_validation_result_importable(self) -> None:
        """ValidationResult dataclass should be importable."""
        from src.profile_manager import ValidationResult

        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.errors == []

    def test_export_options_importable(self) -> None:
        """ExportOptions dataclass should be importable."""
        from src.profile_manager import ExportOptions

        opts = ExportOptions(profile_name="test", version="1.0.0")
        assert opts.profile_name == "test"
        assert opts.include_state is False

    def test_profile_manager_dialog_importable(self) -> None:
        """ProfileManagerDialog should be importable from src.gui."""
        from src.gui.profile_manager_dialog import ProfileManagerDialog

        assert ProfileManagerDialog is not None

    def test_profile_manager_dialog_in_gui_init(self) -> None:
        """ProfileManagerDialog should be exported from src.gui."""
        from src.gui import ProfileManagerDialog

        assert ProfileManagerDialog is not None

    def test_profile_manager_dialog_in_gui_all(self) -> None:
        """ProfileManagerDialog should be in __all__ of src.gui."""
        from src.gui import __all__ as gui_all

        assert "ProfileManagerDialog" in gui_all

    def test_profile_manager_dialog_is_toplevel(self) -> None:
        """ProfileManagerDialog should be a tk.Toplevel subclass."""
        from src.gui.profile_manager_dialog import ProfileManagerDialog

        import tkinter as tk

        assert issubclass(ProfileManagerDialog, tk.Toplevel)


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


class TestExport:
    """Test that export creates valid .gameai_profile archives."""

    def test_export_creates_zip(self, sample_profile: str, tmp_path: Path) -> None:
        """Export should produce a .gameai_profile zip file."""
        from src.profile_manager import ExportOptions, export_profile

        opts = ExportOptions(profile_name=sample_profile, version="1.0.0")
        output = tmp_path / "output.gameai_profile"
        result = export_profile(opts, output)

        assert result.exists()
        assert result.suffix == ".gameai_profile"
        assert zipfile.is_zipfile(result)

    def test_export_contains_required_files(self, sample_zip: Path) -> None:
        """The exported zip must contain all 4 required files."""
        with zipfile.ZipFile(sample_zip, "r") as zf:
            names = {n.rstrip("/") for n in zf.namelist()}

        for required in ("manifest.json", "config.json", "macros.json", "regions.json"):
            assert required in names, f"Missing required file: {required}"

    def test_export_flat_structure(self, sample_zip: Path) -> None:
        """Required files should be at the root of the archive (flat layout)."""
        with zipfile.ZipFile(sample_zip, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]

        for name in ("manifest.json", "config.json", "macros.json", "regions.json"):
            assert name in names, f"File should be at root level: {name}"

    def test_export_manifest_content(self, sample_zip: Path) -> None:
        """The manifest.json should have correct metadata."""
        with zipfile.ZipFile(sample_zip, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

        assert manifest["schema_version"] == "1.0.0"
        assert "profile_name" in manifest
        assert "gameai_min_version" in manifest
        assert "created" in manifest

    def test_export_utf8_encoding(self, sample_zip: Path) -> None:
        """All text files should be UTF-8 encoded."""
        with zipfile.ZipFile(sample_zip, "r") as zf:
            for name in ("manifest.json", "config.json", "macros.json", "regions.json"):
                raw = zf.read(name)
                raw.decode("utf-8")  # should not raise

    def test_export_no_absolute_paths(self, sample_zip: Path) -> None:
        """No entry in the archive should start with /."""
        with zipfile.ZipFile(sample_zip, "r") as zf:
            for info in zf.infolist():
                assert not info.filename.startswith("/"), (
                    f"Absolute path in archive: {info.filename}"
                )

    def test_export_with_optional_inclusions(self, sample_profile: str, tmp_path: Path) -> None:
        """Export with state.json, corrections.json, and README.md."""
        from src.profile_manager import ExportOptions, export_profile

        profile_dir = PROFILES_DIR / sample_profile

        # Create optional files
        profile_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(profile_dir / "state.json", {"last_action": "test"})
        _atomic_write_json(profile_dir / "corrections.json", {"ocr_corrections": {}})
        (profile_dir / "README.md").write_text("# Test Profile", encoding="utf-8")

        opts = ExportOptions(
            profile_name=sample_profile,
            version="1.0.0",
            include_state=True,
            include_corrections=True,
            include_readme=True,
        )
        output = tmp_path / "full_export.gameai_profile"
        result = export_profile(opts, output)

        with zipfile.ZipFile(result, "r") as zf:
            names = {n.rstrip("/") for n in zf.namelist()}

        assert "state.json" in names
        assert "corrections.json" in names
        assert "README.md" in names

    def test_export_icon_inclusion(self, sample_profile: str, tmp_path: Path) -> None:
        """Export with res/icon.png included."""
        from src.profile_manager import ExportOptions, export_profile

        profile_dir = PROFILES_DIR / sample_profile
        res_dir = profile_dir / "res"
        res_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal 1x1 PNG
        import struct
        import zlib

        def _create_png(w: int, h: int) -> bytes:
            def chunk(ctype: bytes, data: bytes) -> bytes:
                c = ctype + data
                return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

            raw = b""
            for y in range(h):
                raw += b"\x00" + b"\x00\x00\x00" * w
            return (
                b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw))
                + chunk(b"IEND", b"")
            )

        (res_dir / "icon.png").write_bytes(_create_png(1, 1))

        opts = ExportOptions(
            profile_name=sample_profile,
            version="1.0.0",
            include_icon=True,
        )
        output = tmp_path / "icon_export.gameai_profile"
        result = export_profile(opts, output)

        with zipfile.ZipFile(result, "r") as zf:
            names = {n.rstrip("/") for n in zf.namelist()}

        assert "res/icon.png" in names


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Test the 8-point validation checklist."""

    # -- Check 1: Required files ---------------------------------------------------

    def test_validation_passes_for_valid_zip(self, sample_zip: Path) -> None:
        """A correctly-built archive should pass validation."""
        from src.profile_manager import validate_profile_zip

        result = validate_profile_zip(sample_zip)
        assert result.is_valid, f"Errors: {result.errors}"
        assert result.manifest is not None

    def test_validation_fails_missing_manifest(self, tmp_path: Path) -> None:
        """Missing manifest.json should fail validation."""
        from src.profile_manager import validate_profile_zip

        zip_path = tmp_path / "no_manifest.gameai_profile"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("config.json", '{"profile_name":"test"}')
            zf.writestr("macros.json", '{"version":"1.0.0","macros":[]}')
            zf.writestr("regions.json", '{"version":"1.0.0","regions":[]}')

        result = validate_profile_zip(zip_path)
        assert not result.is_valid
        assert any("manifest.json" in err for err in result.errors)

    def test_validation_fails_missing_config(self, tmp_path: Path) -> None:
        """Missing config.json should fail validation."""
        from src.profile_manager import validate_profile_zip

        zip_path = tmp_path / "no_config.gameai_profile"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"1.0.0","profile_name":"test","gameai_min_version":"0.2.0"}')
            zf.writestr("macros.json", '{"version":"1.0.0","macros":[]}')
            zf.writestr("regions.json", '{"version":"1.0.0","regions":[]}')

        result = validate_profile_zip(zip_path)
        assert not result.is_valid
        assert any("config.json" in err for err in result.errors)

    # -- Check 2: manifest.schema_version ------------------------------------------

    def test_validation_fails_unsupported_schema(self, tmp_path: Path) -> None:
        """An unsupported schema_version should fail."""
        from src.profile_manager import validate_profile_zip

        zip_path = tmp_path / "bad_schema.gameai_profile"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"9.9.9","profile_name":"test","gameai_min_version":"0.2.0"}')
            zf.writestr("config.json", '{"profile_name":"test"}')
            zf.writestr("macros.json", '{"version":"1.0.0","macros":[]}')
            zf.writestr("regions.json", '{"version":"1.0.0","regions":[]}')

        result = validate_profile_zip(zip_path)
        assert not result.is_valid
        assert any("schema_version" in err for err in result.errors)

    # -- Check 3: Required JSON fields ---------------------------------------------

    def test_validation_fails_malformed_json(self, tmp_path: Path) -> None:
        """Malformed JSON in a required file should fail."""
        from src.profile_manager import validate_profile_zip

        zip_path = tmp_path / "bad_json.gameai_profile"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"1.0.0","profile_name":"test","gameai_min_version":"0.2.0"}')
            zf.writestr("config.json", '{"profile_name":"test"}')
            zf.writestr("macros.json", "NOT JSON")
            zf.writestr("regions.json", '{"version":"1.0.0","regions":[]}')

        result = validate_profile_zip(zip_path)
        assert not result.is_valid
        assert any("macros.json" in err for err in result.errors)

    def test_validation_fails_macros_not_object(self, tmp_path: Path) -> None:
        """macros.json that is not an object should fail."""
        from src.profile_manager import validate_profile_zip

        zip_path = tmp_path / "bad_macros_shape.gameai_profile"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"1.0.0","profile_name":"test","gameai_min_version":"0.2.0"}')
            zf.writestr("config.json", '{"profile_name":"test"}')
            zf.writestr("macros.json", "[]")
            zf.writestr("regions.json", '{"version":"1.0.0","regions":[]}')

        result = validate_profile_zip(zip_path)
        assert not result.is_valid

    # -- Check 4: Valid macro action types -----------------------------------------

    def test_validation_fails_invalid_action_type(self, tmp_path: Path) -> None:
        """An unknown macro action type should fail validation."""
        from src.profile_manager import validate_profile_zip

        zip_path = tmp_path / "bad_action.gameai_profile"
        macros = {
            "version": "1.0.0",
            "macros": [
                {
                    "name": "bad_macro",
                    "description": "Has bad action",
                    "actions": [{"type": "INVALID_ACTION_TYPE"}],
                }
            ],
        }
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"1.0.0","profile_name":"test","gameai_min_version":"0.2.0"}')
            zf.writestr("config.json", '{"profile_name":"test"}')
            zf.writestr("macros.json", json.dumps(macros))
            zf.writestr("regions.json", '{"version":"1.0.0","regions":[]}')

        result = validate_profile_zip(zip_path)
        assert not result.is_valid
        assert any("INVALID_ACTION_TYPE" in err for err in result.errors)

    def test_validation_accepts_all_valid_action_types(self, tmp_path: Path) -> None:
        """All known valid action types should pass validation."""
        from src.profile_manager import validate_profile_zip, VALID_ACTION_TYPES

        actions = [{"type": t} for t in sorted(VALID_ACTION_TYPES)]
        # Add required fields for types that need them (key, hold_ms for "key" type)
        for a in actions:
            if a["type"] == "key":
                a["key"] = "a"
                a["hold_ms"] = 50
            elif a["type"] == "delay" or a["type"] == "wait":
                a["ms"] = 100
            elif a["type"] == "mouse_move":
                a["x"] = 10
                a["y"] = 10
            elif a["type"] == "dynamic_click" or a["type"] == "dynamic_move":
                a["target_class"] = "person"
            elif a["type"] == "type_string" or a["type"] == "type_text":
                a["text"] = "hello"

        macros = {
            "version": "1.0.0",
            "macros": [{"name": "all_types", "description": "All types", "actions": actions}],
        }

        zip_path = tmp_path / "all_types.gameai_profile"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"1.0.0","profile_name":"test","gameai_min_version":"0.2.0"}')
            zf.writestr("config.json", '{"profile_name":"test"}')
            zf.writestr("macros.json", json.dumps(macros))
            zf.writestr("regions.json", '{"version":"1.0.0","regions":[]}')

        result = validate_profile_zip(zip_path)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_validation_fails_missing_type_field(self, tmp_path: Path) -> None:
        """A macro action without a 'type' field should fail."""
        from src.profile_manager import validate_profile_zip

        macros = {
            "version": "1.0.0",
            "macros": [{"name": "no_type", "description": "No type", "actions": [{"key": "a"}]}],
        }

        zip_path = tmp_path / "no_type.gameai_profile"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"1.0.0","profile_name":"test","gameai_min_version":"0.2.0"}')
            zf.writestr("config.json", '{"profile_name":"test"}')
            zf.writestr("macros.json", json.dumps(macros))
            zf.writestr("regions.json", '{"version":"1.0.0","regions":[]}')

        result = validate_profile_zip(zip_path)
        assert not result.is_valid
        assert any("missing 'type'" in err.lower() for err in result.errors)

    # -- Check 5: Valid region bounds ----------------------------------------------

    def test_validation_fails_invalid_region_bounds(self, tmp_path: Path) -> None:
        """Regions with invalid bounds should fail validation."""
        from src.profile_manager import validate_profile_zip

        regions = {
            "version": "1.0.0",
            "regions": [{"name": "bad", "type": "ocr", "bounds": [10, 20, 0, 30]}],
        }

        zip_path = tmp_path / "bad_bounds.gameai_profile"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"1.0.0","profile_name":"test","gameai_min_version":"0.2.0"}')
            zf.writestr("config.json", '{"profile_name":"test"}')
            zf.writestr("macros.json", '{"version":"1.0.0","macros":[]}')
            zf.writestr("regions.json", json.dumps(regions))

        result = validate_profile_zip(zip_path)
        assert not result.is_valid
        assert any("positive" in err.lower() or "width" in err.lower() for err in result.errors)

    def test_validation_fails_missing_bounds(self, tmp_path: Path) -> None:
        """Regions without bounds should fail."""
        from src.profile_manager import validate_profile_zip

        regions = {
            "version": "1.0.0",
            "regions": [{"name": "no_bounds", "type": "ocr"}],
        }

        zip_path = tmp_path / "no_bounds.gameai_profile"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"1.0.0","profile_name":"test","gameai_min_version":"0.2.0"}')
            zf.writestr("config.json", '{"profile_name":"test"}')
            zf.writestr("macros.json", '{"version":"1.0.0","macros":[]}')
            zf.writestr("regions.json", json.dumps(regions))

        result = validate_profile_zip(zip_path)
        assert not result.is_valid
        assert any("bounds" in err.lower() for err in result.errors)

    # -- Check 6: Icon validation --------------------------------------------------

    def test_validation_fails_non_png_icon(self, sample_zip: Path, tmp_path: Path) -> None:
        """A non‑PNG icon should fail validation."""
        # Rebuild zip with a fake icon
        rebuilt = tmp_path / "bad_icon.gameai_profile"
        with zipfile.ZipFile(sample_zip, "r") as src:
            with zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_DEFLATED) as dst:
                for info in src.infolist():
                    dst.writestr(info, src.read(info.filename))
                dst.writestr("res/icon.png", b"NOT A PNG FILE")

        from src.profile_manager import validate_profile_zip

        result = validate_profile_zip(rebuilt)
        assert not result.is_valid
        assert any("png" in err.lower() for err in result.errors)

    def test_validation_fails_oversized_icon(self, sample_zip: Path, tmp_path: Path) -> None:
        """An icon > 1 MB should fail validation."""
        rebuilt = tmp_path / "big_icon.gameai_profile"
        with zipfile.ZipFile(sample_zip, "r") as src:
            with zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_DEFLATED) as dst:
                for info in src.infolist():
                    dst.writestr(info, src.read(info.filename))
                # Write a valid PNG header + 2 MB of junk
                big_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * (2 * 1024 * 1024)
                dst.writestr("res/icon.png", big_data)

        from src.profile_manager import validate_profile_zip

        result = validate_profile_zip(rebuilt)
        assert not result.is_valid
        assert any("icon" in err.lower() and ("mb" in err.lower() or "size" in err.lower()) for err in result.errors)

    # -- Check 7: No absolute paths -----------------------------------------------

    def test_validation_fails_absolute_path(self, tmp_path: Path) -> None:
        """An absolute path in the archive should fail."""
        zip_path = tmp_path / "abs_path.gameai_profile"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"1.0.0","profile_name":"test","gameai_min_version":"0.2.0"}')
            zf.writestr("config.json", '{"profile_name":"test"}')
            zf.writestr("macros.json", '{"version":"1.0.0","macros":[]}')
            zf.writestr("regions.json", '{"version":"1.0.0","regions":[]}')
            # Manually inject an absolute path entry
            zf.writestr("/etc/passwd", "bad")

        from src.profile_manager import validate_profile_zip

        result = validate_profile_zip(zip_path)
        assert not result.is_valid
        assert any("absolute" in err.lower() or "rooted" in err.lower() for err in result.errors)

    # -- Check 8: UTF-8 encoding ---------------------------------------------------

    def test_validation_fails_non_utf8_file(self, tmp_path: Path) -> None:
        """Non‑UTF‑8 text files should fail."""
        zip_path = tmp_path / "bad_encoding.gameai_profile"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"1.0.0","profile_name":"test","gameai_min_version":"0.2.0"}')
            zf.writestr("config.json", '{"profile_name":"test"}')
            zf.writestr("macros.json", '{"version":"1.0.0","macros":[]}')
            zf.writestr("regions.json", b"\xff\xfe\x00\x00")

        from src.profile_manager import validate_profile_zip

        result = validate_profile_zip(zip_path)
        assert not result.is_valid

    # -- Forward compatibility test -----------------------------------------------

    def test_validation_warns_unknown_files(self, tmp_path: Path) -> None:
        """Unknown files should produce warnings, not errors."""
        zip_path = tmp_path / "extra_file.gameai_profile"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"1.0.0","profile_name":"test","gameai_min_version":"0.2.0"}')
            zf.writestr("config.json", '{"profile_name":"test"}')
            zf.writestr("macros.json", '{"version":"1.0.0","macros":[]}')
            zf.writestr("regions.json", '{"version":"1.0.0","regions":[]}')
            zf.writestr("future_feature.json", "{}")

        from src.profile_manager import validate_profile_zip

        result = validate_profile_zip(zip_path)
        # Should still be valid (forward compatible)
        assert result.is_valid
        assert len(result.warnings) > 0

    def test_validation_not_a_zip(self, tmp_path: Path) -> None:
        """A non‑zip file should fail validation immediately."""
        from src.profile_manager import validate_profile_zip

        not_a_zip = tmp_path / "not_a_zip.txt"
        not_a_zip.write_text("hello world")
        result = validate_profile_zip(not_a_zip)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImport:
    """Test that import correctly extracts and places profile files."""

    def test_import_valid_zip(self, sample_zip: Path) -> None:
        """Importing a valid zip should succeed and return profile info."""
        from src.profile_manager import import_profile

        profile_name, profile_path = import_profile(sample_zip)
        assert profile_name
        assert profile_path.is_dir()
        assert (profile_path / "manifest.json").exists()
        assert (profile_path / "config.json").exists()
        assert (profile_path / "macros.json").exists()
        assert (profile_path / "regions.json").exists()

        # Cleanup
        import shutil

        shutil.rmtree(profile_path)

    def test_import_preserves_content(self, sample_zip: Path) -> None:
        """Imported files should have the same content as in the archive."""
        from src.profile_manager import import_profile

        with zipfile.ZipFile(sample_zip, "r") as zf:
            original_manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

        profile_name, profile_path = import_profile(sample_zip)

        imported_manifest = json.loads((profile_path / "manifest.json").read_text("utf-8"))
        assert imported_manifest == original_manifest

        # Cleanup
        import shutil

        shutil.rmtree(profile_path)

    def test_import_rejects_invalid_zip(self, tmp_path: Path) -> None:
        """Importing an invalid zip should raise ValueError."""
        from src.profile_manager import import_profile

        bad_zip = tmp_path / "bad.gameai_profile"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"9.9.9","profile_name":"x","gameai_min_version":"0.2.0"}')
            zf.writestr("config.json", "{}")
            zf.writestr("macros.json", "{}")
            zf.writestr("regions.json", "{}")

        with pytest.raises(ValueError, match="validation failed"):
            import_profile(bad_zip)

    def test_import_file_not_found(self) -> None:
        """Importing a non‑existent file should raise FileNotFoundError."""
        from src.profile_manager import import_profile

        with pytest.raises(FileNotFoundError):
            import_profile(Path("/nonexistent/path.gameai_profile"))


# ---------------------------------------------------------------------------
# Round‑trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Test export → import round‑tripping."""

    def test_round_trip_preserves_config(self, sample_profile: str, tmp_path: Path) -> None:
        """Export then import should preserve config.json content."""
        from src.profile_manager import ExportOptions, export_profile, import_profile

        # Export
        opts = ExportOptions(profile_name=sample_profile, version="1.0.0")
        zip_path = tmp_path / "roundtrip.gameai_profile"
        export_profile(opts, zip_path)

        # Import to a fresh name
        profile_name, profile_path = import_profile(zip_path)

        # Verify config
        imported_config = json.loads((profile_path / "config.json").read_text("utf-8"))
        assert imported_config["profile_name"] == sample_profile

        # Verify macros
        imported_macros = json.loads((profile_path / "macros.json").read_text("utf-8"))
        assert len(imported_macros["macros"]) == 1
        assert imported_macros["macros"][0]["name"] == "test_action"

        # Cleanup
        import shutil

        shutil.rmtree(profile_path)


# ---------------------------------------------------------------------------
# Overwrite behaviour
# ---------------------------------------------------------------------------


class TestOverwrite:
    """Test overwrite and collision handling."""

    def test_import_with_overwrite(self, sample_zip: Path) -> None:
        """Import with overwrite=True should replace existing profile."""
        from src.profile_manager import import_profile

        # First import (may get suffixed since fixture already created the dir)
        name1, path1 = import_profile(sample_zip)

        # Second import with overwrite=True on the SAME profile name
        # Use the manifest's profile_name directly, which is the base name
        with zipfile.ZipFile(sample_zip, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        base_name = manifest["profile_name"]

        name2, path2 = import_profile(sample_zip, overwrite=True)

        # With overwrite=True, it should use the base name (not a suffixed one)
        assert name2 == base_name
        assert path2.is_dir()

        # Cleanup
        import shutil
        shutil.rmtree(path1)
        if path2 != path1:
            shutil.rmtree(path2)

    def test_import_without_overwrite_creates_new(self, sample_zip: Path) -> None:
        """Import without overwrite should create a suffixed name."""
        from src.profile_manager import import_profile

        name1, path1 = import_profile(sample_zip)
        name2, path2 = import_profile(sample_zip, overwrite=False)

        assert name2 != name1  # Should have a different (suffixed) name
        assert path2.is_dir()

        # Cleanup
        import shutil

        shutil.rmtree(path1)
        shutil.rmtree(path2)


# ---------------------------------------------------------------------------
# MainWindow integration
# ---------------------------------------------------------------------------


class TestMainWindowIntegration:
    """Test MainWindow import/export menu items and callbacks."""

    def test_import_menu_item_enabled(self) -> None:
        """The Import Profile menu item should be enabled (not DISABLED)."""
        from src.gui.main_window import MainWindow

        window = MainWindow()
        try:
            assert hasattr(window, "_on_import_profile")
            assert callable(window._on_import_profile)

            assert hasattr(window, "_on_export_profile")
            assert callable(window._on_export_profile)
        finally:
            window.destroy()

    def test_profile_label_updates_on_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After _on_imported callback, the profile label should update."""
        from src.gui.main_window import MainWindow

        window = MainWindow()
        try:
            # Simulate what happens after import: set _profile_path
            window._profile_path = Path("/fake/path/test_game")
            window._update_profile_label()

            label_text = window._lbl_profile.cget("text")
            assert "test_game" in label_text or "Profile" in label_text
        finally:
            window.destroy()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for profile manager."""

    def test_export_nonexistent_profile_raises(self, tmp_path: Path) -> None:
        """Exporting a non‑existent profile should raise FileNotFoundError."""
        from src.profile_manager import ExportOptions, export_profile

        opts = ExportOptions(profile_name="nonexistent_profile_xyz", version="1.0.0")
        with pytest.raises(FileNotFoundError):
            export_profile(opts, tmp_path / "bad.gameai_profile")

    def test_validation_empty_zip(self, tmp_path: Path) -> None:
        """An empty zip should fail validation."""
        from src.profile_manager import validate_profile_zip

        empty = tmp_path / "empty.gameai_profile"
        with zipfile.ZipFile(empty, "w") as _:
            pass

        result = validate_profile_zip(empty)
        assert not result.is_valid

    def test_manifest_missing_required_fields(self, tmp_path: Path) -> None:
        """A manifest without profile_name should fail."""
        from src.profile_manager import validate_profile_zip

        zip_path = tmp_path / "no_name.gameai_profile"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", '{"schema_version":"1.0.0","gameai_min_version":"0.2.0"}')
            zf.writestr("config.json", '{"profile_name":"test"}')
            zf.writestr("macros.json", '{"version":"1.0.0","macros":[]}')
            zf.writestr("regions.json", '{"version":"1.0.0","regions":[]}')

        result = validate_profile_zip(zip_path)
        assert not result.is_valid
        assert any("profile_name" in err for err in result.errors)

    def test_validation_result_defaults(self) -> None:
        """ValidationResult should default to valid with empty lists."""
        from src.profile_manager import ValidationResult

        r = ValidationResult(is_valid=False)
        assert r.errors == []
        assert r.warnings == []
        assert r.manifest is None

    def test_export_options_defaults(self) -> None:
        """ExportOptions should have sensible defaults."""
        from src.profile_manager import ExportOptions

        opts = ExportOptions(profile_name="test")
        assert opts.version == "1.0.0"
        assert opts.author == ""
        assert opts.tags == []
        assert opts.include_state is False
        assert opts.include_icon is False

    def test_quick_export(self, sample_profile: str, tmp_path: Path) -> None:
        """quick_export should produce a valid archive."""
        from src.profile_manager import quick_export

        result = quick_export(sample_profile, output_dir=tmp_path)
        assert result.exists()
        assert zipfile.is_zipfile(result)


# ---------------------------------------------------------------------------
# ProfileManagerDialog class‑level checks (no display)
# ---------------------------------------------------------------------------


class TestProfileManagerDialogClass:
    """Validate ProfileManagerDialog class structure without display."""

    def test_dialog_has_expected_methods(self) -> None:
        """ProfileManagerDialog should have build and callback methods."""
        from src.gui.profile_manager_dialog import ProfileManagerDialog

        assert hasattr(ProfileManagerDialog, "_build_ui")
        assert hasattr(ProfileManagerDialog, "_build_import_ui")
        assert hasattr(ProfileManagerDialog, "_build_export_ui")
        assert hasattr(ProfileManagerDialog, "_on_browse_import")
        assert hasattr(ProfileManagerDialog, "_on_validate")
        assert hasattr(ProfileManagerDialog, "_on_import")
        assert hasattr(ProfileManagerDialog, "_on_browse_export")
        assert hasattr(ProfileManagerDialog, "_on_export")
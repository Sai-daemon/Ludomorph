"""
Profile Manager — Phase 5.7

Import/export `.gameai_profile` zip archives with validation.

Provides:
- export_profile(): Pack a profile directory into a .gameai_profile zip.
- import_profile(): Extract and validate a .gameai_profile zip into ~/.gameai/profiles/.
- validate_profile_zip(): Run the 8-point validation checklist.
- Valid macro action types match what MacroExecutor._execute_steps() supports.

Spec references:
- Implementation_Phases.md §5.7
- gameai_profile_format_research.md (full .gameai_profile schema)
- architecture.md §4.3 (InputController / macro action types)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config_manager import (
    CONFIG_DIR,
    PROFILES_DIR,
    load_json,
    _atomic_write_json,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_SCHEMA_VERSION = "1.0.0"
MAX_ICON_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB

# Required files in every .gameai_profile archive
REQUIRED_FILES = ("manifest.json", "config.json", "macros.json", "regions.json")

# Optional files that MAY be present
OPTIONAL_FILES = ("state.json", "corrections.json", "README.md")

# Valid macro action types (matches MacroExecutor._execute_steps() dispatch)
VALID_ACTION_TYPES = {
    "key",
    "delay",
    "wait",
    "mouse_move",
    "click",
    "mouse_click",
    "type_string",
    "type_text",
    "dynamic_click",
    "dynamic_move",
}

# Allowed shapes for region bounds validation
BAD_PATH_CHARS_RE = re.compile(r"^[a-zA-Z]:\\" if os.name == "nt" else r"^/")


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of validating a .gameai_profile archive."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manifest: dict[str, Any] | None = None


@dataclass
class ExportOptions:
    """Options controlling what's included in a profile export."""

    profile_name: str
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    target_game: str = ""
    tags: list[str] = field(default_factory=list)
    include_state: bool = False
    include_corrections: bool = False
    include_readme: bool = False
    include_icon: bool = False
    icon_path: Path | None = None


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def _build_manifest(opts: ExportOptions) -> dict[str, Any]:
    """Construct manifest.json content for an export."""
    now = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "profile_name": opts.profile_name,
        "target_game": opts.target_game or opts.profile_name,
        "gameai_min_version": "0.2.0",
        "author": opts.author,
        "description": opts.description,
        "created": now,
        "updated": now,
        "tags": opts.tags,
        "requires": {
            "ocr_languages": ["eng"],
            "llm_model": "phi3.5:3.8b-mini-instruct-q4_K_M",
        },
    }
    return manifest


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_profile(opts: ExportOptions, output_path: Path | str) -> Path:
    """Pack a profile into a .gameai_profile zip archive.

    Parameters
    ----------
    opts : ExportOptions
        Configuration for the export (profile name, version, optional inclusions).
    output_path : Path or str
        Where to write the output zip.  If a directory, the filename is derived
        from the profile name and version.

    Returns
    -------
    Path
        Absolute path to the created archive.

    Raises
    ------
    FileNotFoundError
        If the profile directory or a required file is missing.
    ValueError
        If profile_name contains invalid characters.
    """
    output_path = Path(output_path)
    profile_dir = PROFILES_DIR / opts.profile_name

    if not profile_dir.is_dir():
        raise FileNotFoundError(
            f"Profile directory does not exist: {profile_dir}"
        )

    # Derive output filename if output_path is a directory
    if output_path.is_dir() or output_path.suffix != ".gameai_profile":
        safe_name = re.sub(r"[^\w\-_.]", "_", opts.profile_name)
        safe_version = re.sub(r"[^\w\-_.]", "_", opts.version)
        output_path = output_path / f"{safe_name}_v{safe_version}.gameai_profile"
    output_path = output_path.resolve()

    # Collect files for the archive
    files_to_pack: dict[str, Path] = {}

    # Required files
    for fname in REQUIRED_FILES:
        fpath = profile_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(
                f"Required profile file missing: {fpath}"
            )
        files_to_pack[fname] = fpath

    # Manifest — build fresh (overwrites any existing one for export purity)
    manifest = _build_manifest(opts)
    manifest_path = profile_dir / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    files_to_pack["manifest.json"] = manifest_path

    # Optional files
    if opts.include_state:
        state_path = profile_dir / "state.json"
        if state_path.is_file():
            files_to_pack["state.json"] = state_path

    if opts.include_corrections:
        corr_path = profile_dir / "corrections.json"
        if corr_path.is_file():
            files_to_pack["corrections.json"] = corr_path

    if opts.include_readme:
        readme_path = profile_dir / "README.md"
        if readme_path.is_file():
            files_to_pack["README.md"] = readme_path

    if opts.include_icon:
        icon_path = opts.icon_path or (profile_dir / "res" / "icon.png")
        if icon_path.is_file():
            files_to_pack["res/icon.png"] = icon_path

    # Create the zip archive (DEFLATE, compression level 6)
    logger.info(
        "Exporting profile '%s' v%s → %s (%d files)",
        opts.profile_name,
        opts.version,
        output_path,
        len(files_to_pack),
    )

    with zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for arcname, fpath in sorted(files_to_pack.items()):
            # Ensure arcname uses forward slashes (zip standard)
            arcname_normalised = arcname.replace("\\", "/")
            zf.write(fpath, arcname=arcname_normalised)
            logger.debug("  Added: %s", arcname_normalised)

    logger.info("Export complete: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_profile_zip(zip_path: Path | str) -> ValidationResult:
    """Run the full validation checklist against a .gameai_profile archive.

    Parameters
    ----------
    zip_path : Path or str
        Path to the .gameai_profile zip to validate.

    Returns
    -------
    ValidationResult
        With ``is_valid``, ``errors`` list, and ``warnings`` list.
    """
    zip_path = Path(zip_path)
    result = ValidationResult(is_valid=True)

    # Check 0: Must be a readable zip
    if not zipfile.is_zipfile(zip_path):
        result.errors.append("Not a valid zip archive.")
        result.is_valid = False
        return result

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names: set[str] = {n.rstrip("/") for n in zf.namelist()}
            # Build a name→ZipInfo lookup (used for icon size check)
            info_dict: dict[str, zipfile.ZipInfo] = {}
            for info in zf.infolist():
                name = info.filename.rstrip("/")
                info_dict[name] = info

            # --- Check 1: Required files exist ----------------------------
            for required in REQUIRED_FILES:
                if required not in names:
                    result.errors.append(f"Missing required file: {required}")
                    result.is_valid = False

            # --- Check 2: manifest.schema_version is supported ------------
            manifest_data = None
            if "manifest.json" in names:
                try:
                    manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
                    result.manifest = manifest_data

                    sv = manifest_data.get("schema_version", "")
                    if sv != SUPPORTED_SCHEMA_VERSION:
                        result.errors.append(
                            f"Unsupported schema_version '{sv}' (expected '{SUPPORTED_SCHEMA_VERSION}')."
                        )
                        result.is_valid = False

                    # Check required manifest fields
                    for field in ("schema_version", "profile_name", "gameai_min_version"):
                        if field not in manifest_data:
                            result.errors.append(
                                f"manifest.json missing required field: {field}"
                            )
                            result.is_valid = False

                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    result.errors.append(f"manifest.json is not valid UTF-8 JSON: {exc}")
                    result.is_valid = False

            # --- Check 3: config.json, macros.json, regions.json are valid JSON & have required fields
            for json_file in ("config.json", "macros.json", "regions.json"):
                if json_file not in names:
                    continue
                try:
                    data = json.loads(zf.read(json_file).decode("utf-8"))
                    _validate_json_required_fields(json_file, data, result)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    result.errors.append(f"{json_file} is not valid UTF-8 JSON: {exc}")
                    result.is_valid = False

            # --- Check 4: Macros reference only valid action types ---------
            if "macros.json" in names:
                try:
                    macros_data = json.loads(zf.read("macros.json").decode("utf-8"))
                    _validate_macro_actions(macros_data, result)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass  # already reported above

            # --- Check 5: Regions have valid bounds -----------------------
            if "regions.json" in names:
                try:
                    regions_data = json.loads(zf.read("regions.json").decode("utf-8"))
                    _validate_region_bounds(regions_data, result)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass  # already reported above

            # --- Check 6: Optional icon file is PNG ≤ 1 MB ----------------
            icon_name = "res/icon.png"
            if icon_name in names:
                icon_info = info_dict.get(icon_name)
                if icon_info is not None:
                    if icon_info.file_size > MAX_ICON_SIZE_BYTES:
                        result.errors.append(
                            f"Icon file exceeds {MAX_ICON_SIZE_BYTES // (1024 * 1024)} MB "
                            f"(size: {icon_info.file_size} bytes)."
                        )
                        result.is_valid = False
                # Verify it's actually a PNG (check header)
                icon_bytes = zf.read(icon_name)
                if not icon_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                    result.errors.append(
                        "res/icon.png is not a valid PNG file."
                    )
                    result.is_valid = False

            # --- Check 7: No absolute file paths --------------------------
            for name in names:
                if name.startswith("/") or (os.name == "nt" and re.match(r"^[A-Za-z]:\\", name)):
                    result.errors.append(
                        f"Absolute or rooted path not allowed in archive: {name}"
                    )
                    result.is_valid = False

            # --- Check 8: All text files are UTF-8 ------------------------
            for name in names:
                if name.endswith((".json", ".md")):
                    try:
                        raw = zf.read(name)
                        raw.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        result.errors.append(
                            f"File '{name}' is not valid UTF-8: {exc}"
                        )
                        result.is_valid = False

            # --- Forward-compatibility: warn on unknown entries ------------
            known = set(REQUIRED_FILES) | set(OPTIONAL_FILES) | {"res/", "res/icon.png"}
            for name in names:
                if name not in known and not name.startswith("res/"):
                    result.warnings.append(
                        f"Unknown file in archive (will be ignored): {name}"
                    )

    except zipfile.BadZipFile as exc:
        result.errors.append(f"Corrupt zip file: {exc}")
        result.is_valid = False
    except Exception as exc:
        result.errors.append(f"Validation failed: {exc}")
        result.is_valid = False

    return result


def _validate_json_required_fields(
    filename: str, data: Any, result: ValidationResult
) -> None:
    """Check that a JSON file from the archive has its expected shape."""
    if filename == "config.json":
        if not isinstance(data, dict):
            result.errors.append("config.json must be a JSON object.")
            result.is_valid = False
            return
        for field in ("profile_name",):
            if field not in data:
                result.warnings.append(
                    f"config.json missing recommended field: {field}"
                )

    elif filename == "macros.json":
        if not isinstance(data, dict):
            result.errors.append("macros.json must be a JSON object.")
            result.is_valid = False
            return
        macros = data.get("macros")
        if macros is None:
            result.errors.append("macros.json must contain a 'macros' list.")
            result.is_valid = False
        elif not isinstance(macros, list):
            result.errors.append("macros.json 'macros' field must be a list.")
            result.is_valid = False

    elif filename == "regions.json":
        if not isinstance(data, dict):
            result.errors.append("regions.json must be a JSON object.")
            result.is_valid = False
            return
        regions = data.get("regions")
        if regions is None:
            result.errors.append("regions.json must contain a 'regions' list.")
            result.is_valid = False
        elif not isinstance(regions, list):
            result.errors.append("regions.json 'regions' field must be a list.")
            result.is_valid = False


def _validate_macro_actions(
    macros_data: dict[str, Any], result: ValidationResult
) -> None:
    """Check that every macro action type is valid."""
    macros: list[dict[str, Any]] = macros_data.get("macros", [])
    for macro in macros:
        if not isinstance(macro, dict):
            continue
        macro_name = macro.get("name", "<unnamed>")
        actions = macro.get("actions", [])
        if not isinstance(actions, list):
            result.errors.append(
                f"Macro '{macro_name}': 'actions' must be a list."
            )
            result.is_valid = False
            continue
        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                result.errors.append(
                    f"Macro '{macro_name}' action[{i}]: must be an object."
                )
                result.is_valid = False
                continue
            action_type = action.get("type")
            if action_type is None:
                result.errors.append(
                    f"Macro '{macro_name}' action[{i}]: missing 'type' field."
                )
                result.is_valid = False
            elif action_type not in VALID_ACTION_TYPES:
                result.errors.append(
                    f"Macro '{macro_name}' action[{i}]: unknown type '{action_type}'. "
                    f"Valid types: {sorted(VALID_ACTION_TYPES)}"
                )
                result.is_valid = False


def _validate_region_bounds(
    regions_data: dict[str, Any], result: ValidationResult
) -> None:
    """Check that every region has valid [x, y, width, height] bounds."""
    regions: list[dict[str, Any]] = regions_data.get("regions", [])
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_name = region.get("name", "<unnamed>")
        bounds = region.get("bounds")
        if bounds is None:
            result.errors.append(
                f"Region '{region_name}': missing 'bounds' field."
            )
            result.is_valid = False
            continue
        if not isinstance(bounds, list) or len(bounds) != 4:
            result.errors.append(
                f"Region '{region_name}': bounds must be [x, y, width, height]."
            )
            result.is_valid = False
            continue
        if not all(isinstance(v, (int, float)) for v in bounds):
            result.errors.append(
                f"Region '{region_name}': bounds values must be numbers."
            )
            result.is_valid = False
            continue
        x, y, w, h = bounds
        if w <= 0 or h <= 0:
            result.errors.append(
                f"Region '{region_name}': width ({w}) and height ({h}) must be positive."
            )
            result.is_valid = False


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def import_profile(
    zip_path: Path | str, overwrite: bool = False
) -> tuple[str, Path]:
    """Extract and validate a .gameai_profile archive into ~/.gameai/profiles/.

    Parameters
    ----------
    zip_path : Path or str
        Path to the .gameai_profile archive.
    overwrite : bool
        If True, overwrite an existing profile with the same name.
        If False, the profile name gets a suffix to avoid collision.

    Returns
    -------
    tuple[str, Path]
        (profile_name, profile_directory_path)

    Raises
    ------
    ValueError
        If validation fails (the error list is included in the message).
    FileNotFoundError
        If the zip doesn't exist.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    # Step 1: Validate
    logger.info("Validating %s …", zip_path.name)
    validation = validate_profile_zip(zip_path)
    if not validation.is_valid:
        error_summary = "; ".join(validation.errors[:5])  # first 5 errors
        if len(validation.errors) > 5:
            error_summary += f" … and {len(validation.errors) - 5} more"
        raise ValueError(
            f"Profile validation failed: {error_summary}"
        )

    for warning in validation.warnings:
        logger.warning("Validation warning: %s", warning)

    # Step 2: Determine profile name from manifest
    profile_name = _determine_profile_name(validation.manifest)

    # Step 3: Extract to temp dir, then copy to profiles dir
    with tempfile.TemporaryDirectory(prefix="gameai_import_") as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)

        # Resolve target directory
        target_dir = PROFILES_DIR / profile_name
        if target_dir.exists() and not overwrite:
            # Append a unique suffix
            suffix = uuid.uuid4().hex[:8]
            new_name = f"{profile_name}_{suffix}"
            target_dir = PROFILES_DIR / new_name
            logger.info(
                "Profile '%s' already exists — importing as '%s'",
                profile_name,
                new_name,
            )
            profile_name = new_name

        # Copy files
        target_dir.mkdir(parents=True, exist_ok=True)
        files_copied = 0
        for fpath in tmp.rglob("*"):
            if fpath.is_file():
                rel = fpath.relative_to(tmp)
                dest = target_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fpath, dest)
                files_copied += 1

        logger.info(
            "Imported profile '%s' → %s (%d files)",
            profile_name,
            target_dir,
            files_copied,
        )

    return profile_name, target_dir


def _determine_profile_name(manifest: dict[str, Any] | None) -> str:
    """Extract a safe profile name from the manifest."""
    if manifest is None:
        return "imported_profile"

    raw = manifest.get("profile_name", "")
    if not raw:
        raw = manifest.get("target_game", "imported_profile")

    # Sanitise: replace anything not alphanumeric, dash, underscore, dot
    safe = re.sub(r"[^\w\-_. ]", "_", str(raw)).strip()
    safe = re.sub(r"\s+", "_", safe)
    if not safe:
        safe = "imported_profile"
    return safe


# ---------------------------------------------------------------------------
# Quick export convenience
# ---------------------------------------------------------------------------


def quick_export(profile_name: str, output_dir: Path | str | None = None) -> Path:
    """Export a profile with sensible defaults (minimal options).

    Parameters
    ----------
    profile_name : str
        Name of the profile in ~/.gameai/profiles/ to export.
    output_dir : Path, str, or None
        Where to save the archive.  Defaults to the current working directory.

    Returns
    -------
    Path
        Absolute path to the created .gameai_profile file.
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)

    opts = ExportOptions(
        profile_name=profile_name,
        version="1.0.0",
        include_state=True,
        include_corrections=True,
        include_readme=True,
    )
    return export_profile(opts, output_dir)
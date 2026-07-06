"""
Region Profile — load, validate, and query screen region definitions.

Phase 2.1: Each region in regions.json now carries a ``role`` field that
maps it to a state schema slot (e.g. ``"health"``). This module loads those
definitions, validates bounds/type/role, and provides grouping by role for
the StateProcessor to use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Valid constants
# ---------------------------------------------------------------------------

_VALID_REGION_TYPES = frozenset({"color_bar", "ocr"})


# ---------------------------------------------------------------------------
# RegionConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionConfig:
    """A single screen region definition loaded from regions.json.

    Attributes:
        name: Unique human‑readable identifier (e.g. ``"hp_bar"``).
        type: ``"color_bar"`` or ``"ocr"``.
        role: Maps to a state schema slot name (falls back to *name*).
        bounds: Screen‑relative bounding box ``(x1, y1, x2, y2)``.
        preprocess: List of preprocessing step names (e.g. ``["grayscale", "upscale(2x)"]``).
        ocr_config: Per‑region OCR overrides (PSM, OEM, whitelist, confidence threshold, etc.).
        calibration: Colour‑bar calibration data (empty/full colours, bar type, etc.).
    """

    name: str
    type: str
    role: str
    bounds: tuple[int, int, int, int]
    preprocess: list[str] = field(default_factory=list)
    ocr_config: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in _VALID_REGION_TYPES:
            raise ValueError(
                f"Region '{self.name}': invalid type '{self.type}'; "
                f"must be one of {sorted(_VALID_REGION_TYPES)}"
            )

        if len(self.bounds) != 4:
            raise ValueError(
                f"Region '{self.name}': bounds must be [x1, y1, x2, y2] (got {len(self.bounds)} values)"
            )

        x1, y1, x2, y2 = self.bounds
        if x1 >= x2 or y1 >= y2:
            raise ValueError(
                f"Region '{self.name}': invalid bounds {self.bounds} — "
                f"x2 must be > x1 and y2 must be > y1"
            )

    @property
    def width(self) -> int:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> int:
        return self.bounds[3] - self.bounds[1]

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a JSON‑compatible dict."""
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "role": self.role,
            "bounds": list(self.bounds),
            "preprocess": list(self.preprocess),
        }
        if self.ocr_config:
            d["ocr"] = self.ocr_config
        if self.calibration:
            d["calibration"] = self.calibration
        return d


# ---------------------------------------------------------------------------
# RegionProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionProfile:
    """A loaded and validated set of region definitions for a game profile.

    Attributes:
        version: Version string from the JSON file.
        regions: List of validated ``RegionConfig`` entries.
    """

    version: str
    regions: list[RegionConfig] = field(default_factory=list)

    def __iter__(self):
        return iter(self.regions)

    def __len__(self) -> int:
        return len(self.regions)

    def __getitem__(self, index: int) -> RegionConfig:
        return self.regions[index]

    @property
    def region_by_name(self) -> dict[str, RegionConfig]:
        """Lookup map from region name → RegionConfig."""
        return {r.name: r for r in self.regions}

    @property
    def role_set(self) -> set[str]:
        """All unique roles across regions."""
        return {r.role for r in self.regions}

    # ------------------------------------------------------------------
    # Grouping by role (used by StateProcessor)
    # ------------------------------------------------------------------

    def by_role(self) -> dict[str, list[RegionConfig]]:
        """Group regions by their ``role`` field.

        Returns:
            ``{role_name: [RegionConfig, ...]}`` — a single role may have
            multiple regions (e.g. a colour bar AND an OCR region both
            with ``role="health"``).
        """
        groups: dict[str, list[RegionConfig]] = {}
        for r in self.regions:
            groups.setdefault(r.role, []).append(r)
        return groups

    def regions_of_type(self, region_type: str) -> list[RegionConfig]:
        """Filter regions by type (``"color_bar"`` or ``"ocr"``)."""
        return [r for r in self.regions if r.type == region_type]

    def colour_bar_regions(self) -> list[RegionConfig]:
        return self.regions_of_type("color_bar")

    def ocr_regions(self) -> list[RegionConfig]:
        return self.regions_of_type("ocr")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Run all validations and return a list of error/warning messages.

        An empty list means the profile is fully valid.
        Errors are prefixed with ``"ERROR:"``, warnings with ``"WARNING:"``.
        """
        messages: list[str] = []

        if not self.regions:
            messages.append("WARNING: RegionProfile has no regions defined")

        seen_names: set[str] = set()
        for r in self.regions:
            if r.name in seen_names:
                messages.append(f"ERROR: Duplicate region name '{r.name}'")
            seen_names.add(r.name)

        return messages

    # ------------------------------------------------------------------
    # Factory / serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegionProfile":
        """Create a RegionProfile from a deserialised JSON dict.

        Raises:
            ValueError: If the profile is structurally invalid.
        """
        version = data.get("version", "1.0.0")
        raw_regions: list[dict[str, Any]] = data.get("regions", [])

        if not isinstance(raw_regions, list):
            raise ValueError("'regions' must be a list")

        regions: list[RegionConfig] = []
        errors: list[str] = []

        for i, reg in enumerate(raw_regions):
            if not isinstance(reg, dict):
                errors.append(f"Region[{i}]: must be an object")
                continue

            name = reg.get("name")
            if not name:
                errors.append(f"Region[{i}]: missing required field 'name'")
                continue

            region_type = reg.get("type")
            if not region_type:
                errors.append(f"Region '{name}': missing required field 'type'")
                continue

            # Role defaults to region name if not explicitly set
            role = reg.get("role", name)

            raw_bounds = reg.get("bounds")
            if not raw_bounds or len(raw_bounds) != 4:
                errors.append(
                    f"Region '{name}': bounds must be [x1, y1, x2, y2] "
                    f"(got {raw_bounds!r})"
                )
                continue

            try:
                bounds = (int(raw_bounds[0]), int(raw_bounds[1]), int(raw_bounds[2]), int(raw_bounds[3]))
            except (TypeError, ValueError):
                errors.append(f"Region '{name}': bounds values must be integers")
                continue

            preprocess = reg.get("preprocess", [])
            if not isinstance(preprocess, list):
                errors.append(f"Region '{name}': 'preprocess' must be a list")
                preprocess = []

            ocr_config = reg.get("ocr", {})
            calibration = reg.get("calibration", {})

            try:
                regions.append(
                    RegionConfig(
                        name=name,
                        type=region_type,
                        role=role,
                        bounds=bounds,
                        preprocess=preprocess,
                        ocr_config=ocr_config,
                        calibration=calibration,
                    )
                )
            except ValueError as exc:
                errors.append(str(exc))

        if errors:
            raise ValueError("Region profile validation errors:\n  " + "\n  ".join(errors))

        profile = cls(version=version, regions=regions)
        profile_warnings = profile.validate()
        for w in profile_warnings:
            if w.startswith("WARNING:"):
                logger.warning(w)
            else:
                logger.error(w)

        return profile

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a JSON‑compatible dict."""
        return {
            "version": self.version,
            "regions": [r.to_dict() for r in self.regions],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_region_profile_from_path(path: str | Path) -> RegionProfile:
    """Convenience: load a RegionProfile from a JSON file path.

    Returns the validated ``RegionProfile``, or raises ``ValueError`` with a
    human‑readable message on failure.
    """
    import json

    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Region profile file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in region profile {path}: {exc}")

    return RegionProfile.from_dict(data)
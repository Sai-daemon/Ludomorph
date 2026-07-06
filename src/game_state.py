"""
GameState & State Schema — dynamic container for extracted game state.

Phase 2.1: Defines the StateSchema (loaded from state_schema.json) and
GameState (a dynamic dict container keyed by slot name). The schema is
fully user-extensible — adding a new state slot requires only editing
state_schema.json and assigning corresponding region roles.
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

_VALID_SLOT_TYPES = frozenset({"numeric", "text", "boolean"})
_VALID_PRIORITIES = frozenset({"color_first", "ocr_first", "color_only", "ocr_only"})


# ---------------------------------------------------------------------------
# StateSlotDefinition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateSlotDefinition:
    """Single slot definition from state_schema.json.

    Attributes:
        type: One of ``"numeric"``, ``"text"``, or ``"boolean"``.
        priority: Fallback priority — ``"color_first"``, ``"ocr_first"``,
                  ``"color_only"``, or ``"ocr_only"``.
    """

    type: str
    priority: str

    def __post_init__(self) -> None:
        if self.type not in _VALID_SLOT_TYPES:
            raise ValueError(
                f"Invalid slot type '{self.type}'; must be one of {sorted(_VALID_SLOT_TYPES)}"
            )
        if self.priority not in _VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{self.priority}'; must be one of {sorted(_VALID_PRIORITIES)}"
            )


# ---------------------------------------------------------------------------
# StateSchema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateSchema:
    """Loaded and validated state schema.

    Attributes:
        schema_version: Version string from the JSON file.
        slots: Mapping of slot name → StateSlotDefinition.
    """

    schema_version: str
    slots: dict[str, StateSlotDefinition] = field(default_factory=dict)

    def __contains__(self, slot_name: str) -> bool:
        return slot_name in self.slots

    def get(self, slot_name: str) -> StateSlotDefinition | None:
        return self.slots.get(slot_name)

    @property
    def slot_names(self) -> list[str]:
        return list(self.slots.keys())

    # ------------------------------------------------------------------
    # Factory / serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateSchema":
        """Create a StateSchema from a deserialised JSON dict.

        Raises:
            ValueError: If the schema is structurally invalid.
        """
        schema_version = data.get("schema_version", "1.0.0")
        raw_slots: dict[str, dict[str, str]] = data.get("slots", {})

        if not isinstance(raw_slots, dict):
            raise ValueError("'slots' must be a dictionary")

        slots: dict[str, StateSlotDefinition] = {}
        errors: list[str] = []

        for name, slot_cfg in raw_slots.items():
            if not isinstance(slot_cfg, dict):
                errors.append(f"Slot '{name}': value must be an object")
                continue

            slot_type = slot_cfg.get("type")
            priority = slot_cfg.get("priority")

            if not slot_type:
                errors.append(f"Slot '{name}': missing required field 'type'")
                continue
            if not priority:
                errors.append(f"Slot '{name}': missing required field 'priority'")
                continue

            try:
                slots[name] = StateSlotDefinition(type=slot_type, priority=priority)
            except ValueError as exc:
                errors.append(f"Slot '{name}': {exc}")

        if errors:
            raise ValueError("State schema validation errors:\n  " + "\n  ".join(errors))

        if not slots:
            logger.warning("StateSchema has no slots defined — agent will have no state awareness")

        return cls(schema_version=schema_version, slots=slots)

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a JSON-compatible dict."""
        return {
            "schema_version": self.schema_version,
            "slots": {
                name: {"type": slot.type, "priority": slot.priority}
                for name, slot in self.slots.items()
            },
        }

    # ------------------------------------------------------------------
    # Validation of regions against schema
    # ------------------------------------------------------------------

    def validate_region_roles(self, region_roles: set[str]) -> list[str]:
        """Return warnings for region roles that don't map to any schema slot.

        This is *non‑fatal* — unknown roles are allowed so users can add
        slots incrementally. The returned list contains human‑readable
        warning messages.
        """
        warnings: list[str] = []
        for role in region_roles:
            if role not in self.slots:
                warnings.append(
                    f"Region role '{role}' has no matching slot in state_schema; "
                    f"its value will still be available as a raw state entry"
                )
        return warnings


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------


class GameState:
    """Dynamic container for extracted game state values.

    A ``GameState`` is keyed by slot names from the ``StateSchema`` (e.g.
    ``"health"``, ``"mana"``, ``"location"``). Values can be:

    * ``float`` / ``int`` — numeric slot (e.g. colour‑bar percentage)
    * ``str`` — text slot (e.g. OCR output)
    * ``bool`` — boolean slot (e.g. enemy presence flag)
    * ``numpy.ndarray`` — raw image ROI fallback (keyed as ``<slot>_raw_bar``)

    Unknown keys (not in the schema) are allowed so that the StateProcessor
    can attach extras like ``spatial_context``, ``raw_bar`` images, etc.

    Usage::

        schema = StateSchema.from_dict(json.load(open("state_schema.json")))
        state = GameState(schema)
        state.set("health", 78.5)
        state.set("location", "Stormwind")
        print(state.to_dict())
    """

    def __init__(self, schema: StateSchema) -> None:
        self._schema = schema
        self._data: dict[str, Any] = {}
        self._known_slots: set[str] = set(schema.slots.keys())

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, slot_name: str, default: Any = None) -> Any:
        """Return the value for *slot_name*, or *default* if not set."""
        return self._data.get(slot_name, default)

    def get_schema_slot(self, slot_name: str) -> StateSlotDefinition | None:
        """Return the schema definition for a slot, if known."""
        return self._schema.get(slot_name)

    def set(self, slot_name: str, value: Any) -> None:
        """Set a state value.

        Setting a key that exists in the schema does **not** enforce type
        coercion — the StateProcessor is responsible for producing correct
        types. Unknown keys are accepted silently.
        """
        self._data[slot_name] = value

    def __contains__(self, slot_name: str) -> bool:
        return slot_name in self._data

    def __getitem__(self, slot_name: str) -> Any:
        return self._data[slot_name]

    def __setitem__(self, slot_name: str, value: Any) -> None:
        self.set(slot_name, value)

    def __delitem__(self, slot_name: str) -> None:
        del self._data[slot_name]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def schema(self) -> StateSchema:
        return self._schema

    @property
    def known_slots(self) -> set[str]:
        """Set of slot names that exist in the schema (may or may not be populated)."""
        return self._known_slots.copy()

    @property
    def populated_slots(self) -> set[str]:
        """Set of slot names that currently have values set."""
        return set(self._data.keys())

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def missing_schema_slots(self) -> set[str]:
        """Schema slots that have not been populated yet."""
        return self._known_slots - self._data.keys()

    # ------------------------------------------------------------------
    # Serialisation (for prompt building, logging, etc.)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the internal data dict."""
        return dict(self._data)

    def schema_typed_dict(self) -> dict[str, Any]:
        """Return only the keys that are defined in the schema (skipping extras).

        Useful for building LLM prompts where you only want known state fields.
        """
        return {k: v for k, v in self._data.items() if k in self._known_slots}

    def __repr__(self) -> str:
        populated = {k: v for k, v in self._data.items() if k in self._known_slots}
        extras = {k: v for k, v in self._data.items() if k not in self._known_slots}
        parts = [f"{k}={v!r}" for k, v in populated.items()]
        if extras:
            parts.append(f"extras={list(extras.keys())}")
        return f"GameState({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_state_schema_from_path(path: str | Path) -> StateSchema:
    """Convenience: load a StateSchema from a JSON file path.

    Returns the validated ``StateSchema``, or raises ``ValueError`` with a
    human‑readable message on failure.
    """
    import json

    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"State schema file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in state schema {path}: {exc}")

    return StateSchema.from_dict(data)
"""
Macro Resolver — Phase 4.3 / Enhanced Phase 6.8

Dynamic macro resolution that turns vision‑based ``dynamic_*`` macro steps
into concrete ``mouse_move`` / ``click`` steps using real‑time object
detections.

Spec references:
- ``architecture.md`` §4.4B — MacroResolver class design
- ``Implementation_Phases.md`` §4.3 — phase definition

Enhanced (Phase 6.8):
- Multiple *selection strategies* beyond nearest‑to‑center.
- ``reference_point`` support (fixed coords or ``"player"`` token).
"""

from __future__ import annotations

import math
import random as _random_mod
from typing import TYPE_CHECKING, Any, Literal

from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.vision_detector import Detection

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Known dynamic step type prefixes & valid selection strategies
# ---------------------------------------------------------------------------

_DYNAMIC_PREFIX: str = "dynamic_"

SelectionStrategy = Literal[
    "nearest_to_center",
    "nearest_to_point",
    "highest_confidence",
    "largest",
    "random",
]

_VALID_STRATEGIES: frozenset[str] = frozenset(
    {"nearest_to_center", "nearest_to_point", "highest_confidence", "largest", "random"}
)

_DEFAULT_STRATEGY: SelectionStrategy = "nearest_to_center"

ReferencePoint = tuple[int | float, int | float] | Literal["player"] | None

# ---------------------------------------------------------------------------
# MacroResolver
# ---------------------------------------------------------------------------


class MacroResolver:
    """Resolves ``dynamic_*`` macro steps into concrete input steps using
    real‑time vision detections.

    Static steps (``key``, ``delay``, ``mouse_move``, ``click``,
    ``type_string``, ``macro``) pass through unchanged.

    Parameters
    ----------
    detections:
        List of ``Detection`` objects from the current frame's
        ``SpatialContext``.  May be ``None`` or empty when vision is
        disabled or no objects were detected — dynamic steps will be
        skipped in that case.
    screen_size:
        ``(width, height)`` of the game window / capture area, used to
        compute distance‑to‑center when multiple detections match the
        requested class.  If ``None``, the highest‑confidence match is
        used instead.
    window_offset:
        ``(left, top)`` of the game window in screen coordinates.
        Added to detection coordinates so the resulting mouse actions
        target the correct absolute screen position.  Defaults to
        ``(0, 0)`` (no offset).
    dpi_scale_factor:
        Multiplier applied to detection coordinates to convert
        logical (capture) pixels to physical (display) pixels.
        Typical values: ``1.0`` (Linux), ``1.25``/``1.5``/``2.0``
        (Windows with display scaling).  Defaults to ``1.0``.
    player_anchor:
        Optional ``(x, y)`` screen‑relative coordinate of the player
        character, used when a dynamic step requests
        ``reference_point: "player"``.  If ``None`` and ``"player"`` is
        requested, the resolver falls back to ``nearest_to_center``.
    """

    def __init__(
        self,
        detections: list[Any] | None,
        screen_size: tuple[int, int] | None = None,
        window_offset: tuple[int, int] | None = None,
        dpi_scale_factor: float = 1.0,
        player_anchor: tuple[int, int] | None = None,
    ) -> None:
        self.detections: list[Any] = detections or []
        self.screen_size: tuple[int, int] | None = screen_size
        self.window_offset: tuple[int, int] = window_offset or (0, 0)
        self.dpi_scale_factor: float = dpi_scale_factor
        self.player_anchor: tuple[int, int] | None = player_anchor
        self._skipped_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, macro_step: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Resolve a single macro step.

        Parameters
        ----------
        macro_step:
            A step dict, e.g. ``{"type": "dynamic_click", "target_class": "person", ...}``.

        Returns
        -------
        list[dict] | None
            * A list of concrete step dicts ready for ``MacroExecutor``.
            * ``None`` if the step should be **skipped** (target class not
              found, or vision data unavailable).

        Examples
        --------
        >>> resolver = MacroResolver(detections=[...])
        >>> resolver.resolve({"type": "dynamic_click", "target_class": "person", "button": "right"})
        [{"type": "mouse_move", "x": 900, "y": 450}, {"type": "click", "button": "right"}]
        >>> resolver.resolve({"type": "key", "key": "w", "hold_ms": 500})
        [{"type": "key", "key": "w", "hold_ms": 500}]
        """
        step_type: str = macro_step.get("type", "")

        # --- Static step: pass through unchanged ---
        if not step_type.startswith(_DYNAMIC_PREFIX):
            return [macro_step]

        # --- Dynamic step: requires vision data ---
        if not self.detections:
            logger.debug(
                "Dynamic step %r skipped — no vision detections available.",
                step_type,
            )
            self._skipped_count += 1
            return None

        target_class: str = macro_step.get("target_class", "")
        if not target_class:
            logger.warning(
                "Dynamic step %r missing 'target_class' key — step skipped.",
                step_type,
            )
            self._skipped_count += 1
            return None

        # Filter detections by target class
        matching = [
            d
            for d in self.detections
            if getattr(d, "class_name", None) == target_class
        ]

        if not matching:
            logger.debug(
                "Dynamic step %r skipped — target class %r not found in "
                "current detections.",
                step_type,
                target_class,
            )
            self._skipped_count += 1
            return None

        # Determine selection strategy & reference point for this step
        strategy: str = macro_step.get("selection_strategy", _DEFAULT_STRATEGY)
        if strategy not in _VALID_STRATEGIES:
            logger.warning(
                "Dynamic step %r has unknown selection_strategy %r — "
                "falling back to %r.",
                step_type,
                strategy,
                _DEFAULT_STRATEGY,
            )
            strategy = _DEFAULT_STRATEGY

        raw_ref: Any = macro_step.get("reference_point", None)
        ref_point = self._resolve_reference_point(raw_ref, strategy)

        # Pick the best match using the configured strategy
        best = self._pick_best(matching, strategy, ref_point)

        # Build concrete steps based on the dynamic action type
        return self._build_concrete_steps(step_type, best, macro_step)

    def resolve_all(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resolve an entire macro (list of step dicts) and flatten the result.

        Steps that resolve to ``None`` (skipped) are silently dropped.
        The total number of skipped dynamic steps is logged at debug level.

        Parameters
        ----------
        steps:
            A full macro action list, potentially mixing static and
            dynamic steps.

        Returns
        -------
        list[dict]
            Flattened list of concrete step dicts.
        """
        self._skipped_count = 0
        resolved: list[dict[str, Any]] = []

        for step in steps:
            result = self.resolve(step)
            if result is not None:
                resolved.extend(result)

        if self._skipped_count > 0:
            logger.debug(
                "Macro resolution skipped %d dynamic step(s) — "
                "target(s) not detected in current frame.",
                self._skipped_count,
            )

        return resolved

    @property
    def skipped_count(self) -> int:
        """Number of dynamic steps skipped during the last ``resolve_all`` call."""
        return self._skipped_count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Helpers — reference point resolution
    # ------------------------------------------------------------------

    def _resolve_reference_point(
        self,
        raw: Any,
        strategy: str,
    ) -> tuple[float, float] | None:
        """Resolve the ``reference_point`` field into concrete ``(x, y)``.

        Accepts:
        - ``{"x": 960, "y": 540}`` (fixed coordinates)
        - ``"player"`` (resolves to ``self.player_anchor`` if set)
        - ``None`` / omitted (no reference point)
        - ``[x, y]`` list format (same as dict)

        Falls back to ``nearest_to_center`` behaviour if ``"player"`` is
        requested but ``player_anchor`` is not configured.

        Returns:
            ``(x, y)`` tuple or ``None``.
        """
        if raw is None:
            return None

        if isinstance(raw, str):
            if raw == "player":
                if self.player_anchor is not None:
                    return (float(self.player_anchor[0]), float(self.player_anchor[1]))
                logger.warning(
                    "Dynamic step requests reference_point='player' but "
                    "player_anchor is not set — falling back to "
                    "nearest_to_center behaviour."
                )
                return None
            logger.warning(
                "Unknown reference_point string %r — ignoring.", raw
            )
            return None

        if isinstance(raw, dict):
            try:
                x = float(raw.get("x", 0))
                y = float(raw.get("y", 0))
                return (x, y)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid reference_point dict %r — ignoring.", raw
                )
                return None

        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            try:
                return (float(raw[0]), float(raw[1]))
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid reference_point list %r — ignoring.", raw
                )
                return None

        logger.warning(
            "Unrecognised reference_point type %s — ignoring.", type(raw).__name__
        )
        return None

    # ------------------------------------------------------------------
    # Helpers — selection
    # ------------------------------------------------------------------

    def _pick_best(
        self,
        matching: list[Any],
        strategy: str,
        ref_point: tuple[float, float] | None,
    ) -> Any:
        """Pick the best detection from *matching* using *strategy*.

        Parameters
        ----------
        matching:
            Non‑empty list of ``Detection`` objects all sharing the same
            ``class_name``.
        strategy:
            One of the ``_VALID_STRATEGIES`` strings.
        ref_point:
            Resolved reference point for ``nearest_to_point``, or
            ``None`` if not applicable.

        Returns
        -------
        The single best ``Detection`` object.
        """
        if len(matching) == 1:
            return matching[0]

        if strategy == "highest_confidence":
            return max(matching, key=lambda d: d.confidence)

        if strategy == "largest":
            return max(matching, key=lambda d: (
                getattr(d, "area", 0),
                d.confidence,
            ))

        if strategy == "random":
            return _random_mod.choice(matching)

        if strategy == "nearest_to_point":
            if ref_point is not None:
                rx, ry = ref_point
                return min(
                    matching,
                    key=lambda d: (
                        self._distance_sq(d.center, (rx, ry)),
                        -d.confidence,
                    ),
                )
            # No usable reference point → fall back to nearest‑to‑center
            logger.debug(
                "nearest_to_point requested but no reference_point "
                "resolved — falling back to nearest_to_center."
            )

        # --- nearest_to_center (default) ---
        if self.screen_size is not None:
            cx = self.screen_size[0] / 2.0
            cy = self.screen_size[1] / 2.0
            return min(
                matching,
                key=lambda d: (
                    self._distance_sq(d.center, (cx, cy)),
                    -d.confidence,
                ),
            )
        # No screen size — fall back to highest confidence
        return max(matching, key=lambda d: d.confidence)

    # ------------------------------------------------------------------
    # Helpers — concrete step building
    # ------------------------------------------------------------------

    def _build_concrete_steps(
        self,
        step_type: str,
        detection: Any,
        macro_step: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Translate a dynamic step type into concrete ``MacroExecutor`` steps.

        Coordinates from the detection (capture‑relative pixels) are
        converted to absolute screen coordinates by applying
        ``dpi_scale_factor`` and ``window_offset``.

        Parameters
        ----------
        step_type:
            The dynamic step type, e.g. ``"dynamic_click"``.
        detection:
            The ``Detection`` object to target.
        macro_step:
            The original dynamic step dict (for optional parameters like
            ``"button"``).

        Returns
        -------
        list[dict]
        """
        capture_x: int = int(detection.center[0])
        capture_y: int = int(detection.center[1])

        physical_x = capture_x * self.dpi_scale_factor
        physical_y = capture_y * self.dpi_scale_factor

        screen_x = int(physical_x + self.window_offset[0])
        screen_y = int(physical_y + self.window_offset[1])

        if step_type == "dynamic_click":
            return [
                {"type": "mouse_move", "x": screen_x, "y": screen_y},
                {"type": "click", "button": macro_step.get("button", "left")},
            ]
        if step_type == "dynamic_move":
            return [
                {"type": "mouse_move", "x": screen_x, "y": screen_y},
            ]

        logger.warning("Unknown dynamic step type %r — step skipped.", step_type)
        return []

    @staticmethod
    def _distance_sq(
        point: tuple[int | float, int | float],
        center: tuple[float, float],
    ) -> float:
        """Squared Euclidean distance (avoiding ``math.sqrt`` for speed)."""
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return dx * dx + dy * dy


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = ["MacroResolver"]

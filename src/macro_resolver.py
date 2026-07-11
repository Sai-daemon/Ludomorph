"""
Macro Resolver — Phase 4.3

Dynamic macro resolution that turns vision‑based ``dynamic_*`` macro steps
into concrete ``mouse_move`` / ``click`` steps using real‑time object
detections.

Spec references:
- ``architecture.md`` §4.4B — MacroResolver class design
- ``Implementation_Phases.md`` §4.3 — phase definition
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.vision_detector import Detection

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Known dynamic step type prefixes
# ---------------------------------------------------------------------------

_DYNAMIC_PREFIX: str = "dynamic_"

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
    """

    def __init__(
        self,
        detections: list[Any] | None,
        screen_size: tuple[int, int] | None = None,
    ) -> None:
        self.detections: list[Any] = detections or []
        self.screen_size: tuple[int, int] | None = screen_size
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

        # Pick the best match: closest to screen center, breaking ties
        # with higher confidence.
        best = self._pick_best(matching)

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

    def _pick_best(self, matching: list[Any]) -> Any:
        """Pick the best detection from *matching*.

        Strategy: closest to screen center first; ties broken by higher
        confidence.  Falls back to highest‑confidence when *screen_size*
        is ``None``.
        """
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

    @staticmethod
    def _build_concrete_steps(
        step_type: str,
        detection: Any,
        macro_step: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Translate a dynamic step type into concrete ``MacroExecutor`` steps.

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
        target_x: int = int(detection.center[0])
        target_y: int = int(detection.center[1])

        if step_type == "dynamic_click":
            return [
                {"type": "mouse_move", "x": target_x, "y": target_y},
                {
                    "type": "click",
                    "button": macro_step.get("button", "left"),
                },
            ]
        if step_type == "dynamic_move":
            return [
                {"type": "mouse_move", "x": target_x, "y": target_y},
            ]

        # Unknown dynamic type — log and skip
        logger.warning(
            "Unknown dynamic step type %r — step skipped.", step_type
        )
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
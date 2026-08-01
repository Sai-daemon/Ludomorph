"""
Shared utility: normalise region bounds from either format.

regions.json uses ``bounds: [x1, y1, x2, y2]`` (two-point).
The calibration overlay *saves* ``bbox: {x, y, width, height}``.
Both the calibration overlay renderer and the live preview overlay
renderer must handle either format transparently.
"""

from __future__ import annotations

from typing import Any


def normalise_region_bounds(region: dict[str, Any]) -> dict[str, int]:
    """Return ``{x, y, width, height}`` regardless of input format.

    Handles:
    * ``bbox: {x, y, width, height}``   (calibration overlay output)
    * ``bounds: [x1, y1, x2, y2]``     (regions.json storage)
    * Missing/invalid → returns ``{"x": 0, "y": 0, "width": 0, "height": 0}``
    """
    bbox = region.get("bbox")
    if isinstance(bbox, dict):
        try:
            return {
                "x": int(bbox.get("x", 0)),
                "y": int(bbox.get("y", 0)),
                "width": max(int(bbox.get("width", 0)), 1),
                "height": max(int(bbox.get("height", 0)), 1),
            }
        except (TypeError, ValueError):
            pass

    bounds = region.get("bounds")
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        try:
            x1, y1, x2, y2 = (int(v) for v in bounds)
            return {
                "x": x1,
                "y": y1,
                "width": max(x2 - x1, 1),
                "height": max(y2 - y1, 1),
            }
        except (TypeError, ValueError):
            pass

    return {"x": 0, "y": 0, "width": 0, "height": 0}
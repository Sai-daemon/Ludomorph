"""
Synthetic Frame Generator — Phase 2.10 Integration Test Support

Creates simulated game screenshots as numpy arrays using OpenCV drawing
primitives. Supports configurable health/mana bar fill percentages, OCR
text placement, and a rough game‑window border for realism.

The generated frames are designed to match the region bounds defined in
``tests/test_profile/regions.json`` (which mirrors the bundled
``config/regions.json``).

Usage::

    from tests.frame_generator import create_simulated_frame
    import numpy as np

    # 78 % health, 92 % mana, with text
    frame: np.ndarray = create_simulated_frame(
        health_pct=78.0,
        mana_pct=92.0,
        health_text="78/100",
        location_text="Dungeon Entrance",
    )
    # frame.shape → (480, 640, 3), dtype=uint8
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default canvas / region constants — must mirror regions.json
# ---------------------------------------------------------------------------

CANVAS_WIDTH: int = 500
CANVAS_HEIGHT: int = 400

# Health bar region: [100, 200, 300, 230]  → 200×30  (x1=100,y1=200,x2=300,y2=230)
HP_BAR_X1: int = 100
HP_BAR_Y1: int = 200
HP_BAR_X2: int = 300
HP_BAR_Y2: int = 230
HP_BAR_H: int = HP_BAR_Y2 - HP_BAR_Y1  # 30
HP_BAR_W: int = HP_BAR_X2 - HP_BAR_X1  # 200

# Health text region: [100, 230, 300, 260]
HP_TEXT_X1: int = 100
HP_TEXT_Y1: int = 230
HP_TEXT_X2: int = 300
HP_TEXT_Y2: int = 260

# Mana bar region: [100, 270, 300, 300]  → 200×30
MP_BAR_X1: int = 100
MP_BAR_Y1: int = 270
MP_BAR_X2: int = 300
MP_BAR_Y2: int = 300
MP_BAR_H: int = MP_BAR_Y2 - MP_BAR_Y1
MP_BAR_W: int = MP_BAR_X2 - MP_BAR_X1

# Mana text region: [100, 300, 300, 330]
MP_TEXT_X1: int = 100
MP_TEXT_Y1: int = 300
MP_TEXT_X2: int = 300
MP_TEXT_Y2: int = 330

# Location text region: [10, 10, 500, 50]
LOC_X1: int = 10
LOC_Y1: int = 10
LOC_X2: int = 500
LOC_Y2: int = 50

# Objective text region: [10, 60, 500, 100]
OBJ_X1: int = 10
OBJ_Y1: int = 60
OBJ_X2: int = 500
OBJ_Y2: int = 100

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_simulated_frame(
    health_pct: float = 78.0,
    mana_pct: float = 92.0,
    health_text: str = "78/100",
    mana_text: str = "92/100",
    location_text: str = "Training Grounds",
    objective_text: str = "Defeat the dummy",
    background_rgb: tuple[int, int, int] = (30, 30, 40),
    bar_fill_hsv: tuple[int, int, int] = (0, 200, 220),  # bright red
    bar_empty_rgb: tuple[int, int, int] = (20, 20, 20),
    mana_fill_hsv: tuple[int, int, int] = (110, 200, 220),  # bright blue
    mana_empty_rgb: tuple[int, int, int] = (20, 20, 20),
    text_color: tuple[int, int, int] = (240, 240, 240),
) -> np.ndarray:
    """Generate a synthetic BGR game frame suitable for testing the full
    perception pipeline.

    Parameters
    ----------
    health_pct : float
        Fill percentage of the health bar (0–100). Clamped internally.
    mana_pct : float
        Fill percentage of the mana bar (0–100). Clamped internally.
    health_text : str
        OCR text rendered in the HP text region (e.g. ``"78/100"``).
    mana_text : str
        OCR text rendered in the MP text region.
    location_text : str
        OCR text rendered in the location region.
    objective_text : str
        OCR text rendered in the objective region.
    background_rgb : tuple[int, int, int]
        RGB background colour of the canvas.  Internally converted to BGR.
    bar_fill_hsv : tuple[int, int, int]
        Hue / Saturation / Value for the health bar fill colour
        **(must be within the HP calibration range defined in
        ``regions.json``)**.
    bar_empty_rgb : tuple[int, int, int]
        RGB colour for the empty portion of bars.
    mana_fill_hsv : tuple[int, int, int]
        HSV colour for the mana bar fill.
    mana_empty_rgb : tuple[int, int, int]
        RGB colour for the empty portion of the mana bar.
    text_color : tuple[int, int, int]
        RGB colour for all OCR text. Converted to BGR internally.

    Returns
    -------
    np.ndarray
        BGR image of shape ``(CANVAS_HEIGHT, CANVAS_WIDTH, 3)``, dtype
        ``uint8``.
    """
    # OpenCV convention: BGR
    bg_bgr = (background_rgb[2], background_rgb[1], background_rgb[0])
    text_bgr = (text_color[2], text_color[1], text_color[0])
    empty_bgr = (bar_empty_rgb[2], bar_empty_rgb[1], bar_empty_rgb[0])
    mana_empty_bgr = (mana_empty_rgb[2], mana_empty_rgb[1], mana_empty_rgb[0])

    # Create canvas
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), bg_bgr, dtype=np.uint8)

    # ---- Draw window border (decorative) ----
    cv2.rectangle(canvas, (2, 2), (CANVAS_WIDTH - 3, CANVAS_HEIGHT - 3),
                  (80, 80, 100), 2)

    # ---- Health bar ----
    health_pct = float(np.clip(health_pct, 0.0, 100.0))
    _draw_horizontal_bar(
        canvas,
        x1=HP_BAR_X1, y1=HP_BAR_Y1,
        x2=HP_BAR_X2, y2=HP_BAR_Y2,
        fill_pct=health_pct,
        fill_hsv=bar_fill_hsv,
        empty_bgr=empty_bgr,
        bar_label="HP",
    )

    # ---- Mana bar ----
    mana_pct = float(np.clip(mana_pct, 0.0, 100.0))
    _draw_horizontal_bar(
        canvas,
        x1=MP_BAR_X1, y1=MP_BAR_Y1,
        x2=MP_BAR_X2, y2=MP_BAR_Y2,
        fill_pct=mana_pct,
        fill_hsv=mana_fill_hsv,
        empty_bgr=mana_empty_bgr,
        bar_label="MP",
    )

    # ---- OCR text regions ----
    _draw_text_region(canvas, HP_TEXT_X1, HP_TEXT_Y1, HP_TEXT_X2, HP_TEXT_Y2,
                      health_text, text_bgr)
    _draw_text_region(canvas, MP_TEXT_X1, MP_TEXT_Y1, MP_TEXT_X2, MP_TEXT_Y2,
                      mana_text, text_bgr)
    _draw_text_region(canvas, LOC_X1, LOC_Y1, LOC_X2, LOC_Y2,
                      location_text, text_bgr, font_scale=0.6)
    _draw_text_region(canvas, OBJ_X1, OBJ_Y1, OBJ_X2, OBJ_Y2,
                      objective_text, text_bgr, font_scale=0.5)

    return canvas


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _draw_horizontal_bar(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fill_pct: float,
    fill_hsv: tuple[int, int, int],
    empty_bgr: tuple[int, int, int],
    bar_label: str = "",
) -> None:
    """Draw a left‑to‑right horizontal colour bar.

    The bar is first filled entirely with *empty_bgr*, then the filled
    portion (leftmost *fill_pct* % of the width) is over‑painted with a
    solid colour derived from *fill_hsv* (converted to BGR).
    """
    h, s, v = fill_hsv
    hsv_colour = np.uint8([[[h, s, v]]])
    fill_bgr = tuple(int(c) for c in cv2.cvtColor(hsv_colour, cv2.COLOR_HSV2BGR)[0, 0])

    # Empty background
    cv2.rectangle(canvas, (x1, y1), (x2, y2), empty_bgr, -1)

    # Filled portion
    fill_width = int((x2 - x1) * fill_pct / 100.0)
    if fill_width > 0:
        cv2.rectangle(
            canvas,
            (x1, y1),
            (x1 + fill_width, y2),
            fill_bgr,
            -1,
        )

    # Thin border
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # Label (small text left of bar)
    if bar_label:
        cv2.putText(
            canvas,
            bar_label,
            (x1 - 35, y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


def _draw_text_region(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text: str,
    color_bgr: tuple[int, int, int],
    font_scale: float = 0.5,
) -> None:
    """Render *text* in the bounding rectangle using OpenCV ``putText``."""
    # Clear the region with a slightly different dark tone so OCR has
    # contrast against the background
    region_dark = (15, 15, 25)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), region_dark, -1)

    # Thin border to make the region visible
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 70), 1)

    # Position text centred vertically within the region
    text_h = int((y2 - y1) * 0.6)
    baseline_y = y1 + text_h + (y2 - y1 - text_h) // 2

    cv2.putText(
        canvas,
        text,
        (x1 + 5, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color_bgr,
        1,
        cv2.LINE_AA,
    )
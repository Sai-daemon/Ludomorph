"""
Live Preview Overlay — real‑time visualisation of the agent's screen
capture, region boundaries, OCR results, and YOLO detections.

Displays:
- Latest captured frame as background
- Region boundary rectangles (colour‑coded by type)
- OCR text results with confidence inside each OCR region
- Bar fill percentages / fill masks on colour bar regions
- Vision/YOLO detection bounding boxes with class labels

Implemented as a **resizable windowed overlay** (NOT full‑screen transparent)
to avoid the hall‑of‑mirrors feedback loop that occurs when a transparent
overlay is rendered over the same game region being captured by mss.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any

import numpy as np

from src.gui.theme import ThemeManager, resolve_font_stack
from src.logging_config import get_logger
from src.utils.region_normalizer import normalise_region_bounds

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _draw_label(draw: Any, pos: tuple[int, int], text: str, font: Any, colour: str) -> None:
    """Draw a text label at *pos* with shadow for readability."""
    x, y = pos
    draw.text((x + 1, y + 1), text, fill="#000000", font=font)
    draw.text((x, y), text, fill=colour, font=font)


# ---------------------------------------------------------------------------
# LivePreviewOverlay
# ---------------------------------------------------------------------------


class LivePreviewOverlay(tk.Toplevel):
    """A resizable, framed Toplevel that renders the latest captured frame
    and overlays region boundaries, OCR text, bar fill levels, and YOLO
    detection boxes.

    Uses a standard title bar so the user can move it away from the game
    capture region.  Positioned at top‑left of the screen by default.

    Parameters
    ----------
    parent : tk.Widget
        The MainWindow that owns this overlay.
    """

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)

        self._tm = ThemeManager()
        p = self._tm.palette
        self._ui_font = resolve_font_stack(p.ui_font)

        self.title("Live Preview — Ludomorph")
        self.configure(bg=p.bg)

        # Default size and position — larger window for better visibility
        self.geometry("1024x768+50+50")
        self.minsize(500, 400)

        self.is_closed: bool = False
        self._parent = parent

        # Latest data
        self._latest_frame: np.ndarray | None = None
        self._latest_regions: list[dict[str, Any]] = []
        self._latest_state: dict[str, Any] = {}
        self._latest_detections: list[Any] = []
        self._latest_action: str | None = None
        self._latest_confidence: float | None = None
        self._latest_cycle_ms: float | None = None
        self._latest_fps: float | None = None
        self._engine_state: str | None = None

        # Build UI
        self._build_ui()

        # Handle close (window manager close button or Escape)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda e: self._on_close())

        # Fonts for PIL drawing
        try:
            from PIL import ImageFont
            self._font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            self._font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        except (OSError, IOError):
            from PIL import ImageFont
            self._font_sm = ImageFont.load_default()
            self._font_lg = ImageFont.load_default()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the overlay layout."""
        p = self._tm.palette

        # Canvas for drawing
        self._canvas = tk.Canvas(self, bg=p.bg, highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar at the bottom
        self._status_frame = tk.Frame(self, bg=p.status_bar_bg, height=28)
        self._status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self._status_frame.pack_propagate(False)

        self._lbl_status = tk.Label(
            self._status_frame,
            text="Engine not running — start the engine to see live preview.",
            bg=p.status_bar_bg,
            fg=p.fg,
            font=(self._ui_font, 9),
            anchor=tk.W,
            padx=8,
        )
        self._lbl_status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._lbl_fps = tk.Label(
            self._status_frame,
            text="",
            bg=p.status_bar_bg,
            fg=p.fg,
            font=(self._ui_font, 9, "bold"),
            relief=tk.FLAT,
            padx=10,
        )
        self._lbl_fps.pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        """Handle overlay close."""
        self.is_closed = True
        if hasattr(self._parent, "_on_live_preview_closed"):
            self._parent._on_live_preview_closed()
        self.destroy()

    def close(self) -> None:
        """Programmatically close the overlay."""
        self.is_closed = True
        self.destroy()

    # ------------------------------------------------------------------
    # Push frame data
    # ------------------------------------------------------------------

    def push_frame(
        self,
        frame: np.ndarray | None = None,
        *,
        regions: list[dict[str, Any]] | None = None,
        state_data: dict[str, Any] | None = None,
        detections: list[Any] | None = None,
        last_action: str | None = None,
        action_confidence: float | None = None,
        cycle_time_ms: float | None = None,
        fps: float | None = None,
        engine_state: str | None = None,
    ) -> None:
        """Receive the latest data and trigger a redraw."""
        if self.is_closed:
            return
        if frame is not None:
            self._latest_frame = frame
        if regions is not None:
            self._latest_regions = regions
        if state_data is not None:
            self._latest_state = state_data
        if detections is not None:
            self._latest_detections = detections
        if last_action is not None:
            self._latest_action = last_action
        if action_confidence is not None:
            self._latest_confidence = action_confidence
        if cycle_time_ms is not None:
            self._latest_cycle_ms = cycle_time_ms
        if fps is not None:
            self._latest_fps = fps
        if engine_state is not None:
            self._engine_state = engine_state

        self._redraw()

    # ------------------------------------------------------------------
    # Redraw
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        """Render the latest frame and overlays onto the canvas."""
        if self.is_closed:
            return

        p = self._tm.palette

        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 600

        # If no frame yet, show a helpful message on the canvas
        if self._latest_frame is None:
            self._canvas.delete("all")
            self._canvas.create_text(
                cw // 2, ch // 2,
                text="No frame data yet.\nStart the engine to begin capturing.",
                fill=p.disabled_fg,
                font=(self._ui_font, 14),
                justify=tk.CENTER,
                anchor=tk.CENTER,
            )
            self._update_status_line(frames_available=False)
            return

        # Convert frame to PIL Image for drawing
        from PIL import Image, ImageDraw, ImageTk
        frame_bgr = self._latest_frame
        if len(frame_bgr.shape) == 2:
            frame_rgb = np.stack([frame_bgr] * 3, axis=-1)
        elif frame_bgr.shape[2] == 3:
            frame_rgb = frame_bgr[:, :, ::-1]  # BGR → RGB
        else:
            frame_rgb = frame_bgr

        h, w = frame_rgb.shape[:2]

        # Scale to fit canvas (never upscale beyond native size)
        scale = min(cw / w, ch / h, 1.0)
        disp_w = int(w * scale)
        disp_h = int(h * scale)

        # Centre the frame on canvas
        offset_x = (cw - disp_w) // 2
        offset_y = (ch - disp_h) // 2

        # Resize frame
        pil_img = Image.fromarray(frame_rgb)
        if scale != 1.0:
            pil_img = pil_img.resize((disp_w, disp_h), Image.LANCZOS)  # type: ignore[attr-defined]

        draw = ImageDraw.Draw(pil_img)

        # -- Draw region rectangles ------------------------------------------
        for region in self._latest_regions:
            rtype = region.get("type", "ocr")
            name = region.get("name", "")
            nb = normalise_region_bounds(region)  # {x, y, width, height}
            rx = nb["x"]
            ry = nb["y"]
            rw = nb["width"]
            rh = nb["height"]
            dx1 = int(rx * scale)
            dy1 = int(ry * scale)
            dx2 = int((rx + rw) * scale)
            dy2 = int((ry + rh) * scale)

            if rtype == "ocr":
                colour = p.ocr_colour
                draw.rectangle((dx1, dy1, dx2, dy2), outline=colour, width=2)
                state_slots = self._latest_state.get("slots", {})
                slot = next((s for s in state_slots.values() if s.get("region_name") == name), None)
                if slot:
                    ocr_text = slot.get("text", "")
                    conf = slot.get("confidence", 0)
                    if ocr_text:
                        display_text = ocr_text[:30] + ("…" if len(ocr_text) > 30 else "")
                        _draw_label(draw, (dx1, dy1 - 14), display_text, self._font_sm, colour)
                    if conf > 0:
                        _draw_label(draw, (dx1, dy2 + 2), f"{name} ({conf:.0%})", self._font_sm, colour)
                    else:
                        _draw_label(draw, (dx1, dy2 + 2), name, self._font_sm, colour)
                else:
                    _draw_label(draw, (dx1, dy2 + 2), name, self._font_sm, colour)

            elif rtype == "color_bar":
                colour = p.colour_bar_colour
                draw.rectangle((dx1, dy1, dx2, dy2), outline=colour, width=2)
                state_slots = self._latest_state.get("slots", {})
                slot = next((s for s in state_slots.values() if s.get("region_name") == name), None)
                if slot:
                    fill_val = slot.get("fill_percent")
                    if fill_val is not None:
                        label = f"{name}: {fill_val}"
                        _draw_label(draw, (dx1, dy1 - 14), label, self._font_sm, colour)
                    else:
                        _draw_label(draw, (dx1, dy2 + 2), name, self._font_sm, colour)
                else:
                    _draw_label(draw, (dx1, dy2 + 2), name, self._font_sm, colour)

            else:
                colour = "#888888"
                draw.rectangle((dx1, dy1, dx2, dy2), outline=colour, width=1)
                _draw_label(draw, (dx1, dy2 + 2), name, self._font_sm, colour)

        # -- Draw YOLO detections -------------------------------------------
        for det in self._latest_detections:
            bbox = det.get("bbox", {})
            label = det.get("label", "object")
            conf = det.get("confidence", 0)
            dx1 = int(bbox.get("x", 0) * scale)
            dy1 = int(bbox.get("y", 0) * scale)
            dx2 = int((bbox.get("x", 0) + bbox.get("width", 0)) * scale)
            dy2 = int((bbox.get("y", 0) + bbox.get("height", 0)) * scale)
            det_label = f"{label} ({conf:.0%})"
            draw.rectangle((dx1, dy1, dx2, dy2), outline=p.detection_colour, width=2)
            _draw_label(draw, (dx1, dy1 - 14), det_label, self._font_sm, p.detection_colour)

        # Convert back to PhotoImage for tkinter
        tk_img = ImageTk.PhotoImage(pil_img)

        self._canvas.delete("all")
        self._canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=tk_img)
        self._canvas._tk_img = tk_img  # keep a reference

        # -- Update status line ---------------------------------------------
        self._update_status_line(frames_available=True)

    def _update_status_line(self, frames_available: bool) -> None:
        """Update the status bar text at the bottom."""
        p = self._tm.palette

        if not frames_available:
            if self._engine_state == "running":
                self._lbl_status.config(
                    text="Engine running — waiting for frame data …",
                    fg=p.warning,
                )
            elif self._engine_state == "paused":
                self._lbl_status.config(
                    text="Engine paused — no frames being captured.",
                    fg=p.warning,
                )
            elif self._engine_state == "idle":
                self._lbl_status.config(
                    text="Engine idle — start the engine to see live preview.",
                    fg=p.disabled_fg,
                )
            else:
                self._lbl_status.config(
                    text="Engine not running — start the engine to see live preview.",
                    fg=p.disabled_fg,
                )
            self._lbl_fps.config(text="")
            return

        parts: list[str] = []
        if self._latest_action:
            parts.append(f"Last: {self._latest_action}")
        if self._latest_confidence is not None:
            parts.append(f"Conf: {self._latest_confidence:.0%}")
        if self._latest_cycle_ms is not None:
            parts.append(f"Cycle: {self._latest_cycle_ms:.0f}ms")
        if self._latest_fps is not None:
            parts.append(f"FPS: {self._latest_fps:.0f}")
            self._lbl_fps.config(text=f"FPS: {self._latest_fps:.0f}")
        if self._engine_state:
            parts.insert(0, f"Engine: {self._engine_state}")

        self._lbl_status.config(
            text=" | ".join(parts) if parts else "Receiving frame data …",
            fg=p.fg,
        )
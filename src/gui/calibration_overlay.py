"""
Phase 5.2 / 5.3 — Calibration Overlay.

A resizable, fully‑opaque Toplevel window that lets the user draw rectangles
over a game screenshot and assign region roles (OCR, colour bar) with metadata.
Also supports Phase 5.3 colour‑bar calibration via "Capture Empty / Full"
buttons and automatic HSV threshold computation.

Uses a standard title bar so the user can move/resize the window freely.
All controls are rendered on an opaque background for maximum readability.

Uses the centralised theme system from ``src.gui.theme``.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any
from enum import Enum

import numpy as np
from PIL import Image, ImageTk

from src.gui.theme import ThemeManager, resolve_font_stack
from src.logging_config import get_logger
from src.utils.region_normalizer import normalise_region_bounds

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# RegionRole
# ---------------------------------------------------------------------------


class RegionRole(str, Enum):
    OCR = "ocr"
    COLOUR_BAR = "color_bar"


# ---------------------------------------------------------------------------
# CalibrationTool
# ---------------------------------------------------------------------------


class CalibrationTool(tk.Toplevel):
    """Resizable window for game‑state region calibration.

    Parameters
    ----------
    parent : tk.Widget
    screenshot : np.ndarray
        BGR screenshot of the game (used as canvas background).
    ocr_module : optional
        OCR module for live preview.
    existing_regions : list[dict]
        Previously saved regions to display.
    state_schema_slots : dict
        State schema slot names → metadata for the role dropdown.
    on_save : callable
        Called with ``list[dict]`` when user clicks Finish.
    screen_capture : callable or None
        Zero‑argument callable that returns a fresh screenshot (Phase 5.3).
    """

    def __init__(
        self,
        parent: tk.Widget,
        screenshot: np.ndarray,
        ocr_module: Any = None,
        existing_regions: list[dict[str, Any]] | None = None,
        state_schema_slots: dict[str, dict[str, str]] | None = None,
        on_save: Any = None,
        screen_capture: Any = None,
    ) -> None:
        super().__init__(parent)

        self._tm = ThemeManager()
        p = self._tm.palette
        self._ui_font = resolve_font_stack(p.ui_font)
        self._mono_font = resolve_font_stack(p.mono_font)

        self.ocr_module = ocr_module
        self._collected_regions: list[dict[str, Any]] = existing_regions or []
        self._state_schema_slots = state_schema_slots or {}
        self._on_save = on_save
        self._screen_capture = screen_capture

        # Window setup — standard framed, opaque, resizable
        self.title("Region Calibration — Ludomorph")
        self.configure(bg=p.bg)
        self.geometry("1200x800+50+50")
        self.minsize(800, 500)

        # Current drag state
        self._start_x: int = 0
        self._start_y: int = 0
        self._rect_id: int | None = None
        self._drawing: bool = False

        # Scale factor for mapping canvas coords → real screenshot coords
        self._scale: float = 1.0

        # Colour‑bar calibration state (Phase 5.3)
        self._empty_frame: np.ndarray | None = None
        self._full_frame: np.ndarray | None = None
        self._selected_for_cal: dict[str, Any] | None = None
        self._empty_ok: bool = False
        self._full_ok: bool = False

        self._build_ui(screenshot)

        # Bind escape to close
        self.bind("<Escape>", lambda e: self._on_finish())

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self, screenshot: np.ndarray) -> None:
        """Build the layout: scrollable screenshot canvas + control panel."""
        p = self._tm.palette

        h, w = screenshot.shape[:2]

        # Canvas fills most of the window with a frame
        canvas_frame = tk.Frame(self, bg=p.bg, highlightthickness=0)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=(2, 0))

        self._canvas = tk.Canvas(
            canvas_frame,
            bg=p.bg, highlightthickness=0, cursor="cross",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Convert screenshot to PhotoImage at native resolution
        pil_img = Image.fromarray(screenshot[:, :, ::-1])  # BGR → RGB
        self._original_w = w
        self._original_h = h
        self._pil_original = pil_img  # keep full-res for rescaling
        self._tk_img = ImageTk.PhotoImage(pil_img)

        # Place the image; scaling happens in _redraw_image
        self._canvas_img_id = self._canvas.create_image(
            0, 0, anchor=tk.NW, image=self._tk_img
        )

        # Mouse bindings
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        # Control panel at the bottom
        self._build_control_panel()

        # Initial image sizing
        self._redraw_image()

    def _build_control_panel(self) -> None:
        """Build the bottom control bar."""
        p = self._tm.palette

        ctrl = tk.Frame(self, bg=p.bg, padx=8, pady=5)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM)

        # Role dropdown
        role_frame = tk.Frame(ctrl, bg=p.bg)
        role_frame.pack(side=tk.LEFT, padx=4)

        tk.Label(role_frame, text="Role:", bg=p.bg, fg=p.fg,
                 font=(self._ui_font, 9, "bold")).pack(side=tk.LEFT, padx=(0, 2))

        self._role_var = tk.StringVar(value=RegionRole.OCR.value)
        ttk.Combobox(
            role_frame, textvariable=self._role_var,
            values=[r.value for r in RegionRole], state="readonly", width=10,
        ).pack(side=tk.LEFT)

        # Region name entry
        name_frame = tk.Frame(ctrl, bg=p.bg)
        name_frame.pack(side=tk.LEFT, padx=4)

        tk.Label(name_frame, text="Name:", bg=p.bg, fg=p.fg,
                 font=(self._ui_font, 9, "bold")).pack(side=tk.LEFT, padx=(0, 2))

        self._name_var = tk.StringVar()
        tk.Entry(
            name_frame, textvariable=self._name_var,
            bg=p.entry_bg, fg=p.entry_fg, insertbackground=p.entry_insert,
            font=(self._ui_font, 9), width=16, relief=tk.FLAT,
        ).pack(side=tk.LEFT)

        # Bar type / orientation
        type_frame = tk.Frame(ctrl, bg=p.bg)
        type_frame.pack(side=tk.LEFT, padx=4)

        tk.Label(type_frame, text="Type:", bg=p.bg, fg=p.fg,
                 font=(self._ui_font, 9, "bold")).pack(side=tk.LEFT, padx=(0, 2))

        self._type_var = tk.StringVar(value="health")
        ttk.Combobox(
            type_frame, textvariable=self._type_var,
            values=["health", "mana", "stamina", "experience", "other"],
            state="readonly", width=10,
        ).pack(side=tk.LEFT)

        # Orientation
        ori_frame = tk.Frame(ctrl, bg=p.bg)
        ori_frame.pack(side=tk.LEFT, padx=4)

        tk.Label(ori_frame, text="Dir:", bg=p.bg, fg=p.fg,
                 font=(self._ui_font, 8, "bold")).pack(side=tk.LEFT, padx=(0, 1))

        self._ori_var = tk.StringVar(value="horizontal")
        ttk.Combobox(
            ori_frame, textvariable=self._ori_var,
            values=["horizontal", "vertical", "radial"], state="readonly", width=10,
        ).pack(side=tk.LEFT)

        # Capture Empty / Full buttons
        self._btn_capture_empty = tk.Button(
            ctrl, text="Capture Empty", command=self._on_capture_empty,
            bg=p.warning, fg="#1E1E1E", font=(self._ui_font, 9, "bold"),
            relief=tk.FLAT, state=tk.DISABLED,
        )
        self._btn_capture_empty.pack(side=tk.LEFT, padx=4)

        self._btn_capture_full = tk.Button(
            ctrl, text="Capture Full", command=self._on_capture_full,
            bg=p.warning, fg="#1E1E1E", font=(self._ui_font, 9, "bold"),
            relief=tk.FLAT, state=tk.DISABLED,
        )
        self._btn_capture_full.pack(side=tk.LEFT, padx=4)

        # Preview Mask
        self._btn_preview_mask = tk.Button(
            ctrl, text="Preview Mask", command=self._on_preview_mask,
            bg=p.accent, fg="white", font=(self._ui_font, 9, "bold"),
            relief=tk.FLAT, state=tk.DISABLED,
        )
        self._btn_preview_mask.pack(side=tk.LEFT, padx=4)

        # Save Region
        self._btn_save = tk.Button(
            ctrl, text="Save Region", command=self._on_save_region,
            bg=p.success, fg="white", font=(self._ui_font, 9, "bold"),
            relief=tk.FLAT, state=tk.DISABLED,
        )
        self._btn_save.pack(side=tk.LEFT, padx=4)

        # Delete Region
        self._btn_delete = tk.Button(
            ctrl, text="Delete Selected", command=self._on_delete,
            bg=p.danger, fg="white", font=(self._ui_font, 9, "bold"),
            relief=tk.FLAT, state=tk.DISABLED,
        )
        self._btn_delete.pack(side=tk.LEFT, padx=4)

        # Spacer
        tk.Frame(ctrl, bg=p.bg).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Region count
        self._lbl_count = tk.Label(
            ctrl, text=f"Regions: {len(self._collected_regions)}",
            bg=p.bg, fg=p.success, font=(self._mono_font, 9),
        )
        self._lbl_count.pack(side=tk.RIGHT, padx=4)

        # Preview text
        self._preview_text = tk.StringVar(value="Draw a rectangle to begin.")
        tk.Label(
            ctrl, textvariable=self._preview_text,
            bg=p.bg, fg=p.fg,
            font=(self._ui_font, 9),
        ).pack(side=tk.RIGHT, padx=10)

        # Finish button
        tk.Button(
            ctrl, text="Finish", command=self._on_finish,
            bg=p.accent, fg="white", font=(self._ui_font, 9, "bold"),
            relief=tk.FLAT,
        ).pack(side=tk.RIGHT, padx=4)

    # ------------------------------------------------------------------
    # Canvas resize handler
    # ------------------------------------------------------------------

    def _on_canvas_resize(self, event: tk.Event | None = None) -> None:
        """Re‑scale the screenshot to fit the canvas."""
        self._redraw_image()

    def _redraw_image(self) -> None:
        """Scale the screenshot to fit the canvas and update the display.

        The image is centred within the canvas so it is always fully
        visible regardless of aspect‑ratio mismatches.
        """
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 50 or ch < 50:
            # Not yet realized — retry on idle
            self.after(50, self._redraw_image)
            return

        # Compute scale (never upscale beyond native size)
        self._scale = min(cw / self._original_w, ch / self._original_h, 1.0)
        self._img_w = int(self._original_w * self._scale)
        self._img_h = int(self._original_h * self._scale)

        # Resize the PIL image
        pil_scaled = self._pil_original.resize(
            (self._img_w, self._img_h), Image.LANCZOS  # type: ignore[attr-defined]
        )
        self._tk_img = ImageTk.PhotoImage(pil_scaled)

        # Centre the image within the canvas
        self._offset_x = (cw - self._img_w) // 2
        self._offset_y = (ch - self._img_h) // 2
        self._canvas.coords(self._canvas_img_id, self._offset_x, self._offset_y)
        self._canvas.itemconfig(self._canvas_img_id, image=self._tk_img)

        # Redraw saved regions
        self._redraw_saved_regions()

    # ------------------------------------------------------------------
    # Mouse handlers
    # ------------------------------------------------------------------

    def _on_press(self, event: tk.Event) -> None:
        # Check if the click hit a saved-region tag — if so, defer to
        # _on_saved_click and do NOT start a new rectangle draw.
        overlapping = self._canvas.find_overlapping(
            event.x - 1, event.y - 1, event.x + 1, event.y + 1
        )
        for item_id in overlapping:
            tags = self._canvas.gettags(item_id)
            for t in tags:
                if t.startswith("saved_"):
                    return  # handled by tag_bind → _on_saved_click

        self._start_x = event.x
        self._start_y = event.y
        self._drawing = True
        if self._rect_id is not None:
            self._canvas.delete(self._rect_id)
        self._rect_id = self._canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#FFAA00", width=2, dash=(4, 2),
        )

    def _on_drag(self, event: tk.Event) -> None:
        if not self._drawing or self._rect_id is None:
            return
        self._canvas.coords(self._rect_id, self._start_x, self._start_y, event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if not self._drawing:
            return
        self._drawing = False
        self._check_save_enabled()

    # ------------------------------------------------------------------
    # Saved regions
    # ------------------------------------------------------------------

    def _redraw_saved_regions(self) -> None:
        """Redraw all saved region rectangles scaled to current canvas.

        Uses ``normalise_region_bounds`` to accept regions stored in either
        ``bbox: {x, y, width, height}`` or ``bounds: [x1, y1, x2, y2]`` format.
        """
        self._canvas.delete("saved")
        for idx, region in enumerate(self._collected_regions):
            nb = normalise_region_bounds(region)  # {x, y, width, height}
            x1 = int(nb["x"] * self._scale)
            y1 = int(nb["y"] * self._scale)
            x2 = int((nb["x"] + nb["width"]) * self._scale)
            y2 = int((nb["y"] + nb["height"]) * self._scale)
            name = region.get("name", f"Region {idx + 1}")
            selected = region is self._selected_for_cal
            colour = self._tm.palette.selection if selected else self._tm.palette.success
            line_width = 3 if selected else 2
            # Create a unique tag per region for hit-testing on click
            tag = f"saved_{idx}"
            self._canvas.create_rectangle(
                x1, y1, x2, y2, outline=colour, width=line_width,
                tags=(tag, "saved"),
            )
            self._canvas.create_text(
                x1 + 4, y1 + 4, text=name, anchor=tk.NW,
                fill=colour, font=(self._ui_font, 9, "bold"),
                tags=(tag, "saved"),
            )
            # Bind click on each saved region's rectangle for selection
            self._canvas.tag_bind(tag, "<ButtonPress-1>",
                                  lambda e, i=idx: self._on_saved_click(i))

    def _on_saved_click(self, idx: int) -> None:
        """Handle a click on a previously saved region."""
        if idx < 0 or idx >= len(self._collected_regions):
            return
        region = self._collected_regions[idx]
        # Deselect if clicking the already-selected region
        if self._selected_for_cal is region:
            self._selected_for_cal = None
            self._btn_delete.configure(state=tk.DISABLED)
            self._preview_text.set("Selection cleared. Draw a rectangle or click a region.")
        else:
            self._selected_for_cal = region
            name = region.get("name", "unnamed")
            self._btn_delete.configure(state=tk.NORMAL)
            self._preview_text.set(f"Selected '{name}'. Press Delete Selected to remove.")
        self._redraw_saved_regions()

    def _check_save_enabled(self) -> None:
        """Enable/disable Save Region button based on current state."""
        role = self._role_var.get()
        name = self._name_var.get().strip()
        is_custom = name not in self._state_schema_slots

        if role == RegionRole.COLOUR_BAR.value:
            self._check_save_enabled_for_bar()
        elif role and (not is_custom or name):
            self._btn_save.configure(state=tk.NORMAL)
        else:
            self._btn_save.configure(state=tk.DISABLED)

    def _check_save_enabled_for_bar(self) -> None:
        """Enable Save Region if both captures exist."""
        if self._empty_ok and self._full_ok:
            self._btn_save.configure(state=tk.NORMAL)
        else:
            self._btn_save.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Capture Empty / Full (Phase 5.3)
    # ------------------------------------------------------------------

    def _on_capture_empty(self) -> None:
        """Capture an 'empty' frame for colour‑bar calibration."""
        if self._screen_capture is None:
            return
        self.withdraw()
        self.update_idletasks()
        import time as _time
        _time.sleep(0.15)
        try:
            frame = self._screen_capture()
            if frame is not None:
                self._empty_frame = frame
                self._empty_ok = True
        finally:
            self.deiconify()
            self.lift()
            self.focus_force()
        self._update_capture_status()

    def _on_capture_full(self) -> None:
        """Capture a 'full' frame for colour‑bar calibration."""
        if self._screen_capture is None:
            return
        self.withdraw()
        self.update_idletasks()
        import time as _time
        _time.sleep(0.15)
        try:
            frame = self._screen_capture()
            if frame is not None:
                self._full_frame = frame
                self._full_ok = True
        finally:
            self.deiconify()
            self.lift()
            self.focus_force()
        self._update_capture_status()

    def _update_capture_status(self) -> None:
        """Update button colours and labels based on capture state."""
        p = self._tm.palette
        if self._empty_ok:
            self._btn_capture_empty.configure(bg=p.success, fg="#1E1E1E", text="Empty ✓")
        else:
            self._btn_capture_empty.configure(bg=p.warning, fg="#1E1E1E", text="Capture Empty")
        if self._full_ok:
            self._btn_capture_full.configure(bg=p.success, fg="#1E1E1E", text="Full ✓")
        else:
            self._btn_capture_full.configure(bg=p.warning, fg="#1E1E1E", text="Capture Full")
        if self._empty_ok and self._full_ok:
            self._btn_preview_mask.configure(state=tk.NORMAL)
        else:
            self._btn_preview_mask.configure(state=tk.DISABLED)
        self._check_save_enabled_for_bar()

    def _on_preview_mask(self) -> None:
        """Show the HSV mask preview window."""
        if self._empty_frame is None or self._full_frame is None:
            return
        p = self._tm.palette
        preview = tk.Toplevel(self)
        preview.title("HSV Mask Preview")
        preview.configure(bg=p.bg)

        from src.bar_detector import TwoPointCalibrationLoader
        loader = TwoPointCalibrationLoader()
        result = loader.compute(self._empty_frame, self._full_frame)
        msg = f"Lower: {result.get('lower', 'N/A')}\nUpper: {result.get('upper', 'N/A')}"
        tk.Label(
            preview, text=msg, bg=p.bg, fg=p.fg,
            font=(self._mono_font, 9), padx=12, pady=12,
        ).pack()
        tk.Button(
            preview, text="Close", command=preview.destroy,
            bg=p.accent, fg="white", font=(self._ui_font, 9),
        ).pack(pady=(0, 6))

    # ------------------------------------------------------------------
    # Save / Delete / Finish
    # ------------------------------------------------------------------

    def _on_save_region(self) -> None:
        """Save the currently drawn rectangle as a region.

        Converts canvas coordinates back to source‑resolution space using
        the current scale factor.
        """
        if self._rect_id is None:
            return
        role = self._role_var.get()
        name = self._name_var.get().strip()
        coords = self._canvas.coords(self._rect_id)
        x1 = int(coords[0] / self._scale)
        y1 = int(coords[1] / self._scale)
        x2 = int(coords[2] / self._scale)
        y2 = int(coords[3] / self._scale)
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)

        region: dict[str, Any] = {
            "name": name,
            "type": role,
            "bbox": {"x": x1, "y": y1, "width": width, "height": height},
        }
        if role == RegionRole.COLOUR_BAR.value:
            region["bar_type"] = self._type_var.get()
            region["orientation"] = self._ori_var.get()
            if self._empty_frame is not None and self._full_frame is not None:
                region["calibration"] = {
                    "empty_frame": self._empty_frame.tolist(),
                    "full_frame": self._full_frame.tolist(),
                }

        self._collected_regions.append(region)
        self._preview_text.set(f"Saved '{name}' — ({x1}, {y1}, {width}, {height})")
        self._lbl_count.configure(text=f"Regions: {len(self._collected_regions)}")
        self._btn_delete.configure(state=tk.DISABLED)
        self._btn_save.configure(state=tk.DISABLED)
        self._redraw_saved_regions()

    def _on_delete(self) -> None:
        """Delete the currently selected saved region."""
        if self._selected_for_cal is None:
            return
        self._collected_regions.remove(self._selected_for_cal)
        name = self._selected_for_cal.get("name", "unnamed")
        self._selected_for_cal = None
        self._btn_delete.configure(state=tk.DISABLED)
        self._btn_save.configure(state=tk.DISABLED)
        self._lbl_count.configure(text=f"Regions: {len(self._collected_regions)}")
        self._preview_text.set(f"Deleted '{name}'. {len(self._collected_regions)} region(s) remaining.")
        self._redraw_saved_regions()

    def _on_finish(self) -> None:
        """Close the tool and call on_save with collected regions."""
        if self._on_save is not None:
            self._on_save(list(self._collected_regions))
        self.destroy()
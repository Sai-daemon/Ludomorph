"""
Region Calibration Overlay — Phases 5.2 & 5.3

Transparent full‑screen Tkinter overlay for defining screen regions.
Users draw bounding boxes on a game screenshot, assign each region a
**role** (mapped to a state‑schema slot) and a **type** (OCR or Colour
Bar), get a live OCR preview, and save the definitions to regions.json.

Phase 5.3 adds:
- Live screen capture (overlay hides briefly) for empty/full bar captures.
- Delegated HSV threshold computation via ``TwoPointCalibrationLoader``.
- Bar‑type and orientation dropdowns for colour‑bar regions.
- Visual status feedback (button colours, capture‑ready indicators).
- Preview Mask button for verifying calibration quality.

Design follows ``Calibration_UI_research.md`` Problems 2 & 3.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RegionRole(str, Enum):
    """Valid region types (matches regions.json ``type`` field)."""

    OCR = "ocr"
    COLOUR_BAR = "color_bar"


# Human‑readable labels for the dropdown
_ROLE_LABELS: dict[RegionRole, str] = {
    RegionRole.OCR: "OCR (text region)",
    RegionRole.COLOUR_BAR: "Colour Bar (health/mana/etc.)",
}

# Valid bar types and orientations (matches bar_detector constants)
_BAR_TYPES = [
    "solid_horizontal",
    "solid_vertical",
    "gradient",
    "segmented",
    "radial",
]

_ORIENTATIONS = [
    "left_to_right",
    "right_to_left",
    "top_to_bottom",
    "bottom_to_top",
    "radial",
]

# Colour palette
_BG = "#2D2D2D"
_FG = "#D4D4D4"
_ACCENT = "#0078D4"
_SUCCESS = "#50C878"
_DANGER = "#E04040"
_WARNING = "#E8A317"
_AMBER = "#E8A317"
_AMBER_DARK = "#5C3A00"

# ---------------------------------------------------------------------------
# CalibrationTool
# ---------------------------------------------------------------------------


class CalibrationTool(tk.Toplevel):
    """Transparent full‑screen overlay for drawing screen regions.

    Args:
        parent: The root ``AsyncTk`` window.
        screenshot: BGR (H, W, 3) numpy array of the full game window.
        ocr_module: Optional ``OCRModule`` for live OCR preview.
        existing_regions: Previously saved regions to pre‑populate.
        state_schema_slots: Dict of slot‑name → slot‑definition (from
            state_schema.json).  Populates the role dropdown.
        on_save: Optional callback invoked with the collected region list
            when the user clicks "Done".
        screen_capture: Optional async callable that returns a fresh
            BGR screenshot.  Used by Phase 5.3 for live capture of
            empty/full bar states.  Signature: ``async def() -> np.ndarray``.
    """

    def __init__(
        self,
        parent: tk.Tk,
        screenshot: np.ndarray,
        ocr_module: Any = None,
        existing_regions: list[dict[str, Any]] | None = None,
        state_schema_slots: dict[str, dict[str, str]] | None = None,
        on_save: Any = None,
        screen_capture: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self._screenshot = screenshot
        self._ocr_module = ocr_module
        self._on_save = on_save
        self._screen_capture_fn = screen_capture

        # Collected regions — each is a dict ready for regions.json
        self._collected_regions: list[dict[str, Any]] = []
        if existing_regions:
            self._collected_regions = list(existing_regions)

        # Drawing state
        self._dragging: bool = False
        self._start_x: float = 0.0
        self._start_y: float = 0.0
        self._end_x: float = 0.0
        self._end_y: float = 0.0
        self._rect_id: int | None = None

        # Colour‑bar calibration captures (Phase 5.3 — live capture + compute)
        self._empty_capture: np.ndarray | None = None
        self._full_capture: np.ndarray | None = None

        # Widget references for status feedback (Phase 5.3)
        self._btn_capture_empty: tk.Button | None = None
        self._btn_capture_full: tk.Button | None = None
        self._btn_preview_mask: tk.Button | None = None

        # Bar type / orientation overrides (Phase 5.3 dropdowns)
        self._bar_type_var: tk.StringVar | None = None
        self._orientation_var: tk.StringVar | None = None

        # Slot names for the role dropdown
        self._slot_names: list[str] = []
        if state_schema_slots:
            self._slot_names = sorted(state_schema_slots.keys())

        self._setup_ui()
        self._logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._logger.info("Calibration overlay opened.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{sw}x{sh}+0+0")
        self.overrideredirect(True)
        self.attributes("-alpha", 0.75)
        self.attributes("-topmost", True)
        self.configure(bg="black")

        # -- Canvas (the screenshot + drawing surface) ------------------------
        self._canvas = tk.Canvas(
            self, width=sw, height=sh, highlightthickness=0, bg="black"
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Convert BGR screenshot to RGB for PIL
        from PIL import Image, ImageTk

        rgb = cv2.cvtColor(self._screenshot, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Scale screenshot to fill screen while preserving aspect ratio
        img_w, img_h = img.size
        scale = min(sw / img_w, sh / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        self._scale = scale
        self._offset_x = (sw - new_w) // 2
        self._offset_y = (sh - new_h) // 2

        img_resized = img.resize((new_w, new_h), Image.NEAREST)
        self._bg_image = ImageTk.PhotoImage(img_resized)
        self._canvas.create_image(self._offset_x, self._offset_y, image=self._bg_image, anchor=tk.NW)

        # Store dimensions for coordinate translation
        self._img_x = self._offset_x
        self._img_y = self._offset_y
        self._img_w = new_w
        self._img_h = new_h

        # Draw existing regions
        for region in self._collected_regions:
            self._draw_existing_region(region)

        # Bind mouse events
        self._canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self._canvas.bind("<B1-Motion>", self._on_mouse_move)
        self._canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        # -- Floating toolbar -------------------------------------------------
        self._build_toolbar()

        # Keyboard shortcuts
        self.bind("<Escape>", lambda _e: self._on_cancel())
        self.bind("<Return>", lambda _e: self._on_save_region())

    def _build_toolbar(self) -> None:
        """Build the floating bottom toolbar."""
        self._toolbar = tk.Frame(self, bg=_BG, relief=tk.RAISED, borderwidth=1)
        self._toolbar.place(relx=0.5, rely=0.93, anchor=tk.CENTER)

        # -- Role dropdown ----------------------------------------------------
        role_frame = tk.Frame(self._toolbar, bg=_BG)
        role_frame.pack(side=tk.LEFT, padx=4, pady=4)

        tk.Label(role_frame, text="Role:", bg=_BG, fg=_FG, font=("Segoe UI", 9)).pack(
            side=tk.LEFT, padx=(0, 2)
        )

        self._role_var = tk.StringVar(value="")
        role_menu = ttk.Combobox(
            role_frame,
            textvariable=self._role_var,
            values=self._slot_names,
            state="readonly" if self._slot_names else tk.DISABLED,
            width=18,
        )
        if self._slot_names:
            role_menu.current(0)
        role_menu.pack(side=tk.LEFT)
        role_menu.bind("<<ComboboxSelected>>", self._on_role_changed)

        # -- Type dropdown ----------------------------------------------------
        type_frame = tk.Frame(self._toolbar, bg=_BG)
        type_frame.pack(side=tk.LEFT, padx=4, pady=4)

        tk.Label(type_frame, text="Type:", bg=_BG, fg=_FG, font=("Segoe UI", 9)).pack(
            side=tk.LEFT, padx=(0, 2)
        )

        self._type_var = tk.StringVar(value=RegionRole.OCR.value)
        type_menu = ttk.Combobox(
            type_frame,
            textvariable=self._type_var,
            values=[r.value for r in RegionRole],
            state="readonly",
            width=14,
        )
        type_menu.current(0)
        type_menu.pack(side=tk.LEFT)
        type_menu.bind("<<ComboboxSelected>>", self._on_type_changed)

        # -- Preview text -----------------------------------------------------
        self._preview_text = tk.StringVar(value="Draw a rectangle on the screen")
        preview_label = tk.Label(
            self._toolbar,
            textvariable=self._preview_text,
            bg="#3C3C3C",
            fg=_SUCCESS,
            font=("Cascadia Code", 9),
            width=42,
            anchor=tk.W,
            padx=4,
        )
        preview_label.pack(side=tk.LEFT, padx=4, pady=4)

        # -- Dynamic action buttons -------------------------------------------
        self._dynamic_btn_frame = tk.Frame(self._toolbar, bg=_BG)
        self._dynamic_btn_frame.pack(side=tk.LEFT, padx=2, pady=4)

        # Separator
        ttk.Separator(self._toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        # -- Save / Cancel / Done buttons ------------------------------------
        self._btn_save = tk.Button(
            self._toolbar,
            text="Save Region",
            command=self._on_save_region,
            bg=_ACCENT,
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=2,
            state=tk.DISABLED,
        )
        self._btn_save.pack(side=tk.LEFT, padx=2)

        self._btn_cancel = tk.Button(
            self._toolbar,
            text="Cancel",
            command=self._on_cancel,
            bg=_DANGER,
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=2,
        )
        self._btn_cancel.pack(side=tk.LEFT, padx=2)

        self._btn_done = tk.Button(
            self._toolbar,
            text="Done (Save All)",
            command=self._on_done,
            bg=_SUCCESS,
            fg="#1E1E1E",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=2,
        )
        self._btn_done.pack(side=tk.LEFT, padx=2)

        # Region count label
        self._lbl_count = tk.Label(
            self._toolbar,
            text=f"Regions: {len(self._collected_regions)}",
            bg=_BG,
            fg=_FG,
            font=("Segoe UI", 9),
        )
        self._lbl_count.pack(side=tk.LEFT, padx=4)

        # Initial dynamic button state
        self._update_dynamic_buttons()

    # ------------------------------------------------------------------
    # Mouse event handlers
    # ------------------------------------------------------------------

    def _on_mouse_down(self, event: tk.Event) -> None:
        """Start drawing a rectangle."""
        self._dragging = True
        self._start_x = event.x
        self._start_y = event.y
        self._end_x = event.x
        self._end_y = event.y

        # Remove previous temporary rectangle
        if self._rect_id is not None:
            self._canvas.delete(self._rect_id)

        self._rect_id = self._canvas.create_rectangle(
            self._start_x,
            self._start_y,
            self._end_x,
            self._end_y,
            outline="#FF4444",
            width=2,
            dash=(4, 2),
            tag="selection",
        )

    def _on_mouse_move(self, event: tk.Event) -> None:
        """Update rectangle while dragging."""
        if not self._dragging or self._rect_id is None:
            return
        self._end_x = event.x
        self._end_y = event.y
        self._canvas.coords(self._rect_id, self._start_x, self._start_y, self._end_x, self._end_y)

    def _on_mouse_up(self, event: tk.Event) -> None:
        """Finalise the rectangle."""
        self._dragging = False
        self._end_x = event.x
        self._end_y = event.y

        width = abs(self._end_x - self._start_x)
        height = abs(self._end_y - self._start_y)

        if width < 5 or height < 5:
            # Too small — discard
            if self._rect_id is not None:
                self._canvas.delete(self._rect_id)
                self._rect_id = None
            self._preview_text.set("Rectangle too small — redraw")
            return

        # Enable Save button (role must also be set)
        role = self._role_var.get()
        if role:
            self._btn_save.configure(state=tk.NORMAL)

        self._update_preview_from_rect()

    # ------------------------------------------------------------------
    # Toolbar callbacks
    # ------------------------------------------------------------------

    def _on_role_changed(self, event: tk.Event) -> None:
        """Called when the role dropdown selection changes."""
        self._check_save_enabled()

    def _on_type_changed(self, event: tk.Event) -> None:
        """Called when the type dropdown selection changes."""
        self._update_dynamic_buttons()
        self._update_preview_from_rect()

    def _update_dynamic_buttons(self) -> None:
        """Replace dynamic action buttons based on the selected type.

        Phase 5.3: Colour Bar type now includes bar‑type & orientation
        dropdowns, capture buttons with status feedback, and a Preview
        Mask button.
        """
        for w in self._dynamic_btn_frame.winfo_children():
            w.destroy()

        # Reset references
        self._btn_capture_empty = None
        self._btn_capture_full = None
        self._btn_preview_mask = None
        self._bar_type_var = None
        self._orientation_var = None

        region_type = self._type_var.get()
        if region_type == RegionRole.OCR.value:
            btn = tk.Button(
                self._dynamic_btn_frame,
                text="Test OCR",
                command=self._test_ocr,
                bg=_ACCENT,
                fg="white",
                font=("Segoe UI", 9),
                relief=tk.FLAT,
                padx=10,
                pady=2,
            )
            btn.pack(side=tk.LEFT)
        elif region_type == RegionRole.COLOUR_BAR.value:
            # -- Bar type dropdown ------------------------------------------------
            bar_type_frame = tk.Frame(self._dynamic_btn_frame, bg=_BG)
            bar_type_frame.pack(side=tk.LEFT, padx=2)

            tk.Label(bar_type_frame, text="Bar:", bg=_BG, fg=_FG, font=("Segoe UI", 8)).pack(
                side=tk.LEFT, padx=(0, 1)
            )
            self._bar_type_var = tk.StringVar(value=_BAR_TYPES[0])
            bar_type_menu = ttk.Combobox(
                bar_type_frame,
                textvariable=self._bar_type_var,
                values=_BAR_TYPES,
                state="readonly",
                width=14,
            )
            bar_type_menu.pack(side=tk.LEFT)

            # -- Orientation dropdown ---------------------------------------------
            ori_frame = tk.Frame(self._dynamic_btn_frame, bg=_BG)
            ori_frame.pack(side=tk.LEFT, padx=2)

            tk.Label(ori_frame, text="Dir:", bg=_BG, fg=_FG, font=("Segoe UI", 8)).pack(
                side=tk.LEFT, padx=(0, 1)
            )
            self._orientation_var = tk.StringVar(value=_ORIENTATIONS[0])
            ori_menu = ttk.Combobox(
                ori_frame,
                textvariable=self._orientation_var,
                values=_ORIENTATIONS,
                state="readonly",
                width=13,
            )
            ori_menu.pack(side=tk.LEFT)

            # -- Capture Empty button ----------------------------------------------
            self._btn_capture_empty = tk.Button(
                self._dynamic_btn_frame,
                text="Capture Empty",
                command=self._on_capture_empty,
                bg=_AMBER,
                fg="#1E1E1E",
                font=("Segoe UI", 9),
                relief=tk.FLAT,
                padx=8,
                pady=2,
            )
            self._btn_capture_empty.pack(side=tk.LEFT, padx=1)

            # -- Capture Full button -----------------------------------------------
            self._btn_capture_full = tk.Button(
                self._dynamic_btn_frame,
                text="Capture Full",
                command=self._on_capture_full,
                bg=_AMBER,
                fg="#1E1E1E",
                font=("Segoe UI", 9),
                relief=tk.FLAT,
                padx=8,
                pady=2,
            )
            self._btn_capture_full.pack(side=tk.LEFT, padx=1)

            # -- Preview Mask button (disabled until both captures exist) ----------
            self._btn_preview_mask = tk.Button(
                self._dynamic_btn_frame,
                text="Preview Mask",
                command=self._on_preview_mask,
                bg="#6A5ACD",
                fg="white",
                font=("Segoe UI", 9),
                relief=tk.FLAT,
                padx=8,
                pady=2,
                state=tk.DISABLED,
            )
            self._btn_preview_mask.pack(side=tk.LEFT, padx=1)

            # Reflect any existing captures
            self._update_capture_status()

    # ------------------------------------------------------------------
    # Region save
    # ------------------------------------------------------------------

    def _on_save_region(self) -> None:
        """Save the currently drawn rectangle as a region definition."""
        role = self._role_var.get().strip()
        region_type = self._type_var.get()

        if not role:
            self._preview_text.set("ERROR: Select a role first")
            return
        if not region_type:
            self._preview_text.set("ERROR: Select a type first")
            return

        # Compute screen‑relative bounds from the selection
        bounds = self._get_selection_bounds()
        if bounds is None:
            self._preview_text.set("ERROR: No valid rectangle drawn")
            return

        # Convert from display coords to screenshot image coords
        img_bounds = self._display_to_image_bounds(bounds)
        if img_bounds is None:
            self._preview_text.set("ERROR: Rectangle outside image area")
            return

        # Generate a name from the role
        region_name = self._generate_region_name(role)

        region_dict: dict[str, Any] = {
            "name": region_name,
            "type": region_type,
            "role": role,
            "bounds": list(img_bounds),
            "preprocess": self._default_preprocess(region_type),
        }

        if region_type == RegionRole.OCR.value:
            region_dict["ocr"] = {
                "confidence_threshold": 0.6,
                "cache_ttl_seconds": 2.0,
            }
        elif region_type == RegionRole.COLOUR_BAR.value:
            # Build calibration dict — use computed HSV if both captures exist
            if self._empty_capture is not None and self._full_capture is not None:
                bar_type = self._bar_type_var.get() if self._bar_type_var else "solid_horizontal"
                orientation = self._orientation_var.get() if self._orientation_var else "left_to_right"
                calib = self._compute_bar_hsv_thresholds(
                    self._empty_capture,
                    self._full_capture,
                    bar_type=bar_type,
                    orientation=orientation,
                )
                self._logger.info(
                    f"Computed HSV calibration for '{region_name}': "
                    f"bar_type={bar_type}, orientation={orientation}"
                )
            else:
                # Placeholder defaults — user hasn't calibrated yet
                calib = {
                    "enabled": True,
                    "bar_type": self._bar_type_var.get() if self._bar_type_var else "solid_horizontal",
                    "orientation": self._orientation_var.get() if self._orientation_var else "left_to_right",
                    "fill_hsv_lower": [0, 100, 100],
                    "fill_hsv_upper": [10, 255, 255],
                    "empty_hsv_lower": [0, 0, 0],
                    "empty_hsv_upper": [179, 30, 40],
                    "use_fill_mask": True,
                    "method": "projection",
                    "confidence_threshold": 0.6,
                    "dynamic_adjustment": True,
                }
            region_dict["calibration"] = calib

        # Add to collected list (replace existing with same name)
        self._remove_region_by_name(region_name)
        self._collected_regions.append(region_dict)

        # Draw a persistent green rectangle
        x1, y1, x2, y2 = bounds
        self._canvas.create_rectangle(
            x1, y1, x2, y2, outline=_SUCCESS, width=2, tag="saved"
        )

        # Clear selection
        if self._rect_id is not None:
            self._canvas.delete(self._rect_id)
            self._rect_id = None

        self._btn_save.configure(state=tk.DISABLED)
        self._lbl_count.configure(text=f"Regions: {len(self._collected_regions)}")
        self._preview_text.set(f"Saved '{region_name}' — {img_bounds}")

        self._logger.info(
            f"Region saved: name={region_name}, role={role}, type={region_type}, "
            f"bounds={img_bounds}"
        )

        # Reset capture buffers and status
        self._empty_capture = None
        self._full_capture = None
        self._update_capture_status()

    def _on_cancel(self) -> None:
        """Close the overlay without saving."""
        self._logger.info("Calibration overlay cancelled.")
        self.destroy()

    def _on_done(self) -> None:
        """Finish calibration and invoke the save callback."""
        self._logger.info(f"Calibration done — {len(self._collected_regions)} regions collected.")

        if self._on_save is not None:
            self._on_save(self._collected_regions)

        self.destroy()

    # ------------------------------------------------------------------
    # OCR preview
    # ------------------------------------------------------------------

    def _test_ocr(self) -> None:
        """Run OCR on the currently selected rectangle and show result."""
        bounds = self._get_selection_bounds()
        if bounds is None:
            self._preview_text.set("Draw a rectangle first, then test")
            return

        if self._ocr_module is None:
            self._preview_text.set("OCR module not available")
            return

        img_bounds = self._display_to_image_bounds(bounds)
        if img_bounds is None:
            self._preview_text.set("Rectangle outside image area")
            return

        # Crop the screenshot
        x1, y1, x2, y2 = img_bounds
        cropped = self._screenshot[y1:y2, x1:x2]
        if cropped.size == 0:
            self._preview_text.set("Empty crop — redraw")
            return

        self._preview_text.set("Running OCR …")

        # Run OCR synchronously (via thread pool internally)
        import asyncio

        async def _run() -> None:
            from src.region_profile import RegionConfig

            region_config = RegionConfig(
                name="_preview_",
                type="ocr",
                role="preview",
                bounds=img_bounds,
                preprocess=["grayscale"],
            )
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    result = await self._ocr_module.recognize_region(
                        self._screenshot, region_config
                    )
                else:
                    result = await asyncio.ensure_future(
                        self._ocr_module.recognize_region(self._screenshot, region_config)
                    )

                if result.success:
                    self._preview_text.set(
                        f"OCR: '{result.text}' (conf={result.confidence:.2f})"
                    )
                else:
                    self._preview_text.set(
                        f"OCR low confidence: '{result.text}' ({result.confidence:.2f})"
                    )
            except Exception as exc:
                self._preview_text.set(f"OCR error: {exc}")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_run())
            else:
                asyncio.run(_run())
        except RuntimeError:
            # No event loop available — run in a thread
            import threading

            def _run_thread() -> None:
                asyncio.run(_run())

            t = threading.Thread(target=_run_thread, daemon=True)
            t.start()

    # ------------------------------------------------------------------
    # Colour bar capture — live screen capture (Phase 5.3)
    # ------------------------------------------------------------------

    def _on_capture_empty(self) -> None:
        """Capture empty bar — uses live screen capture if available."""
        self._capture_bar_reference(is_empty=True)

    def _on_capture_full(self) -> None:
        """Capture full bar — uses live screen capture if available."""
        self._capture_bar_reference(is_empty=False)

    def _capture_bar_reference(self, *, is_empty: bool) -> None:
        """Grab a screenshot (live or static), crop to selection, and store.

        Phase 5.3: If ``_screen_capture_fn`` is provided, the overlay hides
        briefly, captures a fresh frame, then reapplies the overlay.  This
        lets the user Alt+Tab to the game, change the bar state, then
        Alt+Tab back and click the capture button.
        """
        bounds = self._get_image_selection()
        if bounds is None:
            self._preview_text.set("Draw a rectangle first")
            return

        x1, y1, x2, y2 = bounds

        # Try live capture
        captured_img: np.ndarray | None = None
        if self._screen_capture_fn is not None:
            try:
                captured_img = self._do_live_capture()
            except Exception as exc:
                self._logger.warning(f"Live screen capture failed: {exc}")
                self._preview_text.set(f"Live capture failed: {exc} — using static image")

        # Fall back to the static screenshot
        if captured_img is None:
            captured_img = self._screenshot

        # Crop
        h, w = captured_img.shape[:2]
        ix1 = max(0, min(x1, w))
        iy1 = max(0, min(y1, h))
        ix2 = max(0, min(x2, w))
        iy2 = max(0, min(y2, h))
        crop = captured_img[iy1:iy2, ix1:ix2].copy()

        if is_empty:
            self._empty_capture = crop
        else:
            self._full_capture = crop

        # Update status
        self._update_capture_status()
        self._check_save_enabled_for_bar()
        self._logger.info(
            f"{'Empty' if is_empty else 'Full'} bar captured: "
            f"shape={crop.shape}, live={captured_img is not self._screenshot}"
        )

    def _do_live_capture(self) -> np.ndarray | None:
        """Hide overlay, grab a fresh screenshot, show overlay.

        Works with both sync and async screen_capture callables.
        """
        import asyncio

        # Hide overlay so it doesn't appear in the capture
        self.withdraw()
        self.update_idletasks()  # process the withdraw

        try:
            if self._screen_capture_fn is None:
                return None

            # The capture callable may be async or sync
            result = self._screen_capture_fn()
            if asyncio.iscoroutine(result):
                # Need an event loop
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                captured = loop.run_until_complete(result)
            else:
                captured = result

            if captured is None:
                return None

            # Ensure BGR format
            if len(captured.shape) == 2:
                return cv2.cvtColor(captured, cv2.COLOR_GRAY2BGR)
            if captured.shape[2] == 4:
                return captured[:, :, :3]  # drop alpha
            return captured
        finally:
            # Always restore the overlay
            self.deiconify()
            self.lift()
            self.focus_force()

    # ------------------------------------------------------------------
    # Capture status feedback (Phase 5.3)
    # ------------------------------------------------------------------

    def _update_capture_status(self) -> None:
        """Update button colours and labels based on capture state.

        - Empty captured → button turns green, label shows ✓
        - Full captured → button turns green, label shows ✓
        - Both captured → Preview Mask enabled, Save ready
        """
        empty_ok = self._empty_capture is not None
        full_ok = self._full_capture is not None

        if self._btn_capture_empty is not None:
            if empty_ok:
                self._btn_capture_empty.configure(
                    bg=_SUCCESS, fg="#1E1E1E", text="Empty ✓"
                )
            else:
                self._btn_capture_empty.configure(
                    bg=_AMBER, fg="#1E1E1E", text="Capture Empty"
                )

        if self._btn_capture_full is not None:
            if full_ok:
                self._btn_capture_full.configure(
                    bg=_SUCCESS, fg="#1E1E1E", text="Full ✓"
                )
            else:
                self._btn_capture_full.configure(
                    bg=_AMBER, fg="#1E1E1E", text="Capture Full"
                )

        # Enable Preview Mask if both exist
        if self._btn_preview_mask is not None:
            if empty_ok and full_ok:
                self._btn_preview_mask.configure(state=tk.NORMAL)
            else:
                self._btn_preview_mask.configure(state=tk.DISABLED)

        # Update preview text
        e_label = "✓" if empty_ok else "—"
        f_label = "✓" if full_ok else "—"
        if empty_ok and full_ok:
            self._preview_text.set(f"Empty: {e_label} | Full: {f_label} — Ready to save ✓")
        elif empty_ok or full_ok:
            self._preview_text.set(f"Empty: {e_label} | Full: {f_label} — capture the other")
        else:
            self._preview_text.set(f"Empty: {e_label} | Full: {f_label} — capture both")

    def _check_save_enabled_for_bar(self) -> None:
        """Enable Save Region if both captures exist (for colour bar type)."""
        if self._type_var.get() != RegionRole.COLOUR_BAR.value:
            return
        if self._empty_capture is not None and self._full_capture is not None:
            role = self._role_var.get()
            if role:
                self._btn_save.configure(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Preview Mask (Phase 5.3)
    # ------------------------------------------------------------------

    def _on_preview_mask(self) -> None:
        """Show a popup with the computed HSV mask applied to the full capture.

        Lets the user verify calibration quality before saving.
        """
        if self._empty_capture is None or self._full_capture is None:
            return

        bar_type = self._bar_type_var.get() if self._bar_type_var else "solid_horizontal"
        orientation = self._orientation_var.get() if self._orientation_var else "left_to_right"
        calib = self._compute_bar_hsv_thresholds(
            self._empty_capture, self._full_capture,
            bar_type=bar_type, orientation=orientation,
        )

        full_img = self._full_capture
        hsv = cv2.cvtColor(full_img, cv2.COLOR_BGR2HSV)

        fill_lower = np.array(calib["fill_hsv_lower"], dtype=np.uint8)
        fill_upper = np.array(calib["fill_hsv_upper"], dtype=np.uint8)
        mask = cv2.inRange(hsv, fill_lower, fill_upper)

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Build a side‑by‑side preview: original | mask | overlay
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = full_img.copy()
        overlay[mask > 0] = (0, 255, 0)  # bright green fill
        blended = cv2.addWeighted(full_img, 0.4, overlay, 0.6, 0)

        # Stack horizontally: [Original | Mask | Overlay]
        stacked = np.hstack([full_img, mask_bgr, blended])
        # Downscale if too wide
        h_s, w_s = stacked.shape[:2]
        max_w = 1200
        if w_s > max_w:
            scale = max_w / w_s
            stacked = cv2.resize(stacked, (int(w_s * scale), int(h_s * scale)))

        # Show in a Toplevel window
        preview = tk.Toplevel(self)
        preview.title("HSV Mask Preview")
        preview.configure(bg="#1E1E1E")

        from PIL import Image, ImageTk
        rgb = cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._mask_preview_image = ImageTk.PhotoImage(img)

        canvas = tk.Canvas(preview, width=img.width, height=img.height, highlightthickness=0, bg="#1E1E1E")
        canvas.pack(padx=4, pady=4)
        canvas.create_image(0, 0, image=self._mask_preview_image, anchor=tk.NW)

        info = tk.Label(
            preview,
            text=(
                f"  Original  |  Fill Mask  |  Overlay  "
                f"  —  bar_type={calib.get('bar_type', '?')}, "
                f"orientation={calib.get('orientation', '?')}"
            ),
            bg="#1E1E1E", fg=_FG, font=("Segoe UI", 9),
        )
        info.pack(pady=(0, 4))

        close_btn = tk.Button(preview, text="Close", command=preview.destroy, bg=_ACCENT, fg="white", font=("Segoe UI", 9))
        close_btn.pack(pady=(0, 6))

        preview.transient(self)
        preview.grab_set()
        self.wait_window(preview)
        self._logger.info("Mask preview closed.")

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _get_selection_bounds(self) -> tuple[int, int, int, int] | None:
        """Return the current selection as (x1, y1, x2, y2) in display coords."""
        x1 = min(self._start_x, self._end_x)
        y1 = min(self._start_y, self._end_y)
        x2 = max(self._start_x, self._end_x)
        y2 = max(self._start_y, self._end_y)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        return (x1, y1, x2, y2)

    def _display_to_image_bounds(
        self, bounds: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int] | None:
        """Convert display‑coordinate bounds to screenshot image coordinates."""
        x1, y1, x2, y2 = bounds

        # Convert to image‑relative coords
        ix1 = int((x1 - self._img_x) / self._scale)
        iy1 = int((y1 - self._img_y) / self._scale)
        ix2 = int((x2 - self._img_x) / self._scale)
        iy2 = int((y2 - self._img_y) / self._scale)

        # Clamp to image dimensions
        h, w = self._screenshot.shape[:2]
        ix1 = max(0, min(ix1, w))
        iy1 = max(0, min(iy1, h))
        ix2 = max(0, min(ix2, w))
        iy2 = max(0, min(iy2, h))

        if ix1 >= ix2 or iy1 >= iy2:
            return None

        return (ix1, iy1, ix2, iy2)

    def _get_image_selection(self) -> tuple[int, int, int, int] | None:
        """Get the current selection in image coordinates."""
        bounds = self._get_selection_bounds()
        if bounds is None:
            return None
        return self._display_to_image_bounds(bounds)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_save_enabled(self) -> None:
        """Enable Save if we have a valid rect AND a role."""
        role = self._role_var.get()
        bounds = self._get_selection_bounds()
        if role and bounds is not None:
            self._btn_save.configure(state=tk.NORMAL)
        else:
            self._btn_save.configure(state=tk.DISABLED)

    def _update_preview_from_rect(self) -> None:
        """Show coordinates of the current selection in the preview label."""
        bounds = self._get_selection_bounds()
        if bounds is None:
            return
        x1, y1, x2, y2 = bounds
        w = x2 - x1
        h = y2 - y1
        img_bounds = self._display_to_image_bounds(bounds)
        if img_bounds:
            ix1, iy1, ix2, iy2 = img_bounds
            self._preview_text.set(
                f"Selection: ({x1},{y1}) {w}x{h}px "
                f"→ Image: [{ix1}, {iy1}, {ix2}, {iy2}]"
            )
        else:
            self._preview_text.set(f"Selection: ({x1},{y1}) {w}x{h}px")

    @staticmethod
    def _default_preprocess(region_type: str) -> list[str]:
        """Return sensible default preprocessing steps per region type."""
        if region_type == RegionRole.OCR.value:
            return ["grayscale", "upscale(2x)", "denoise"]
        return ["grayscale", "threshold"]

    def _generate_region_name(self, role: str) -> str:
        """Create a unique region name from the role.

        If ``role`` is not yet used, returns the role name directly.
        Otherwise appends a suffix (e.g. ``health_2``).
        """
        existing = {r["name"] for r in self._collected_regions}
        if role not in existing:
            return role

        counter = 2
        while f"{role}_{counter}" in existing:
            counter += 1
        return f"{role}_{counter}"

    def _remove_region_by_name(self, name: str) -> None:
        """Remove a previously saved region by name."""
        self._collected_regions = [r for r in self._collected_regions if r["name"] != name]
        # Also clear saved rectangles and redraw
        self._canvas.delete("saved")
        for region in self._collected_regions:
            self._draw_existing_region(region)

    def _draw_existing_region(self, region: dict[str, Any]) -> None:
        """Draw a green rectangle for an already‑saved region."""
        bounds = region.get("bounds", [])
        if len(bounds) != 4:
            return
        ix1, iy1, ix2, iy2 = bounds

        # Convert image coords to display coords
        x1 = int(ix1 * self._scale + self._img_x)
        y1 = int(iy1 * self._scale + self._img_y)
        x2 = int(ix2 * self._scale + self._img_x)
        y2 = int(iy2 * self._scale + self._img_y)

        self._canvas.create_rectangle(
            x1, y1, x2, y2, outline=_SUCCESS, width=2, tag="saved"
        )

        # Label with role name
        self._canvas.create_text(
            x1 + 4,
            y1 + 4,
            text=region.get("role", region.get("name", "")),
            fill=_SUCCESS,
            anchor=tk.NW,
            font=("Segoe UI", 9, "bold"),
            tag="saved",
        )

    # ------------------------------------------------------------------
    # HSV threshold computation (Phase 5.3 — delegates to TwoPointCalibrationLoader)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bar_hsv_thresholds(
        empty_img: np.ndarray,
        full_img: np.ndarray,
        bar_type: str = "solid_horizontal",
        orientation: str = "left_to_right",
    ) -> dict[str, Any]:
        """Compute HSV calibration thresholds from empty/full bar captures.

        Delegates to ``TwoPointCalibrationLoader.compute_bar_hsv_thresholds``
        from ``bar_detector.py`` for a single source of truth.  Includes
        ``total_length_px`` calculation, calibration sample SHA‑256 hashes,
        and segment/radial auto‑detection.
        """
        from src.bar_detector import TwoPointCalibrationLoader

        result = TwoPointCalibrationLoader.compute_bar_hsv_thresholds(
            empty_img=empty_img,
            full_img=full_img,
            bar_type=bar_type,
            orientation=orientation,
        )
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_collected_regions(self) -> list[dict[str, Any]]:
        """Return the list of region dicts collected so far."""
        return list(self._collected_regions)
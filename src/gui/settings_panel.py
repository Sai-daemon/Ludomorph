"""
Phase 5.5 — Settings Panel.

Provides a modal Toplevel dialog with tabbed settings for:
- General (Ollama, MCP, log level, auto‑focus)
- AI / LLM (timeout, tokens, cooldown, summarisation)
- Performance (frame skip, cache TTLs)
- Vision (enable, model, thresholds, backend)
- Input (backend selection)
- Theme / UI (theme mode, custom background colour) ← NEW

Reads from config.json via ConfigManager, validates user input, and
atomically writes changes back to disk.
"""

from __future__ import annotations

import copy
from pathlib import Path
from tkinter import messagebox
from typing import Any

import tkinter as tk
from tkinter import ttk

from src.gui.theme import ThemeManager, resolve_font_stack

# ---------------------------------------------------------------------------
# Helpers for nested dict access (e.g., "vision.enabled" → config["vision"]["enabled"])
# ---------------------------------------------------------------------------


def _get_nested(data: dict[str, Any], key_path: str) -> Any:
    """Retrieve a value from a nested dict using a dotted key path."""
    keys = key_path.split(".")
    current: Any = data
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return None
    return current


def _set_nested(data: dict[str, Any], key_path: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key path, creating intermediate dicts as needed."""
    keys = key_path.split(".")
    current = data
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


# ---------------------------------------------------------------------------
# SettingsPanel
# ---------------------------------------------------------------------------


class SettingsPanel(tk.Toplevel):
    """Modal settings dialog with tabbed notebook layout.

    Layout::

        ┌───────────────────────────────────────────────┐
        │  Settings                                       │
        ├───────────────────────────────────────────────┤
        │  [General] [AI/LLM] [Performance] [Vision] …   │  ← Notebook tabs
        ├───────────────────────────────────────────────┤
        │                                                 │
        │  (active tab content – labelled frames)         │
        │                                                 │
        ├───────────────────────────────────────────────┤
        │  [💾 Save] [↻ Reset to Defaults] [✖ Cancel]   │
        └───────────────────────────────────────────────┘
    """

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)

        # -- Theme ------------------------------------------------------------
        self._tm = ThemeManager()
        self._tm.apply_ttk_styles()
        p = self._tm.palette
        self._ui_font = resolve_font_stack(p.ui_font)
        self._mono_font = resolve_font_stack(p.mono_font)

        self.title("Settings")
        self.configure(bg=p.bg)
        self.geometry("700x520")
        self.minsize(600, 400)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Logger
        from src.logging_config import get_logger

        self._logger = get_logger(__name__)

        # Load current config (deep copy so we don't mutate the live object)
        self._original_config = self._load_config()
        self._modified_config = copy.deepcopy(self._original_config)

        # Build UI
        self._build_toolbar()
        self._build_notebook()
        self._build_status_bar()

        # Handle close
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._logger.info("Settings Panel opened.")

    # ------------------------------------------------------------------
    # Colour helpers (delegate to theme)
    # ------------------------------------------------------------------

    @property
    def _p(self):
        """Shortcut to the active palette."""
        return self._tm.palette

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self) -> dict[str, Any]:
        """Load global config via ConfigManager.

        Falls back to DEFAULT_CONFIG (built-in defaults) only when
        ConfigManager itself fails — never to the *bundled* config file,
        which is a static snapshot and not the user's live config.
        """
        try:
            from src.config_manager import load_global_config

            return load_global_config()
        except Exception as exc:
            self._logger.warning(
                "Could not load config via ConfigManager: %s — using built-in defaults.",
                exc,
            )
            from src.config_manager import DEFAULT_CONFIG

            return dict(DEFAULT_CONFIG)

    def _save_config(self, config: dict[str, Any]) -> None:
        """Persist config via ConfigManager."""
        from src.config_manager import save_global_config

        save_global_config(config)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        """Title bar label — just shows 'Settings'."""
        p = self._p
        header = tk.Frame(self, bg=p.bg, padx=12, pady=8)
        header.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            header,
            text="⚙  Settings",
            bg=p.bg,
            fg=p.fg,
            font=(self._ui_font, 14, "bold"),
        ).pack(side=tk.LEFT)

        tk.Label(
            header,
            text="config.json",
            bg=p.bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 9),
        ).pack(side=tk.RIGHT, padx=8)

    def _build_notebook(self) -> None:
        """Create the tabbed notebook and populate each tab.

        Also initialises the widget tracking lists used by _collect_changes.
        """
        self._tracked_entries: list[tk.Entry] = []
        self._tracked_combos: list[ttk.Combobox] = []
        self._tracked_checkboxes: list[tk.Checkbutton] = []
        self._tracked_spinboxes: list[ttk.Spinbox] = []
        self._tracked_sliders: list[tk.Scale] = []

        p = self._p
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # Notebook styling is already handled by ThemeManager.apply_ttk_styles()

        # Create each tab
        self._build_general_tab()
        self._build_llm_tab()
        self._build_performance_tab()
        self._build_vision_tab()
        self._build_input_tab()
        self._build_theme_tab()  # ← NEW

    def _build_status_bar(self) -> None:
        """Bottom bar with action buttons and status text."""
        p = self._p
        bottom = tk.Frame(self, bg=p.status_bar_bg, padx=8, pady=6)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)

        ttk.Button(bottom, text="💾  Save", style="Settings.TButton",
                   command=self._on_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom, text="↻  Reset to Defaults", style="Settings.TButton",
                   command=self._on_reset_defaults).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom, text="✖  Cancel", style="SettingsDanger.TButton",
                   command=self._on_cancel).pack(side=tk.RIGHT, padx=2)

        self._lbl_status = tk.Label(
            bottom,
            text="",
            bg=p.status_bar_bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 9),
            padx=10,
        )
        self._lbl_status.pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _create_tab_frame(self, notebook: ttk.Notebook, title: str) -> tk.Frame:
        """Create a scrollable frame inside a notebook tab.

        Returns the inner content frame (the one to pack widgets into).
        """
        p = self._p
        outer = tk.Frame(notebook, bg=p.bg)
        notebook.add(outer, text=title)

        # Canvas + scrollbar for scrollable content
        canvas = tk.Canvas(outer, bg=p.bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg=p.bg)

        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=inner, anchor=tk.NW, tags="inner")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Make the inner frame expand to fill the canvas width
        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfig("inner", width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel scrolling
        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")
        # Linux scroll events
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"), add="+")
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"), add="+")

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Unbind scroll when this tab is destroyed
        def _cleanup() -> None:
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        outer.bind("<Destroy>", lambda e: _cleanup())

        return inner

    # -- Helpers for building labelled form rows --------------------------

    @staticmethod
    def _add_section_label(parent: tk.Widget, text: str) -> tk.Label:
        """Add a bold section header."""
        p = ThemeManager().palette
        lbl = tk.Label(
            parent,
            text=text,
            bg=p.bg,
            fg=p.fg,
            font=("Segoe UI", 11, "bold"),
            anchor=tk.W,
        )
        lbl.pack(fill=tk.X, padx=12, pady=(8, 2))
        return lbl

    def _add_entry_row(self, parent: tk.Widget, label: str, key_path: str, width: int = 40) -> tk.Entry:
        """Add a labelled Entry row. Returns the Entry widget."""
        p = self._p
        row = tk.Frame(parent, bg=p.bg)
        row.pack(fill=tk.X, padx=12, pady=2)

        tk.Label(row, text=label, bg=p.bg, fg=p.fg,
                 font=(self._ui_font, 10), width=24, anchor=tk.W).pack(
            side=tk.LEFT, padx=(0, 4)
        )

        var = tk.StringVar(value=str(_get_nested(self._modified_config, key_path)))
        entry = tk.Entry(
            row,
            textvariable=var,
            bg=p.entry_bg,
            fg=p.entry_fg,
            insertbackground=p.entry_insert,
            font=(self._mono_font, 10),
            width=width,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=p.disabled_bg,
            highlightcolor=p.accent,
        )
        entry.pack(side=tk.LEFT)

        # Store the variable reference so we can read it back on save
        entry._key_path = key_path  # type: ignore[attr-defined]
        entry._var = var  # type: ignore[attr-defined]
        self._tracked_entries.append(entry)

        return entry

    def _add_combo_row(
        self, parent: tk.Widget, label: str, key_path: str, values: list[str], width: int = 20
    ) -> ttk.Combobox:
        """Add a labelled Combobox row. Returns the Combobox."""
        p = self._p
        row = tk.Frame(parent, bg=p.bg)
        row.pack(fill=tk.X, padx=12, pady=2)

        tk.Label(row, text=label, bg=p.bg, fg=p.fg,
                 font=(self._ui_font, 10), width=24, anchor=tk.W).pack(
            side=tk.LEFT, padx=(0, 4)
        )

        current = str(_get_nested(self._modified_config, key_path))
        var = tk.StringVar(value=current)
        combo = ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=width)
        combo.pack(side=tk.LEFT)

        combo._key_path = key_path  # type: ignore[attr-defined]
        combo._var = var  # type: ignore[attr-defined]
        self._tracked_combos.append(combo)

        return combo

    def _add_checkbox_row(self, parent: tk.Widget, label: str, key_path: str) -> tk.Checkbutton:
        """Add a labelled Checkbox row. Returns the Checkbutton."""
        p = self._p
        row = tk.Frame(parent, bg=p.bg)
        row.pack(fill=tk.X, padx=12, pady=2)

        current = _get_nested(self._modified_config, key_path)
        var = tk.BooleanVar(value=bool(current))
        cb = tk.Checkbutton(
            row,
            text=label,
            variable=var,
            bg=p.bg,
            fg=p.fg,
            selectcolor=p.bg,
            activebackground=p.bg,
            activeforeground=p.fg,
            font=(self._ui_font, 10),
            anchor=tk.W,
        )
        cb.pack(side=tk.LEFT)

        cb._key_path = key_path  # type: ignore[attr-defined]
        cb._var = var  # type: ignore[attr-defined]
        self._tracked_checkboxes.append(cb)

        return cb

    def _add_spinbox_row(
        self,
        parent: tk.Widget,
        label: str,
        key_path: str,
        from_: float,
        to: float,
        increment: float = 1.0,
        width: int = 10,
    ) -> ttk.Spinbox:
        """Add a labelled Spinbox row. Returns the Spinbox."""
        p = self._p
        row = tk.Frame(parent, bg=p.bg)
        row.pack(fill=tk.X, padx=12, pady=2)

        tk.Label(row, text=label, bg=p.bg, fg=p.fg,
                 font=(self._ui_font, 10), width=24, anchor=tk.W).pack(
            side=tk.LEFT, padx=(0, 4)
        )

        current = _get_nested(self._modified_config, key_path)
        var = tk.StringVar(value=str(current))
        spin = ttk.Spinbox(
            row, textvariable=var, from_=from_, to=to, increment=increment, width=width
        )
        spin.pack(side=tk.LEFT)

        spin._key_path = key_path  # type: ignore[attr-defined]
        spin._var = var  # type: ignore[attr-defined]
        self._tracked_spinboxes.append(spin)

        return spin

    def _add_slider_row(
        self,
        parent: tk.Widget,
        label: str,
        key_path: str,
        from_: float,
        to: float,
        resolution: float = 0.05,
    ) -> tuple[tk.Scale, tk.Label]:
        """Add a labelled Scale (slider) row with a value readout. Returns (scale, value_label)."""
        p = self._p
        row = tk.Frame(parent, bg=p.bg)
        row.pack(fill=tk.X, padx=12, pady=2)

        tk.Label(row, text=label, bg=p.bg, fg=p.fg,
                 font=(self._ui_font, 10), width=24, anchor=tk.W).pack(
            side=tk.LEFT, padx=(0, 4)
        )

        current = _get_nested(self._modified_config, key_path)
        var = tk.DoubleVar(value=float(current))

        scale = tk.Scale(
            row,
            variable=var,
            from_=from_,
            to=to,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            bg=p.bg,
            fg=p.fg,
            highlightbackground=p.bg,
            troughcolor=p.disabled_bg,
            activebackground=p.accent,
            length=200,
        )
        scale.pack(side=tk.LEFT)

        val_lbl = tk.Label(
            row, text=f"{current:.2f}", bg=p.bg, fg=p.accent,
            font=(self._mono_font, 10), width=5
        )
        val_lbl.pack(side=tk.LEFT, padx=6)

        # Update readout on change
        def _update_readout(*args: Any) -> None:
            val_lbl.config(text=f"{var.get():.2f}")

        var.trace_add("write", _update_readout)

        scale._key_path = key_path  # type: ignore[attr-defined]
        scale._var = var  # type: ignore[attr-defined]
        self._tracked_sliders.append(scale)

        return scale, val_lbl

    # -- Tab content --------------------------------------------------------

    def _build_general_tab(self) -> None:
        """Build the General tab: Ollama, MCP, log level, auto‑focus."""
        inner = self._create_tab_frame(self._notebook, "General")

        # --- Ollama ---
        self._add_section_label(inner, "Ollama")
        self._add_entry_row(inner, "Ollama URL:", "ollama_url")
        self._add_entry_row(inner, "Ollama Model:", "ollama_model")

        # --- MCP Memory ---
        self._add_section_label(inner, "MCP Memory")
        self._add_checkbox_row(inner, "Enable MCP memory server", "mcp_enabled")
        self._add_entry_row(inner, "MCP URL:", "mcp_url")
        self._add_spinbox_row(inner, "Memory Max Events:", "memory_max_events", 100, 100000, 1000)

        # --- General ---
        self._add_section_label(inner, "General")
        self._add_combo_row(
            inner, "Log Level:", "log_level",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
        )
        self._add_checkbox_row(inner, "Auto‑focus game window", "auto_focus_window")

    def _build_llm_tab(self) -> None:
        """Build the AI / LLM tab: timeouts, tokens, cooldown, summarisation."""
        inner = self._create_tab_frame(self._notebook, "AI / LLM")

        self._add_section_label(inner, "Decision Loop")
        self._add_spinbox_row(inner, "LLM Timeout (ms):", "llm_timeout_ms", 50, 5000, 50)
        self._add_spinbox_row(inner, "LLM Max Tokens:", "llm_max_tokens", 64, 8192, 64)
        self._add_spinbox_row(inner, "LLM Vision Max Tokens:", "llm_vision_max_tokens", 64, 8192, 64)
        self._add_spinbox_row(inner, "Action Cooldown (ms):", "action_cooldown_ms", 0, 5000, 50)

        self._add_section_label(inner, "Memory Summarisation")
        self._add_checkbox_row(inner, "Enable summarisation", "enable_summarization")
        self._add_entry_row(inner, "Summarisation Model:", "summarization_model")

    def _build_performance_tab(self) -> None:
        """Build the Performance tab: frame skip, cache TTLs."""
        inner = self._create_tab_frame(self._notebook, "Performance")

        self._add_section_label(inner, "Frame Skipping")
        self._add_spinbox_row(inner, "Frame Skip:", "frame_skip", 0, 60, 1)

        self._add_section_label(inner, "Caching")
        self._add_spinbox_row(inner, "State Cache TTL (s):", "state_cache_ttl_seconds", 0.0, 10.0, 0.1)
        self._add_spinbox_row(inner, "OCR Cache TTL (s):", "ocr_cache_ttl_seconds", 0.0, 10.0, 0.1)

    def _build_vision_tab(self) -> None:
        """Build the Vision tab: enable, model, thresholds, backend, etc."""
        inner = self._create_tab_frame(self._notebook, "Vision")

        self._add_section_label(inner, "Vision Module (optional – Appendix A)")
        self._add_checkbox_row(inner, "Enable Vision module", "vision.enabled")

        self._add_section_label(inner, "Model")
        self._add_entry_row(inner, "Model Path:", "vision.model_path")

        self._add_section_label(inner, "Detection Parameters")
        self._add_spinbox_row(inner, "Detection Interval:", "vision.detection_interval", 1, 30, 1)
        self._add_spinbox_row(inner, "Max Detections:", "vision.max_detections", 1, 200, 5)
        self._add_combo_row(
            inner, "Input Size:", "vision.input_size", ["320", "416", "640"], width=10
        )

        self._add_section_label(inner, "Thresholds")
        self._add_slider_row(inner, "Confidence Threshold:", "vision.confidence_threshold", 0.1, 0.9, 0.05)
        self._add_slider_row(inner, "IOU Threshold:", "vision.iou_threshold", 0.1, 0.95, 0.05)

        self._add_section_label(inner, "Backend")
        self._add_combo_row(
            inner, "Inference Backend:", "vision.backend",
            ["auto", "cpu", "openvino", "cuda"], width=12,
        )

    def _build_input_tab(self) -> None:
        """Build the Input tab: backend selection and mouse smoothing."""
        p = self._p
        inner = self._create_tab_frame(self._notebook, "Input")

        # === Input Injection Backend =====================================
        self._add_section_label(inner, "Input Injection Backend")
        self._add_combo_row(
            inner, "Input Backend:", "input_backend",
            ["auto", "pynput", "ydotool", "dotool"], width=12,
        )

        # Backend help text
        backend_help = tk.Frame(inner, bg=p.bg)
        backend_help.pack(fill=tk.X, padx=12, pady=(0, 12))
        tk.Label(
            backend_help,
            text=(
                "auto   → detects platform automatically\n"
                "pynput → recommended for Windows & X11\n"
                "ydotool → required for Linux Wayland\n"
                "dotool → fallback if ydotoold is unavailable"
            ),
            bg=p.bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 9),
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        # === Mouse Smoothing ==============================================
        self._add_section_label(inner, "Mouse Smoothing")
        self._add_slider_row(
            inner, "Mouse Speed:", "mouse_speed",
            from_=0.0, to=2.0, resolution=0.1,
        )

        # Snap-point labels — aligned directly below the 200px slider
        # Layout: [24-char label] [slider 200px] [value label]
        # We use a frame with the same offset + width as the slider row
        snap_frame = tk.Frame(inner, bg=p.bg)
        snap_frame.pack(fill=tk.X, padx=12, pady=(0, 4))

        # Matching the label width from _add_slider_row (width=24)
        tk.Label(snap_frame, text="", bg=p.bg, width=24).pack(side=tk.LEFT, padx=(0, 4))

        # Bar container matching slider length=200
        bar_frame = tk.Frame(snap_frame, bg=p.bg, width=200, height=28)
        bar_frame.pack(side=tk.LEFT)
        bar_frame.pack_propagate(False)

        # Slow — left edge
        tk.Label(
            bar_frame, text="Slow\n· Human-like", bg=p.bg,
            fg=p.disabled_fg, font=(self._ui_font, 7),
            justify=tk.LEFT,
        ).pack(side=tk.LEFT)

        # Spacer in the middle
        tk.Frame(bar_frame, bg=p.bg).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Normal — roughly centered
        tk.Label(
            bar_frame, text="Normal\n· Smooth", bg=p.bg,
            fg=p.disabled_fg, font=(self._ui_font, 7),
            justify=tk.CENTER,
        ).pack(side=tk.LEFT, padx=4)

        # Another spacer
        tk.Frame(bar_frame, bg=p.bg).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Fast — right edge
        tk.Label(
            bar_frame, text="Fast\n· Instant", bg=p.bg,
            fg=p.disabled_fg, font=(self._ui_font, 7),
            justify=tk.RIGHT,
        ).pack(side=tk.RIGHT)

        # Speed description — aligned under the slider via a label-width spacer
        speed_help = tk.Frame(inner, bg=p.bg)
        speed_help.pack(fill=tk.X, padx=12, pady=(8, 8))
        
        # Spacer matching the 24-char label + 4px gap from _add_slider_row
        tk.Label(speed_help, text="", bg=p.bg, width=24).pack(side=tk.LEFT, padx=(0, 4))
        
        tk.Label(
            speed_help,
            text=(
                "0.0 (Slow)   → ~600 px/s  — human-like deliberate glide\n"
                "1.0 (Normal) → ~3000 px/s — smooth, faster than human\n"
                "2.0 (Fast)    → instant teleport, no interpolation"
            ),
            bg=p.bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 9),
            justify=tk.LEFT,
        ).pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # NEW: Theme / UI tab
    # ------------------------------------------------------------------

    def _build_theme_tab(self) -> None:
        """Build the Theme / UI tab: theme mode, custom background colour."""
        p = self._p
        inner = self._create_tab_frame(self._notebook, "Theme / UI")

        self._add_section_label(inner, "Theme Mode")

        # -- Theme mode combo -------------------------------------------------
        row = tk.Frame(inner, bg=p.bg)
        row.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(row, text="Theme:", bg=p.bg, fg=p.fg,
                 font=(self._ui_font, 10), width=24, anchor=tk.W).pack(
            side=tk.LEFT, padx=(0, 4)
        )

        current_theme = self._modified_config.get("theme", "auto")
        if current_theme not in ("auto", "dark", "light"):
            current_theme = "auto"
        self._theme_var = tk.StringVar(value=current_theme)
        combo = ttk.Combobox(
            row, textvariable=self._theme_var,
            values=["auto", "dark", "light"],
            state="readonly", width=20,
        )
        combo.pack(side=tk.LEFT)
        self._theme_combo = combo

        # Theme help text
        theme_help = tk.Frame(inner, bg=p.bg)
        theme_help.pack(fill=tk.X, padx=12, pady=(0, 12))
        tk.Label(
            theme_help,
            text=(
                "auto   → detect system preference automatically\n"
                "dark   → always use dark theme\n"
                "light  → always use light theme"
            ),
            bg=p.bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 9),
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        # --- Custom Background Colour ---
        self._add_section_label(inner, "Custom Background Colour")

        row2 = tk.Frame(inner, bg=p.bg)
        row2.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(row2, text="Background:", bg=p.bg, fg=p.fg,
                 font=(self._ui_font, 10), width=24, anchor=tk.W).pack(
            side=tk.LEFT, padx=(0, 4)
        )

        custom_bg = self._modified_config.get("theme_custom_bg") or ""
        self._custom_bg_var = tk.StringVar(value=custom_bg)
        entry = tk.Entry(
            row2,
            textvariable=self._custom_bg_var,
            bg=p.entry_bg,
            fg=p.entry_fg,
            insertbackground=p.entry_insert,
            font=(self._mono_font, 10),
            width=10,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=p.disabled_bg,
            highlightcolor=p.accent,
        )
        entry.pack(side=tk.LEFT, padx=(0, 4))
        self._custom_bg_entry = entry

        # Preview swatch
        self._swatch = tk.Label(
            row2, text="    ", bg=p.bg, relief=tk.SOLID,
            borderwidth=1, width=4,
        )
        self._swatch.pack(side=tk.LEFT, padx=8)
        self._update_swatch()

        # Swatch help
        tk.Label(
            inner,
            text="Enter a hex colour (e.g. #0F141E) and click Apply to preview.\n"
                 "Leave empty to use the theme's default background.",
            bg=p.bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 9),
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=12, pady=(4, 0))

        # Apply Preview button
        row3 = tk.Frame(inner, bg=p.bg)
        row3.pack(fill=tk.X, padx=12, pady=(6, 12))
        ttk.Button(
            row3, text="🎨  Apply Preview",
            style="Settings.TButton",
            command=self._on_preview_theme,
        ).pack(side=tk.LEFT)

        # Reset custom bg button
        ttk.Button(
            row3, text="↩  Use Default",
            style="Settings.TButton",
            command=self._on_reset_custom_bg,
        ).pack(side=tk.LEFT, padx=4)

        # Theme note
        note = tk.Frame(inner, bg=p.bg)
        note.pack(fill=tk.X, padx=12, pady=(0, 12))
        tk.Label(
            note,
            text="⚠ Theme changes take effect when you close and reopen the app.",
            bg=p.bg,
            fg=p.warning,
            font=(self._ui_font, 8),
            wraplength=400,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

    def _update_swatch(self) -> None:
        """Update the colour preview swatch from the custom bg entry."""
        val = self._custom_bg_var.get().strip()
        if val.startswith("#") and len(val) == 7:
            self._swatch.configure(bg=val)
        else:
            self._swatch.configure(bg=self._p.bg)

    def _on_preview_theme(self) -> None:
        """Apply the theme changes as a live preview."""
        mode = self._theme_var.get()
        custom_bg_raw = self._custom_bg_var.get().strip()
        custom_bg = custom_bg_raw if (custom_bg_raw.startswith("#") and len(custom_bg_raw) == 7) else None

        self._tm.set_theme(mode, custom_bg)
        # Re-apply styles immediately for preview
        self._tm.apply_ttk_styles()
        # Update our palette reference
        self._tm = ThemeManager()  # refresh singleton
        self._update_swatch()
        self._set_status("Theme preview applied.", self._p.success)

    def _on_reset_custom_bg(self) -> None:
        """Clear the custom background colour."""
        self._custom_bg_var.set("")
        self._update_swatch()
        self._set_status("Custom background cleared.", self._p.success)

    # ------------------------------------------------------------------
    # Collect widget values into modified config
    # ------------------------------------------------------------------

    def _collect_changes(self) -> None:
        """Read all widget values back into self._modified_config."""
        # String entries
        for entry in self._tracked_entries:
            key_path = getattr(entry, "_key_path", None)
            if key_path is None:
                continue
            value = entry._var.get()
            _set_nested(self._modified_config, key_path, value)

        # Combos
        for combo in self._tracked_combos:
            key_path = getattr(combo, "_key_path", None)
            if key_path is None:
                continue
            value = combo._var.get()
            _set_nested(self._modified_config, key_path, value)

        # Checkboxes
        for cb in self._tracked_checkboxes:
            key_path = getattr(cb, "_key_path", None)
            if key_path is None:
                continue
            value = cb._var.get()
            _set_nested(self._modified_config, key_path, value)

        # Spinboxes
        for spin in self._tracked_spinboxes:
            key_path = getattr(spin, "_key_path", None)
            if key_path is None:
                continue
            raw = spin._var.get()
            try:
                value = float(raw) if "." in raw else int(raw)
            except ValueError:
                value = raw
            _set_nested(self._modified_config, key_path, value)

        # Sliders
        for scale in self._tracked_sliders:
            key_path = getattr(scale, "_key_path", None)
            if key_path is None:
                continue
            value = scale._var.get()
            _set_nested(self._modified_config, key_path, value)

        # Theme fields
        self._modified_config["theme"] = self._theme_var.get()
        custom_bg = self._custom_bg_var.get().strip()
        if custom_bg.startswith("#") and len(custom_bg) == 7:
            self._modified_config["theme_custom_bg"] = custom_bg
        else:
            self._modified_config["theme_custom_bg"] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> bool:
        """Validate all fields. Returns True if valid, False otherwise.

        Shows the first error via status bar and returns False.
        """
        # Ollama URL: must start with http:// or https://
        url = str(_get_nested(self._modified_config, "ollama_url") or "")
        if url and not (url.startswith("http://") or url.startswith("https://")):
            self._set_status("Ollama URL must start with http:// or https://", self._p.danger)
            return False

        # MCP URL: must start with http:// or https://
        mcp_url = str(_get_nested(self._modified_config, "mcp_url") or "")
        if mcp_url and not (mcp_url.startswith("http://") or mcp_url.startswith("https://")):
            self._set_status("MCP URL must start with http:// or https://", self._p.danger)
            return False

        # Numeric range checks
        timeout = _get_nested(self._modified_config, "llm_timeout_ms")
        if isinstance(timeout, (int, float)) and (timeout < 50 or timeout > 5000):
            self._set_status("LLM Timeout must be between 50 and 5000 ms", self._p.danger)
            return False

        fps = _get_nested(self._modified_config, "frame_skip")
        if isinstance(fps, (int, float)) and (fps < 0 or fps > 60):
            self._set_status("Frame Skip must be between 0 and 60", self._p.danger)
            return False

        viz_conf = _get_nested(self._modified_config, "vision.confidence_threshold")
        if isinstance(viz_conf, (int, float)) and (viz_conf < 0.1 or viz_conf > 0.9):
            self._set_status("Vision confidence threshold must be between 0.1 and 0.9", self._p.danger)
            return False

        # Theme custom bg validation
        custom_bg = self._modified_config.get("theme_custom_bg")
        if custom_bg and isinstance(custom_bg, str):
            if not (custom_bg.startswith("#") and len(custom_bg) == 7):
                self._set_status("Custom background must be a hex colour like #0F141E", self._p.danger)
                return False

        return True

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        """Validate and save settings."""
        self._collect_changes()

        if not self._validate():
            return

        try:
            self._save_config(self._modified_config)
            self._original_config = copy.deepcopy(self._modified_config)

            # Also push theme changes to ThemeManager
            mode = self._modified_config.get("theme", "auto")
            custom_bg = self._modified_config.get("theme_custom_bg")
            self._tm.set_theme(mode, custom_bg)

            self._set_status("Settings saved successfully.", self._p.success)
            self._logger.info("Settings saved to config.json.")
        except Exception as exc:
            self._set_status(f"Failed to save: {exc}", self._p.danger)
            self._logger.error(f"Failed to save settings: {exc}")

    def _on_reset_defaults(self) -> None:
        """Reset all settings to built‑in defaults."""
        if not messagebox.askyesno(
            "Reset to Defaults",
            "This will reset all settings to their default values and close the dialog. Continue?",
            parent=self,
        ):
            return

        from src.config_manager import DEFAULT_CONFIG

        self._modified_config = copy.deepcopy(DEFAULT_CONFIG)
        self._set_status("Reset to defaults. Re‑open to customise.", self._p.warning)
        self._logger.info("Settings reset to defaults.")

        # We can't easily repopulate all widgets, so close & let user reopen
        self.destroy()

    def _on_cancel(self) -> None:
        """Close without saving."""
        self._set_status("Cancelled — no changes saved.", self._p.warning)
        self._logger.info("Settings panel cancelled.")
        self.destroy()

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str, colour: str | None = None) -> None:
        """Update the status bar text and colour."""
        if colour is None:
            colour = self._p.disabled_fg
        self._lbl_status.config(text=text, fg=colour)
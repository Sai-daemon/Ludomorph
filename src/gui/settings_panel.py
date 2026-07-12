"""
Phase 5.5 — Settings Panel.

Provides a modal Toplevel dialog with tabbed settings for:
- General (Ollama, MCP, log level, auto‑focus)
- AI / LLM (timeout, tokens, cooldown, summarisation)
- Performance (frame skip, cache TTLs)
- Vision (enable, model, thresholds, backend)
- Input (backend selection)

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

# ---------------------------------------------------------------------------
# Colour palette (matches main_window.py dark theme)
# ---------------------------------------------------------------------------

_BG = "#1E1E1E"
_FG = "#D4D4D4"
_ACCENT = "#0078D4"
_SUCCESS = "#50C878"
_DANGER = "#E04040"
_WARNING = "#E8A317"
_DISABLED_BG = "#3C3C3C"
_DISABLED_FG = "#808080"
_FRAME_BG = "#252526"
_LABEL_BG = "#1E1E1E"
_ENTRY_BG = "#2D2D2D"
_ENTRY_FG = "#D4D4D4"
_ENTRY_INSERT = "#D4D4D4"

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

        self.title("Settings")
        self.configure(bg=_BG)
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
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self) -> dict[str, Any]:
        """Load global config via ConfigManager, or from disk if not available."""
        try:
            from src.config_manager import load_global_config

            return load_global_config()
        except Exception:
            # Fallback: load directly from the config dir
            config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.json"
            if config_path.exists():
                import json

                try:
                    return json.loads(config_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            # Ultimate fallback: built‑in defaults
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
        header = tk.Frame(self, bg=_BG, padx=12, pady=8)
        header.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            header,
            text="⚙  Settings",
            bg=_BG,
            fg=_FG,
            font=("Segoe UI", 14, "bold"),
        ).pack(side=tk.LEFT)

        tk.Label(
            header,
            text="config.json",
            bg=_BG,
            fg=_DISABLED_FG,
            font=("Segoe UI", 9),
        ).pack(side=tk.RIGHT, padx=8)

    def _build_notebook(self) -> None:
        """Create the tabbed notebook and populate each tab."""
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # Style the notebook for dark theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=_DISABLED_BG, foreground=_FG, padding=(12, 4), font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", _ACCENT)], foreground=[("selected", "#FFFFFF")])

        # Create each tab
        self._build_general_tab()
        self._build_llm_tab()
        self._build_performance_tab()
        self._build_vision_tab()
        self._build_input_tab()

    def _build_status_bar(self) -> None:
        """Bottom bar with action buttons and status text."""
        bottom = tk.Frame(self, bg="#2D2D2D", padx=8, pady=6)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)

        # Style for bottom buttons
        style = ttk.Style()
        style.configure(
            "Settings.TButton",
            background=_ACCENT,
            foreground="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            borderwidth=1,
            padding=(12, 3),
        )
        style.map("Settings.TButton", background=[("active", "#005A9E")])
        style.configure(
            "SettingsDanger.TButton",
            background=_DANGER,
            foreground="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            borderwidth=1,
            padding=(12, 3),
        )
        style.map("SettingsDanger.TButton", background=[("active", "#C03030")])

        ttk.Button(bottom, text="💾  Save", style="Settings.TButton", command=self._on_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom, text="↻  Reset to Defaults", style="Settings.TButton", command=self._on_reset_defaults).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(bottom, text="✖  Cancel", style="SettingsDanger.TButton", command=self._on_cancel).pack(
            side=tk.RIGHT, padx=2
        )

        self._lbl_status = tk.Label(
            bottom,
            text="",
            bg="#2D2D2D",
            fg=_DISABLED_FG,
            font=("Segoe UI", 9),
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
        outer = tk.Frame(notebook, bg=_BG)
        notebook.add(outer, text=title)

        # Canvas + scrollbar for scrollable content
        canvas = tk.Canvas(outer, bg=_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg=_BG)

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
        lbl = tk.Label(
            parent,
            text=text,
            bg=_BG,
            fg=_FG,
            font=("Segoe UI", 11, "bold"),
            anchor=tk.W,
        )
        lbl.pack(fill=tk.X, padx=12, pady=(8, 2))
        return lbl

    def _add_entry_row(self, parent: tk.Widget, label: str, key_path: str, width: int = 40) -> tk.Entry:
        """Add a labelled Entry row. Returns the Entry widget."""
        row = tk.Frame(parent, bg=_BG)
        row.pack(fill=tk.X, padx=12, pady=2)

        tk.Label(row, text=label, bg=_BG, fg=_FG, font=("Segoe UI", 10), width=24, anchor=tk.W).pack(
            side=tk.LEFT, padx=(0, 4)
        )

        var = tk.StringVar(value=str(_get_nested(self._modified_config, key_path)))
        entry = tk.Entry(
            row,
            textvariable=var,
            bg=_ENTRY_BG,
            fg=_ENTRY_FG,
            insertbackground=_ENTRY_INSERT,
            font=("Consolas", 10),
            width=width,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=_DISABLED_BG,
            highlightcolor=_ACCENT,
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
        row = tk.Frame(parent, bg=_BG)
        row.pack(fill=tk.X, padx=12, pady=2)

        tk.Label(row, text=label, bg=_BG, fg=_FG, font=("Segoe UI", 10), width=24, anchor=tk.W).pack(
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
        row = tk.Frame(parent, bg=_BG)
        row.pack(fill=tk.X, padx=12, pady=2)

        current = _get_nested(self._modified_config, key_path)
        var = tk.BooleanVar(value=bool(current))
        cb = tk.Checkbutton(
            row,
            text=label,
            variable=var,
            bg=_BG,
            fg=_FG,
            selectcolor=_BG,
            activebackground=_BG,
            activeforeground=_FG,
            font=("Segoe UI", 10),
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
        row = tk.Frame(parent, bg=_BG)
        row.pack(fill=tk.X, padx=12, pady=2)

        tk.Label(row, text=label, bg=_BG, fg=_FG, font=("Segoe UI", 10), width=24, anchor=tk.W).pack(
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
        row = tk.Frame(parent, bg=_BG)
        row.pack(fill=tk.X, padx=12, pady=2)

        tk.Label(row, text=label, bg=_BG, fg=_FG, font=("Segoe UI", 10), width=24, anchor=tk.W).pack(
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
            bg=_BG,
            fg=_FG,
            highlightbackground=_BG,
            troughcolor=_DISABLED_BG,
            activebackground=_ACCENT,
            length=200,
        )
        scale.pack(side=tk.LEFT)

        val_lbl = tk.Label(
            row, text=f"{current:.2f}", bg=_BG, fg=_ACCENT, font=("Consolas", 10), width=5
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
            inner,
            "Log Level:",
            "log_level",
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
            inner,
            "Inference Backend:",
            "vision.backend",
            ["auto", "cpu", "openvino", "cuda"],
            width=12,
        )

    def _build_input_tab(self) -> None:
        """Build the Input tab: backend selection."""
        inner = self._create_tab_frame(self._notebook, "Input")

        self._add_section_label(inner, "Input Injection Backend")
        self._add_combo_row(
            inner,
            "Input Backend:",
            "input_backend",
            ["auto", "pynput", "ydotool", "dotool"],
            width=12,
        )

        # Help text
        help_frame = tk.Frame(inner, bg=_BG)
        help_frame.pack(fill=tk.X, padx=12, pady=(16, 8))

        help_text = (
            "auto → detects platform automatically\n"
            "pynput → recommended for Windows & X11\n"
            "ydotool → required for Linux Wayland\n"
            "dotool → fallback if ydotoold is unavailable"
        )
        tk.Label(
            help_frame,
            text=help_text,
            bg=_BG,
            fg=_DISABLED_FG,
            font=("Segoe UI", 9),
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

    # ------------------------------------------------------------------
    # Initialisation – track all widget references for save
    # ------------------------------------------------------------------

    def _build_notebook(self) -> None:
        """Create the tabbed notebook and populate each tab.

        Also initialises the widget tracking lists used by _collect_changes.
        """
        self._tracked_entries: list[tk.Entry] = []
        self._tracked_combos: list[ttk.Combobox] = []
        self._tracked_checkboxes: list[tk.Checkbutton] = []
        self._tracked_spinboxes: list[ttk.Spinbox] = []
        self._tracked_sliders: list[tk.Scale] = []

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # Style the notebook for dark theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=_BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=_DISABLED_BG,
            foreground=_FG,
            padding=(12, 4),
            font=("Segoe UI", 10),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", _ACCENT)],
            foreground=[("selected", "#FFFFFF")],
        )

        # Create each tab
        self._build_general_tab()
        self._build_llm_tab()
        self._build_performance_tab()
        self._build_vision_tab()
        self._build_input_tab()

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
            self._set_status("Ollama URL must start with http:// or https://", _DANGER)
            return False

        # MCP URL: must start with http:// or https://
        mcp_url = str(_get_nested(self._modified_config, "mcp_url") or "")
        if mcp_url and not (mcp_url.startswith("http://") or mcp_url.startswith("https://")):
            self._set_status("MCP URL must start with http:// or https://", _DANGER)
            return False

        # Numeric range checks
        timeout = _get_nested(self._modified_config, "llm_timeout_ms")
        if isinstance(timeout, (int, float)) and (timeout < 50 or timeout > 5000):
            self._set_status("LLM Timeout must be between 50 and 5000 ms", _DANGER)
            return False

        fps = _get_nested(self._modified_config, "frame_skip")
        if isinstance(fps, (int, float)) and (fps < 0 or fps > 60):
            self._set_status("Frame Skip must be between 0 and 60", _DANGER)
            return False

        viz_conf = _get_nested(self._modified_config, "vision.confidence_threshold")
        if isinstance(viz_conf, (int, float)) and (viz_conf < 0.1 or viz_conf > 0.9):
            self._set_status("Vision confidence threshold must be between 0.1 and 0.9", _DANGER)
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
            self._set_status("Settings saved successfully.", _SUCCESS)
            self._logger.info("Settings saved to config.json.")
        except Exception as exc:
            self._set_status(f"Failed to save: {exc}", _DANGER)
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
        self._set_status("Reset to defaults. Re‑open to customise.", _WARNING)
        self._logger.info("Settings reset to defaults.")

        # We can't easily repopulate all widgets, so close & let user reopen
        self.destroy()

    def _on_cancel(self) -> None:
        """Close without saving."""
        self._set_status("Cancelled — no changes saved.", _WARNING)
        self._logger.info("Settings panel cancelled.")
        self.destroy()

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str, colour: str = _DISABLED_FG) -> None:
        """Update the status bar text and colour."""
        self._lbl_status.config(text=text, fg=colour)
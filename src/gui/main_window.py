"""
Main application window for AI Game Master.

Provides:
- AsyncTk-derived root window with Start/Stop/Pause controls
- Embedded log dashboard
- Status bar (engine state, Ollama, MCP)
- Menu bar skeleton (File, Settings, Help)
- Calibration overlay launcher (Phases 5.2 & 5.3)
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable

import numpy as np

from src.gui.async_tk import AsyncTk
from src.gui.health_panel import HealthPanel
from src.gui.log_dashboard import LogDashboard

# ---------------------------------------------------------------------------
# Colour palette (dark theme)
# ---------------------------------------------------------------------------

_BG = "#1E1E1E"          # main background
_FG = "#D4D4D4"          # default foreground
_ACCENT = "#0078D4"       # accent blue (buttons, highlights)
_SUCCESS = "#50C878"      # green — running / healthy
_DANGER = "#E04040"       # red — stopped / unhealthy
_WARNING = "#E8A317"      # amber — degraded
_DISABLED_BG = "#3C3C3C"  # disabled button background
_DISABLED_FG = "#808080"  # disabled button foreground


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------


class MainWindow(AsyncTk):
    """Top-level application window.

    Layout::

        ┌──────────────────────────────────────────────┐
        │  Menu bar  (File | Settings | Help)           │
        ├──────────────────────────────────────────────┤
        │  Toolbar  [▶ Start] [■ Stop] [⏸ Pause]       │
        │           [📐 Calibrate Regions]              │
        ├──────────────────────────────────────────────┤
        │                                               │
        │  Log Dashboard  (scrollable, real-time)       │
        │                                               │
        ├──────────────────────────────────────────────┤
        │  Status bar  Engine: Idle | Ollama: ? | MCP: ?│
        └──────────────────────────────────────────────┘
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.title("AI Game Master")
        self.configure(bg=_BG)

        # Minimum size; user can resize larger
        self.minsize(900, 600)
        self.geometry("1000x700")

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # -- Component references (wired in Phase 6.1) -----------------------
        self._ocr_module: Any = None
        self._screen_capture: Any = None
        self._config_manager: Any = None

        # Current profile path
        self._profile_path: Path | None = None

        # Health panel (Phase 5.6)
        self._health_panel: HealthPanel | None = None
        self._health_panel_visible: bool = False

        # -- Build UI sections -----------------------------------------------
        self._build_menu()
        self._build_toolbar()
        self._build_log_dashboard()
        self._build_status_bar()

        # Logger for the main window
        from src.logging_config import get_logger

        self._logger = get_logger(__name__)
        self._logger.info("Main window initialised.")

    # ------------------------------------------------------------------
    # Component injection (Phase 6.1 wiring)
    # ------------------------------------------------------------------

    def set_ocr_module(self, ocr_module: Any) -> None:
        """Inject the OCR module for calibration preview."""
        self._ocr_module = ocr_module

    def set_screen_capture(self, screen_capture: Any) -> None:
        """Inject the screen capture module for calibration."""
        self._screen_capture = screen_capture

    def set_config_manager(self, config_manager: Any) -> None:
        """Inject the config manager for saving profiles."""
        self._config_manager = config_manager

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        """Create the menu bar."""
        menubar = tk.Menu(self, bg=_BG, fg=_FG, activebackground=_ACCENT, activeforeground="#FFFFFF")

        # -- File menu -------------------------------------------------------
        file_menu = tk.Menu(menubar, tearoff=0, bg=_BG, fg=_FG, activebackground=_ACCENT)
        file_menu.add_command(label="New Profile …", command=self._on_new_profile)
        file_menu.add_command(label="Import Profile …", command=self._on_import_profile)
        file_menu.add_command(label="Export Profile …", command=self._on_export_profile)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        # -- View menu -------------------------------------------------------
        view_menu = tk.Menu(menubar, tearoff=0, bg=_BG, fg=_FG, activebackground=_ACCENT)
        view_menu.add_command(label="Health Monitor …", command=self._on_health_monitor)
        menubar.add_cascade(label="View", menu=view_menu)

        # -- Settings menu ---------------------------------------------------
        settings_menu = tk.Menu(menubar, tearoff=0, bg=_BG, fg=_FG, activebackground=_ACCENT)
        settings_menu.add_command(label="Preferences …", command=self._on_preferences)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        # -- Help menu -------------------------------------------------------
        help_menu = tk.Menu(menubar, tearoff=0, bg=_BG, fg=_FG, activebackground=_ACCENT)
        help_menu.add_command(label="About", state=tk.DISABLED, command=self._placeholder)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _build_toolbar(self) -> None:
        """Create the Start / Stop / Pause button bar with calibration."""
        toolbar = tk.Frame(self, bg=_BG, padx=8, pady=6)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        # -- Style for themed buttons ----------------------------------------
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Start.TButton",
            background=_SUCCESS,
            foreground="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            borderwidth=1,
            padding=(16, 4),
        )
        style.map("Start.TButton", background=[("active", "#3DA75A")])
        style.configure(
            "Stop.TButton",
            background=_DANGER,
            foreground="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            borderwidth=1,
            padding=(16, 4),
        )
        style.map("Stop.TButton", background=[("active", "#C03030")])
        style.configure(
            "Pause.TButton",
            background=_WARNING,
            foreground="#1E1E1E",
            font=("Segoe UI", 10, "bold"),
            borderwidth=1,
            padding=(16, 4),
        )
        style.map("Pause.TButton", background=[("active", "#D09010")])
        style.configure(
            "Calibrate.TButton",
            background=_ACCENT,
            foreground="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            borderwidth=1,
            padding=(16, 4),
        )
        style.map("Calibrate.TButton", background=[("active", "#005A9E")])

        self._btn_start = ttk.Button(toolbar, text="▶  Start", style="Start.TButton", command=self._on_start)
        self._btn_start.pack(side=tk.LEFT, padx=(0, 4))

        self._btn_stop = ttk.Button(
            toolbar, text="■  Stop", style="Stop.TButton", command=self._on_stop, state=tk.DISABLED
        )
        self._btn_stop.pack(side=tk.LEFT, padx=4)

        self._btn_pause = ttk.Button(
            toolbar, text="⏸  Pause", style="Pause.TButton", command=self._on_pause, state=tk.DISABLED
        )
        self._btn_pause.pack(side=tk.LEFT, padx=4)

        # Separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # Calibration button (Phase 5.2)
        self._btn_calibrate = ttk.Button(
            toolbar,
            text="📐  Calibrate Regions",
            style="Calibrate.TButton",
            command=self._on_calibrate,
        )
        self._btn_calibrate.pack(side=tk.LEFT, padx=4)

        # Macro editor button (Phase 5.4)
        self._btn_macros = ttk.Button(
            toolbar,
            text="📝  Edit Macros",
            style="Calibrate.TButton",
            command=self._on_edit_macros,
        )
        self._btn_macros.pack(side=tk.LEFT, padx=4)

        # Spacer
        tk.Frame(toolbar, bg=_BG).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Profile label
        self._lbl_profile = tk.Label(
            toolbar,
            text="No profile loaded",
            bg=_BG,
            fg=_DISABLED_FG,
            font=("Segoe UI", 9),
        )
        self._lbl_profile.pack(side=tk.RIGHT, padx=8)

    def _build_log_dashboard(self) -> None:
        """Create the log dashboard that fills the remaining space."""
        self.log_dashboard = LogDashboard(self, bg=_BG)
        self.log_dashboard.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))

    def _build_status_bar(self) -> None:
        """Create the bottom status bar."""
        status_frame = tk.Frame(self, bg="#2D2D2D", height=28)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        # Engine state indicator
        self._lbl_engine = tk.Label(
            status_frame,
            text="●  Idle",
            bg="#2D2D2D",
            fg=_DISABLED_FG,
            font=("Segoe UI", 9),
            padx=10,
        )
        self._lbl_engine.pack(side=tk.LEFT)

        # Separator
        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        # Ollama status
        self._lbl_ollama = tk.Label(
            status_frame,
            text="Ollama: —",
            bg="#2D2D2D",
            fg=_DISABLED_FG,
            font=("Segoe UI", 9),
            padx=10,
        )
        self._lbl_ollama.pack(side=tk.LEFT)

        # Separator
        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        # MCP status
        self._lbl_mcp = tk.Label(
            status_frame,
            text="MCP: —",
            bg="#2D2D2D",
            fg=_DISABLED_FG,
            font=("Segoe UI", 9),
            padx=10,
        )
        self._lbl_mcp.pack(side=tk.LEFT)

        # Spacer
        tk.Frame(status_frame, bg="#2D2D2D").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Version label
        from src import __version__

        tk.Label(
            status_frame,
            text=f"v{__version__}",
            bg="#2D2D2D",
            fg=_DISABLED_FG,
            font=("Segoe UI", 9),
            padx=10,
        ).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Engine control callbacks (wired in Phase 6.1)
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        """Start the AI engine."""
        self._logger.info("▶ Start pressed — engine launching (wire in Phase 6.1).")
        self._set_engine_state("Running", _SUCCESS)
        self._btn_start.configure(state=tk.DISABLED)
        self._btn_stop.configure(state=tk.NORMAL)
        self._btn_pause.configure(state=tk.NORMAL)

    def _on_stop(self) -> None:
        """Stop the AI engine."""
        self._logger.info("■ Stop pressed — shutting down (wire in Phase 6.1).")
        self._set_engine_state("Stopping", _WARNING)
        self._btn_start.configure(state=tk.NORMAL)
        self._btn_stop.configure(state=tk.DISABLED)
        self._btn_pause.configure(state=tk.DISABLED)

    def _on_pause(self) -> None:
        """Pause / resume the AI engine."""
        self._logger.info("⏸ Pause pressed — toggling (wire in Phase 6.1).")
        self._set_engine_state("Paused", _WARNING)
        self._btn_pause.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        """Graceful shutdown on window close."""
        self._logger.info("Window close requested — shutting down.")
        self._set_engine_state("Shutting down", _WARNING)
        self.destroy()

    # ------------------------------------------------------------------
    # Calibration (Phases 5.2 & 5.3)
    # ------------------------------------------------------------------

    def _on_new_profile(self) -> None:
        """Create a new profile → launches region calibration."""
        self._on_calibrate()

    def _on_calibrate(self) -> None:
        """Launch the region calibration overlay.

        Takes a screenshot, then opens the transparent overlay for the
        user to draw and assign regions.
        """
        self._logger.info("Launching region calibration overlay …")

        # 1. Get a screenshot
        screenshot = self._get_screenshot()
        if screenshot is None:
            self._logger.error("Could not capture screenshot for calibration.")
            return

        # 2. Load state schema slots for the role dropdown
        state_schema_slots = self._load_state_schema()

        # 3. Load existing regions (if editing an existing profile)
        existing_regions = self._load_existing_regions()

        # 4. Build a live capture callable for Phase 5.3 colour‑bar calibration
        screen_capture_fn = self._build_capture_callable()

        # 5. Open the overlay (modal Toplevel)
        from src.gui.calibration_overlay import CalibrationTool

        tool = CalibrationTool(
            parent=self,
            screenshot=screenshot,
            ocr_module=self._ocr_module,
            existing_regions=existing_regions,
            state_schema_slots=state_schema_slots,
            on_save=self._on_calibration_save,
            screen_capture=screen_capture_fn,
        )

        # Grab focus (modal behaviour)
        tool.grab_set()
        tool.focus_force()

        # Wait for the tool to close
        self.wait_window(tool)

    def _get_screenshot(self) -> "np.ndarray | None":
        """Capture a screenshot for calibration.

        Uses ScreenCapture if wired, otherwise falls back to mss full‑screen.
        """
        try:
            if self._screen_capture is not None:
                # Use the engine's screen capture
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't use await in a tkinter callback — do a direct mss cap
                    return self._fallback_screenshot()
                else:
                    frame = loop.run_until_complete(self._screen_capture.capture())
                    if frame is not None:
                        # If grayscale, convert back to BGR for display
                        import cv2

                        if len(frame.shape) == 2:
                            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        return frame
            return self._fallback_screenshot()
        except Exception as exc:
            self._logger.warning(f"Screenshot capture via engine failed: {exc}")
            return self._fallback_screenshot()

    @staticmethod
    def _fallback_screenshot() -> "np.ndarray | None":
        """Take a full‑screen screenshot using mss directly."""
        try:
            import mss
            import numpy as np

            with mss.MSS() as sct:
                monitor = sct.monitors[0]
                img = sct.grab(monitor)
                return np.array(img)[:, :, :3]  # drop alpha
        except ImportError:
            return None
        except Exception:
            return None

    def _build_capture_callable(self) -> "Callable[[], np.ndarray | None] | None":
        """Build a sync callable for live screen capture (Phase 5.3).

        Returns a zero‑argument callable that grabs a fresh screenshot.
        Used by ``CalibrationTool`` to capture empty/full bar states
        while the overlay is temporarily hidden.

        Returns ``None`` if no capture method is available.
        """
        # Prefer the engine's screen capture if wired
        if self._screen_capture is not None:
            engine = self._screen_capture  # capture reference

            def _engine_capture() -> "np.ndarray | None":
                import asyncio
                import cv2

                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                if loop.is_running():
                    # In a running loop, use mss fallback
                    return MainWindow._fallback_screenshot()
                try:
                    frame = loop.run_until_complete(engine.capture())
                except Exception:
                    return MainWindow._fallback_screenshot()
                if frame is None:
                    return None
                if len(frame.shape) == 2:
                    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if frame.shape[2] == 4:
                    return frame[:, :, :3]
                return frame

            return _engine_capture

        # Fallback: use mss directly
        try:
            import mss  # noqa: F401

            def _mss_capture() -> "np.ndarray | None":
                return MainWindow._fallback_screenshot()

            return _mss_capture
        except ImportError:
            return None

    def _load_state_schema(self) -> dict[str, dict[str, str]]:
        """Load the state schema slots for the role dropdown."""
        if self._config_manager is not None:
            try:
                schema = self._config_manager.load_state_schema()
                return schema.get("slots", {})
            except Exception:
                pass

        # Fallback: load directly from config dir
        schema_path = Path(__file__).resolve().parent.parent.parent / "config" / "state_schema.json"
        if not schema_path.exists():
            schema_path = Path("config") / "state_schema.json"

        try:
            if schema_path.exists():
                data = json.loads(schema_path.read_text(encoding="utf-8"))
                return data.get("slots", {})
        except (json.JSONDecodeError, FileNotFoundError):
            pass

        return {}

    def _load_existing_regions(self) -> list[dict[str, Any]]:
        """Load existing region definitions if editing a profile."""
        if self._config_manager is not None:
            try:
                regions = self._config_manager.load_regions()
                return regions.get("regions", [])
            except Exception:
                pass

        # Fallback: load from profile path
        if self._profile_path is not None:
            regions_path = self._profile_path / "regions.json"
        else:
            regions_path = Path(__file__).resolve().parent.parent.parent / "config" / "regions.json"
            if not regions_path.exists():
                regions_path = Path("config") / "regions.json"

        try:
            if regions_path.exists():
                data = json.loads(regions_path.read_text(encoding="utf-8"))
                return data.get("regions", [])
        except (json.JSONDecodeError, FileNotFoundError):
            pass

        return []

    def _on_calibration_save(self, collected_regions: list[dict[str, Any]]) -> None:
        """Handle the regions collected from the calibration overlay.

        Writes them to regions.json and updates the profile label.
        """
        self._logger.info(f"Saving {len(collected_regions)} region(s) from calibration.")

        region_data = {
            "version": "1.0.0",
            "regions": collected_regions,
        }

        # Try to save via ConfigManager first
        if self._config_manager is not None:
            try:
                self._config_manager.save_regions(region_data)
            except Exception as exc:
                self._logger.warning(f"ConfigManager save failed: {exc} — using direct write")
                self._write_regions_direct(region_data)
        else:
            self._write_regions_direct(region_data)

        self._update_profile_label()

    def _write_regions_direct(self, region_data: dict[str, Any]) -> None:
        """Write regions directly to the config JSON file."""
        if self._profile_path is not None:
            regions_path = self._profile_path / "regions.json"
        else:
            regions_path = Path(__file__).resolve().parent.parent.parent / "config" / "regions.json"
            if not regions_path.parent.exists():
                regions_path = Path("config") / "regions.json"

        regions_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write
        tmp_path = regions_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(region_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp_path.replace(regions_path)
        self._logger.info(f"Regions written to {regions_path}")

    def _update_profile_label(self) -> None:
        """Update the profile indicator in the toolbar."""
        if self._profile_path is not None:
            self._lbl_profile.config(
                text=f"Profile: {self._profile_path.name}", fg=_FG
            )
        else:
            self._lbl_profile.config(
                text=f"Regions configured ({len(self._load_existing_regions())} regions)", fg=_SUCCESS
            )

    # ------------------------------------------------------------------
    # Status bar helpers (public API for Phase 6.1 wiring)
    # ------------------------------------------------------------------

    def _set_engine_state(self, text: str, colour: str) -> None:
        """Update the engine status indicator."""
        self._lbl_engine.config(text=f"●  {text}", fg=colour)

    def set_ollama_status(self, status: str, healthy: bool) -> None:
        """Update the Ollama status label."""
        colour = _SUCCESS if healthy else _DANGER
        self._lbl_ollama.config(text=f"Ollama: {status}", fg=colour)

    def set_mcp_status(self, status: str, healthy: bool) -> None:
        """Update the MCP status label."""
        colour = _SUCCESS if healthy else _DANGER
        self._lbl_mcp.config(text=f"MCP: {status}", fg=colour)

    # ------------------------------------------------------------------
    # Macro editor (Phase 5.4)
    # ------------------------------------------------------------------

    def _on_edit_macros(self) -> None:
        """Open the macro JSON editor as a modal window."""
        self._logger.info("Opening Macro Editor …")
        from src.gui.macro_editor import MacroEditor

        editor = MacroEditor(parent=self)
        editor.grab_set()
        editor.focus_force()
        self.wait_window(editor)
        self._logger.info("Macro Editor closed.")

    # ------------------------------------------------------------------
    # Settings panel (Phase 5.5)
    # ------------------------------------------------------------------

    def _on_preferences(self) -> None:
        """Open the settings panel as a modal dialog."""
        self._logger.info("Opening Settings Panel …")
        from src.gui.settings_panel import SettingsPanel

        panel = SettingsPanel(parent=self)
        panel.grab_set()
        panel.focus_force()
        self.wait_window(panel)
        self._logger.info("Settings Panel closed.")

    # ------------------------------------------------------------------
    # Health Monitor UI (Phase 5.6)
    # ------------------------------------------------------------------

    def _on_health_monitor(self) -> None:
        """Toggle the Health Monitor panel visibility."""
        if self._health_panel is None:
            self._health_panel = HealthPanel(self)
        if self._health_panel_visible:
            self._health_panel.pack_forget()
            self._health_panel_visible = False
            self._logger.info("Health Monitor panel hidden.")
        else:
            # Insert between log dashboard and status bar
            self._health_panel.pack(
                fill=tk.X,
                side=tk.BOTTOM,
                before=self.log_dashboard.master.nametowidget(
                    self.log_dashboard.master.winfo_children()[-1]._name
                ) if False else None,
            )
            # Repack in correct order: toolbar → log dashboard → health panel → status bar
            self._health_panel.pack_forget()
            self._health_panel.pack(
                fill=tk.X,
                side=tk.TOP,
                after=self.log_dashboard,
                before=self._lbl_engine.master,
            )
            self._health_panel_visible = True
            self._logger.info("Health Monitor panel shown.")

    def set_health_monitor(self, health_monitor_instance: Any) -> None:
        """Inject a HealthMonitor and wire its queue to the HealthPanel (Phase 6.1)."""
        if self._health_panel is None:
            self._health_panel = HealthPanel(self)
            self._health_panel.pack_forget()  # hidden by default
        # Store reference for the engine to push status updates
        self._health_monitor = health_monitor_instance

    @property
    def health_panel(self) -> "HealthPanel | None":
        """Public accessor for the HealthPanel widget (used by Phase 6.1 wiring)."""
        return self._health_panel

    # ------------------------------------------------------------------
    # Profile Import/Export (Phase 5.7)
    # ------------------------------------------------------------------

    def _on_import_profile(self) -> None:
        """Open the import profile dialog."""
        self._logger.info("Opening Import Profile dialog …")
        from src.gui.profile_manager_dialog import ProfileManagerDialog

        def _on_imported(profile_name: str, profile_path: Path) -> None:
            """Callback after a successful import."""
            self._profile_path = Path(profile_path)
            self._update_profile_label()
            self._logger.info("Profile '%s' imported and set as active.", profile_name)

        dialog = ProfileManagerDialog(
            parent=self,
            mode="import",
            on_imported=_on_imported,
        )
        dialog.grab_set()
        dialog.focus_force()
        self.wait_window(dialog)

    def _on_export_profile(self) -> None:
        """Open the export profile dialog."""
        self._logger.info("Opening Export Profile dialog …")
        from src.gui.profile_manager_dialog import ProfileManagerDialog

        # Pre-fill profile name if one is active
        profile_name = None
        if self._profile_path is not None:
            profile_name = self._profile_path.name

        dialog = ProfileManagerDialog(
            parent=self,
            mode="export",
            profile_name=profile_name,
        )
        dialog.grab_set()
        dialog.focus_force()
        self.wait_window(dialog)

    # ------------------------------------------------------------------
    # Placeholder
    # ------------------------------------------------------------------

    @staticmethod
    def _placeholder() -> None:
        """No-op for disabled menu items."""
        pass

"""
Main application window for Ludomorph.

Provides:
- AsyncTk-derived root window with Start/Stop/Pause controls
- Embedded log dashboard
- Status bar (engine state, Ollama, MCP)
- Menu bar skeleton (File, Settings, Help)
- Calibration overlay launcher (Phases 5.2 & 5.3)
- Capture Target Panel (Phase 6.8 — window selection or fixed region)
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable

import numpy as np

from src.about import ABOUT_TEXT
from src.gui.async_tk import AsyncTk
from src.gui.capture_target_panel import CaptureTargetPanel
from src.gui.health_panel import HealthPanel
from src.gui.log_dashboard import LogDashboard
from src.gui.theme import ThemeManager, resolve_font_stack
from src.hotkey_listener import GlobalHotkeyListener


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------


class MainWindow(AsyncTk):
    """Top-level application window.

    Layout::

        ┌──────────────────────────────────────────────────────────────┐
        │  Menu bar  (File | Settings | Help)                           │
        ├──────────────────────────────────────────────────────────────┤
        │  Toolbar  [▶ Start] [■ Stop] [⏸ Pause]                       │
        │           [📐 Calibrate Regions] [📝 Edit Macros]              │
        ├──────────────────────────────────────────────────────────────┤
        │  📌 Capture Target    Win/X11 — Window Selection              │
        │  Window: [▼ Game              ] [🔄 Refresh]                   │
        │  Status: Tracking "Game" (1920×1080)                          │
        ├──────────────────────────────────────────────────────────────┤
        │                                                               │
        │  Log Dashboard  (scrollable, real-time)                       │
        │                                                               │
        ├──────────────────────────────────────────────────────────────┤
        │  Status bar  Engine: Idle | Ollama: ? | MCP: ?                │
        └──────────────────────────────────────────────────────────────┘
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # -- Theme ------------------------------------------------------------
        self._tm = ThemeManager()
        self._tm.apply_ttk_styles()
        p = self._tm.palette
        # Resolve fonts for use in tk widgets
        self._ui_font = resolve_font_stack(p.ui_font)
        self._mono_font = resolve_font_stack(p.mono_font)

        self.title("Ludomorph")
        self.configure(bg=p.bg, highlightthickness=0, highlightbackground=p.bg)

        # Minimum size; user can resize larger
        self.minsize(950, 650)
        self.geometry("1000x700")

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Logger for the main window (must come before anything that logs)
        from src.logging_config import get_logger

        self._logger = get_logger(__name__)

        # -- Component references (wired in Phase 6.1) -----------------------
        self._ocr_module: Any = None
        self._screen_capture: Any = None
        self._config_manager: Any = None
        self._engine: Any = None  # GUIEngineBridge (Phase 6.1)
        self._focus_manager: Any = None  # WindowFocusManager (Phase 6.8)

        # Current profile path
        self._profile_path: Path | None = None

        # Health panel (Phase 5.6)
        self._health_panel: HealthPanel | None = None
        self._health_panel_visible: bool = False

        # Capture target panel (Phase 6.8)
        self._capture_target_panel: CaptureTargetPanel | None = None

        # Live preview overlay (User-requested feature)
        self._preview_overlay: Any = None
        self._preview_active: bool = False

        # Debug overlay (Phase 6 extension — ~ hotkey)
        self._debug_overlay: Any = None
        self._hotkey_listener: GlobalHotkeyListener | None = None

        # -- Setup global hotkey listener for debug overlay ------------------
        self._setup_debug_overlay()

        # -- Build UI sections -----------------------------------------------
        self._build_menu()
        self._build_toolbar()
        self._build_capture_target_panel()
        self._build_log_dashboard()
        self._build_status_bar()

        self._logger.info("Main window initialised.")

        # Listen for theme changes
        self.bind("<<ThemeChanged>>", lambda e: self._on_theme_changed())

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

    def set_engine_bridge(self, engine: Any) -> None:
        """Inject the GUIEngineBridge (Phase 6.1).

        Must be called before Start/Stop buttons function correctly.
        """
        self._engine = engine
        # Wire health monitor ref to the health panel
        if engine is not None and hasattr(engine, "health_monitor"):
            self.set_health_monitor(engine.health_monitor)
        # Wire macro data to the debug overlay (if already created)
        if (
            engine is not None
            and self._debug_overlay is not None
            and hasattr(engine, "macro_executor")
            and hasattr(engine, "macros_data")
        ):
            self._debug_overlay.set_macro_data(
                engine.macros_data,
                engine.macro_executor,
                engine._loop,
            )

    def set_focus_manager(self, focus_mgr: Any) -> None:
        """Inject the WindowFocusManager (Phase 6.8).

        Wires it into the CaptureTargetPanel so selecting a window
        triggers a focus attempt.
        """
        self._focus_manager = focus_mgr
        if self._capture_target_panel is not None:
            self._capture_target_panel.set_focus_manager(focus_mgr)

    # ------------------------------------------------------------------
    # Debug Overlay (Phase 6 extension — ~ hotkey)
    # ------------------------------------------------------------------

    def _setup_debug_overlay(self) -> None:
        """Create the global hotkey listener that toggles the debug overlay.

        The listener runs on a daemon thread and dispatches back to the
        GUI thread via ``AsyncTk.call_async()`` when ``~`` is tapped.
        """
        def _on_hotkey() -> None:
            self.call_async(lambda: self._toggle_debug_overlay())

        self._hotkey_listener = GlobalHotkeyListener(_on_hotkey)
        self._logger.info("Debug hotkey listener active (~ to toggle debug overlay).")

    def _toggle_debug_overlay(self) -> None:
        """Show or hide the debug overlay (called on the GUI thread)."""
        if self._debug_overlay is None:
            self._debug_overlay = self._create_debug_overlay()
        self._debug_overlay.toggle()

    def _create_debug_overlay(self) -> Any:
        """Instantiate the DebugOverlay and wire up macro data if available."""
        from src.gui.debug_overlay import DebugOverlay

        overlay = DebugOverlay(parent=self)
        if self._engine is not None and hasattr(self._engine, "macro_executor"):
            overlay.set_macro_data(
                self._engine.macros_data,
                self._engine.macro_executor,
                self._engine._loop,
            )
        self._logger.info("Debug overlay created.")
        return overlay

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        """Create the menu bar."""
        p = self._tm.palette
        menubar = tk.Menu(self, bg=p.bg, fg=p.fg,
                          activebackground=p.accent, activeforeground="#FFFFFF",
                          borderwidth=0, activeborderwidth=0)

        # -- File menu -------------------------------------------------------
        file_menu = tk.Menu(menubar, tearoff=0, bg=p.bg, fg=p.fg,
                            activebackground=p.accent)
        file_menu.add_command(label="New Profile …", command=self._on_new_profile)
        file_menu.add_command(label="Import Profile …", command=self._on_import_profile)
        file_menu.add_command(label="Export Profile …", command=self._on_export_profile)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        # -- View menu -------------------------------------------------------
        view_menu = tk.Menu(menubar, tearoff=0, bg=p.bg, fg=p.fg,
                            activebackground=p.accent)
        view_menu.add_command(label="Live Preview Overlay", command=self._on_toggle_live_preview)
        view_menu.add_command(label="Health Monitor …", command=self._on_health_monitor)
        menubar.add_cascade(label="View", menu=view_menu)

        # Store the View menu for updating the Live Preview checkmark
        self._view_menu = view_menu

        # -- Settings menu ---------------------------------------------------
        settings_menu = tk.Menu(menubar, tearoff=0, bg=p.bg, fg=p.fg,
                                activebackground=p.accent)
        settings_menu.add_command(label="Preferences …", command=self._on_preferences)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        # -- Help menu -------------------------------------------------------
        help_menu = tk.Menu(menubar, tearoff=0, bg=p.bg, fg=p.fg,
                            activebackground=p.accent)
        help_menu.add_command(label="About", command=self._on_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _build_toolbar(self) -> None:
        """Create the Start / Stop / Pause button bar with calibration."""
        p = self._tm.palette
        toolbar = tk.Frame(self, bg=p.bg, padx=8, pady=6)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        # ttk styles are already applied by _on_theme_changed via ThemeManager

        self._btn_start = ttk.Button(toolbar, text="▶  Start",
                                     style="Start.TButton", command=self._on_start)
        self._btn_start.pack(side=tk.LEFT, padx=(0, 4))

        self._btn_stop = ttk.Button(
            toolbar, text="■  Stop", style="Stop.TButton",
            command=self._on_stop, state=tk.DISABLED
        )
        self._btn_stop.pack(side=tk.LEFT, padx=4)

        self._btn_pause = ttk.Button(
            toolbar, text="⏸  Pause", style="Pause.TButton",
            command=self._on_pause, state=tk.DISABLED
        )
        self._btn_pause.pack(side=tk.LEFT, padx=4)

        # Separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # Calibration button (Phase 5.2) — game-state region calibration
        self._btn_calibrate = ttk.Button(
            toolbar, text="📐  Calibrate Game Regions",
            style="Calibrate.TButton", command=self._on_calibrate,
        )
        self._btn_calibrate.pack(side=tk.LEFT, padx=4)

        # Macro editor button (Phase 5.4)
        self._btn_macros = ttk.Button(
            toolbar, text="📝  Edit Macros",
            style="Calibrate.TButton", command=self._on_edit_macros,
        )
        self._btn_macros.pack(side=tk.LEFT, padx=4)

        # Live Preview overlay toggle button (User-requested feature)
        self._btn_preview = ttk.Button(
            toolbar, text="👁  Preview",
            style="Calibrate.TButton", command=self._on_toggle_live_preview,
        )
        self._btn_preview.pack(side=tk.LEFT, padx=4)

        # Spacer
        tk.Frame(toolbar, bg=p.bg).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Profile label (compact, right-aligned)
        self._lbl_profile = tk.Label(
            toolbar,
            text="",
            bg=p.bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 9),
        )
        self._lbl_profile.pack(side=tk.RIGHT, padx=8)

    def _build_capture_target_panel(self) -> None:
        """Create the Capture Target Panel (Phase 6.8).

        Placed between the toolbar and the log dashboard.  Auto-detects
        platform and shows the appropriate mode (Win/X11 or Wayland).
        """
        self._capture_target_panel = CaptureTargetPanel(
            parent=self,
            config_manager=self._get_config_provider,
            focus_manager=self._focus_manager,
        )
        self._capture_target_panel.pack(fill=tk.X, side=tk.TOP, padx=4, pady=(0, 2))

    def _get_config_provider(self) -> tuple[dict[str, Any], Callable[[dict[str, Any]], None]]:
        """Return (config_dict, save_func) for the CaptureTargetPanel.

        Uses the ConfigManager if available, otherwise loads from disk.
        """
        from src.config_manager import load_global_config, save_global_config

        config = load_global_config()
        return config, save_global_config

    def _build_log_dashboard(self) -> None:
        """Create the log dashboard that fills the remaining space."""
        self.log_dashboard = LogDashboard(self)
        self.log_dashboard.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))

    def _build_status_bar(self) -> None:
        """Create the bottom status bar."""
        p = self._tm.palette
        status_frame = tk.Frame(self, bg=p.status_bar_bg, height=28)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        # Engine state indicator
        self._lbl_engine = tk.Label(
            status_frame,
            text="●  Idle",
            bg=p.status_bar_bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 9),
            padx=10,
        )
        self._lbl_engine.pack(side=tk.LEFT)

        # Separator
        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        # Ollama status
        self._lbl_ollama = tk.Label(
            status_frame,
            text="Ollama: —",
            bg=p.status_bar_bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 9),
            padx=10,
        )
        self._lbl_ollama.pack(side=tk.LEFT)

        # Separator
        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        # MCP status
        self._lbl_mcp = tk.Label(
            status_frame,
            text="MCP: —",
            bg=p.status_bar_bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 9),
            padx=10,
        )
        self._lbl_mcp.pack(side=tk.LEFT)

        # Spacer
        tk.Frame(status_frame, bg=p.status_bar_bg).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        # Version label
        from src import __version__

        tk.Label(
            status_frame,
            text=f"v{__version__}",
            bg=p.status_bar_bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 9),
            padx=10,
        ).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Theme change handler
    # ------------------------------------------------------------------

    def _on_theme_changed(self) -> None:
        """Re-apply colours to all widgets after a theme switch."""
        self._tm.apply_ttk_styles()
        self._tm = ThemeManager()  # refresh singleton
        p = self._tm.palette
        self._ui_font = resolve_font_stack(p.ui_font)
        self._mono_font = resolve_font_stack(p.mono_font)
        self.configure(bg=p.bg)
        # Rebuild menu bar colours
        self._rebuild_menu_colours()
        # Rebuild toolbar colours
        self._rebuild_toolbar_colours()
        # Update status bar
        self._rebuild_status_bar_colours()
        # Update profile label
        self._update_profile_label()
        # Rebuild log dashboard
        self.log_dashboard._apply_theme()

    def _rebuild_menu_colours(self) -> None:
        """Update menu bar colours after theme change."""
        p = self._tm.palette
        # tk.Menu doesn't easily allow reconfiguring; we rebuild it
        if hasattr(self, "_view_menu"):
            self._build_menu()

    def _rebuild_toolbar_colours(self) -> None:
        """Update toolbar background colours after theme change."""
        p = self._tm.palette
        for child in self.winfo_children():
            if isinstance(child, tk.Frame) and child.winfo_manager() == "pack":
                # Toolbar frame is the first packed frame after menu
                try:
                    info = child.pack_info()
                    if info.get("side") == "top" and info.get("fill") == "x":
                        child.configure(bg=p.bg)
                        for sub in child.winfo_children():
                            if isinstance(sub, tk.Frame):
                                sub.configure(bg=p.bg)
                            elif isinstance(sub, tk.Label):
                                sub.configure(bg=p.bg)
                except tk.TclError:
                    pass

    def _rebuild_status_bar_colours(self) -> None:
        """Update status bar colours after theme change."""
        p = self._tm.palette
        bar_bg = p.status_bar_bg
        self._lbl_engine.configure(bg=bar_bg)
        self._lbl_ollama.configure(bg=bar_bg)
        self._lbl_mcp.configure(bg=bar_bg)
        # Also update any children of the status_frame
        for child in self._lbl_engine.master.winfo_children():
            if isinstance(child, tk.Frame):
                child.configure(bg=bar_bg)

    # ------------------------------------------------------------------
    # Engine control callbacks (wired in Phase 6.1)
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        """Start the AI engine."""
        if self._engine is None:
            self._logger.warning("Start pressed but engine bridge not wired.")
            return

        self._logger.info("▶ Start pressed — launching engine.")
        self._btn_start.configure(state=tk.DISABLED)
        self._btn_stop.configure(state=tk.NORMAL)
        self._btn_pause.configure(state=tk.NORMAL)
        # Trigger engine start on the asyncio thread
        self._engine.start(profile_path=self._profile_path)

    def _on_stop(self) -> None:
        """Stop the AI engine."""
        if self._engine is None:
            self._logger.warning("Stop pressed but engine bridge not wired.")
            return

        self._logger.info("■ Stop pressed — shutting down engine.")
        self._btn_start.configure(state=tk.NORMAL)
        self._btn_stop.configure(state=tk.DISABLED)
        self._btn_pause.configure(state=tk.DISABLED)
        self._engine.stop()

    def _on_pause(self) -> None:
        """Pause / resume the AI engine."""
        if self._engine is None:
            self._logger.warning("Pause pressed but engine bridge not wired.")
            return

        from src.gui.engine_bridge import EngineState

        if self._engine.state == EngineState.RUNNING:
            self._logger.info("⏸ Pause pressed — pausing engine.")
            self._engine.pause()
            self._btn_pause.configure(text="▶  Resume")
        elif self._engine.state == EngineState.PAUSED:
            self._logger.info("▶ Resume pressed — resuming engine.")
            self._engine.pause()  # toggle back
            self._btn_pause.configure(text="⏸  Pause")

    def _on_close(self) -> None:
        """Full application shutdown: engine, Ollama, GUI, MCP, everything.

        We do NOT call ``remove_gui_sink()`` here — that function blocks
        indefinitely because ``logger.remove()`` internally joins the
        loguru writer thread, which never finishes draining its backlog
        of enqueued messages.  Instead we shut down the engine (stopping
        the periodic background tasks that generate log output), then
        tear down tkinter.  ``launch_gui()`` calls ``os._exit(0)`` as a
        hard guarantee that kills all remaining threads.
        """
        p = self._tm.palette
        self._logger.info("Window close requested — full shutdown initiated.")
        self._set_engine_state("Shutting down …", p.warning)

        # 0. Stop the hotkey listener
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
            except Exception:
                pass
            self._hotkey_listener = None

        # 0b. Close the debug overlay
        if self._debug_overlay is not None:
            try:
                self._debug_overlay.destroy()
            except Exception:
                pass
            self._debug_overlay = None

        # 1. Stop engine — this cancels the decision loop, health polling,
        #    preview polling, and summariser.  Once these stop, no more log
        #    messages are generated, so the loguru writer thread can drain
        #    naturally.
        if self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception as exc:
                self._logger.warning(f"Engine shutdown error: {exc}")

        # 2. Unload Ollama models to free GPU/system memory
        self._unload_ollama()

        # 3. Tear down tkinter window and break mainloop.
        #    quit() causes app.mainloop() to return in launch_gui(),
        #    which will then call os._exit() as a hard guarantee.
        self.destroy()
        self.quit()

    def _unload_ollama(self) -> None:
        """Request Ollama to unload all models from memory.

        Sends a keep_alive=0 request to the Ollama API so models
        are freed rather than lingering in GPU VRAM.
        """
        import json
        import urllib.request

        ollama_url = ""
        ollama_model = ""
        try:
            from src.config_manager import load_global_config
            config = load_global_config()
            ollama_url = config.get("ollama_url", "http://localhost:11434/v1")
            ollama_model = config.get("ollama_model", "")
        except Exception:
            pass

        if not ollama_model:
            self._logger.info("No Ollama model configured — skipping unload.")
            return

        # Build the Ollama native API URL (not the v1 endpoint)
        base_url = ollama_url.rstrip("/").replace("/v1", "")
        if "/v1" not in ollama_url:
            base_url = ollama_url.rstrip("/")
        unload_url = f"{base_url}/api/generate"

        payload = json.dumps({
            "model": ollama_model,
            "keep_alive": 0,
            "prompt": "",  # minimal prompt — just to trigger unload
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                unload_url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            self._logger.info(f"Ollama unload request sent for model '{ollama_model}'.")
        except Exception as exc:
            self._logger.debug(f"Ollama unload request failed (non-critical): {exc}")

    # ------------------------------------------------------------------
    # Engine lifecycle callbacks (called from GUIEngineBridge via call_async)
    # ------------------------------------------------------------------

    def _on_engine_started(self) -> None:
        """Update button states when the engine transitions to RUNNING."""
        self._btn_start.configure(state=tk.DISABLED)
        self._btn_stop.configure(state=tk.NORMAL)
        self._btn_pause.configure(text="⏸  Pause", state=tk.NORMAL)

    def _on_engine_stopped(self) -> None:
        """Update button states when the engine transitions to IDLE."""
        p = self._tm.palette
        self._set_engine_state("Idle", p.disabled_fg)
        self._btn_start.configure(state=tk.NORMAL)
        self._btn_stop.configure(state=tk.DISABLED)
        self._btn_pause.configure(text="⏸  Pause", state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Calibration (Phases 5.2 & 5.3)
    # ------------------------------------------------------------------

    def _on_new_profile(self) -> None:
        """Create a new profile → launches region calibration."""
        self._on_calibrate()

    def _on_calibrate(self) -> None:
        """Launch the region calibration overlay.

        Hides the Game Master window temporarily, takes a screenshot of
        the game underneath, then opens the transparent overlay for the
        user to draw and assign regions.
        """
        self._logger.info("Launching region calibration overlay …")

        # Hide this window so it doesn't appear in the game screenshot.
        # Some window managers (KWin, Mutter) need extra time to repaint
        # and may also composite the withdraw asynchronously.  We poll
        # winfo_viewable() until the window is confirmed hidden, then
        # add an extra 200 ms safety margin.
        self.withdraw()
        self.update_idletasks()
        import time as _time

        # Wait up to 1 s for the WM to confirm the window is hidden
        for _ in range(20):
            self.update_idletasks()
            _time.sleep(0.05)
            if not self.winfo_viewable():
                break
        # Safety margin: extra time for compositor to finish repainting
        _time.sleep(0.2)

        try:
            # 1. Get a screenshot of just the game
            screenshot = self._get_screenshot()
        finally:
            # Restore the Game Master window before opening the overlay
            self.deiconify()
            self.lift()
            self.focus_force()

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
        """Capture a full-colour screenshot of the game for calibration.

        Goes through the configured capture target (window title, fixed
        region, or active window via the engine's WindowTracker) so the
        calibration overlay shows the actual game content — not the full
        desktop.

        Falls back to full-screen mss capture only when no game window
        target is configured.

        Stores the captured resolution in ``self._calibration_source_resolution``
        so it can be saved into regions.json for downstream coordinate scaling.
        """
        result: np.ndarray | None = None

        # 1. Try the engine's ScreenCapture with a fresh raw capture
        if self._screen_capture is not None:
            try:
                tracker = getattr(self._screen_capture, "_tracker", None)
                if tracker is not None:
                    rect = tracker.get_active_window_rect()
                    if rect is not None:
                        result = self._capture_region_sync(*rect)
                        if result is not None:
                            self._calibration_source_resolution = (result.shape[1], result.shape[0])
                            return result
            except Exception:
                pass

        # 2. Try config-based capture_region
        if self._config_manager is not None:
            try:
                config = self._config_manager.load_global_config() if callable(getattr(self._config_manager, "load_global_config", None)) else {}
            except Exception:
                config = {}
        else:
            config = {}

        capture_region = config.get("capture_region")
        if isinstance(capture_region, list) and len(capture_region) == 4:
            result = self._capture_region_sync(*capture_region)
            if result is not None:
                self._calibration_source_resolution = (result.shape[1], result.shape[0])
            return result

        # 3. Try window_title from config
        window_title = config.get("window_title")
        if window_title:
            try:
                import pywinctl as pwc
                windows = pwc.getWindowsWithTitle(window_title)
                if windows:
                    win = windows[0]
                    left, top, right, bottom = win.getClientFrame()
                    width = right - left
                    height = bottom - top
                    if width > 20 and height > 20:
                        result = self._capture_region_sync(left, top, width, height)
                        if result is not None:
                            self._calibration_source_resolution = (result.shape[1], result.shape[0])
                        return result
            except Exception:
                pass

        # 4. Full-desktop fallback (last resort)
        result = self._fallback_screenshot()
        if result is not None:
            self._calibration_source_resolution = (result.shape[1], result.shape[0])
        return result

    @staticmethod
    def _capture_region_sync(left: int, top: int, width: int, height: int) -> "np.ndarray | None":
        """Synchronously capture a specific region using mss."""
        try:
            import mss
            import numpy as np

            with mss.MSS() as sct:
                monitor = {"top": top, "left": left, "width": max(width, 1), "height": max(height, 1)}
                img = sct.grab(monitor)
                return np.array(img)[:, :, :3]  # drop alpha
        except ImportError:
            return None
        except Exception:
            return None

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
        Also records the source resolution of the screenshot so the engine
        can correctly scale region coordinates when frames are downsampled.

        Converts the calibration tool's ``bbox`` format to the canonical
        ``bounds`` format expected by ``RegionProfile.from_dict()``.
        """
        self._logger.info(f"Saving {len(collected_regions)} region(s) from calibration.")

        # Record source resolution from the last calibration screenshot
        source_resolution = getattr(self, "_calibration_source_resolution", None)

        # Normalise regions from calibration tool's bbox format to the
        # canonical bounds [x1, y1, x2, y2] format stored in regions.json
        from src.utils.region_normalizer import normalise_region_bounds
        normalised_regions: list[dict[str, Any]] = []
        for r in collected_regions:
            nb = normalise_region_bounds(r)
            normalised = {
                "name": r.get("name", "unnamed"),
                "type": r.get("type", "ocr"),
                "role": r.get("role", r.get("name", "unnamed")),
                "bounds": [nb["x"], nb["y"], nb["x"] + nb["width"], nb["y"] + nb["height"]],
            }
            # Preserve any calibration data for colour bars
            if "calibration" in r:
                normalised["calibration"] = r["calibration"]
            if "bar_type" in r:
                normalised["calibration"] = {**(r.get("calibration", {})), "bar_type": r["bar_type"], "orientation": r.get("orientation", "horizontal")}
            if "preprocess" in r:
                normalised["preprocess"] = r["preprocess"]
            if "ocr" in r:
                normalised["ocr"] = r["ocr"]
            normalised_regions.append(normalised)

        region_data: dict[str, Any] = {
            "version": "1.0.0",
            "regions": normalised_regions,
        }
        if source_resolution is not None and isinstance(source_resolution, tuple) and len(source_resolution) == 2:
            region_data["source_resolution"] = {"width": source_resolution[0], "height": source_resolution[1]}
            self._logger.info(f"Source resolution recorded: {source_resolution[0]}×{source_resolution[1]}")

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
        """Update the profile indicator in the toolbar and calibrate button."""
        p = self._tm.palette
        if self._profile_path is not None:
            self._lbl_profile.config(
                text=f"📁 {self._profile_path.name}", fg=p.fg
            )
            # Show region count on the calibrate button
            self._btn_calibrate.configure(
                text=f"📐  Calibrate Game Regions ({len(self._load_existing_regions())} regions)"
            )
        else:
            num_regions = len(self._load_existing_regions())
            self._lbl_profile.config(text="")
            self._btn_calibrate.configure(
                text=f"📐  Calibrate Game Regions ({num_regions} regions)"
            )

    # ------------------------------------------------------------------
    # Status bar helpers (public API for Phase 6.1 wiring)
    # ------------------------------------------------------------------

    def _set_engine_state(self, text: str, colour: str) -> None:
        """Update the engine status indicator."""
        self._lbl_engine.config(text=f"●  {text}", fg=colour)

    def set_ollama_status(self, status: str, healthy: bool) -> None:
        """Update the Ollama status label."""
        p = self._tm.palette
        colour = p.success if healthy else p.danger
        self._lbl_ollama.config(text=f"Ollama: {status}", fg=colour)

    def set_mcp_status(self, status: str, healthy: bool) -> None:
        """Update the MCP status label."""
        p = self._tm.palette
        colour = p.success if healthy else p.danger
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
            # Pack the health panel between capture target panel and the
            # log dashboard so it never overlaps the bottom status bar.
            # Using before=self.log_dashboard ensures the log dashboard
            # stays below and can still expand/shrink properly.
            self._health_panel.pack(
                fill=tk.X,
                side=tk.TOP,
                before=self.log_dashboard,
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
    # Live Preview Overlay (User-requested feature)
    # ------------------------------------------------------------------

    def _on_toggle_live_preview(self) -> None:
        """Toggle the Live Preview Overlay window on/off."""
        if self._preview_active:
            self._close_live_preview()
        else:
            self._open_live_preview()

    def _open_live_preview(self) -> None:
        """Create and show the Live Preview Overlay."""
        if self._preview_overlay is not None and not self._preview_overlay.is_closed:
            self._preview_overlay.lift()
            self._preview_overlay.focus_force()
            self._preview_active = True
            return

        from src.gui.live_preview_overlay import LivePreviewOverlay

        self._preview_overlay = LivePreviewOverlay(parent=self)
        self._preview_active = True

        # Force the window to realize so canvas dimensions are valid
        self._preview_overlay.update_idletasks()

        self._logger.info("Live Preview Overlay opened.")

        # When the engine is already running, push current regions after
        # a short delay so the canvas is fully mapped.
        if self._engine is not None:
            regions = self._load_existing_regions()
            overlay_ref = self._preview_overlay  # capture for closure safety
            self.after(150, lambda: overlay_ref.push_frame(regions=regions) if overlay_ref and not overlay_ref.is_closed else None)

    def _close_live_preview(self) -> None:
        """Close the Live Preview Overlay."""
        self._preview_active = False
        if self._preview_overlay is not None:
            try:
                self._preview_overlay.close()
            except Exception:
                pass
            self._preview_overlay = None
        self._logger.info("Live Preview Overlay closed.")

    def _on_live_preview_closed(self) -> None:
        """Called by LivePreviewOverlay when the user closes its window."""
        self._preview_active = False
        self._preview_overlay = None
        self._logger.info("Live Preview Overlay closed by user.")

    def push_preview_frame(
        self,
        frame: "np.ndarray | None" = None,
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
        """Push preview data to the Live Preview Overlay.

        This is the public API called from ``GUIEngineBridge`` via
        ``AsyncTk.call_async()``.  If the overlay is not active, the
        call is a no-op.

        Args:
            frame: BGR numpy array of the latest captured frame.
            regions: Region definitions from the active profile.
            state_data: Extracted game state (OCR text, bar fill %).
            detections: Vision/YOLO detection objects.
            last_action: Name of the macro the AI decided to execute.
            action_confidence: LLM confidence score (0‑1).
            cycle_time_ms: Decision loop cycle time in milliseconds.
            fps: Frames per second.
            engine_state: ``"running"``, ``"paused"``, or ``"stopped"``.
        """
        if not self._preview_active or self._preview_overlay is None:
            return
        try:
            self._preview_overlay.push_frame(
                frame=frame,
                regions=regions,
                state_data=state_data,
                detections=detections,
                last_action=last_action,
                action_confidence=action_confidence,
                cycle_time_ms=cycle_time_ms,
                fps=fps,
                engine_state=engine_state,
            )
        except Exception:
            pass  # Window may have been destroyed between check and push

    # ------------------------------------------------------------------
    # Placeholder
    # ------------------------------------------------------------------

    def _on_about(self) -> None:
        """Show the About dialog."""
        from tkinter import messagebox

        messagebox.showinfo("About Ludomorph", ABOUT_TEXT)

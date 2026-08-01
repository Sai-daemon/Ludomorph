"""
GUI package for Ludomorph — Phase 5 / Phase 6.1.

Provides the Tkinter-based desktop shell with:
- AsyncTk bridge (tkinter + asyncio interop)
- Real-time log dashboard
- Main application window with Start/Stop controls
- GUIEngineBridge — wires GUI to the async engine (Phase 6.1)
"""

from __future__ import annotations

from src.gui.async_tk import AsyncTk
from src.gui.engine_bridge import GUIEngineBridge, EngineState
from src.gui.log_dashboard import LogDashboard, setup_gui_sink, remove_gui_sink
from src.gui.main_window import MainWindow
from src.gui.calibration_overlay import CalibrationTool, RegionRole
from src.gui.health_panel import HealthPanel
from src.gui.macro_editor import MacroEditor
from src.gui.macro_builder import MacroBuilder
from src.gui.settings_panel import SettingsPanel
from src.gui.profile_manager_dialog import ProfileManagerDialog

__all__ = [
    "AsyncTk",
    "CalibrationTool",
    "EngineState",
    "GUIEngineBridge",
    "HealthPanel",
    "LogDashboard",
    "MacroEditor",
    "MainWindow",
    "ProfileManagerDialog",
    "RegionRole",
    "SettingsPanel",
    "setup_gui_sink",
    "remove_gui_sink",
    "launch_gui",
]


def launch_gui() -> None:
    """Entry point to launch the GUI application with full engine wiring (Phase 6.1).

    1. Loads the global config.
    2. Creates InputController, WindowFocusManager.
    3. Creates MainWindow (tkinter root) + log dashboard sink.
    4. Creates GUIEngineBridge (asyncio in background thread).
    5. Wires the bridge into MainWindow.
    6. Starts the tkinter mainloop (blocks until window closes).
    7. On close: engine shutdown → destroy window.
    """
    import sys

    from src.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Launching Ludomorph GUI ...")

    # ------------------------------------------------------------------
    # 1. Load global config
    # ------------------------------------------------------------------
    from src.config_manager import load_global_config

    try:
        config = load_global_config()
    except Exception as exc:
        logger.error(f"Failed to load config: {exc}")
        sys.exit(1)

    logger.info(
        "Config loaded. Ollama URL: %s, Model: %s",
        config.get("ollama_url"),
        config.get("ollama_model"),
    )

    # Apply configured log level
    configured_level = config.get("log_level", "INFO")
    if configured_level != "INFO":
        from src.logging_config import setup_logging

        setup_logging(log_level=configured_level)
        logger.debug("Log level updated to %s", configured_level)

    # ------------------------------------------------------------------
    # 2. Create InputController
    # ------------------------------------------------------------------
    from src.input_controller import InputController, InputError

    try:
        input_ctrl = InputController(config)
        logger.info("Input backend active: %s", input_ctrl.backend_name)
    except InputError as exc:
        logger.error(f"Input initialisation failed: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Create WindowFocusManager
    # ------------------------------------------------------------------
    from src.window_focus import WindowFocusManager

    try:
        focus_mgr = WindowFocusManager(config, input_controller=input_ctrl)
        logger.info("Window focus manager ready. Compositor: %s", focus_mgr.compositor.value)
    except Exception as exc:
        import traceback as _tb
        logger.error("Failed to create WindowFocusManager: %s\n%s", exc, _tb.format_exc())
        focus_mgr = None  # Non‑fatal — engine can run without focus mgr
        logger.warning("Continuing without WindowFocusManager — auto‑focus unavailable.")

    # ------------------------------------------------------------------
    # 4. Create MainWindow
    # ------------------------------------------------------------------
    app = MainWindow()
    setup_gui_sink(app.log_dashboard)

    # ------------------------------------------------------------------
    # 5. Create GUIEngineBridge (spawns asyncio background thread)
    # ------------------------------------------------------------------
    try:
        engine = GUIEngineBridge(
            config=config,
            input_ctrl=input_ctrl,
            focus_mgr=focus_mgr,
            main_window=app,
            profile_path=None,  # user can import/select a profile later
        )
        # Wire the bridge into MainWindow so buttons work
        app.set_engine_bridge(engine)
        # Wire the focus manager for the Capture Target Panel (Phase 6.8)
        app.set_focus_manager(focus_mgr)
        logger.info("GUIEngineBridge created and wired to MainWindow.")
    except Exception as exc:
        logger.error(f"Failed to create engine bridge: {exc}")
        # Still launch the GUI — the user can calibrate regions etc.
        # Start/Stop will show warnings if used without an engine.
        app.set_engine_bridge(None)

    # ------------------------------------------------------------------
    # 6. Run the tkinter mainloop (blocks until window closed)
    # ------------------------------------------------------------------
    try:
        app.mainloop()
    except KeyboardInterrupt:
        remove_gui_sink()
        logger.info("GUI interrupted by user.")
    finally:
        # Engine was already shut down and the GUI sink removed
        # in MainWindow._on_close() — nothing left to clean up.
        logger.info("GUI closed.")

    # 7. Hard process exit guarantee.
    #    _on_close() calls self.quit() which causes mainloop() to return.
    #    os._exit() forces the process to terminate immediately even if
    #    daemon threads or Tcl cleanup are still pending.
    import os as _os
    _os._exit(0)

"""
GUI package for AI Game Master — Phase 5.

Provides the Tkinter-based desktop shell with:
- AsyncTk bridge (tkinter + asyncio interop)
- Real-time log dashboard
- Main application window with Start/Stop controls
"""

from __future__ import annotations

from src.gui.async_tk import AsyncTk
from src.gui.log_dashboard import LogDashboard, setup_gui_sink, remove_gui_sink
from src.gui.main_window import MainWindow
from src.gui.calibration_overlay import CalibrationTool, RegionRole
from src.gui.health_panel import HealthPanel
from src.gui.macro_editor import MacroEditor
from src.gui.settings_panel import SettingsPanel
from src.gui.profile_manager_dialog import ProfileManagerDialog

__all__ = [
    "AsyncTk",
    "CalibrationTool",
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
    """Entry point to launch the GUI application.

    Creates the main window, wires the log dashboard, and starts the
    tkinter mainloop. The asyncio event loop is run in a background
    thread via ``AsyncTk``.
    """
    from src.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Launching AI Game Master GUI ...")

    app = MainWindow()
    setup_gui_sink(app.log_dashboard)

    try:
        app.mainloop()
    except KeyboardInterrupt:
        remove_gui_sink()
        logger.info("GUI interrupted by user.")
    finally:
        remove_gui_sink()

"""
Phase 6.8 — Capture Target Panel.

A distinct toolbar section that lets the user choose which window the agent
operates on.  Auto-detects the platform on construction and renders either:

* **Win / X11** — combobox of enumerated window titles + refresh button
* **Wayland / unknown** — fixed-region label + "Calibrate Capture Region" button

Saves the selection to ``config.json`` via ``ConfigManager`` and triggers a
focus attempt via ``WindowFocusManager``.

Uses the centralised theme system from ``src.gui.theme``.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

import os
import sys

from src.logging_config import get_logger
from src.gui.theme import ThemeManager, resolve_font_stack

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------


def _is_wayland() -> bool:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


def _platform_label() -> str:
    """Human-readable name for the active platform."""
    if sys.platform == "win32":
        return "Windows"
    if _is_wayland():
        return "Wayland"
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "x11":
        return "X11"
    return "Unknown"


# ---------------------------------------------------------------------------
# CaptureTargetPanel
# ---------------------------------------------------------------------------


class CaptureTargetPanel(tk.Frame):
    """Platform-aware capture-target selection panel.

    Built as a horizontal frame intended to sit in the main toolbar
    below the Start/Stop buttons and above the log dashboard.

    Parameters
    ----------
    parent : tk.Widget
        Parent widget (MainWindow toolbar frame).
    config_manager : callable or None
        Callable that returns ``(config_dict, save_func)``, or ``None``
        for read‑only mode.  *config_dict* is the current global config,
        *save_func* accepts a dict to persist.
    focus_manager : ``WindowFocusManager`` or None
        Used to focus a selected window on Win/X11.  If ``None``, the
        "Focus" step after selection is silently skipped.
    """

    def __init__(
        self,
        parent: tk.Widget,
        config_manager: Any = None,
        focus_manager: Any = None,
    ) -> None:
        self._tm = ThemeManager()
        p = self._tm.palette
        self._ui_font = resolve_font_stack(p.ui_font)
        self._mono_font = resolve_font_stack(p.mono_font)

        super().__init__(parent, bg=p.bg, padx=8, pady=4)
        self._config_manager = config_manager
        self._focus_manager = focus_manager

        # Determine platform
        self._is_wayland = _is_wayland()
        self._platform_name = _platform_label()

        # Load current config
        self._current_window_title: str | None = None
        self._current_capture_region: tuple[int, int, int, int] | None = None
        self._reload_config()

        # Build the appropriate layout
        self._build_separator()
        self._build_section_header()

        if self._is_wayland:
            self._build_wayland_layout()
        else:
            self._build_windows_x11_layout()

        # Status label (shared)
        self._build_status_line()

    # ------------------------------------------------------------------
    # Convenience palette accessor
    # ------------------------------------------------------------------

    @property
    def _p(self):
        """Return the active palette for concise access."""
        return self._tm.palette

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _reload_config(self) -> None:
        """Read window_title and capture_region from the live config."""
        config = self._get_config_dict()
        self._current_window_title = config.get("window_title") or None
        region = config.get("capture_region")
        if isinstance(region, list) and len(region) == 4:
            self._current_capture_region = tuple(region)  # type: ignore[arg-type]
        else:
            self._current_capture_region = None

    def _get_config_dict(self) -> dict[str, Any]:
        """Return the current global config dict, or an empty dict."""
        if self._config_manager is not None:
            try:
                cfg, _ = self._config_manager()
                return cfg
            except Exception:
                pass
        # Fallback: read from disk
        try:
            from src.config_manager import load_global_config
            return load_global_config()
        except Exception:
            return {}

    def _save_config(self, updates: dict[str, Any]) -> None:
        """Atomically update the global config with *updates*."""
        config = self._get_config_dict()
        config.update(updates)

        if self._config_manager is not None:
            try:
                _, save = self._config_manager()
                save(config)
                logger.info("Capture target saved via ConfigManager.")
                return
            except Exception as exc:
                logger.warning(f"ConfigManager save failed: {exc}")

        # Direct write fallback
        try:
            from src.config_manager import save_global_config
            save_global_config(config)
        except Exception as exc:
            logger.error(f"Failed to persist capture target: {exc}")

    # ------------------------------------------------------------------
    # Layout sections
    # ------------------------------------------------------------------

    def _build_separator(self) -> None:
        """Horizontal rule between controls and capture panel."""
        p = self._p
        sep = tk.Frame(self, bg=p.separator, height=1)
        sep.pack(fill=tk.X, pady=(0, 6))

    def _build_section_header(self) -> None:
        """Section title: '📌 Capture Target' + platform badge."""
        p = self._p
        header = tk.Frame(self, bg=p.bg)
        header.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            header,
            text="📌  Capture Target",
            bg=p.bg,
            fg=p.fg,
            font=(self._ui_font, 11, "bold"),
        ).pack(side=tk.LEFT)

        mode_text = (
            f"{self._platform_name} detected — Fixed Region Only"
            if self._is_wayland
            else f"{self._platform_name} detected — Window Selection"
        )
        self._lbl_mode = tk.Label(
            header,
            text=mode_text,
            bg=p.bg,
            fg=p.subtle,
            font=(self._ui_font, 9),
            padx=12,
        )
        self._lbl_mode.pack(side=tk.LEFT)

    def _build_status_line(self) -> None:
        """Persistent status line showing the current capture target."""
        p = self._p
        self._lbl_status = tk.Label(
            self,
            text="",
            bg=p.bg,
            fg=p.subtle,
            font=(self._ui_font, 9),
            anchor=tk.W,
            padx=4,
        )
        self._lbl_status.pack(fill=tk.X, pady=(2, 0))
        self._refresh_status()

    # ------------------------------------------------------------------
    # Win / X11 layout
    # ------------------------------------------------------------------

    def _build_windows_x11_layout(self) -> None:
        """Combobox + Refresh button for window selection."""
        p = self._p
        row = tk.Frame(self, bg=p.bg)
        row.pack(fill=tk.X, pady=(2, 0))

        tk.Label(
            row,
            text="Window:",
            bg=p.bg,
            fg=p.fg,
            font=(self._ui_font, 10),
            width=8,
            anchor=tk.W,
        ).pack(side=tk.LEFT, padx=(0, 4))

        # Combobox (editable so the user can type partial titles)
        self._window_var = tk.StringVar()
        self._combo = ttk.Combobox(
            row,
            textvariable=self._window_var,
            values=[],
            state="normal",  # allow typing
            font=(self._ui_font, 10),
            width=32,
        )
        self._combo.pack(side=tk.LEFT, padx=(0, 4))
        self._combo.bind("<<ComboboxSelected>>", self._on_window_selected)
        # Also allow pressing Enter after typing
        self._combo.bind("<Return>", self._on_window_selected)

        # Refresh button
        self._btn_refresh = ttk.Button(
            row,
            text="🔄  Refresh",
            style="Capture.TButton",
            command=self._on_refresh_windows,
        )
        self._btn_refresh.pack(side=tk.LEFT, padx=4)

        # Populate on first show
        self._populate_window_list()

        # Restore current value
        if self._current_window_title:
            self._window_var.set(self._current_window_title)

    def _populate_window_list(self) -> None:
        """Fetch all window titles and update the combobox."""
        try:
            from src.utils.window_list import get_window_list

            titles = get_window_list()
            self._combo["values"] = titles
            logger.debug(f"Window list refreshed: {len(titles)} title(s).")
        except Exception as exc:
            logger.warning(f"Failed to populate window list: {exc}")

    def _on_refresh_windows(self) -> None:
        """Refresh button clicked — repopulate list."""
        self._populate_window_list()
        logger.info("Window list manually refreshed.")

    def _on_window_selected(self, event: tk.Event | None = None) -> None:
        """Save the selected/typed window title and attempt focus."""
        title = self._window_var.get().strip()
        if not title:
            return

        self._current_window_title = title
        # Save to config
        self._save_config({"window_title": title, "capture_region": None})
        self._refresh_status()

        # Attempt focus
        if self._focus_manager is not None:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        self._focus_manager.focus_window(title)
                    )
                else:
                    result = loop.run_until_complete(
                        self._focus_manager.focus_window(title)
                    )
                    logger.info(f"Focus result: {result.method} — {result.message}")
            except Exception as exc:
                logger.warning(f"Focus attempt skipped: {exc}")

        logger.info(f"Capture target set to: {title!r}")

    # ------------------------------------------------------------------
    # Wayland layout
    # ------------------------------------------------------------------

    def _build_wayland_layout(self) -> None:
        """Fixed-region display + calibration button."""
        p = self._p
        row = tk.Frame(self, bg=p.bg)
        row.pack(fill=tk.X, pady=(2, 0))

        # Region display
        tk.Label(
            row,
            text="Region:",
            bg=p.bg,
            fg=p.fg,
            font=(self._ui_font, 10),
            width=8,
            anchor=tk.W,
        ).pack(side=tk.LEFT, padx=(0, 4))

        region_text = self._format_region(self._current_capture_region)
        self._lbl_region = tk.Label(
            row,
            text=region_text,
            bg=p.entry_bg,
            fg=p.entry_fg,
            font=(self._mono_font, 10),
            padx=8,
            pady=2,
            anchor=tk.CENTER,
            relief=tk.FLAT,
        )
        self._lbl_region.pack(side=tk.LEFT, padx=(0, 6))

        # Calibrate Capture Region button
        self._btn_calibrate_capture = ttk.Button(
            row,
            text="📐  Calibrate Capture Region",
            style="Capture.TButton",
            command=self._on_calibrate_capture_region,
        )
        self._btn_calibrate_capture.pack(side=tk.LEFT, padx=4)

    @staticmethod
    def _format_region(region: tuple[int, int, int, int] | None) -> str:
        """Return a human-readable region string."""
        if region is None:
            return "None set"
        left, top, width, height = region
        return f"{width}×{height}  @ ({left}, {top})"

    def _on_calibrate_capture_region(self) -> None:
        """Launch a simplified calibration for the capture region only."""
        from tkinter import messagebox

        logger.info("Launching capture region calibration …")

        # Get a screenshot
        screenshot = self._get_screenshot()
        if screenshot is None:
            messagebox.showwarning(
                "Capture Failed",
                "Could not capture a screenshot.\n"
                "Ensure a game window is visible and try again.",
                parent=self,
            )
            return

        # Load state schema (needed for CalibrationTool role dropdown)
        state_schema_slots = self._load_state_schema()

        # Open the overlay in a simplified mode
        from src.gui.calibration_overlay import CalibrationTool

        top = self.winfo_toplevel()
        tool = CalibrationTool(
            parent=top,
            screenshot=screenshot,
            ocr_module=None,
            existing_regions=[],
            state_schema_slots=state_schema_slots,
            on_save=self._on_capture_region_saved,
            screen_capture=None,
        )
        tool.grab_set()
        tool.focus_force()
        self.wait_window(tool)

    def _on_capture_region_saved(self, collected_regions: list[dict[str, Any]]) -> None:
        """Callback from CalibrationTool — extract the first region's bbox."""
        if not collected_regions:
            logger.warning("No regions collected from capture calibration.")
            return

        # Use the first drawn region's bounding box
        first = collected_regions[0]
        bbox = first.get("bbox", {})
        left = bbox.get("x", 0)
        top = bbox.get("y", 0)
        width = bbox.get("width", 0)
        height = bbox.get("height", 0)

        if width <= 0 or height <= 0:
            logger.warning("Invalid capture region drawn — zero size.")
            return

        region = [left, top, width, height]
        self._current_capture_region = (left, top, width, height)
        self._save_config({"capture_region": region, "window_title": None})
        self._lbl_region.config(text=self._format_region(self._current_capture_region))
        self._refresh_status()
        logger.info(f"Capture region set to {self._format_region(self._current_capture_region)}")

    @staticmethod
    def _get_screenshot() -> "np.ndarray | None":  # noqa: F821
        """Take a full-screen screenshot for calibration."""
        try:
            import mss
            import numpy as np

            with mss.mss() as sct:
                monitor = sct.monitors[0]
                img = sct.grab(monitor)
                return np.array(img)[:, :, :3]
        except ImportError:
            return None
        except Exception as exc:
            logger.warning(f"Screenshot failed: {exc}")
            return None

    @staticmethod
    def _load_state_schema() -> dict[str, dict[str, str]]:
        """Load state schema slots for the CalibrationTool dropdown."""
        import json
        from pathlib import Path

        try:
            schema_path = Path(__file__).resolve().parent.parent.parent / "config" / "state_schema.json"
            if schema_path.exists():
                data = json.loads(schema_path.read_text(encoding="utf-8"))
                return data.get("slots", {})
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    # Shared status line
    # ------------------------------------------------------------------

    def _refresh_status(self) -> None:
        """Update the status label with the current capture target."""
        p = self._p
        if self._is_wayland:
            region = self._current_capture_region
            if region:
                self._lbl_status.config(
                    text=f"Status: Fixed region {self._format_region(region)}",
                    fg=p.success,
                )
            else:
                self._lbl_status.config(
                    text="Status: No capture region set — use Calibrate above",
                    fg=p.warning,
                )
        else:
            title = self._current_window_title
            if title:
                # Try to get dimensions via PyWinCtl
                dims_str = ""
                try:
                    import pywinctl as pwc

                    windows = pwc.getWindowsWithTitle(title)
                    if windows:
                        w = windows[0]
                        l, t, r, b = w.getClientFrame()
                        dims_str = f" ({r - l}×{b - t})"
                except Exception:
                    pass
                self._lbl_status.config(
                    text=f'Status: Tracking "{title}"{dims_str}',
                    fg=p.success,
                )
            else:
                self._lbl_status.config(
                    text="Status: No window selected — using active window",
                    fg=p.warning,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_config_provider(self, provider: Any) -> None:
        """Replace the config provider at runtime."""
        self._config_manager = provider

    def set_focus_manager(self, mgr: Any) -> None:
        """Inject / replace the focus manager."""
        self._focus_manager = mgr
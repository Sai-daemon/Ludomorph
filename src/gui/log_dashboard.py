"""
Log Dashboard — real-time loguru output displayed in a tkinter Text widget.

Provides:
- ``LogDashboard(tk.Frame)`` — scrollable, read-only text area for log lines.
- ``setup_gui_sink(dashboard)`` — attaches a loguru sink that streams formatted
  log records into the dashboard via ``AsyncTk.call_async()``.
- ``remove_gui_sink()`` — detaches the sink on shutdown.
"""

from __future__ import annotations

import sys
import textwrap
import tkinter as tk
from datetime import datetime, timezone
from tkinter import ttk
from typing import Any

from src.gui.theme import ThemeManager, resolve_font_stack

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_LINES: int = 5_000  # hard cap to prevent unbounded memory growth
_LOG_FORMAT: str = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
)
# Map loguru level names to palette-derived colours.
# Call _build_level_colours() after ThemeManager is initialised.
_LEVEL_COLOURS: dict[str, str] = {}


def _build_level_colours(palette: Any) -> dict[str, str]:
    """Return log-level → foreground colour mappings from *palette*.
    
    Called once at construction and again on theme change so log text
    stays readable in both dark and light modes.
    """
    return {
        "TRACE": palette.disabled_fg,
        "DEBUG": palette.accent,
        "INFO": palette.fg,
        "SUCCESS": palette.success,
        "WARNING": palette.warning,
        "ERROR": palette.danger,
        "CRITICAL": "#FF00FF",  # magenta — works on both dark & light
    }

# ---------------------------------------------------------------------------
# module-level reference to the active GUI sink id
# ---------------------------------------------------------------------------

_sink_id: int | None = None

# ---------------------------------------------------------------------------
# LogDashboard widget
# ---------------------------------------------------------------------------


class LogDashboard(tk.Frame):  # type: ignore[misc]
    """Scrollable, read-only text widget that displays log lines.

    Designed to be embedded in the main window.  New lines are appended
    via :meth:`append_line` (thread-safe when called through
    ``AsyncTk.call_async``).
    """

    def __init__(self, parent: tk.Widget, max_lines: int = _MAX_LINES, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self._max_lines = max_lines
        self._line_count: int = 0

        self._tm = ThemeManager()
        p = self._tm.palette
        self._mono_font = resolve_font_stack(p.mono_font)

        # -- Build level colour map from palette -----------------------------
        self._level_colours = _build_level_colours(p)

        # -- Build widgets ---------------------------------------------------
        self._text = tk.Text(
            self,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg=p.bg,
            fg=p.fg,
            insertbackground=p.fg,
            font=(self._mono_font, 10),
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=6,
            pady=4,
        )
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._text.yview)
        self._text.configure(yscrollcommand=scrollbar.set)

        self._text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # -- Configure text tags for colouring -------------------------------
        self._configure_tags(p)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_line(self, level: str, timestamp: str, location: str, message: str) -> None:
        """Append a single formatted log line to the dashboard.

        This method **must** be called on the tkinter main thread.  Use
        ``AsyncTk.call_async(lambda: dashboard.append_line(...))`` from
        asyncio tasks.
        """
        self._text.configure(state=tk.NORMAL)

        # Build the line segment by segment with appropriate tags
        self._text.insert(tk.END, timestamp, ("timestamp",))
        self._text.insert(tk.END, " | ")

        level_tag = level if level in self._level_colours else None
        if level_tag:
            self._text.insert(tk.END, f"{level:<8}", (level_tag,))
        else:
            self._text.insert(tk.END, f"{level:<8}")
        self._text.insert(tk.END, " | ")

        self._text.insert(tk.END, location, ("location",))
        self._text.insert(tk.END, " | ")

        if level_tag:
            self._text.insert(tk.END, message + "\n", (level_tag,))
        else:
            self._text.insert(tk.END, message + "\n")

        self._text.configure(state=tk.DISABLED)

        # -- Enforce line cap ------------------------------------------------
        self._line_count += 1
        if self._line_count > self._max_lines:
            self._trim_head()

        # -- Auto-scroll to bottom ------------------------------------------
        self._text.see(tk.END)

    # ------------------------------------------------------------------
    # Theme support
    # ------------------------------------------------------------------

    def _configure_tags(self, p: Any) -> None:
        """(Re)configure all text tags from the active palette."""
        self._level_colours = _build_level_colours(p)
        for level_name, colour in self._level_colours.items():
            self._text.tag_configure(level_name, foreground=colour)
        self._text.tag_configure("timestamp", foreground=p.disabled_fg)
        self._text.tag_configure("location", foreground=p.disabled_fg)

    def _apply_theme(self) -> None:
        """Update colours after a theme change."""
        p = self._tm.palette
        self._mono_font = resolve_font_stack(p.mono_font)
        self._text.configure(bg=p.bg, fg=p.fg, insertbackground=p.fg,
                             font=(self._mono_font, 10))
        # Re-tag all text with palette-aware colours
        self._configure_tags(p)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim_head(self) -> None:
        """Remove the oldest line(s) when the buffer exceeds *max_lines*."""
        self._text.configure(state=tk.NORMAL)
        # Delete the first logical line
        self._text.delete("1.0", "2.0")
        self._line_count -= 1
        self._text.configure(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# loguru sink factory
# ---------------------------------------------------------------------------


def _make_sink(dashboard: LogDashboard) -> Any:
    """Return a callable suitable for use as a loguru sink.

    The returned function formats the log record and pushes it to the
    dashboard on the tkinter main thread.
    """

    def _sink(message: Any) -> None:
        record: Any = message.record
        level_name: str = record["level"].name
        # Build a compact time string
        ts = record["time"]
        if isinstance(ts, datetime):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        else:
            ts_str = str(ts)
        # Build location string
        location = f"{record['name']}:{record['function']}:{record['line']}"
        # Get the formatted message (loguru passes the formatted string in
        # message when using a raw sink, but with a structured handler we
        # reconstruct it).
        formatted = str(record["message"]).rstrip("\n")

        try:
            # Push to tkinter main thread via the top-level window's call_async
            root = dashboard.winfo_toplevel()
            if hasattr(root, "call_async"):
                root.call_async(
                    lambda lvl=level_name, ts=ts_str, loc=location, msg=formatted: dashboard.append_line(lvl, ts, loc, msg)  # type: ignore[misc]
                )
            else:
                # Fallback: not an AsyncTk — just append directly (test / mock)
                dashboard.append_line(level_name, ts_str, location, formatted)
        except (RuntimeError, tk.TclError):
            # Dashboard has been destroyed or mainloop has exited —
            # silently drop the message.  This happens during graceful
            # shutdown when loguru flushes remaining messages after
            # remove_gui_sink() is called.
            pass

    return _sink


def setup_gui_sink(dashboard: LogDashboard) -> None:
    """Attach a loguru sink that streams into *dashboard*.

    Safe to call multiple times — any previously registered GUI sink is
    removed first.
    """
    global _sink_id
    from loguru import logger

    remove_gui_sink()
    _sink_id = logger.add(
        _make_sink(dashboard),
        level="DEBUG",
        format=_LOG_FORMAT,
        colorize=False,
        enqueue=True,  # thread-safe: loguru uses its own internal queue
    )


def remove_gui_sink() -> None:
    """Remove the GUI loguru sink (no-op if not registered)."""
    global _sink_id
    if _sink_id is not None:
        from loguru import logger

        try:
            logger.remove(_sink_id)
        except ValueError:
            pass
        _sink_id = None
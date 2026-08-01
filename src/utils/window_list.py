"""
Window enumeration helper — cross-platform.

Provides :func:`get_window_list` which returns a sorted list of visible
window titles on Windows and X11.  Returns an empty list on Wayland
(native window enumeration is blocked by the compositor).

Spec reference:
  ``Extra_research01.md`` §1 — Window-Specific Capture
"""

from __future__ import annotations

import os
import sys

from src.logging_config import get_logger

logger = get_logger(__name__)


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_wayland() -> bool:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


def get_window_list() -> list[str]:
    """Return a sorted list of visible window titles on the system.

    Windows / X11:
        Enumerates all windows via PyWinCtl (``getAllWindows()``) and
        returns unique, sorted, non-empty titles.

    Wayland:
        Returns an empty list — native Wayland window enumeration is not
        possible without compositor-specific APIs (which are not
        supported for window *listing* by this helper).

    Returns
    -------
    list[str]
        Sorted list of window titles.  Empty on Wayland or if PyWinCtl
        is unavailable.
    """
    if _is_wayland():
        logger.debug("Wayland — window enumeration not supported.")
        return []

    try:
        import pywinctl as pwc

        windows = pwc.getAllWindows()
        titles = sorted({
            w.title.strip()
            for w in windows
            if w.title and w.title.strip()
        })
        logger.debug(f"Found {len(titles)} window title(s).")
        return titles
    except ImportError:
        logger.warning("pywinctl not available — returning empty window list.")
        return []
    except Exception as exc:
        logger.warning(f"Window enumeration failed: {exc}")
        return []
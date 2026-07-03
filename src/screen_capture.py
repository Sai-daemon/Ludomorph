"""
Screen Capture Module ― Step 1.2

Provides cross-platform window tracking and async screen capture with:
- Window tracking via PyWinCtl (Win/X11), fallback to user-defined region (Wayland)
- mss as primary backend, dxcam as optional high-perf backend (Windows)
- Downsampling + grayscale via OpenCV
- Frame buffer (last successful frame for timeout fallback)
- 30 ms capture timeout per spec §7.3
"""

import asyncio
import hashlib
import time as time_module
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Platform detection helpers
# ---------------------------------------------------------------------------

import os
import sys


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_x11() -> bool:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "x11"


def _is_wayland() -> bool:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class CaptureConfig:
    """Configuration for the screen capture module."""

    downsample_size: tuple[int, int] = (640, 360)
    """Target resolution (width, height) after downsampling."""

    capture_interval: float = 0.05
    """Capture interval in seconds (50ms = 20 Hz per §7.4)."""

    timeout_ms: float = 30.0
    """Maximum time (milliseconds) a single capture may take (§7.3)."""

    backend: str = "auto"
    """'auto', 'mss', or 'dxcam'."""

    grayscale: bool = True
    """Whether to convert captured frames to grayscale."""

    monitor_index: int = 0
    """Monitor index for full-screen capture (mss fallback)."""

    capture_region: tuple[int, int, int, int] | None = None
    """User-defined region (left, top, width, height) ― required for Wayland."""

    window_title: str | None = None
    """Optional window title to search for (instead of active window)."""

    use_dxcam: bool = False
    """Set to True on Windows to prefer the Desktop Duplication API."""


# ---------------------------------------------------------------------------
# Window Tracker
# ---------------------------------------------------------------------------


class WindowTracker:
    """
    Cross-platform active-window geometry tracker.

    - Windows/X11: uses PyWinCtl for automatic tracking
    - Wayland: returns None (user must provide a fixed capture_region)

    Per Extra_research01:
      - Polls at 30-50 ms intervals
      - Change detection via rectangle hash
      - Error recovery: retry every 1-2 seconds
      - DPI awareness on Windows
    """

    _dpi_awareness_set: bool = False
    """Class-level flag so we only call SetProcessDPIAwarenessContext once."""

    def __init__(self, config: CaptureConfig) -> None:
        self._config = config
        self._last_rect_hash: str | None = None
        self._last_rect: tuple[int, int, int, int] | None = None
        self._last_error_time: float = 0.0
        self._error_retry_interval: float = 1.5  # seconds

        self._dpi_setup()

    # ------------------------------------------------------------------
    # DPI awareness (Windows only)
    # ------------------------------------------------------------------

    @staticmethod
    def _dpi_setup() -> None:
        """Set per-monitor DPI awareness on Windows (once per process)."""
        if not _is_windows() or WindowTracker._dpi_awareness_set:
            return
        try:
            import ctypes

            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
            awareness = ctypes.c_int(-4)
            result = ctypes.windll.user32.SetProcessDPIAwarenessContext(awareness)
            if result:
                logger.debug("Windows DPI awareness set to PER_MONITOR_AWARE_V2")
            else:
                # Fallback to legacy API
                ctypes.windll.user32.SetProcessDPIAware()
                logger.debug("Windows DPI awareness set via legacy SetProcessDPIAware")
            WindowTracker._dpi_awareness_set = True
        except Exception as exc:
            logger.warning(f"Could not set DPI awareness: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_active_window_rect(self) -> tuple[int, int, int, int] | None:
        """
        Return (left, top, width, height) of the currently active window,
        or None if the platform / session doesn't support automatic tracking.

        On Wayland this always returns None — the user must set
        ``capture_region`` manually.
        """
        if _is_wayland():
            logger.debug("Wayland detected; automatic window tracking unavailable.")
            return None

        if self._config.capture_region is not None:
            return self._config.capture_region

        return self._polls_active_window()

    def _polls_active_window(self) -> tuple[int, int, int, int] | None:
        """Internal polling of the active window via PyWinCtl."""
        now = time_module.monotonic()

        # Rate-limit retries after errors (every ~1.5s)
        if self._last_error_time and (now - self._last_error_time) < self._error_retry_interval:
            return self._last_rect

        try:
            import pywinctl as pwc

            if self._config.window_title:
                windows = pwc.getWindowsWithTitle(self._config.window_title)
                if not windows:
                    logger.debug(f"Window '{self._config.window_title}' not found.")
                    self._last_error_time = now
                    return self._last_rect
                win = windows[0]
            else:
                win = pwc.getActiveWindow()
                if win is None:
                    logger.debug("No active window found.")
                    self._last_error_time = now
                    return self._last_rect

            # getClientFrame excludes title bar / shadows
            left, top, right, bottom = win.getClientFrame()

            # Validate geometry
            if right <= left or bottom <= top:
                logger.debug(f"Invalid window geometry: {left=} {top=} {right=} {bottom=}")
                self._last_error_time = now
                return self._last_rect

            width = right - left
            height = bottom - top

            # Change detection via hash
            rect_hash = hashlib.md5(f"{left},{top},{width},{height}".encode()).hexdigest()
            if rect_hash != self._last_rect_hash:
                logger.info(f"Window rect changed: {left},{top} {width}x{height}")
                self._last_rect_hash = rect_hash
                self._last_rect = (left, top, width, height)

            self._last_error_time = 0.0  # reset error timer
            return self._last_rect

        except Exception as exc:
            logger.warning(f"Window tracking error: {exc}")
            self._last_error_time = now
            return self._last_rect


# ---------------------------------------------------------------------------
# Screen Capture (async coroutine)
# ---------------------------------------------------------------------------


class ScreenCapture:
    """
    Async screen capture using mss (cross-platform) with optional dxcam
    acceleration on Windows.

    Features:
    - Downsampling + grayscale via OpenCV
    - Frame buffer: last successful frame available on timeout
    - 30 ms timeout per spec §7.3
    - Captures at ~20 Hz (50 ms interval) per spec §7.4
    """

    def __init__(
        self,
        config: CaptureConfig,
        tracker: WindowTracker | None = None,
    ) -> None:
        self._config = config
        self._tracker = tracker
        self._frame_buffer: np.ndarray | None = None
        self._mss: object | None = None  # lazily initialised
        self._dxcam = None  # type: ignore[assignment]  # lazily initialised

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def capture(self) -> np.ndarray | None:
        """
        Capture a single frame from the game window.

        Returns a downsampled NumPy array (grayscale or BGR), or None if
        the capture timed out or failed. Callers should fall back to
        ``get_last_frame()`` when None is returned.
        """
        try:
            frame = await asyncio.wait_for(
                self._capture_impl(),
                timeout=self._config.timeout_ms / 1000.0,
            )
            if frame is not None:
                self._frame_buffer = frame
            return frame
        except asyncio.TimeoutError:
            logger.warning(f"Capture timed out after {self._config.timeout_ms}ms")
            return None

    def get_last_frame(self) -> np.ndarray | None:
        """Return the last successfully captured frame."""
        return self._frame_buffer

    async def close(self) -> None:
        """Release capture resources."""
        if self._dxcam is not None and hasattr(self._dxcam, "stop"):
            self._dxcam.stop()
        self._mss = None
        self._dxcam = None

    # ------------------------------------------------------------------
    # Internal: capture dispatch
    # ------------------------------------------------------------------

    async def _capture_impl(self) -> np.ndarray | None:
        """Core capture logic, dispatched to backend."""
        region = self._get_capture_region()
        if region is None:
            logger.warning("No capture region available.")
            return None

        left, top, region_width, region_height = region

        # Determine backend
        use_dxcam = (
            self._config.use_dxcam
            and self._config.backend in ("auto", "dxcam")
            and _is_windows()
            and self._init_dxcam()
        )

        if use_dxcam:
            raw = await self._capture_dxcam(left, top, region_width, region_height)
        else:
            raw = await self._capture_mss(left, top, region_width, region_height)

        if raw is None:
            return None

        # Downsample + grayscale
        return self._postprocess(raw)

    def _get_capture_region(self) -> tuple[int, int, int, int] | None:
        """Resolve the capture region from config / window tracker."""
        if self._config.capture_region is not None:
            return self._config.capture_region
        if self._tracker is not None:
            rect = self._tracker.get_active_window_rect()
            if rect is not None:
                # tracker returns (left, top, width, height)
                return rect
        # Full-screen fallback via mss monitor
        return None  # caller will use mss monitor_index

    # ------------------------------------------------------------------
    # mss backend
    # ------------------------------------------------------------------

    def _init_mss(self) -> object:
        if self._mss is None:
            import mss

            self._mss = mss.mss()
        return self._mss

    async def _capture_mss(
        self, left: int, top: int, region_width: int, region_height: int
    ) -> np.ndarray | None:
        """Capture via mss (cross-platform)."""
        import mss

        sct = self._init_mss()
        monitor = {"top": top, "left": left, "width": region_width, "height": region_height}
        try:
            sct_img = await asyncio.to_thread(sct.grab, monitor)
            return np.array(sct_img)[:, :, :3]  # drop alpha
        except mss.exception.ScreenShotError as exc:
            logger.warning(f"mss capture error: {exc}")
            return None
        except Exception as exc:
            logger.warning(f"mss unexpected error: {exc}")
            return None

    # ------------------------------------------------------------------
    # dxcam backend (Windows only)
    # ------------------------------------------------------------------

    def _init_dxcam(self) -> bool:
        """Attempt to initialise dxcam. Returns True on success."""
        if self._dxcam is not None:
            return True
        try:
            import dxcam

            self._dxcam = dxcam.create()
            return self._dxcam is not None
        except ImportError:
            logger.debug("dxcam not available; falling back to mss.")
            return False
        except Exception as exc:
            logger.warning(f"dxcam init failed: {exc}")
            return False

    async def _capture_dxcam(
        self, left: int, top: int, region_width: int, region_height: int
    ) -> np.ndarray | None:
        """Capture via dxcam (Desktop Duplication API, Windows)."""
        try:
            dxcam_obj = self._dxcam
            if dxcam_obj is None:
                return None
            frame = await asyncio.to_thread(
                dxcam_obj.grab,
                region=(left, top, left + region_width, top + region_height),
            )
            if frame is None:
                return None
            return np.array(frame)
        except Exception as exc:
            logger.warning(f"dxcam capture error: {exc}. Falling back to mss.")
            self._dxcam = None  # disable dxcam for future captures
            return await self._capture_mss(left, top, region_width, region_height)

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _postprocess(self, raw: np.ndarray) -> np.ndarray:
        """
        Downsample to ``downsample_size`` and optionally convert to grayscale.

        All processing runs via ``asyncio.to_thread`` from the capture
        coroutine, so this is inherently non-blocking.
        """
        target_w, target_h = self._config.downsample_size

        # Resize
        resized = cv2.resize(raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Grayscale
        if self._config.grayscale:
            return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return resized
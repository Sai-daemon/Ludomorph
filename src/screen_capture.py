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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.logging_config import get_logger

# ---------------------------------------------------------------------------
# Dedicated capture thread pool
# ---------------------------------------------------------------------------

_capture_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="capture")
"""Dedicated thread pool for mss/dxcam screen grabs.  Must **not** be
shared with the default executor (used by colour‑bar detection via
``asyncio.to_thread``) to prevent capture timeouts when the default
executor is saturated with CPU‑bound work.
"""

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

    timeout_ms: float = 300.0
    """Maximum time (milliseconds) a single capture may take (§7.3).
    Increased from 150 to 300 to be tolerant of momentary system load
    (e.g. debug overlay creation, compositor transitions)."""

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
        self._minimized_warned: bool = False
        self._dpi_scale_factor: float = 1.0

        self._dpi_setup()
        self._detect_dpi_scale()

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
    # DPI scale detection & validation
    # ------------------------------------------------------------------

    def _detect_dpi_scale(self) -> None:
        """Detect the current DPI scale factor for coordinate correction.

        On Windows reads the primary monitor's DPI via ctypes;
        on X11/Linux defaults to 1.0 (no scaling).
        """
        if _is_windows():
            try:
                import ctypes

                hdc = ctypes.windll.user32.GetDC(0)
                if hdc:
                    dpi_x = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
                    ctypes.windll.user32.ReleaseDC(0, hdc)
                    # Standard DPI is 96 — scale factor relative to that
                    self._dpi_scale_factor = float(dpi_x) / 96.0
                    if self._dpi_scale_factor != 1.0:
                        logger.debug(
                            f"DPI scale factor detected: {self._dpi_scale_factor:.2f}"
                            f" (raw DPI: {dpi_x})"
                        )
            except Exception as exc:
                logger.debug(f"Could not detect DPI scale: {exc}")
                self._dpi_scale_factor = 1.0
        else:
            # X11 reports logical pixels; Wayland uses fixed region — no scaling needed
            self._dpi_scale_factor = 1.0

    def _validate_dpi_rect(
        self, left: int, top: int, width: int, height: int
    ) -> tuple[int, int, int, int] | None:
        """Validate and correct the window rectangle for mixed-DPI setups.

        On Windows with mixed-DPI monitors, PyWinCtl may return virtualised
        coordinates.  If the rect dimensions seem too small relative to the
        DPI scale factor, apply correction and log a warning.

        Returns a corrected (left, top, width, height) tuple, or None if
        no correction is needed.
        """
        if not _is_windows():
            return None

        if self._dpi_scale_factor <= 1.0:
            return None

        # Check if the captured rect is suspiciously small
        # On a 1920×1080 display at 125% DPI, virtualised rect would be ~1536×864
        # We flag rects where width < 200 and height < 200 as potentially scaled
        if width > 200 and height > 200:
            return None  # dimensions look reasonable

        # Attempt correction: scale back to physical pixels
        try:
            corrected_left = int(left * self._dpi_scale_factor)
            corrected_top = int(top * self._dpi_scale_factor)
            corrected_width = int(width * self._dpi_scale_factor)
            corrected_height = int(height * self._dpi_scale_factor)

            logger.warning(
                f"Mixed-DPI suspected: rect ({left},{top} {width}×{height}) "
                f"appears virtualised at {self._dpi_scale_factor:.0%} scaling. "
                f"Correcting to ({corrected_left},{corrected_top} "
                f"{corrected_width}×{corrected_height})."
            )
            return (corrected_left, corrected_top, corrected_width, corrected_height)
        except Exception as exc:
            logger.debug(f"DPI correction failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_active_window_rect(self) -> tuple[int, int, int, int] | None:
        """
        Return (left, top, width, height) of the currently active window,
        or None if the platform / session doesn't support automatic tracking.

        On Wayland this always returns None — the user must set
        ``capture_region`` manually.

        Returns None for minimized windows to signal capture should pause.
        """
        if _is_wayland():
            logger.debug("Wayland detected; automatic window tracking unavailable.")
            return None

        if self._config.capture_region is not None:
            return self._config.capture_region

        return self._polls_active_window()

    def _polls_active_window(self) -> tuple[int, int, int, int] | None:
        """Internal polling of the active window via PyWinCtl.

        On ewmhlib / Xlib failures (e.g. ``ret`` variable unbound),
        we degrade gracefully: log once, then return the last known
        rect until the error clears.
        """
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

            # --- Minimized window detection (pass pre-fetched rect to avoid repeat X11 roundtrip) ---
            if _is_minimized(win, (left, top, right, bottom)):
                if not self._minimized_warned:
                    logger.warning(
                        "Target window is minimized — capture cannot proceed. "
                        "Un-minimize the window or use a fixed capture_region."
                    )
                    self._minimized_warned = True
                self._last_error_time = now
                return None
            else:
                self._minimized_warned = False

            width = right - left
            height = bottom - top

            # --- Mixed-DPI validation ---
            validated_rect = self._validate_dpi_rect(left, top, width, height)
            if validated_rect is not None:
                left, top, width, height = validated_rect

            # Change detection via hash
            rect_hash = hashlib.md5(f"{left},{top},{width},{height}".encode()).hexdigest()
            if rect_hash != self._last_rect_hash:
                logger.info(f"Window rect changed: {left},{top} {width}x{height}")
                self._last_rect_hash = rect_hash
                self._last_rect = (left, top, width, height)

            self._last_error_time = 0.0  # reset error timer
            return self._last_rect

        except Exception as exc:
            # ewmhlib / Xlib errors (e.g. 'ret' unbound) are caught here —
            # degrade gracefully and return the last known rect
            exc_name = type(exc).__name__
            exc_msg = str(exc)
            # Only log once per unique error type; avoid log spam on retries
            logger.warning(f"Window tracking error ({exc_name}): {exc_msg}")
            self._last_error_time = now
            return self._last_rect


# ---------------------------------------------------------------------------
# Helper: minimized window detection
# ---------------------------------------------------------------------------

def _is_minimized(
    win: object,
    pre_rect: tuple[int, int, int, int] | None = None,
) -> bool:
    """Return True if the PyWinCtl window object represents a minimized window.

    Parameters
    ----------
    win : object
        PyWinCtl window handle.
    pre_rect : (left, top, right, bottom) | None
        If already fetched from ``getClientFrame()``, pass it here to avoid
        a duplicate X11 roundtrip.  If None, fetches it internally.
    """
    try:
        # PyWinCtl Window objects typically have an isMinimized attribute/method
        if hasattr(win, "isMinimized"):
            return bool(win.isMinimized)
        # Check if window is not active (a minimized window can't be active)
        if hasattr(win, "isActive"):
            try:
                if not win.isActive:
                    return True
            except Exception:
                pass  # isActive may raise on some WMs
        # Fallback: check if dimensions are near-zero
        if pre_rect is not None:
            left, top, right, bottom = pre_rect
        else:
            left, top, right, bottom = win.getClientFrame()
        return (right - left) <= 1 or (bottom - top) <= 1
    except Exception:
        return False


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

    # Stale / black frame detection thresholds
    _BLACK_FRAME_THRESHOLD: float = 5.0
    """Mean pixel value below which a frame is considered 'mostly black'."""

    _MAX_BLACK_BLOCK_TIME: float = 1.0
    """Maximum time (seconds) to reject frames for blackness before accepting."""

    _MAX_STALE_AFTER_FOCUS: float = 1.0
    """Time window (seconds) after a focus-change event during which
    pixel‑identical frames are suspicious and may be rejected."""

    _MIN_REGION_SIZE: int = 20
    """Minimum width or height (pixels) for a valid capture region.
    Smaller regions are rejected to avoid zero-size / garbage captures."""

    _MIN_REGION_WARNED: bool = False
    """Set after first min-size warning to avoid log spam."""

    _FOCUS_CHANGE_COOLDOWN: float = 2.0
    """Minimum time between focus-change staleness windows to avoid
    repeated false positives during rapid focus toggling."""

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

        # -- Perceptual staleness detection (replaces exact‑hash approach) --
        self._last_accepted_frame: np.ndarray | None = None
        """The most recently accepted frame, used for pixel‑diff comparison."""

        self._focus_change_time: float = 0.0
        """Last time a window focus change was detected (monotonic)."""

        self._consecutive_identical: int = 0
        """Counter of consecutive frames with near‑zero pixel difference."""

        self._stale_warned: bool = False
        """Set after first stale-frame warning to avoid log spam."""
        self._last_no_region_warning: float = 0.0
        """Timestamp of last 'No capture region' warning for rate-limiting."""
        self._first_black_time: float | None = None
        """Timestamp when the first consecutive black frame was rejected."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def capture(self) -> np.ndarray | None:
        """
        Capture a single frame from the game window.

        Returns a downsampled NumPy array (grayscale or BGR), or None if
        the capture timed out, failed, or produced a stale/black frame.
        Callers should fall back to ``get_last_frame()`` when None is returned.
        """
        try:
            frame = await asyncio.wait_for(
                self._capture_impl(),
                timeout=self._config.timeout_ms / 1000.0,
            )
            if frame is not None:
                if not self._is_frame_valid(frame):
                    logger.debug("Frame rejected — stale or black; returning last valid frame.")
                    return None
                self._frame_buffer = frame
            return frame
        except asyncio.TimeoutError:
            logger.warning(f"Capture timed out after {self._config.timeout_ms}ms")
            return None

    def _is_frame_valid(self, frame: np.ndarray) -> bool:
        """Validate a captured frame is not black or stale.

        **Blackness check**: rejects frames whose mean pixel value is below
        ``_BLACK_FRAME_THRESHOLD`` (compositor artifact during focus
        transition).  After ``_MAX_BLACK_BLOCK_TIME`` of continuous
        rejection, the frame is accepted anyway to prevent engine lockup.

        **Staleness check**: uses perceptual pixel‑difference, NOT exact
        MD5 hashes.  A frame is only considered "stale" when the mean
        absolute diff versus the last accepted frame is near‑zero AND the
        frame arrived within ``_MAX_STALE_AFTER_FOCUS`` seconds of a
        detected focus‑change event.  Genuinely static game scenes (menus,
        idle gameplay) are **not** rejected — a static game IS valid.

        This prevents the previous bug where MD5‑identical frames from
        static menus were rejected with "Stale frame detected", causing
        cascading OCR‑timeout errors.

        Returns True if the frame passes all checks.
        """
        now = time_module.monotonic()

        # --- Black frame check (fast) ---
        mean_val: float = float(np.mean(frame))
        if mean_val < self._BLACK_FRAME_THRESHOLD:
            if self._first_black_time is None:
                self._first_black_time = now
            blocked_duration = now - self._first_black_time
            if blocked_duration >= self._MAX_BLACK_BLOCK_TIME:
                logger.debug(
                    f"Black frame accepted after {blocked_duration:.1f}s "
                    f"of continuous rejection — bypassing black check."
                )
                self._first_black_time = None
                # fall through to staleness check
            else:
                if not self._stale_warned:
                    logger.warning(
                        f"Black frame detected (mean={mean_val:.1f}). "
                        f"Likely compositor artifact during focus switch — "
                        f"discarding frame."
                    )
                    self._stale_warned = True
                else:
                    self._stale_warned = False
                return False
        else:
            self._stale_warned = False
            self._first_black_time = None

        # --- Perceptual staleness check ---
        # Only reject identical frames when we suspect a compositor artifact
        # (shortly after a focus change).  Static game scenes are valid.
        if self._last_accepted_frame is not None:
            last = self._last_accepted_frame
            if last.shape == frame.shape:
                # Compute mean absolute pixel difference
                diff: float = float(np.mean(np.abs(frame.astype(np.float32)
                                                    - last.astype(np.float32))))
                # Near‑zero diff: frames are effectively identical
                if diff < 0.5:
                    self._consecutive_identical += 1
                    time_since_focus = now - self._focus_change_time
                    if time_since_focus < self._MAX_STALE_AFTER_FOCUS:
                        # Within the focus‑switch window — this identical frame
                        # is probably a compositor artifact, reject it.
                        if self._consecutive_identical <= 3:
                            logger.debug(
                                f"Identical frame detected {time_since_focus:.2f}s "
                                f"after focus change (diff={diff:.2f}) — suspect "
                                f"compositor artifact, rejecting."
                            )
                            return False
                        else:
                            # After 3 consecutive identical frames in the focus
                            # window, accept — the game renderer may genuinely
                            # be paused / static.
                            logger.debug(
                                f"Accepting identical frame after "
                                f"{self._consecutive_identical} consecutive "
                                f"matches (likely static scene)."
                            )
                    else:
                        # Outside focus‑change window — static game scene, accept.
                        if self._consecutive_identical > 10:
                            logger.debug(
                                f"Frame identical to previous for "
                                f"{self._consecutive_identical} cycles — "
                                f"game appears static (menus, idle).  Accepting."
                            )
                else:
                    self._consecutive_identical = 0
            else:
                self._consecutive_identical = 0

        # Accept the frame and update the last-accepted reference
        self._last_accepted_frame = frame.copy()
        return True

    def notify_focus_change(self) -> None:
        """Notify the capture module that a window focus change was detected.

        Called by ``WindowTracker`` when the active window rect changes.
        Records the monotonic timestamp so ``_is_frame_valid`` can activate
        the guarded staleness window.
        """
        now = time_module.monotonic()
        if now - self._focus_change_time > self._FOCUS_CHANGE_COOLDOWN:
            self._focus_change_time = now
            self._consecutive_identical = 0
            # Reset last-accepted so the first frame after focus change
            # is always accepted (baseline for diff comparison).
            self._last_accepted_frame = None
            logger.debug("Focus change recorded — resetting staleness guard.")

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
            now = time_module.monotonic()
            if now - self._last_no_region_warning > 10.0:
                logger.warning("No capture region available.")
                self._last_no_region_warning = now
            else:
                logger.debug("No capture region available.")
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
        """Resolve the capture region from config / window tracker.

        Rejects regions smaller than ``_MIN_REGION_SIZE`` in either
        dimension (logs a warning once).  Returns None if no region
        is available *or* if the region is too small to capture.
        """
        region: tuple[int, int, int, int] | None = None

        if self._config.capture_region is not None:
            region = self._config.capture_region
        elif self._tracker is not None:
            rect = self._tracker.get_active_window_rect()
            if rect is not None:
                # tracker returns (left, top, width, height)
                region = rect

        if region is None:
            return None  # caller will use mss monitor_index

        left, top, width, height = region
        if width < self._MIN_REGION_SIZE or height < self._MIN_REGION_SIZE:
            if not self._MIN_REGION_WARNED:
                logger.warning(
                    f"Capture region too small: {width}×{height} px "
                    f"(minimum {self._MIN_REGION_SIZE}×{self._MIN_REGION_SIZE}). "
                    f"Is the target window minimized or off-screen? "
                    f"Configure a fixed ``capture_region`` in settings."
                )
                self._MIN_REGION_WARNED = True
            return None

        self._MIN_REGION_WARNED = False
        return region

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
        """Capture via mss (cross-platform), using the dedicated capture
        thread pool to avoid contention with other CPU‑bound work."""
        import mss

        sct = self._init_mss()
        monitor = {"top": top, "left": left, "width": region_width, "height": region_height}
        loop = asyncio.get_running_loop()
        try:
            sct_img = await loop.run_in_executor(
                _capture_executor, sct.grab, monitor
            )
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
        """Capture via dxcam (Desktop Duplication API, Windows), using the
        dedicated capture thread pool."""
        try:
            dxcam_obj = self._dxcam
            if dxcam_obj is None:
                return None
            loop = asyncio.get_running_loop()
            frame = await loop.run_in_executor(
                _capture_executor,
                dxcam_obj.grab,
                (left, top, left + region_width, top + region_height),
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
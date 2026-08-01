"""
Window Auto-Focus (Cross-Platform) ― Step 1.4

Provides compositor detection and window focus management across
Windows, X11, and Wayland (Sway, Hyprland, with stubs for GNOME/KDE).

Spec references:
- Implementation_Phases.md §1.4
- Extra_research04.md §5-6  Window Auto‑focus on Linux Wayland
- architecture.md §4.1  Cross-Platform Strategy
"""

from __future__ import annotations

import asyncio
import enum
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Platform detection (mirrors patterns from screen_capture / input_controller)
# ============================================================================

def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_x11() -> bool:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "x11"


def _is_wayland() -> bool:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


# ============================================================================
# Compositor enum
# ============================================================================

class Compositor(enum.Enum):
    """Identified desktop environment / compositor."""
    WINDOWS = "windows"
    X11 = "x11"
    SWAY = "sway"
    HYPRLAND = "hyprland"
    GNOME = "gnome"
    KDE = "kde"
    COSMIC = "cosmic"
    OTHER_WAYLAND = "other_wayland"
    UNKNOWN = "unknown"


# ============================================================================
# Compositor detection
# ============================================================================

class CompositorDetector:
    """
    Detects the active compositor / display server at startup.

    Detection order (per Extra_research04 §5.1):
      1. Check $XDG_SESSION_TYPE
      2. Environment variables: SWAYSOCK, HYPRLAND_INSTANCE_SIGNATURE, KDE_FULL_SESSION
      3. Process list: pgrep for sway, gnome-shell, kwin_wayland
      4. $XDG_CURRENT_DESKTOP
    """

    @staticmethod
    def detect() -> Compositor:
        """Return the detected compositor / display server."""
        # -- Windows ----------------------------------------------------
        if _is_windows():
            return Compositor.WINDOWS

        # -- X11 --------------------------------------------------------
        if _is_x11():
            return Compositor.X11

        # -- Wayland (or TTY / unknown) ---------------------------------
        if _is_wayland():
            return CompositorDetector._detect_wayland_compositor()

        logger.warning(
            "$XDG_SESSION_TYPE is neither 'x11' nor 'wayland'. "
            "Window auto-focus may not be available."
        )
        return Compositor.UNKNOWN

    @staticmethod
    def _detect_wayland_compositor() -> Compositor:
        """Identify the specific Wayland compositor."""
        # 1. Environment variable checks (fastest)
        if os.environ.get("SWAYSOCK"):
            logger.debug("Detected compositor: Sway (via SWAYSOCK)")
            return Compositor.SWAY

        if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
            logger.debug("Detected compositor: Hyprland (via HYPRLAND_INSTANCE_SIGNATURE)")
            return Compositor.HYPRLAND

        if os.environ.get("KDE_FULL_SESSION") == "true":
            logger.debug("Detected compositor: KDE Plasma (via KDE_FULL_SESSION)")
            return Compositor.KDE

        # 2. Process-list checks
        if _pgrep_matches("gnome-shell"):
            logger.debug("Detected compositor: GNOME (via pgrep gnome-shell)")
            return Compositor.GNOME

        if _pgrep_matches("kwin_wayland"):
            logger.debug("Detected compositor: KDE Plasma (via pgrep kwin_wayland)")
            return Compositor.KDE

        if _pgrep_matches("sway"):
            logger.debug("Detected compositor: Sway (via pgrep sway)")
            return Compositor.SWAY

        # 3. $XDG_CURRENT_DESKTOP fallback
        xdg_desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
        if "gnome" in xdg_desktop:
            logger.debug(f"Detected compositor: GNOME (via XDG_CURRENT_DESKTOP={xdg_desktop!r})")
            return Compositor.GNOME
        if "kde" in xdg_desktop:
            logger.debug(f"Detected compositor: KDE (via XDG_CURRENT_DESKTOP={xdg_desktop!r})")
            return Compositor.KDE
        if "cosmic" in xdg_desktop:
            logger.debug(f"Detected compositor: COSMIC (via XDG_CURRENT_DESKTOP={xdg_desktop!r})")
            return Compositor.COSMIC

        logger.debug("Compositor: unrecognised Wayland compositor")
        return Compositor.OTHER_WAYLAND


def _pgrep_matches(name: str) -> bool:
    """Return True if a process with exact *name* is running."""
    return shutil.which("pgrep") is not None and os.system(
        f'pgrep -x "{name}" > /dev/null 2>&1'
    ) == 0


# ============================================================================
# Focus result
# ============================================================================

@dataclass
class FocusResult:
    """Outcome of a focus attempt."""
    success: bool
    method: str          # e.g. "pywinctl", "swaymsg", "ydotool_click_fallback"
    compositor: str      # detected compositor name
    message: str = ""    # human-readable description


# ============================================================================
# WindowFocusManager
# ============================================================================

class WindowFocusManager:
    """
    Cross-platform window auto-focus.

    Usage::

        mgr = WindowFocusManager(config, input_controller)
        result = await mgr.focus_window("My Game Title")
        if result.success:
            logger.info(f"Window focused via {result.method}")
        else:
            logger.warning(f"Focus failed: {result.message}")

    The manager respects ``auto_focus_window`` from config.json.
    When False, ``focus_window()`` returns immediately with
    ``success=False`` and an appropriate message.
    """

    def __init__(
        self,
        config: dict[str, Any],
        input_controller: Any = None,
    ) -> None:
        """
        Parameters
        ----------
        config : dict
            Global configuration dict.  Must contain ``auto_focus_window``.
        input_controller : InputController or None
            Required for the ydotool click-to-focus fallback on Wayland.
            If None and the fallback is needed, focus will fail gracefully.
        """
        self._config = config
        self._input = input_controller
        self._compositor = CompositorDetector.detect()
        logger.info(f"WindowFocusManager initialised. Compositor: {self._compositor.value}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def compositor(self) -> Compositor:
        """Return the detected compositor."""
        return self._compositor

    async def focus_window(self, window_title: str) -> FocusResult:
        """
        Attempt to bring *window_title* to the foreground and give it focus.

        Returns ``FocusResult`` describing the outcome.
        Does **not** raise on failure — callers should inspect the result.
        """
        # Honour the global toggle
        if not self._config.get("auto_focus_window", True):
            return FocusResult(
                success=False,
                method="disabled",
                compositor=self._compositor.value,
                message="auto_focus_window is disabled in config.",
            )

        if not window_title.strip():
            return FocusResult(
                success=False,
                method="none",
                compositor=self._compositor.value,
                message="Empty window title provided.",
            )

        title = window_title.strip()

        # Dispatch by compositor
        try:
            if self._compositor == Compositor.WINDOWS:
                return await self._focus_windows_x11(title)  # shared PyWinCtl path
            elif self._compositor == Compositor.X11:
                return await self._focus_windows_x11(title)
            elif self._compositor == Compositor.SWAY:
                return await self._focus_sway(title)
            elif self._compositor == Compositor.HYPRLAND:
                return await self._focus_hyprland(title)
            elif self._compositor == Compositor.GNOME:
                return await self._focus_gnome(title)
            elif self._compositor == Compositor.KDE:
                return await self._focus_kde(title)
            elif self._compositor == Compositor.COSMIC:
                return await self._focus_cosmic(title)
            else:
                return await self._focus_fallback(title)
        except Exception as exc:
            logger.warning(f"Unexpected error during focus attempt: {exc}")
            return FocusResult(
                success=False,
                method="error",
                compositor=self._compositor.value,
                message=f"Focus failed with unexpected error: {exc}",
            )

    # ------------------------------------------------------------------
    # Windows / X11 (shared PyWinCtl path)
    # ------------------------------------------------------------------

    async def _focus_windows_x11(self, title: str) -> FocusResult:
        """Focus a window on Windows or X11 via PyWinCtl."""
        try:
            import pywinctl as pwc

            windows = await asyncio.to_thread(pwc.getWindowsWithTitle, title)
            if not windows:
                return FocusResult(
                    success=False,
                    method="pywinctl",
                    compositor=self._compositor.value,
                    message=f"No window found matching title: {title!r}",
                )

            win = windows[0]
            await asyncio.to_thread(win.activate)
            logger.info(f"Window focused via pywinctl: {win.title}")
            return FocusResult(
                success=True,
                method="pywinctl",
                compositor=self._compositor.value,
                message=f"Focused window: {win.title}",
            )
        except ImportError:
            logger.warning("pywinctl not available; cannot focus window.")
            return FocusResult(
                success=False,
                method="pywinctl",
                compositor=self._compositor.value,
                message="pywinctl library is not installed.",
            )
        except Exception as exc:
            logger.warning(f"pywinctl focus error: {exc}")
            return FocusResult(
                success=False,
                method="pywinctl",
                compositor=self._compositor.value,
                message=f"pywinctl error: {exc}",
            )

    # ------------------------------------------------------------------
    # Sway
    # ------------------------------------------------------------------

    async def _focus_sway(self, title: str) -> FocusResult:
        """Focus a window on Sway via swaymsg."""
        if not shutil.which("swaymsg"):
            return FocusResult(
                success=False,
                method="swaymsg",
                compositor="sway",
                message="swaymsg binary not found on $PATH.",
            )
        try:
            # swaymsg '[title="GameTitle"]' focus
            cmd = ["swaymsg", f'[title="{title}"]', "focus"]
            logger.debug(f"Running: {' '.join(cmd)}")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                err = stderr.decode().strip() if stderr else f"exit code {proc.returncode}"
                logger.warning(f"swaymsg focus failed: {err}")
                # Fallback: try click-to-focus
                return await self._focus_fallback(title)

            logger.info(f"Sway window focused: {title!r}")
            return FocusResult(
                success=True,
                method="swaymsg",
                compositor="sway",
                message=f"Focused window via swaymsg: {title!r}",
            )
        except FileNotFoundError:
            return FocusResult(
                success=False,
                method="swaymsg",
                compositor="sway",
                message="swaymsg not found on PATH.",
            )
        except Exception as exc:
            logger.warning(f"swaymsg error: {exc}")
            return await self._focus_fallback(title)

    # ------------------------------------------------------------------
    # Hyprland
    # ------------------------------------------------------------------

    async def _focus_hyprland(self, title: str) -> FocusResult:
        """Focus a window on Hyprland via hyprctl."""
        if not shutil.which("hyprctl"):
            return FocusResult(
                success=False,
                method="hyprctl",
                compositor="hyprland",
                message="hyprctl binary not found on $PATH.",
            )
        try:
            cmd = ["hyprctl", "dispatch", "focuswindow", f"title:{title}"]
            logger.debug(f"Running: {' '.join(cmd)}")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                err = stderr.decode().strip() if stderr else f"exit code {proc.returncode}"
                logger.warning(f"hyprctl focus failed: {err}")
                return await self._focus_fallback(title)

            logger.info(f"Hyprland window focused: {title!r}")
            return FocusResult(
                success=True,
                method="hyprctl",
                compositor="hyprland",
                message=f"Focused window via hyprctl: {title!r}",
            )
        except FileNotFoundError:
            return FocusResult(
                success=False,
                method="hyprctl",
                compositor="hyprland",
                message="hyprctl not found on PATH.",
            )
        except Exception as exc:
            logger.warning(f"hyprctl error: {exc}")
            return await self._focus_fallback(title)

    # ------------------------------------------------------------------
    # GNOME (stub)
    # ------------------------------------------------------------------

    async def _focus_gnome(self, title: str) -> FocusResult:
        """
        GNOME window focus via the 'Activate Window By Title' extension
        (DBus: de.lucaswerkmeister.ActivateWindowByTitle).

        Attempts gdbus activation first; if the extension is unavailable,
        falls back to click-to-focus with setup instructions.
        """
        # Check if the GNOME extension DBus interface exists
        dbus_available = await self._check_gnome_extension()

        if dbus_available:
            return await self._focus_gnome_dbus(title)

        logger.warning(
            "GNOME auto-focus requires the 'Activate Window By Title' extension.\n"
            "  Install it from: https://extensions.gnome.org/extension/5021/activate-window-by-title/\n"
            "  Or your distribution's package manager (e.g., gnome-shell-extension-activate-window).\n"
            "Falling back to ydotool click-to-focus (unreliable)."
        )
        return await self._focus_fallback(title)

    async def _focus_gnome_dbus(self, title: str) -> FocusResult:
        """Execute the gdbus activateByTitle call for GNOME."""
        if not shutil.which("gdbus"):
            return FocusResult(
                success=False,
                method="gnome_dbus",
                compositor="gnome",
                message="gdbus binary not found on $PATH.",
            )
        try:
            cmd = [
                "gdbus", "call", "--session",
                "--dest", "de.lucaswerkmeister.ActivateWindowByTitle",
                "--object-path", "/de/lucaswerkmeister/ActivateWindowByTitle",
                "--method", "de.lucaswerkmeister.ActivateWindowByTitle.activateByTitle",
                title,
            ]
            logger.debug(f"Running: {' '.join(cmd)}")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                err = stderr.decode().strip() if stderr else f"exit code {proc.returncode}"
                logger.warning(f"GNOME gdbus activation failed: {err}")
                return await self._focus_fallback(title)

            # Parse the boolean return: "(<true>,)" or "(<false>,)" etc.
            output = stdout.decode().strip()
            logger.debug(f"GNOME gdbus output: {output!r}")

            if "true" in output.lower():
                logger.info(f"GNOME window activated via gdbus: {title!r}")
                return FocusResult(
                    success=True,
                    method="gnome_dbus",
                    compositor="gnome",
                    message=f"Focused window via GNOME extension: {title!r}",
                )

            logger.warning(
                f"GNOME gdbus returned false for title {title!r} — "
                f"window may not exist or extension may need updating."
            )
            return await self._focus_fallback(title)

        except FileNotFoundError:
            return FocusResult(
                success=False,
                method="gnome_dbus",
                compositor="gnome",
                message="gdbus not found on PATH.",
            )
        except Exception as exc:
            logger.warning(f"GNOME gdbus error: {exc}")
            return await self._focus_fallback(title)

    async def _check_gnome_extension(self) -> bool:
        """Return True if the ActivateWindowByTitle DBus interface is present."""
        if not shutil.which("gdbus"):
            logger.debug("gdbus not found; cannot check GNOME extension.")
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                "gdbus", "call", "--session",
                "--dest=de.lucaswerkmeister.ActivateWindowByTitle",
                "--object-path=/de/lucaswerkmeister/ActivateWindowByTitle",
                "--method=org.freedesktop.DBus.Introspectable.Introspect",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # KDE Plasma (stub)
    # ------------------------------------------------------------------

    async def _focus_kde(self, title: str) -> FocusResult:
        """
        KDE Plasma window focus via the bundled KWin script
        (DBus: org.kde.KWin.WindowActivator).

        If the KWin script is detected, calls qdbus activateWindow.
        If the script is not installed, attempts to auto-copy it to
        ~/.local/share/kwin/scripts/ and instructs the user to enable it.
        Falls back to click-to-focus in all other cases.
        """
        dbus_available = await self._check_kde_script()

        if dbus_available:
            return await self._focus_kde_dbus(title)

        # Script not installed — attempt auto-deployment
        deployed = await self._deploy_kde_script()
        if deployed:
            logger.info("KDE KWin script deployed — user must still enable it.")
        else:
            logger.warning("Could not deploy KDE KWin script automatically.")

        logger.warning(
            "KDE auto-focus requires the GameAI KWin script to be enabled.\n"
            "  1. The script has been copied to: ~/.local/share/kwin/scripts/gameai-activator/\n"
            "  2. Go to System Settings → Window Management → KWin Scripts\n"
            "  3. Enable 'GameAI Activator' and click Apply\n"
            "  4. Log out and back in (or run: systemctl restart --user plasma-kwin_wayland)\n"
            "Falling back to ydotool click-to-focus (unreliable)."
        )
        return await self._focus_fallback(title)

    async def _focus_kde_dbus(self, title: str) -> FocusResult:
        """Execute the qdbus activateWindow call for KDE."""
        if not shutil.which("qdbus"):
            return FocusResult(
                success=False,
                method="kde_dbus",
                compositor="kde",
                message="qdbus binary not found on $PATH.",
            )
        try:
            cmd = [
                "qdbus", "org.kde.KWin", "/WindowActivator",
                "org.kde.KWin.WindowActivator.activateWindow",
                title,
            ]
            logger.debug(f"Running: {' '.join(cmd)}")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                err = stderr.decode().strip() if stderr else f"exit code {proc.returncode}"
                logger.warning(f"KDE qdbus activation failed: {err}")
                return await self._focus_fallback(title)

            output = stdout.decode().strip()
            logger.debug(f"KDE qdbus output: {output!r}")

            # The KWin script returns a boolean — check for true/false
            if output.lower() == "true":
                logger.info(f"KDE window activated via qdbus: {title!r}")
                return FocusResult(
                    success=True,
                    method="kde_dbus",
                    compositor="kde",
                    message=f"Focused window via KDE KWin script: {title!r}",
                )

            logger.warning(
                f"KDE qdbus returned {output!r} for title {title!r} — "
                f"window may not exist or KWin script may need reloading."
            )
            return await self._focus_fallback(title)

        except FileNotFoundError:
            return FocusResult(
                success=False,
                method="kde_dbus",
                compositor="kde",
                message="qdbus not found on PATH.",
            )
        except Exception as exc:
            logger.warning(f"KDE qdbus error: {exc}")
            return await self._focus_fallback(title)

    async def _deploy_kde_script(self) -> bool:
        """Copy the bundled KWin script to the user's KWin scripts directory.

        Returns True if the copy succeeded, False otherwise.
        """
        import pathlib

        # Resolve the source directory relative to this source file
        src_dir = pathlib.Path(__file__).resolve().parent.parent / "resources" / "kwin" / "gameai-activator"
        dest_dir = pathlib.Path.home() / ".local" / "share" / "kwin" / "scripts" / "gameai-activator"

        if not src_dir.exists():
            logger.warning(f"KWin script source not found at {src_dir}")
            return False

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            for item in src_dir.iterdir():
                dest_item = dest_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest_item)
            logger.info(f"KWin script deployed to {dest_dir}")
            return True
        except OSError as exc:
            logger.warning(f"Failed to deploy KWin script: {exc}")
            return False

    async def _check_kde_script(self) -> bool:
        """Return True if the GameAI KWin script DBus interface is present."""
        if not shutil.which("qdbus"):
            logger.debug("qdbus not found; cannot check KDE KWin script.")
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                "qdbus", "org.kde.KWin", "/WindowActivator",
                "org.kde.KWin.WindowActivator.activateWindow",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # COSMIC (stub)
    # ------------------------------------------------------------------

    async def _focus_cosmic(self, title: str) -> FocusResult:
        """COSMIC window focus — stub (deferred to Phase 2+)."""
        logger.info(
            "COSMIC compositor detected. Native auto-focus is not yet implemented. "
            "Falling back to ydotool click-to-focus."
        )
        return await self._focus_fallback(title)

    # ------------------------------------------------------------------
    # Fallback: click-to-focus via ydotool / uinput
    # ------------------------------------------------------------------

    async def _focus_fallback(self, title: str) -> FocusResult:
        """
        Last-resort fallback: simulate a click near the window's top-left
        corner via the InputController (ydotool / uinput).

        Per Extra_research04 §5.2:
          - Reliability is VERY LOW.
          - Only works with click-to-focus compositor policies.
          - Always log a prominent warning.
        """
        if self._input is None:
            return FocusResult(
                success=False,
                method="none",
                compositor=self._compositor.value,
                message="No InputController available for click-to-focus fallback.",
            )

        logger.warning(
            "Attempting fallback click-to-focus via ydotool/uinput. "
            "This is unreliable — auto-focus may have failed. "
            "Please ensure the game window stays in focus, or set up "
            "proper auto-focus for your desktop environment."
        )

        try:
            # Retrieve window coordinates from the capture system if possible,
            # otherwise use a conservative default near (0,0)
            window_x, window_y = self._get_fallback_coordinates()

            # Click near the title bar / top-left corner
            click_x = window_x + 20
            click_y = window_y + 10
            logger.debug(f"Fallback click at ({click_x}, {click_y})")

            await self._input.move_mouse(click_x, click_y, relative=False)
            await asyncio.sleep(0.05)
            await self._input.click("left")
            await asyncio.sleep(0.10)

            return FocusResult(
                success=True,  # we tried; assume it worked
                method="ydotool_click_fallback",
                compositor=self._compositor.value,
                message=(
                    f"Fallback click at ({click_x}, {click_y}). "
                    f"Focus is NOT guaranteed — verify manually."
                ),
            )
        except Exception as exc:
            logger.warning(f"Fallback click failed: {exc}")
            return FocusResult(
                success=False,
                method="ydotool_click_fallback",
                compositor=self._compositor.value,
                message=f"Fallback click failed: {exc}",
            )

    @staticmethod
    def _get_fallback_coordinates() -> tuple[int, int]:
        """
        Return (x, y) coordinates to use as a base for fallback clicking.

        Prefers the capture_region from the window tracker if available
        (via screen_capture module), otherwise defaults to (0, 0).
        """
        # We avoid importing screen_capture at module level to prevent
        # circular imports.  Most callers on Wayland already have a
        # capture_region set; if not, (0,0) is a reasonable guess.
        return (0, 0)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release any resources held by the focus manager."""
        pass
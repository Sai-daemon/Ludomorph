"""
Input Abstraction Layer ― Step 1.3

Provides a unified async InputController that abstracts away the differences
between pynput (Windows / X11), python-ydotool (Wayland), dotool CLI fallback,
and pyautogui (emergency fallback).

Spec references:
- architecture.md §4.1  Cross-Platform Strategy
- architecture.md §4.2  Linux Wayland Setup Requirements
- architecture.md §4.3  Python Interface Abstraction
- Implementation_Phases.md §1.3
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Platform detection helpers
# ============================================================================

def _is_windows() -> bool:
    """Return True when running on Windows."""
    return sys.platform == "win32"


def _is_x11() -> bool:
    """Return True when the XDG session type is X11."""
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "x11"


def _is_wayland() -> bool:
    """Return True when the XDG session type is Wayland."""
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


# ============================================================================
# Custom exception
# ============================================================================

class InputError(Exception):
    """
    Raised when an input operation cannot be completed.

    This covers backend initialisation failures, unsupported platforms,
    invalid key names, macro step errors, and timeouts.
    """


# ============================================================================
# Backend detection & initialisation
# ============================================================================

@dataclass(frozen=True)
class _BackendInfo:
    """Describes a resolved input backend."""

    name: str          # "pynput" | "ydotool" | "dotool" | "pyautogui"
    is_wayland: bool   # True when the backend drives Wayland
    details: str       # human-readable description for logging


def _check_ydotoold_socket() -> bool:
    """
    Return True if the ydotoold socket exists.

    Per §4.2 the expected location is /run/user/<uid>/ydotoold/socket.
    """
    uid = os.getuid()
    socket_path = f"/run/user/{uid}/ydotoold/socket"
    exists = os.path.exists(socket_path)
    logger.debug(f"ydotoold socket check: {socket_path} → {'found' if exists else 'missing'}")
    return exists


def _dotool_available() -> bool:
    """Return True if the `dotool` CLI is on $PATH."""
    return shutil.which("dotool") is not None


def _resolve_backend(desired: str | None = None) -> _BackendInfo:
    """
    Resolve which input backend to use.

    Parameters
    ----------
    desired : str or None
        Value of ``input_backend`` from config.json.  Supported values:
        ``"auto"``, ``"pynput"``, ``"ydotool"``, ``"dotool"``.

    Returns
    -------
    _BackendInfo

    Raises
    ------
    InputError
        When the desired backend is unavailable on the current platform or
        no usable backend could be auto-detected.
    """
    desired = (desired or "auto").strip().lower()

    if desired not in ("auto", "pynput", "ydotool", "dotool"):
        raise InputError(
            f"Unknown input_backend value: {desired!r}. "
            f"Expected one of: auto, pynput, ydotool, dotool."
        )

    # -- Windows ----------------------------------------------------------
    if _is_windows():
        if desired in ("auto", "pynput"):
            return _BackendInfo(name="pynput", is_wayland=False, details="Windows → pynput")
        if desired == "ydotool":
            raise InputError("ydotool backend is not supported on Windows.")
        if desired == "dotool":
            raise InputError("dotool backend is not supported on Windows.")
        raise InputError("No usable input backend for Windows.")  # unreachable

    # -- Linux X11 --------------------------------------------------------
    if _is_x11():
        if desired in ("auto", "pynput"):
            return _BackendInfo(name="pynput", is_wayland=False, details="Linux X11 → pynput")
        if desired in ("ydotool", "dotool"):
            raise InputError(
                f"On X11 the recommended backend is pynput. "
                f"'{desired}' was requested but is intended for Wayland."
            )
        raise InputError("No usable input backend for Linux X11.")  # unreachable

    # -- Linux Wayland (or TTY / unknown) ---------------------------------
    if desired == "pynput":
        raise InputError(
            "pynput does not support Wayland. "
            "Set input_backend to 'auto', 'ydotool', or 'dotool'."
        )

    if desired == "ydotool":
        if not _check_ydotoold_socket():
            raise InputError(
                "ydotoold socket not found at /run/user/<uid>/ydotoold/socket.\n"
                "Please ensure ydotool is installed and the ydotoold daemon is running.\n"
                "See the user guide for Wayland setup instructions "
                "(architecture.md §4.2)."
            )
        return _BackendInfo(name="ydotool", is_wayland=True, details="Linux Wayland → ydotool")

    if desired == "dotool":
        if not _dotool_available():
            raise InputError(
                "dotool binary not found on $PATH.\n"
                "Install it via your package manager, e.g.: sudo apt install dotool"
            )
        return _BackendInfo(name="dotool", is_wayland=True, details="Linux Wayland → dotool")

    # desired == "auto"
    if _check_ydotoold_socket():
        return _BackendInfo(name="ydotool", is_wayland=True, details="Linux Wayland → ydotool (auto)")
    if _dotool_available():
        return _BackendInfo(name="dotool", is_wayland=True, details="Linux Wayland → dotool (auto)")
    raise InputError(
        "No usable input backend found for Linux Wayland.\n"
        "Neither the ydotoold socket nor the dotool CLI are available.\n"
        "Please install ydotool and start ydotoold, or install dotool.\n"
        "See the user guide for Wayland setup instructions (architecture.md §4.2)."
    )


# ============================================================================
# InputController
# ============================================================================

class InputController:
    """
    Cross-platform async input injection.

    Usage::

        ctrl = InputController(config)
        await ctrl.press_key("w", duration=0.5)
        await ctrl.move_mouse(100, 200, relative=False)
        await ctrl.click("left")
        await ctrl.type_string("hello")
        await ctrl.macro([
            {"type": "key", "key": "space", "duration": 0.1},
            {"type": "mouse_move", "x": 300, "y": 150, "relative": False},
            {"type": "click", "button": "left"},
        ])
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Parameters
        ----------
        config : dict
            The global configuration dict (at minimum must contain
            ``"input_backend"``).
        """
        self._config = config
        desired = config.get("input_backend", "auto")
        self._backend = _resolve_backend(desired)
        logger.info(f"InputController initialised with backend: {self._backend.details}")

    # -- public properties ------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Return the resolved backend name (for logging / UI)."""
        return self._backend.name

    # ------------------------------------------------------------------
    # key_down / key_up (press-only and release-only)
    # ------------------------------------------------------------------

    async def key_down(self, key: str) -> None:
        """
        Press *key* and hold it down.  Does NOT release.

        The caller is responsible for eventually calling ``key_up()``
        to release the key, otherwise it will stick.

        Raises InputError on failure.
        """
        try:
            if self._backend.name == "pynput":
                await self._pynput_key_down(key)
            elif self._backend.name == "ydotool":
                await self._ydotool_key_down(key)
            elif self._backend.name == "dotool":
                await self._dotool_key_down(key)
            else:
                raise InputError(f"Unsupported backend: {self._backend.name}")
        except InputError:
            raise
        except Exception as exc:
            raise InputError(f"key_down({key!r}) failed: {exc}") from exc

    async def key_up(self, key: str) -> None:
        """
        Release *key* that was previously pressed with ``key_down()``.

        Safe to call even if the key was not pressed (no-op on most backends).

        Raises InputError on failure.
        """
        try:
            if self._backend.name == "pynput":
                await self._pynput_key_up(key)
            elif self._backend.name == "ydotool":
                await self._ydotool_key_up(key)
            elif self._backend.name == "dotool":
                await self._dotool_key_up(key)
            else:
                raise InputError(f"Unsupported backend: {self._backend.name}")
        except InputError:
            raise
        except Exception as exc:
            raise InputError(f"key_up({key!r}) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # press_key
    # ------------------------------------------------------------------

    async def press_key(self, key: str, duration: float = 0.05) -> None:
        """
        Press and hold *key* for *duration* seconds, then release.

        This is a convenience wrapper around ``key_down()`` + sleep + ``key_up()``.
        For precise hold timing the caller should use ``key_down()`` / ``key_up()``
        directly together with ``macro_executor.accurate_hold()``.

        Raises InputError on failure.
        """
        try:
            if self._backend.name == "pynput":
                await self._pynput_press_key(key, duration)
            elif self._backend.name == "ydotool":
                await self._ydotool_press_key(key, duration)
            elif self._backend.name == "dotool":
                await self._dotool_press_key(key, duration)
            else:
                raise InputError(f"Unsupported backend: {self._backend.name}")
        except InputError:
            raise
        except Exception as exc:
            raise InputError(f"press_key({key!r}, duration={duration}) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # move_mouse
    # ------------------------------------------------------------------

    async def move_mouse(self, x: int, y: int, relative: bool = False) -> None:
        """
        Move the mouse cursor.

        If *relative* is False (default) the coordinates are absolute screen
        positions.  When True they are interpreted as deltas.

        Raises InputError on failure.
        """
        try:
            if self._backend.name == "pynput":
                await self._pynput_move_mouse(x, y, relative)
            elif self._backend.name == "ydotool":
                await self._ydotool_move_mouse(x, y, relative)
            elif self._backend.name == "dotool":
                await self._dotool_move_mouse(x, y, relative)
            else:
                raise InputError(f"Unsupported backend: {self._backend.name}")
        except InputError:
            raise
        except Exception as exc:
            raise InputError(f"move_mouse({x}, {y}, relative={relative}) failed: {exc}") from exc

    async def move_mouse_smooth(
        self, x: int, y: int, speed: float = 1.0, relative: bool = False,
    ) -> None:
        """
        Move the mouse cursor smoothly using linear interpolation.

        Speed controls the movement pace on a 0.0–2.0 scale:

        - ``0.0`` → Slow (~600 px/s, human-like deliberate movement)
        - ``1.0`` → Normal (~3000 px/s, faster than human but visibly smooth)
        - ``2.0`` → Fast / Instant (teleports directly, no interpolation)

        Values between these snap points scale linearly.  The method reads
        the current cursor position, computes the straight-line distance,
        divides it into micro-steps, and moves incrementally with small
        sleeps so the event loop is not starved.

        If *relative* is True, *x* and *y* are treated as deltas from the
        current position.

        Raises InputError on failure (falls back to instant move if the
        current position cannot be read).
        """
        if speed >= 2.0:
            # Instant / teleport — delegate to the normal move
            await self.move_mouse(x, y, relative=relative)
            return

        # Read current cursor position -------------------------------------------------
        try:
            cx, cy = await self._get_current_mouse_position()
        except Exception:
            # Cannot read position — fall back to instant move
            logger.debug("Cannot read mouse position; falling back to instant move.")
            await self.move_mouse(x, y, relative=relative)
            return

        # Compute target -----------------------------------------------------------------
        if relative:
            target_x = cx + x
            target_y = cy + y
        else:
            target_x = x
            target_y = y

        # Early exit if already at target
        if cx == target_x and cy == target_y:
            return

        # Compute distance and step count ------------------------------------------------
        import math

        dist = math.hypot(target_x - cx, target_y - cy)
        if dist < 1.0:
            await self.move_mouse(target_x, target_y, relative=False)
            return

        # Base pixels-per-second mapped from speed (0.0→600, 1.0→3000, 2.0→instant)
        base_pps = 600.0 + (speed / 2.0) * (3000.0 - 600.0) * 2.0  # linear: 600→3000→6000
        # Step interval: target ~120 updates/s for smoothness, but at least 2 steps
        steps_per_second = 120.0
        step_interval = 1.0 / steps_per_second
        pixels_per_step = base_pps / steps_per_second
        num_steps = max(2, int(dist / pixels_per_step))

        dx = (target_x - cx) / num_steps
        dy = (target_y - cy) / num_steps

        for i in range(1, num_steps + 1):
            interp_x = round(cx + dx * i)
            interp_y = round(cy + dy * i)
            # Use relative movement for backends that support it well
            if self._backend.name == "pynput":
                step_dx = round(dx)
                step_dy = round(dy)
                await self._pynput_move_mouse(step_dx, step_dy, relative=True)
            else:
                # ydotool / dotool: use absolute positioning per step
                await self.move_mouse(interp_x, interp_y, relative=False)

            await asyncio.sleep(step_interval)

        # Final correction: ensure we land exactly on target
        await self.move_mouse(target_x, target_y, relative=False)

    async def _get_current_mouse_position(self) -> tuple[int, int]:
        """Return the current (x, y) screen coordinates of the mouse cursor.

        Uses the most reliable method available for the active backend.
        """
        if self._backend.name == "pynput":
            from pynput.mouse import Controller as MouseController
            mc = MouseController()
            pos = await asyncio.to_thread(lambda: mc.position)
            return int(pos[0]), int(pos[1])

        # For ydotool / dotool backends try to read via pynput (reading
        # usually works even on Wayland where injection doesn't).
        try:
            from pynput.mouse import Controller as MouseController
            mc = MouseController()
            pos = await asyncio.to_thread(lambda: mc.position)
            return int(pos[0]), int(pos[1])
        except Exception:
            raise InputError("Cannot determine current mouse position.")

    # ------------------------------------------------------------------
    # click
    # ------------------------------------------------------------------

    async def click(self, button: str = "left") -> None:
        """
        Click the given mouse *button* (``"left"``, ``"right"``, ``"middle"``).

        Raises InputError on failure.
        """
        try:
            if self._backend.name == "pynput":
                await self._pynput_click(button)
            elif self._backend.name == "ydotool":
                await self._ydotool_click(button)
            elif self._backend.name == "dotool":
                await self._dotool_click(button)
            else:
                raise InputError(f"Unsupported backend: {self._backend.name}")
        except InputError:
            raise
        except Exception as exc:
            raise InputError(f"click({button!r}) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # type_string
    # ------------------------------------------------------------------

    async def type_string(self, text: str) -> None:
        """
        Type *text* as keyboard input (character by character).

        Raises InputError on failure.
        """
        try:
            if self._backend.name == "pynput":
                await self._pynput_type_string(text)
            elif self._backend.name == "ydotool":
                await self._ydotool_type_string(text)
            elif self._backend.name == "dotool":
                await self._dotool_type_string(text)
            else:
                raise InputError(f"Unsupported backend: {self._backend.name}")
        except InputError:
            raise
        except Exception as exc:
            raise InputError(f"type_string({text!r}) failed: {exc}") from exc

    # ------------------------------------------------------------------
    # macro
    # ------------------------------------------------------------------

    async def macro(self, steps: list[dict[str, Any]]) -> None:
        """
        Execute a list of macro *steps* sequentially.

        Each step is a dict with a ``"type"`` key and per-type payload:

        - ``{"type": "key", "key": "w", "duration": 0.5}``
        - ``{"type": "mouse_move", "x": 100, "y": 200, "relative": false}``
        - ``{"type": "click", "button": "left"}``

        Raises InputError if any step fails.
        """
        for i, step in enumerate(steps):
            step_type = step.get("type")
            try:
                if step_type == "key":
                    await self.press_key(
                        key=step["key"],
                        duration=float(step.get("duration", 0.05)),
                    )
                elif step_type == "mouse_move":
                    await self.move_mouse(
                        x=int(step["x"]),
                        y=int(step["y"]),
                        relative=bool(step.get("relative", False)),
                    )
                elif step_type == "click":
                    await self.click(button=step.get("button", "left"))
                else:
                    raise InputError(f"Unknown macro step type at index {i}: {step_type!r}")
            except InputError:
                raise
            except KeyError as exc:
                raise InputError(
                    f"Macro step at index {i} missing required key: {exc}"
                ) from exc
            except Exception as exc:
                raise InputError(
                    f"Macro step at index {i} ({step_type}) failed: {exc}"
                ) from exc

    # ==================================================================
    # pynput backend (Windows / X11)
    # ==================================================================

    async def _pynput_key_down(self, key: str) -> None:
        from pynput.keyboard import Controller as KeyboardController, Key

        kc = KeyboardController()
        resolved = self._resolve_pynput_key(key, Key)
        await asyncio.to_thread(kc.press, resolved)

    async def _pynput_key_up(self, key: str) -> None:
        from pynput.keyboard import Controller as KeyboardController, Key

        kc = KeyboardController()
        resolved = self._resolve_pynput_key(key, Key)
        await asyncio.to_thread(kc.release, resolved)

    async def _pynput_press_key(self, key: str, duration: float) -> None:
        from pynput.keyboard import Controller as KeyboardController, Key

        kc = KeyboardController()
        resolved = self._resolve_pynput_key(key, Key)
        await asyncio.to_thread(kc.press, resolved)
        await asyncio.sleep(duration)
        await asyncio.to_thread(kc.release, resolved)

    async def _pynput_move_mouse(self, x: int, y: int, relative: bool) -> None:
        from pynput.mouse import Controller as MouseController

        mc = MouseController()
        if relative:
            await asyncio.to_thread(mc.move, x, y)
        else:
            await asyncio.to_thread(self._set_mouse_position, mc, x, y)

    async def _pynput_click(self, button: str) -> None:
        from pynput.mouse import Button, Controller as MouseController

        mc = MouseController()
        resolved = self._resolve_pynput_button(button, Button)
        await asyncio.to_thread(mc.press, resolved)
        await asyncio.sleep(0.01)
        await asyncio.to_thread(mc.release, resolved)

    async def _pynput_type_string(self, text: str) -> None:
        from pynput.keyboard import Controller as KeyboardController

        kc = KeyboardController()
        await asyncio.to_thread(kc.type, text)

    @staticmethod
    def _resolve_pynput_key(key: str, key_enum: Any) -> Any:
        """Map a human-readable key name to a pynput Key / KeyCode."""
        # pynput key names are stored as Key.foo.name (lowercase).
        # We normalise the input and try to match.
        lower = key.lower().strip()
        for attr in dir(key_enum):
            if attr.startswith("_"):
                continue
            val = getattr(key_enum, attr)
            if hasattr(val, "name") and val.name == lower:
                return val
        # Plain character key
        if len(key) == 1:
            from pynput.keyboard import KeyCode
            return KeyCode.from_char(key)
        raise InputError(f"Unknown pynput key: {key!r}")

    @staticmethod
    def _set_mouse_position(mc: Any, x: int, y: int) -> None:
        """Set absolute mouse position (callable wrapper for asyncio.to_thread)."""
        mc.position = (x, y)

    @staticmethod
    def _resolve_pynput_button(button: str, button_enum: Any) -> Any:
        """Map a human-readable button name to a pynput Button."""
        mapping = {
            "left": button_enum.left,
            "right": button_enum.right,
            "middle": button_enum.middle,
        }
        btn = mapping.get(button.lower().strip())
        if btn is None:
            raise InputError(f"Unknown mouse button: {button!r}. Expected left, right, or middle.")
        return btn

    # ==================================================================
    # python-ydotool backend (Wayland primary)
    # ==================================================================

    async def _ydotool_key_down(self, key: str) -> None:
        import ydotool

        code = self._key_to_ydotool_code(key)
        await asyncio.to_thread(ydotool.key_press, [code])

    async def _ydotool_key_up(self, key: str) -> None:
        import ydotool

        code = self._key_to_ydotool_code(key)
        await asyncio.to_thread(ydotool.key_release, [code])

    async def _ydotool_press_key(self, key: str, duration: float) -> None:
        import ydotool

        code = self._key_to_ydotool_code(key)
        await asyncio.to_thread(ydotool.key_press, [code])
        await asyncio.sleep(duration)
        await asyncio.to_thread(ydotool.key_release, [code])

    async def _ydotool_move_mouse(self, x: int, y: int, relative: bool) -> None:
        import ydotool

        if relative:
            await asyncio.to_thread(ydotool.mouse_move_relative, x, y)
        else:
            await asyncio.to_thread(ydotool.mouse_move_absolute, x, y)

    async def _ydotool_click(self, button: str) -> None:
        import ydotool

        btn_code = self._button_to_ydotool_code(button)
        await asyncio.to_thread(ydotool.mouse_click, btn_code)

    async def _ydotool_type_string(self, text: str) -> None:
        import ydotool

        await asyncio.to_thread(ydotool.type_text, text)

    @staticmethod
    def _key_to_ydotool_code(key: str) -> int:
        """
        Map a human-readable key name to a Linux input event code.

        This is a best-effort mapping.  For the full list see
        /usr/include/linux/input-event-codes.h.
        """
        lower = key.lower().strip()
        mapping: dict[str, int] = {
            "a": 30, "b": 48, "c": 46, "d": 32, "e": 18, "f": 33, "g": 34,
            "h": 35, "i": 23, "j": 36, "k": 37, "l": 38, "m": 50, "n": 49,
            "o": 24, "p": 25, "q": 16, "r": 19, "s": 31, "t": 20, "u": 22,
            "v": 47, "w": 17, "x": 45, "y": 21, "z": 44,
            "0": 11, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7,
            "7": 8, "8": 9, "9": 10,
            "space": 57,
            "enter": 28, "return": 28,
            "escape": 1, "esc": 1,
            "tab": 15,
            "backspace": 14,
            "shift": 42, "shift_r": 54,
            "ctrl": 29, "ctrl_r": 97,
            "alt": 56, "alt_r": 100,
            "left": 105, "right": 106, "up": 103, "down": 108,
            "f1": 59, "f2": 60, "f3": 61, "f4": 62, "f5": 63, "f6": 64,
            "f7": 65, "f8": 66, "f9": 67, "f10": 68, "f11": 87, "f12": 88,
        }
        if lower in mapping:
            return mapping[lower]
        if len(key) == 1 and key.isprintable():
            # Fallback: compute from ASCII-ish range
            # This is approximate; ydotool expects proper scan codes.
            raise InputError(
                f"Cannot reliably map key {key!r} to ydotool scan code. "
                f"Use a named key from the supported list."
            )
        raise InputError(f"Unknown ydotool key: {key!r}")

    @staticmethod
    def _button_to_ydotool_code(button: str) -> int:
        mapping: dict[str, int] = {
            "left": 0x110,   # BTN_LEFT
            "right": 0x111,  # BTN_RIGHT
            "middle": 0x112, # BTN_MIDDLE
        }
        btn = mapping.get(button.lower().strip())
        if btn is None:
            raise InputError(f"Unknown mouse button: {button!r}. Expected left, right, or middle.")
        return btn

    # ==================================================================
    # dotool CLI fallback (Wayland secondary)
    # ==================================================================

    async def _dotool_key_down(self, key: str) -> None:
        # dotool uses `keydown` command to press without releasing
        await self._run_dotool(["keydown", key])

    async def _dotool_key_up(self, key: str) -> None:
        await self._run_dotool(["keyup", key])

    async def _dotool_press_key(self, key: str, duration: float) -> None:
        await self._run_dotool(["key", f"{key}:{duration}"])

    async def _dotool_move_mouse(self, x: int, y: int, relative: bool) -> None:
        mode = "mousemove_relative" if relative else "mousemove"
        await self._run_dotool([mode, str(x), str(y)])

    async def _dotool_click(self, button: str) -> None:
        await self._run_dotool(["click", button])

    async def _dotool_type_string(self, text: str) -> None:
        await self._run_dotool(["type", text])

    async def _run_dotool(self, args: list[str]) -> None:
        """Execute a dotool CLI command asynchronously."""
        cmd = ["dotool"] + args
        logger.debug(f"dotool: {' '.join(cmd)}")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                err = stderr.decode().strip() if stderr else f"exit code {proc.returncode}"
                raise InputError(f"dotool command failed: {' '.join(cmd)}\n{err}")
        except InputError:
            raise
        except FileNotFoundError:
            raise InputError(
                "dotool binary not found on $PATH. Install it via your package manager."
            ) from None
        except Exception as exc:
            raise InputError(f"dotool error: {exc}") from exc

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release any resources held by the input backend."""
        # pynput, ydotool, and dotool are stateless; nothing to close.
        pass
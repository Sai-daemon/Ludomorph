"""
Global Hotkey Listener — monitors for the debug overlay toggle key (~ / grave)
using pynput.keyboard.Listener in a daemon thread.

Communicates with the GUI main thread via a callback that is invoked
via ``AsyncTk.call_async()`` so all tkinter operations stay on the
main thread.

Key handling
------------
* Linux (X11 / Wayland) — ``KeyCode.from_vk(0xC0)`` maps to the
  grave/tilde key on US/UK layouts.
* The key ***must*** be pressed and released within 0.25 s (not held)
  to avoid triggering on normal typing or key‑repeat events.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from pynput.keyboard import KeyCode, Listener, Key

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Virtual key‑code for the grave/tilde key (US layout — the key to the
# left of ``1``).  ``0xC0`` = 192 decimal.
_GRAVE_VK = 0xC0

# Maximum time (seconds) between key‑press and key‑release to count as
# a deliberate toggle.  Longer holds are ignored to avoid false triggers
# when the user is holding the key for in‑game purposes.
_MAX_PRESS_DURATION = 0.25


class GlobalHotkeyListener:
    """Listens for the ``~`` (grave) key globally via ``pynput``.

    Runs the listener in a daemon thread so the main application never
    blocks.  The *callback* is invoked (with no arguments) whenever the
    key is tapped.

    Parameters
    ----------
    callback : Callable[[], None]
        Called on the **pynput listener thread** when the hotkey is
        detected.  The caller is responsible for dispatching to the
        GUI thread (e.g. via ``AsyncTk.call_async``).
    start : bool
        If ``True`` (the default) the listener thread is started
        immediately.  Set to ``False`` to defer.
    """

    def __init__(
        self,
        callback: Callable[[], None],
        *,
        start: bool = True,
    ) -> None:
        self._callback = callback
        self._running = False
        self._listener: Listener | None = None
        self._thread: threading.Thread | None = None

        # Track key‑down time so we only fire on a quick tap.
        self._graves_down_at: float | None = None

        if start:
            self.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the listener thread.

        Idempotent — safe to call if already running.
        """
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._listen, name="gameai-hotkey", daemon=True,
        )
        self._thread.start()
        logger.info("Global hotkey listener started (~ / grave).")

    def stop(self) -> None:
        """Stop the listener thread.

        Blocks for up to 2 s waiting for the thread to join.  Idempotent.
        """
        if not self._running:
            return
        self._running = False
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.debug("Hotkey listener thread did not join within 2 s.")
            self._thread = None
        logger.info("Global hotkey listener stopped.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _listen(self) -> None:
        """Run the blocking pynput listener.

        Any unhandled exception is logged and the listener restarted
        (with a short back‑off) as long as ``self._running`` is True.
        """
        while self._running:
            try:
                self._listener = Listener(
                    on_press=self._on_press,
                    on_release=self._on_release,
                )
                self._listener.start()
                self._listener.join()  # blocks until stopped
            except Exception as exc:
                logger.warning(f"Hotkey listener thread error: {exc}")
                if self._running:
                    time.sleep(1.0)  # brief back‑off before restart
            finally:
                self._listener = None

    def _on_press(self, key: KeyCode | Key | None) -> None:
        """Track when the grave key is pressed."""
        if self._is_grave(key) and self._graves_down_at is None:
            self._graves_down_at = time.monotonic()

    def _on_release(self, key: KeyCode | Key | None) -> None:
        """Fire the callback if the grave key was a quick tap."""
        if not self._is_grave(key):
            return
        if self._graves_down_at is None:
            return

        duration = time.monotonic() - self._graves_down_at
        self._graves_down_at = None

        if duration <= _MAX_PRESS_DURATION:
            logger.debug("Debug hotkey (~) tapped (%.0f ms).", duration * 1000)
            try:
                self._callback()
            except Exception as exc:
                logger.warning(f"Hotkey callback raised: {exc}")

    @staticmethod
    def _is_grave(key: KeyCode | Key | None) -> bool:
        """Return ``True`` if *key* is the grave/tilde virtual key."""
        if isinstance(key, KeyCode) and key.vk == _GRAVE_VK:
            return True
        # Some platforms may report it as char '`'
        if isinstance(key, KeyCode) and key.char == "`":
            return True
        return False
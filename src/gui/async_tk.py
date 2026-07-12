"""
AsyncTk — tkinter + asyncio bridge.

Implements the polling-based bridge from architecture.md §7.5.
Because tkinter is not async‑safe and runs its own event loop, all
asyncio→GUI communication goes through ``call_async(callback)``,
which places callbacks onto an ``asyncio.Queue`` drained on the
tkinter main thread every 100 ms.
"""

from __future__ import annotations

import asyncio
import tkinter as tk
from typing import Any


class AsyncTk(tk.Tk):
    """Tkinter root window with an asyncio bridge for thread-safe GUI updates.

    Usage from asyncio tasks::

        app.call_async(lambda: label.config(text="Done!"))

    ``call_async`` is the **only** safe way to touch tkinter widgets from
    asyncio coroutines or background threads.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._async_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._poll_async_queue()

    # ------------------------------------------------------------------
    # Core bridge
    # ------------------------------------------------------------------

    def _poll_async_queue(self) -> None:
        """Drain the async callback queue on the tkinter main thread."""
        try:
            while True:
                callback = self._async_queue.get_nowait()
                callback()
        except asyncio.QueueEmpty:
            pass
        self.after(100, self._poll_async_queue)

    def call_async(self, callback: Any) -> None:
        """Thread-safe: schedule *callback* to run on the tkinter main thread."""
        self._async_queue.put_nowait(callback)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Gracefully tear down the bridge on window close."""
        super().destroy()
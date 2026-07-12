"""
Phase 5.6 — HealthPanel UI Widget.

A ttk.Frame-based panel that displays the live health status of all
dependent services (Ollama, Tesseract/OCR, MCP memory server, game window)
with colour-coded indicators and expandable actionable error messages.

Uses the ``asyncio.Queue`` → tkinter polling pattern from
``Calibration_UI_research.md`` §Problem 1 (``StatusUIHook``) so the
async ``HealthMonitor`` can push updates to the tkinter main thread
without blocking.

Widget layout::

    ┌──────────────────────────────────────────────┐
    │  ● All Systems Nominal            [⏻ Refresh]│
    ├──────────────────────────────────────────────┤
    │  🧠 Ollama LLM            ● Connected         │
    │     Ollama 0.5.4 — phi3.5:3.8b              │
    ├──────────────────────────────────────────────┤
    │  📖 Tesseract OCR         ● Ready             │
    │     OCR initialised                          │
    ├──────────────────────────────────────────────┤
    │  💾 MCP Memory            ● Reachable         │
    │     MCP server reachable (port 8000)         │
    ├──────────────────────────────────────────────┤
    │  🪟 Game Window           ● Present           │
    │     Game window present                      │
    └──────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import tkinter as tk
from datetime import datetime, timezone
from tkinter import ttk
from typing import Any

# ---------------------------------------------------------------------------
# Colour palette (matches main_window.py dark theme)
# ---------------------------------------------------------------------------

_BG = "#1E1E1E"
_FG = "#D4D4D4"
_ACCENT = "#0078D4"
_SUCCESS = "#50C878"
_DANGER = "#E04040"
_WARNING = "#E8A317"
_DISABLED_FG = "#808080"
_CARD_BG = "#2D2D2D"
_HEADER_BG = "#252526"
_SEPARATOR = "#3C3C3C"

# Per-service icons
_SERVICE_ICONS: dict[str, str] = {
    "ollama": "🧠",
    "tesseract": "📖",
    "mcp": "💾",
    "game_window": "🪟",
}

# Human-readable service names
_SERVICE_NAMES: dict[str, str] = {
    "ollama": "Ollama LLM",
    "tesseract": "Tesseract OCR",
    "mcp": "MCP Memory",
    "game_window": "Game Window",
}


# ---------------------------------------------------------------------------
# HealthPanel
# ---------------------------------------------------------------------------


class HealthPanel(ttk.Frame):  # type: ignore[misc]
    """Collapsible panel showing live health status of all services.

    Receives updates through :meth:`update_health` (called from the
    tkinter main thread via ``AsyncTk.call_async``) or by polling an
    ``asyncio.Queue`` filled by the backend ``HealthMonitor``.
    """

    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(parent, style="HealthPanel.TFrame", **kwargs)
        self.configure(style="HealthPanel.TFrame")

        # -- Style setup -------------------------------------------------------
        self._setup_styles()

        # -- Async queue for receiving updates from the health monitor ---------
        self._status_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        # -- Per-service card widgets (lazily created) -------------------------
        self._cards: dict[str, dict[str, tk.Widget]] = {}

        # -- Build sections ----------------------------------------------------
        self._build_header()
        self._build_separator()
        self._build_service_cards()

        # -- Start async poll --------------------------------------------------
        self._poll_queue()

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def _setup_styles(self) -> None:
        """Configure ttk styles for the health panel."""
        style = ttk.Style()
        style.configure(
            "HealthPanel.TFrame",
            background=_BG,
        )
        style.configure(
            "HealthCard.TFrame",
            background=_CARD_BG,
        )
        style.configure(
            "HealthHeader.TFrame",
            background=_HEADER_BG,
        )
        style.configure(
            "HealthStatus.TLabel",
            background=_CARD_BG,
            foreground=_FG,
            font=("Segoe UI", 9),
        )
        style.configure(
            "HealthService.TLabel",
            background=_CARD_BG,
            foreground=_FG,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "HealthReason.TLabel",
            background=_CARD_BG,
            foreground=_DISABLED_FG,
            font=("Segoe UI", 8),
            wraplength=380,
        )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        """Top bar: overall status + refresh button."""
        header = ttk.Frame(self, style="HealthHeader.TFrame", padding=(10, 6))
        header.pack(fill=tk.X)

        self._lbl_overall = tk.Label(
            header,
            text="●  Not yet checked",
            bg=_HEADER_BG,
            fg=_DISABLED_FG,
            font=("Segoe UI", 11, "bold"),
        )
        self._lbl_overall.pack(side=tk.LEFT)

        self._btn_refresh = ttk.Button(
            header,
            text="⏻  Refresh",
            command=self._on_refresh,
        )
        self._btn_refresh.pack(side=tk.RIGHT, padx=(8, 0))

        # Timestamp label
        self._lbl_timestamp = tk.Label(
            header,
            text="",
            bg=_HEADER_BG,
            fg=_DISABLED_FG,
            font=("Segoe UI", 8),
        )
        self._lbl_timestamp.pack(side=tk.RIGHT, padx=8)

    @staticmethod
    def _build_separator() -> None:
        """Visual separator (handled by cards themselves)."""
        pass

    # ------------------------------------------------------------------
    # Service cards
    # ------------------------------------------------------------------

    def _build_service_cards(self) -> None:
        """Create a card for each of the four monitored services."""
        for name in ("ollama", "tesseract", "mcp", "game_window"):
            self._create_card(name)

    def _create_card(self, name: str) -> None:
        """Create a single service status card."""
        icon = _SERVICE_ICONS.get(name, "❓")
        label = _SERVICE_NAMES.get(name, name.title())

        # Card frame
        card = ttk.Frame(self, style="HealthCard.TFrame", padding=(10, 5))
        card.pack(fill=tk.X, padx=2, pady=1)

        # Row 0: icon + service name + status dot
        row0 = ttk.Frame(card, style="HealthCard.TFrame")
        row0.pack(fill=tk.X)

        lbl_icon = tk.Label(
            row0, text=icon, bg=_CARD_BG, font=("Segoe UI", 14)
        )
        lbl_icon.pack(side=tk.LEFT, padx=(0, 6))

        lbl_name = tk.Label(
            row0,
            text=label,
            bg=_CARD_BG,
            fg=_FG,
            font=("Segoe UI", 10, "bold"),
        )
        lbl_name.pack(side=tk.LEFT)

        # Status dot (moved to right side)
        lbl_dot = tk.Label(
            row0,
            text="●",
            bg=_CARD_BG,
            fg=_DISABLED_FG,
            font=("Segoe UI", 12),
        )
        lbl_dot.pack(side=tk.RIGHT, padx=(8, 0))

        lbl_status = tk.Label(
            row0,
            text="Unknown",
            bg=_CARD_BG,
            fg=_DISABLED_FG,
            font=("Segoe UI", 9),
        )
        lbl_status.pack(side=tk.RIGHT, padx=(2, 0))

        # Row 1: reason / error message (expandable)
        lbl_reason = tk.Label(
            card,
            text="",
            bg=_CARD_BG,
            fg=_DISABLED_FG,
            font=("Segoe UI", 8),
            justify=tk.LEFT,
            anchor=tk.W,
        )
        lbl_reason.pack(fill=tk.X, padx=(28, 0), pady=(2, 0))

        self._cards[name] = {
            "card": card,
            "dot": lbl_dot,
            "status": lbl_status,
            "reason": lbl_reason,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_health(self, status: dict[str, Any]) -> None:
        """Apply a full health status dict to all cards.

        Called on the tkinter main thread.  The *status* dict must have
        the shape returned by ``HealthMonitor.get_overall_status()``::

            {
                "level": "OK" | "DEGRADED" | "STOPPING",
                "reason": "All systems nominal",
                "services": {
                    "ollama": {"healthy": True, "reason": "..."},
                    ...
                }
            }
        """
        level = status.get("level", "UNKNOWN")
        reason = status.get("reason", "")
        services = status.get("services", {})

        # -- Update overall header -----------------------------------------
        if level == "OK":
            self._lbl_overall.config(
                text=f"●  {reason}", fg=_SUCCESS
            )
        elif level == "DEGRADED":
            self._lbl_overall.config(
                text=f"⚠  {reason}", fg=_WARNING
            )
        elif level == "STOPPING":
            self._lbl_overall.config(
                text=f"⛔  {reason}", fg=_DANGER
            )
        else:
            self._lbl_overall.config(
                text=f"●  {reason}", fg=_DISABLED_FG
            )

        # Timestamp
        self._lbl_timestamp.config(
            text=datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        )

        # -- Update per-service cards --------------------------------------
        for name, svc_data in services.items():
            if name not in self._cards:
                self._create_card(name)

            card = self._cards[name]
            healthy = svc_data.get("healthy", False)
            svc_reason = svc_data.get("reason", "")

            if healthy:
                dot_colour = _SUCCESS
                status_text = "Healthy"
                reason_colour = _DISABLED_FG
            else:
                dot_colour = _DANGER
                status_text = "Unhealthy"
                reason_colour = _WARNING

            card["dot"].config(fg=dot_colour)
            card["status"].config(text=status_text, fg=dot_colour)
            card["reason"].config(text=svc_reason, fg=reason_colour)

    def get_queue(self) -> asyncio.Queue[dict[str, Any]]:
        """Return the ``asyncio.Queue`` that the backend pushes status into.

        The UI polls this queue every 200 ms and calls
        :meth:`update_health` with each new status dict.
        """
        return self._status_queue

    def push_status(self, status: dict[str, Any]) -> None:
        """Thread-safe: push a status update from an asyncio task.

        Equivalent to ``self._status_queue.put_nowait(status)`` — use
        this from ``HealthMonitor`` background tasks.
        """
        self._status_queue.put_nowait(status)

    # ------------------------------------------------------------------
    # Internal: async queue polling
    # ------------------------------------------------------------------

    def _poll_queue(self) -> None:
        """Drain the async status queue and apply updates."""
        try:
            while True:
                status = self._status_queue.get_nowait()
                self.update_health(status)
        except asyncio.QueueEmpty:
            pass
        self.after(200, self._poll_queue)

    # ------------------------------------------------------------------
    # Refresh button
    # ------------------------------------------------------------------

    def _on_refresh(self) -> None:
        """User clicked Refresh — push a sentinel that the poller interprets."""
        # The actual refresh happens in the async HealthMonitor;
        # here we just signal the UI that a manual check was requested.
        self._status_queue.put_nowait({"_refresh_request": True})


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "HealthPanel",
]
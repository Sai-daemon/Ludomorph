"""
Phase 5.6 — Health Monitor UI Panel.

Displays the real‑time status of every monitored service (Ollama,
Tesseract/OCR, MCP memory, game window) as compact colour‑coded cards
with expandable actionable error messages.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from src.gui.theme import ThemeManager, resolve_font_stack


class HealthPanel(tk.Frame):  # type: ignore[misc]
    """Collapsible panel showing live service health.

    Each service is rendered as a "card" with:
    - Icon + name
    - Healthy / Unhealthy dot (green / red)
    - Short reason text
    - Full error message (expandable)

    Services are registered via :meth:`set_service` and updated via
    :meth:`push_status` (called from the engine's health‑polling task).
    """

    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        # NOTE: tk.Frame does NOT support 'style' — we apply bg manually.
        # The ttk styles (HealthCard.TFrame etc.) are pre‑registered in theme.py.
        super().__init__(parent, **kwargs)

        self._tm = ThemeManager()
        self._tm.apply_ttk_styles()
        p = self._tm.palette
        self.configure(bg=p.bg)

        # Service cards — keyed by service name
        self._cards: dict[str, dict[str, Any]] = {}

        # -- Build layout -----------------------------------------------------
        self._build_header()
        self._build_service_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        """Top bar: overall status + refresh button."""
        p = self._tm.palette
        header = tk.Frame(self, bg=p.header_bg, padx=10, pady=6)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="🏥  System Health",
            bg=p.header_bg,
            fg=p.disabled_fg,
            font=(resolve_font_stack(p.ui_font), 11, "bold"),
        ).pack(side=tk.LEFT)

        self._lbl_overall = tk.Label(
            header,
            text="",
            bg=p.header_bg,
            fg=p.disabled_fg,
            font=(resolve_font_stack(p.ui_font), 8),
        )
        self._lbl_overall.pack(side=tk.RIGHT, padx=8)

    def _build_service_list(self) -> None:
        """Container frame for the individual service cards."""
        p = self._tm.palette
        self._cards_frame = tk.Frame(self, bg=p.bg)
        self._cards_frame.pack(fill=tk.X, padx=2, pady=2)

    # ------------------------------------------------------------------
    # Card factory
    # ------------------------------------------------------------------

    def _create_card(self, name: str, icon: str) -> dict[str, Any]:
        """Create a single service health card and return its widget references."""
        p = self._tm.palette
        card = tk.Frame(self._cards_frame, bg=p.card_bg, padx=10, pady=5)
        card.pack(fill=tk.X, padx=2, pady=1)

        # Row 0: icon + service name + status dot
        row0 = tk.Frame(card, bg=p.card_bg)
        row0.pack(fill=tk.X)

        ui_font = resolve_font_stack(p.ui_font)
        lbl_icon = tk.Label(
            row0, text=icon, bg=p.card_bg, font=(ui_font, 14),
        )
        lbl_icon.pack(side=tk.LEFT, padx=(0, 6))

        lbl_name = tk.Label(
            row0, text=name, bg=p.card_bg, fg=p.fg,
            font=(ui_font, 10, "bold"),
        )
        lbl_name.pack(side=tk.LEFT)

        lbl_dot = tk.Label(
            row0, text="●", bg=p.card_bg, fg=p.disabled_fg,
            font=(ui_font, 12),
        )
        lbl_dot.pack(side=tk.RIGHT, padx=4)

        lbl_status = tk.Label(
            row0, text="Unknown", bg=p.card_bg, fg=p.disabled_fg,
            font=(ui_font, 9),
        )
        lbl_status.pack(side=tk.RIGHT, padx=2)

        # Row 1: reason / error detail
        lbl_reason = tk.Label(
            card, text="", bg=p.card_bg, fg=p.disabled_fg,
            font=(ui_font, 8), justify=tk.LEFT, wraplength=380,
        )
        lbl_reason.pack(fill=tk.X, padx=(20, 0), pady=(2, 0))

        widgets = {
            "frame": card,
            "icon": lbl_icon,
            "name": lbl_name,
            "dot": lbl_dot,
            "status": lbl_status,
            "reason": lbl_reason,
        }
        self._cards[name] = widgets
        return widgets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_status(self, status: dict[str, Any]) -> None:
        """Update all service cards from a health status dict.

        Called from the engine's health‑polling task (via ``call_async``).
        """
        p = self._tm.palette
        services = status.get("services", {})
        level = status.get("level", "UNKNOWN")

        # Overall status
        if level == "OK":
            overall_colour = p.success
            overall_text = "All systems healthy"
        elif level == "DEGRADED":
            overall_colour = p.warning
            overall_text = "Some services degraded"
        else:
            overall_colour = p.danger
            overall_text = status.get("reason", "System unhealthy")

        self._lbl_overall.config(text=overall_text, fg=overall_colour)

        # Update each service card
        for svc_name, svc_data in services.items():
            if svc_name not in self._cards:
                icon = self._icon_for_service(svc_name)
                self._create_card(svc_name, icon)

            card = self._cards[svc_name]
            healthy = svc_data.get("healthy", False)
            svc_reason = svc_data.get("reason", "")

            if healthy:
                dot_colour = p.success
                status_text = "Healthy"
                reason_colour = p.disabled_fg
            else:
                dot_colour = p.danger
                status_text = "Unhealthy"
                reason_colour = p.warning

            card["dot"].config(fg=dot_colour)
            card["status"].config(text=status_text, fg=dot_colour)
            card["reason"].config(text=svc_reason, fg=reason_colour)

    @staticmethod
    def _icon_for_service(name: str) -> str:
        """Return an emoji icon for a service name."""
        icons = {
            "ollama": "🦙",
            "ocr": "🔍",
            "tesseract": "🔍",
            "mcp": "🧠",
            "game_window": "🪟",
            "vision": "👁",
        }
        return icons.get(name.lower(), "⚙")

    # ------------------------------------------------------------------
    # Theme support
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        """Re‑apply colours after a theme change."""
        self._tm = ThemeManager()
        p = self._tm.palette
        ui_font = resolve_font_stack(p.ui_font)

        # Update header label
        self._lbl_overall.configure(bg=p.header_bg, font=(ui_font, 8))

        # Update all cards
        for name, card in self._cards.items():
            card["frame"].configure(bg=p.card_bg)
            card["icon"].configure(bg=p.card_bg)
            card["name"].configure(bg=p.card_bg, fg=p.fg)
            card["status"].configure(bg=p.card_bg)
            card["reason"].configure(bg=p.card_bg, fg=p.disabled_fg)
            card["dot"].configure(bg=p.card_bg)
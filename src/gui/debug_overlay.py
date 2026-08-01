"""
Debug Overlay — in‑game semi‑transparent debug menu triggered by the ~ hotkey.

Provides a tabbed ``ttk.Notebook`` interface that floats above the game
window without stealing focus.  The initial "Macros" tab lets the user
select and execute any macro from the active profile for testing.

All styling follows the centralised theme system (``src.gui.theme``).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from src.gui.theme import ThemeManager, resolve_font_stack
from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OVERLAY_WIDTH = 420
_OVERLAY_HEIGHT = 520
_HEADER_HEIGHT = 32
_ALPHA = 0.90


# ---------------------------------------------------------------------------
# DebugOverlay
# ---------------------------------------------------------------------------


class DebugOverlay(tk.Toplevel):
    """A frameless, semi‑transparent debug panel that floats above the
    game and does **not** steal keyboard focus.

    Usage::

        overlay = DebugOverlay(parent=main_window)
        overlay.set_macro_data(macros_list, executor_ref)
        overlay.toggle()   # or press ~ hotkey

    Parameters
    ----------
    parent : tk.Widget
        The owning ``AsyncTk`` main window.
    """

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)

        # -- Theme ------------------------------------------------------------
        self._tm = ThemeManager()
        self._tm.apply_ttk_styles()
        p = self._tm.palette
        self._ui_font = resolve_font_stack(p.ui_font)
        self._mono_font = resolve_font_stack(p.mono_font)

        # -- Window chrome ----------------------------------------------------
        self.overrideredirect(True)                     # no title bar
        self.attributes("-alpha", _ALPHA)
        self.attributes("-topmost", True)
        self.configure(bg=p.bg, highlightthickness=1, highlightbackground=p.separator)

        # On X11, "splash" type prevents the WM from giving us focus
        try:
            self.attributes("-type", "splash")
        except tk.TclError:
            pass  # macOS / Windows don't have -type

        # -- Size & position --------------------------------------------------
        # Default: top‑right corner, ~80 px from top, ~40 px from right
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self._saved_x = sw - _OVERLAY_WIDTH - 40
        self._saved_y = 80
        self.geometry(
            f"{_OVERLAY_WIDTH}x{_OVERLAY_HEIGHT}+{self._saved_x}+{self._saved_y}"
        )
        self.minsize(320, 300)

        # -- State ------------------------------------------------------------
        self._visible: bool = False
        self._parent = parent

        # Drag support
        self._drag_x: int = 0
        self._drag_y: int = 0
        self._dragging: bool = False

        # Macro data
        self._macros: list[dict[str, Any]] = []
        self._macro_executor: Any = None
        self._engine_loop: Any = None  # reference to the asyncio loop

        # -- Build UI ---------------------------------------------------------
        self._build_ui()

        # -- Show / hide ------------------------------------------------------
        self.withdraw()  # hidden by default; toggle() shows it

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct the draggable header, tab bar, and content area."""
        p = self._tm.palette

        # -- Header bar (grip for dragging) -----------------------------------
        self._header = tk.Frame(
            self,
            bg=p.header_bg,
            height=_HEADER_HEIGHT,
            cursor="fleur",
        )
        self._header.pack(fill=tk.X, side=tk.TOP)
        self._header.pack_propagate(False)

        # Drag bindings
        self._header.bind("<ButtonPress-1>", self._on_drag_start)
        self._header.bind("<B1-Motion>", self._on_drag_move)
        self._header.bind("<ButtonRelease-1>", self._on_drag_stop)

        # Title label
        tk.Label(
            self._header,
            text="🔧  Debug Menu",
            bg=p.header_bg,
            fg=p.fg,
            font=(self._ui_font, 10, "bold"),
            padx=10,
            pady=0,
        ).pack(side=tk.LEFT)

        # Close button
        tk.Label(
            self._header,
            text="✕",
            bg=p.header_bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 12),
            padx=12,
            pady=0,
            cursor="hand2",
        ).pack(side=tk.RIGHT)
        # Bind close to the rightmost label
        for child in self._header.winfo_children():
            if isinstance(child, tk.Label) and child.cget("text") == "✕":
                child.bind("<Button-1>", lambda e: self.close())
                child.bind("<Enter>", lambda e: child.configure(fg=p.danger))
                child.bind("<Leave>", lambda e: child.configure(fg=p.disabled_fg))

        # -- Notebook tabs ----------------------------------------------------
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Tab frames
        self._tab_macros = self._build_macros_tab()
        self._tab_toggles = self._build_placeholder_tab("Toggles", "Feature toggles and runtime switches coming soon.")
        self._tab_state = self._build_placeholder_tab("State", "Live game state inspector coming soon.")
        self._tab_perf = self._build_placeholder_tab("Perf", "Performance metrics and frame timing coming soon.")

        self._notebook.add(self._tab_macros, text="  Macros  ")
        self._notebook.add(self._tab_toggles, text="  Toggles  ")
        self._notebook.add(self._tab_state, text="  State  ")
        self._notebook.add(self._tab_perf, text="  Perf  ")

        # Bring Macros tab to front by default
        self._notebook.select(0)

    def _build_macros_tab(self) -> tk.Frame:
        """Build the Macros tab: scrollable list of macro entries."""
        p = self._tm.palette
        frame = tk.Frame(self, bg=p.bg)

        # -- Scrollable container -----------------------------------------------
        canvas = tk.Canvas(frame, bg=p.bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)

        self._macros_list_frame = tk.Frame(canvas, bg=p.bg)
        self._macros_list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=self._macros_list_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse‑wheel scrolling
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>",
                     lambda ev: canvas.yview_scroll(int(-1 * (ev.delta / 120)), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        return frame

    def _build_placeholder_tab(self, title: str, message: str) -> tk.Frame:
        """Build a placeholder tab with a centred informational message."""
        p = self._tm.palette
        frame = tk.Frame(self, bg=p.bg)
        tk.Label(
            frame,
            text=f"🚧  {title}",
            bg=p.bg,
            fg=p.fg,
            font=(self._ui_font, 12, "bold"),
        ).pack(pady=(60, 6))
        tk.Label(
            frame,
            text=message,
            bg=p.bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 9),
            wraplength=_OVERLAY_WIDTH - 60,
        ).pack(pady=(0, 20))
        return frame

    # ------------------------------------------------------------------
    # Dragging
    # ------------------------------------------------------------------

    def _on_drag_start(self, event: tk.Event) -> None:
        """Begin window drag."""
        self._drag_x = event.x_root - self.winfo_x()
        self._drag_y = event.y_root - self.winfo_y()
        self._dragging = True

    def _on_drag_move(self, event: tk.Event) -> None:
        """Move window while dragging."""
        if not self._dragging:
            return
        new_x = event.x_root - self._drag_x
        new_y = event.y_root - self._drag_y
        self.geometry(f"+{new_x}+{new_y}")

    def _on_drag_stop(self, event: tk.Event) -> None:
        """End drag — save position for session memory."""
        self._dragging = False
        self._saved_x = self.winfo_x()
        self._saved_y = self.winfo_y()

    # ------------------------------------------------------------------
    # Show / Hide
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        """Toggle visibility of the debug overlay."""
        if self._visible:
            self.close()
        else:
            self.show()

    def show(self) -> None:
        """Bring the overlay on‑screen (restore to last saved position)."""
        if self._visible:
            return
        self.geometry(f"+{self._saved_x}+{self._saved_y}")
        self.deiconify()
        self.lift()
        self._visible = True
        logger.debug("Debug overlay shown.")

    def close(self) -> None:
        """Hide the overlay, preserving its last position."""
        if not self._visible:
            return
        self._saved_x = self.winfo_x()
        self._saved_y = self.winfo_y()
        self.withdraw()
        self._visible = False
        logger.debug("Debug overlay hidden.")

    @property
    def visible(self) -> bool:
        return self._visible

    # ------------------------------------------------------------------
    # Macro data & execution
    # ------------------------------------------------------------------

    def set_macro_data(
        self,
        macros: list[dict[str, Any]],
        macro_executor: Any,
        engine_loop: Any,
    ) -> None:
        """Inject the macro list, executor, and asyncio event loop.

        Called by ``MainWindow`` after the engine bridge is wired up.
        """
        self._macros = macros
        self._macro_executor = macro_executor
        self._engine_loop = engine_loop
        self._populate_macros_tab()

    def _populate_macros_tab(self) -> None:
        """Rebuild the macro list UI from ``self._macros``."""
        p = self._tm.palette

        # Clear existing rows
        for widget in self._macros_list_frame.winfo_children():
            widget.destroy()

        if not self._macros:
            tk.Label(
                self._macros_list_frame,
                text="No macros found in profile.",
                bg=p.bg,
                fg=p.disabled_fg,
                font=(self._ui_font, 9),
                pady=20,
            ).pack()
            return

        for macro in self._macros:
            name = macro.get("name", "unnamed")
            description = macro.get("description", "")
            actions = macro.get("actions", [])

            self._add_macro_row(name, description, actions)

    def _add_macro_row(
        self, name: str, description: str, actions: list[dict[str, Any]]
    ) -> None:
        """Insert a single macro entry into the list."""
        p = self._tm.palette

        row = tk.Frame(self._macros_list_frame, bg=p.bg, padx=8, pady=4)
        row.pack(fill=tk.X, pady=(0, 1))

        # Left side: name + description
        text_frame = tk.Frame(row, bg=p.bg)
        text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(
            text_frame,
            text=name,
            bg=p.bg,
            fg=p.fg,
            font=(self._mono_font, 9, "bold"),
            anchor=tk.W,
        ).pack(anchor=tk.W)

        if description:
            tk.Label(
                text_frame,
                text=description,
                bg=p.bg,
                fg=p.subtle,
                font=(self._ui_font, 8),
                anchor=tk.W,
                wraplength=260,
            ).pack(anchor=tk.W)

        # Action count badge
        action_count = len(actions)
        tk.Label(
            text_frame,
            text=f"{action_count} action{'s' if action_count != 1 else ''}",
            bg=p.bg,
            fg=p.disabled_fg,
            font=(self._ui_font, 7),
            anchor=tk.W,
        ).pack(anchor=tk.W)

        # Right side: Run button
        run_btn = ttk.Button(
            row,
            text="▶  Run",
            style="Start.TButton",
            command=lambda n=name, a=actions: self._on_run_macro(n, a),
        )
        run_btn.pack(side=tk.RIGHT, padx=(4, 0))

        # Subtle separator
        tk.Frame(
            self._macros_list_frame, bg=p.separator, height=1
        ).pack(fill=tk.X, padx=8)

    def _on_run_macro(self, name: str, actions: list[dict[str, Any]]) -> None:
        """Submit a macro for execution via the engine's MacroExecutor."""
        if self._macro_executor is None or self._engine_loop is None:
            logger.warning(
                "Cannot run macro '%s' — executor or event loop not wired.", name
            )
            return

        import asyncio

        from src.macro_executor import MacroPriority, MacroRequest

        request = MacroRequest(
            name=name,
            actions=list(actions),
            priority=MacroPriority.HIGH,
        )

        logger.info("Debug overlay — triggering macro '%s' (priority=HIGH).", name)

        try:
            asyncio.run_coroutine_threadsafe(
                self._macro_executor.submit_and_preempt(request),
                self._engine_loop,
            )
        except Exception as exc:
            logger.error("Failed to submit macro '%s': %s", name, exc)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Cleanly destroy the overlay window."""
        self._visible = False
        super().destroy()
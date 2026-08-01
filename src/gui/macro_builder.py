"""
Phase 6.8 — Macro Builder GUI.

Provides an interactive visual builder for constructing macro actions:
- Add actions (key_press, mouse_click, wait, etc.)
- Reorder with ▲▼ buttons
- Real-time JSON preview
- Dynamic targeting with spatial references

Uses the centralised theme system from ``src.gui.theme``.
"""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from src.gui.theme import ThemeManager, ThemePalette, resolve_font_stack


# ---------------------------------------------------------------------------
# MacroBuilder
# ---------------------------------------------------------------------------


class MacroBuilder(tk.Frame):
    """Visual macro action builder with card‑based action editing.

    Parameters
    ----------
    parent : tk.Widget
        Parent frame.
    on_change : callable or None
        Called whenever the macro actions change.
    """

    def __init__(self, parent: tk.Widget, on_change: Callable[..., Any] | None = None) -> None:
        self._tm = ThemeManager()
        super().__init__(parent, bg=self._tm.palette.bg)

        self._ui_font = resolve_font_stack(self._tm.palette.ui_font)
        self._mono_font = resolve_font_stack(self._tm.palette.mono_font)
        self._on_change = on_change

        # Current macro data
        self._macro_obj: dict[str, Any] = {"name": "", "description": "", "actions": []}

        # Action cards
        self._action_widgets: list[dict[str, Any]] = []

        # Build UI
        self._build_ui()

    # ------------------------------------------------------------------
    # Palette shortcut
    # ------------------------------------------------------------------

    @property
    def _p(self):
        return self._tm.palette

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the builder layout."""
        p = self._p

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # Meta fields
        meta_frame = tk.Frame(self, bg=p.bg)
        meta_frame.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        meta_frame.columnconfigure(1, weight=1)

        tk.Label(meta_frame, text="Name:", bg=p.bg, fg=p.fg,
                 font=(self._ui_font, 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 4))
        self._entry_name = tk.Entry(
            meta_frame, bg=p.card_bg, fg=p.fg, insertbackground=p.fg,
            font=(self._ui_font, 10), relief="flat",
        )
        self._entry_name.grid(row=0, column=1, sticky="ew", pady=2)
        self._entry_name.insert(0, self._macro_obj.get("name", ""))

        tk.Label(meta_frame, text="Desc:", bg=p.bg, fg=p.fg,
                 font=(self._ui_font, 10, "bold")).grid(row=1, column=0, sticky="w", padx=(0, 4))
        self._entry_desc = tk.Entry(
            meta_frame, bg=p.card_bg, fg=p.fg, insertbackground=p.fg,
            font=(self._ui_font, 10), relief="flat",
        )
        self._entry_desc.grid(row=1, column=1, sticky="ew", pady=2)
        self._entry_desc.insert(0, self._macro_obj.get("description", ""))

        # Toolbar — add action buttons
        toolbar = tk.Frame(self, bg=p.bg)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(4, 2), padx=4)

        tk.Label(toolbar, text="Add Action:", bg=p.bg, fg=p.fg,
                 font=(self._ui_font, 10)).pack(side=tk.LEFT, padx=(0, 4))

        action_types = [
            ("⌨ Key", self._add_key_press),
            ("🖱 Click", self._add_mouse_click),
            ("🎯 DynClick", self._add_dynamic_click),
            ("📍 DynMove", self._add_dynamic_move),
            ("⏳ Wait", self._add_wait),
            ("⌨ Type", self._add_type_text),
        ]
        for label, cmd in action_types:
            ttk.Button(
                toolbar, text=label, command=cmd,
                style="Editor.TButton",
            ).pack(side=tk.LEFT, padx=1)

        # Action list (scrollable)
        list_container = tk.Frame(self, bg=p.bg)
        list_container.grid(row=2, column=0, sticky="nsew")
        list_container.columnconfigure(0, weight=1)
        list_container.rowconfigure(0, weight=1)

        canvas = tk.Canvas(list_container, bg=p.bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=canvas.yview)
        self._action_list_frame = tk.Frame(canvas, bg=p.bg)

        self._action_list_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self._action_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")

        # JSON preview toggle
        toggle_frame = tk.Frame(self, bg=p.bg, padx=4, pady=2)
        toggle_frame.grid(row=3, column=0, sticky="ew")

        self._show_preview = tk.BooleanVar(value=True)
        tk.Checkbutton(
            toggle_frame, text="Show JSON Preview", variable=self._show_preview,
            bg=p.bg, fg=p.disabled_fg,
            selectcolor=p.bg,
            activebackground=p.bg, activeforeground=p.fg,
            command=self._toggle_json_preview,
        ).pack(side=tk.LEFT)

        # JSON preview area
        json_frame = tk.Frame(self, bg=p.bg)
        json_frame.grid(row=4, column=0, sticky="ew", padx=4, pady=(0, 4))
        json_frame.columnconfigure(0, weight=1)

        self._json_preview = tk.Text(
            json_frame,
            bg=p.card_bg,
            fg=p.disabled_fg,
            font=(self._mono_font, 10),
            height=6,
            state=tk.DISABLED,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=6,
            pady=4,
        )
        self._json_preview.grid(row=0, column=0, sticky="ew")

        # Populate initial macro
        self.load_macro(self._macro_obj)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_macro(self, macro: dict[str, Any]) -> None:
        """Load a macro dict into the builder."""
        self._macro_obj = macro
        self._entry_name.delete(0, tk.END)
        self._entry_name.insert(0, macro.get("name", ""))
        self._entry_desc.delete(0, tk.END)
        self._entry_desc.insert(0, macro.get("description", ""))
        self._rebuild_action_cards()
        self._update_preview()

    def clear(self) -> None:
        """Clear all actions and reset the builder."""
        self._macro_obj = {"name": "", "description": "", "actions": []}
        self._entry_name.delete(0, tk.END)
        self._entry_desc.delete(0, tk.END)
        self._rebuild_action_cards()
        self._update_preview()

    # ------------------------------------------------------------------
    # Action cards
    # ------------------------------------------------------------------

    def _rebuild_action_cards(self) -> None:
        """Destroy and recreate all action cards."""
        for child in self._action_list_frame.winfo_children():
            child.destroy()
        self._action_widgets.clear()

        actions = self._macro_obj.get("actions", [])
        if not actions:
            p = self._p
            tk.Label(
                self._action_list_frame,
                text="No actions yet — click a button above to add one.",
                bg=p.card_bg, fg=p.disabled_fg,
                font=(self._ui_font, 11, "italic"),
                padx=16, pady=16,
            ).pack(fill=tk.X, padx=4, pady=4)
            return

        for idx, action in enumerate(actions):
            self._create_action_card(action, idx)

    def _create_action_card(self, action: dict[str, Any], idx: int) -> None:
        """Create a single action card."""
        p = self._p
        info = _ACTION_INFO.get(action.get("type", ""), {})

        card_bg = p.card_bg
        card = tk.Frame(self._action_list_frame, bg=card_bg, relief=tk.FLAT, bd=0)
        card.pack(fill=tk.X, padx=4, pady=2)

        # Left colour accent (dynamic vs static)
        accent_color = p.warning if (info and info.get("is_dynamic")) else p.accent
        accent = tk.Frame(card, bg=accent_color, width=4)
        accent.pack(side=tk.LEFT, fill=tk.Y)

        # Content
        body = tk.Frame(card, bg=card_bg, padx=8, pady=4)
        body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Type label
        tk.Label(
            body, text=action.get("type", "unknown"), bg=card_bg, fg=p.fg,
            font=(self._ui_font, 10, "bold"),
        ).pack(anchor="w")

        # Description
        desc = info.get("desc", "") if isinstance(info, dict) else ""
        if desc:
            tk.Label(
                body, text=desc, bg=card_bg, fg=p.disabled_fg,
                font=(self._ui_font, 9),
            ).pack(anchor="w")

        # Control buttons
        ctrl = tk.Frame(card, bg=card_bg)
        ctrl.pack(side=tk.RIGHT, padx=4, pady=4)

        up_btn = tk.Label(ctrl, text="▲", bg=card_bg, fg=p.fg, cursor="hand2",
                          font=(self._ui_font, 10))
        up_btn.pack(side=tk.LEFT, padx=1)
        up_btn.bind("<Button-1>", lambda e, i=idx: self._move_up(i))

        down_btn = tk.Label(ctrl, text="▼", bg=card_bg, fg=p.fg, cursor="hand2",
                            font=(self._ui_font, 10))
        down_btn.bind("<Button-1>", lambda e, i=idx: self._move_down(i))
        down_btn.pack(side=tk.LEFT, padx=1)

        del_btn = tk.Label(ctrl, text="✕", bg=card_bg, fg=p.danger, cursor="hand2",
                           font=(self._ui_font, 10, "bold"))
        del_btn.bind("<Button-1>", lambda e, i=idx: self._delete_action(i))
        del_btn.pack(side=tk.LEFT, padx=2)

        # Store widget refs
        self._action_widgets.append({"card": card, "action": action, "index": idx})

    # ------------------------------------------------------------------
    # Action adders
    # ------------------------------------------------------------------

    def _add_key_press(self) -> None:
        self._macro_obj.setdefault("actions", []).append({
            "type": "key_press", "key": "space", "hold_ms": 50,
        })
        self._rebuild_action_cards()
        self._update_preview()

    def _add_mouse_click(self) -> None:
        self._macro_obj.setdefault("actions", []).append({
            "type": "mouse_click", "button": "left",
            "x": 960, "y": 540, "relative": False,
        })
        self._rebuild_action_cards()
        self._update_preview()

    def _add_dynamic_click(self) -> None:
        self._macro_obj.setdefault("actions", []).append({
            "type": "dynamic_click", "target": "player",
            "button": "left", "offset_x": 0, "offset_y": 0,
        })
        self._rebuild_action_cards()
        self._update_preview()

    def _add_dynamic_move(self) -> None:
        self._macro_obj.setdefault("actions", []).append({
            "type": "dynamic_move", "target": "player",
            "offset_x": 0, "offset_y": 0,
        })
        self._rebuild_action_cards()
        self._update_preview()

    def _add_wait(self) -> None:
        self._macro_obj.setdefault("actions", []).append({
            "type": "wait", "duration_ms": 1000,
        })
        self._rebuild_action_cards()
        self._update_preview()

    def _add_type_text(self) -> None:
        self._macro_obj.setdefault("actions", []).append({
            "type": "type_text", "text": "",
        })
        self._rebuild_action_cards()
        self._update_preview()

    # ------------------------------------------------------------------
    # Action manipulation
    # ------------------------------------------------------------------

    def _move_up(self, idx: int) -> None:
        actions = self._macro_obj.get("actions", [])
        if idx <= 0 or idx >= len(actions):
            return
        actions[idx], actions[idx - 1] = actions[idx - 1], actions[idx]
        self._rebuild_action_cards()
        self._update_preview()

    def _move_down(self, idx: int) -> None:
        actions = self._macro_obj.get("actions", [])
        if idx < 0 or idx >= len(actions) - 1:
            return
        actions[idx], actions[idx + 1] = actions[idx + 1], actions[idx]
        self._rebuild_action_cards()
        self._update_preview()

    def _delete_action(self, idx: int) -> None:
        actions = self._macro_obj.get("actions", [])
        if 0 <= idx < len(actions):
            del actions[idx]
        self._rebuild_action_cards()
        self._update_preview()

    # ------------------------------------------------------------------
    # JSON preview
    # ------------------------------------------------------------------

    def _toggle_json_preview(self) -> None:
        if self._show_preview.get():
            self._json_preview.grid()
        else:
            self._json_preview.grid_remove()

    def _update_preview(self) -> None:
        """Update the JSON preview text."""
        macro_obj = {
            "name": self._entry_name.get(),
            "description": self._entry_desc.get(),
            "actions": self._macro_obj.get("actions", []),
        }
        text = json.dumps(macro_obj, indent=2, ensure_ascii=False)
        self._json_preview.configure(state=tk.NORMAL)
        self._json_preview.delete("1.0", "end")
        self._json_preview.insert("1.0", text)
        self._json_preview.configure(state=tk.DISABLED)
        if self._on_change:
            self._on_change()


# ---------------------------------------------------------------------------
# Action type metadata
# ---------------------------------------------------------------------------

_ACTION_INFO: dict[str, dict[str, Any]] = {
    "key_press": {"desc": "Press a key", "is_dynamic": False},
    "key_hold": {"desc": "Hold a key for duration", "is_dynamic": False},
    "mouse_click": {"desc": "Click at absolute coordinates", "is_dynamic": False},
    "mouse_move": {"desc": "Move to absolute coordinates", "is_dynamic": False},
    "dynamic_click": {"desc": "Click on a detected object", "is_dynamic": True},
    "dynamic_move": {"desc": "Move to a detected object", "is_dynamic": True},
    "wait": {"desc": "Wait for duration", "is_dynamic": False},
    "type_text": {"desc": "Type a text string", "is_dynamic": False},
}
"""
Phase 5.4 — Macro JSON Editor.

Provides a modal Toplevel with:
- Sidebar list of macros
- Visual builder (via MacroBuilder) OR raw JSON editor
- Add, Delete, Save, Refresh toolbar
- Syntax‑highlighted JSON view

Uses the centralised theme system from ``src.gui.theme``.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

from src.gui.theme import ThemeManager, resolve_font_stack
from src.logging_config import get_logger

logger = get_logger(__name__)


class MacroEditor(tk.Toplevel):
    """Modal macro editor with visual builder + raw JSON toggle."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)

        self._tm = ThemeManager()
        self._tm.apply_ttk_styles()
        p = self._tm.palette
        self._ui_font = resolve_font_stack(p.ui_font)
        self._mono_font = resolve_font_stack(p.mono_font)

        self._macros_path = self._resolve_macros_path()
        self._macros: list[dict[str, Any]] = self._load_macros()

        self.title("Macro Editor")
        self.configure(bg=p.bg)
        self.geometry("1050x680")
        self.minsize(800, 500)

        self.transient(parent)

        # Current macro index
        self._current_idx: int | None = None

        # View mode: "visual" or "json"
        self._view_mode: str = "visual"

        # Build UI
        self._build_toolbar()
        self._build_layout()

        # Logger
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Palette shortcut
    # ------------------------------------------------------------------

    @property
    def _p(self):
        return self._tm.palette

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_macros_path() -> Path:
        path = Path(__file__).resolve().parent.parent.parent / "config" / "macros.json"
        if not path.exists():
            path = Path("config") / "macros.json"
        return path

    def _load_macros(self) -> list[dict[str, Any]]:
        try:
            if self._macros_path.exists():
                data = json.loads(self._macros_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "macros" in data:
                    return data["macros"]
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        return []

    def _save_macros(self) -> None:
        self._macros_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._macros_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._macros, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._macros_path)

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        """Top toolbar with action buttons."""
        p = self._p
        toolbar = tk.Frame(self, bg=p.bg, padx=8, pady=4)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        ttk.Button(toolbar, text="＋ Add", style="Editor.TButton",
                   command=self._on_add).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="− Delete", style="Danger.TButton",
                   command=self._on_delete).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Button(toolbar, text="💾 Save", style="Editor.TButton",
                   command=self._on_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="↻ Refresh", style="Editor.TButton",
                   command=self._on_refresh).pack(side=tk.LEFT, padx=2)

        self._toggle_btn = ttk.Button(
            toolbar, text="🔧 JSON Mode",
            style="Mode.TButton",
            command=self._toggle_view_mode,
        )
        self._toggle_btn.pack(side=tk.RIGHT, padx=4)

        tk.Label(toolbar, text=str(self._macros_path.name),
                 bg=p.bg, fg=p.disabled_fg,
                 font=(self._ui_font, 8)).pack(side=tk.RIGHT, padx=8)

    def _build_layout(self) -> None:
        """Main layout: sidebar + content area."""
        p = self._p
        body = tk.Frame(self, bg=p.bg)
        body.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        body.columnconfigure(0, minsize=180)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Sidebar
        sidebar = tk.Frame(body, bg=p.sidebar_bg)
        sidebar.grid(row=0, column=0, sticky="ns")
        tk.Label(sidebar, text="Macros", bg=p.sidebar_bg, fg=p.fg,
                 font=(self._ui_font, 10, "bold"), anchor=tk.W, padx=8, pady=4).pack(fill=tk.X)

        list_frame = tk.Frame(sidebar, bg=p.sidebar_bg)
        list_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._macro_listbox = tk.Listbox(
            list_frame, bg=p.entry_bg, fg=p.fg, selectbackground=p.accent,
            selectforeground="#FFFFFF", font=(self._mono_font, 10),
            yscrollcommand=scrollbar.set, activestyle="none", borderwidth=0, highlightthickness=0,
        )
        self._macro_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar.configure(command=self._macro_listbox.yview)
        self._macro_listbox.bind("<<ListboxSelect>>", self._on_list_select)
        self._refresh_list()

        # Content area
        self._content_frame = tk.Frame(body, bg=p.bg)
        self._content_frame.grid(row=0, column=1, sticky="nsew")

        # Visual builder
        from src.gui.macro_builder import MacroBuilder
        self._builder_frame = tk.Frame(self._content_frame, bg=p.bg)
        self._builder = MacroBuilder(self._builder_frame, on_change=self._on_builder_change)
        self._builder._tm = self._tm  # inject theme
        self._builder.pack(fill=tk.BOTH, expand=True)
        self._builder_frame.pack(fill=tk.BOTH, expand=True)

        # JSON editor (hidden by default)
        self._json_editor_frame = tk.Frame(self._content_frame, bg=p.bg)
        self._json_editor = tk.Text(
            self._json_editor_frame, bg=p.entry_bg, fg=p.fg,
            insertbackground=p.fg, font=(self._mono_font, 11), wrap=tk.NONE,
            undo=True, padx=8, pady=8, borderwidth=0, highlightthickness=0,
        )
        self._json_editor.pack(fill=tk.BOTH, expand=True)

        # Syntax highlight tags
        self._json_editor.tag_configure("key", foreground=p.syntax_key)
        self._json_editor.tag_configure("string", foreground=p.syntax_string)
        self._json_editor.tag_configure("number", foreground=p.syntax_number)
        self._json_editor.tag_configure("bool_null", foreground=p.syntax_bool_null)
        self._json_editor.tag_configure("bracket", foreground=p.syntax_bracket)
        self._json_editor.bind("<KeyRelease>", self._schedule_highlight)

        # Status bar
        status_frame = tk.Frame(self, bg=p.status_bar_bg, height=28)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        self._lbl_status = tk.Label(status_frame, text="Ready", bg=p.status_bar_bg,
                                    fg=p.disabled_fg, font=(self._ui_font, 9), padx=10)
        self._lbl_status.pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # View toggle
    # ------------------------------------------------------------------

    def _toggle_view_mode(self) -> None:
        if self._view_mode == "visual":
            self._view_mode = "json"
            self._toggle_btn.configure(text="🖼 Visual Mode")
            self._builder_frame.pack_forget()
            self._json_editor_frame.pack(fill=tk.BOTH, expand=True)
            self._populate_json_editor()
        else:
            self._view_mode = "visual"
            self._toggle_btn.configure(text="🔧 JSON Mode")
            self._json_editor_frame.pack_forget()
            self._builder_frame.pack(fill=tk.BOTH, expand=True)
            # Reload builder from current macro
            if self._current_idx is not None and self._current_idx < len(self._macros):
                self._builder.load_macro(self._macros[self._current_idx])

    def _populate_json_editor(self) -> None:
        self._json_editor.configure(state=tk.NORMAL)
        self._json_editor.delete("1.0", "end")
        if self._current_idx is not None and self._current_idx < len(self._macros):
            text = json.dumps(self._macros[self._current_idx], indent=2, ensure_ascii=False)
            self._json_editor.insert("1.0", text)
        self._json_editor.configure(state=tk.DISABLED)
        self._schedule_highlight()

    # ------------------------------------------------------------------
    # List refresh
    # ------------------------------------------------------------------

    def _refresh_list(self) -> None:
        self._macro_listbox.delete(0, tk.END)
        for macro in self._macros:
            name = macro.get("name", "Unnamed")
            self._macro_listbox.insert(tk.END, name)

    def _on_list_select(self, event: tk.Event | None = None) -> None:
        sel = self._macro_listbox.curselection()
        if not sel:
            return
        self._current_idx = sel[0]
        if self._view_mode == "visual":
            self._builder.load_macro(self._macros[self._current_idx])
        else:
            self._populate_json_editor()

    def _on_builder_change(self) -> None:
        """Called when the visual builder changes the macro."""
        pass

    # ------------------------------------------------------------------
    # Add / Delete
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        new_macro: dict[str, Any] = {
            "name": "new_macro",
            "description": "",
            "actions": [],
        }
        self._macros.append(new_macro)
        self._current_idx = len(self._macros) - 1
        self._refresh_list()
        self._macro_listbox.selection_clear(0, tk.END)
        self._macro_listbox.selection_set(self._current_idx)
        self._on_list_select()
        self._set_status("New macro added.", self._p.success)

    def _on_delete(self) -> None:
        if self._current_idx is None or self._current_idx >= len(self._macros):
            return
        name = self._macros[self._current_idx].get("name", "Unnamed")
        del self._macros[self._current_idx]
        self._current_idx = None
        self._refresh_list()
        self._builder.clear()
        self._set_status(f"Deleted '{name}'.", self._p.warning)

    # ------------------------------------------------------------------
    # Save / Refresh
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        # If in JSON mode, read back from editor
        if self._view_mode == "json" and self._current_idx is not None:
            try:
                raw = self._json_editor.get("1.0", "end-1c")
                data = json.loads(raw)
                self._macros[self._current_idx] = data
            except json.JSONDecodeError as exc:
                self._set_status(f"Invalid JSON: {exc}", self._p.danger)
                return

        self._save_macros()
        self._refresh_list()
        self._set_status("Macros saved.", self._p.success)

    def _on_refresh(self) -> None:
        self._macros = self._load_macros()
        self._current_idx = None
        self._refresh_list()
        self._builder.clear()
        self._set_status("Macros reloaded from disk.", self._p.success)

    # ------------------------------------------------------------------
    # Syntax highlight (simple regex‑based)
    # ------------------------------------------------------------------

    def _schedule_highlight(self, event: tk.Event | None = None) -> None:
        """Re-apply syntax highlighting after each key release."""
        self._json_editor.after(100, self._do_highlight)

    def _do_highlight(self) -> None:
        """Apply regex‑based syntax highlighting to the JSON editor."""
        import re
        text = self._json_editor.get("1.0", "end-1c")
        self._json_editor.configure(state=tk.NORMAL)
        # Clear all tags
        for tag in ("key", "string", "number", "bool_null", "bracket"):
            self._json_editor.tag_remove(tag, "1.0", "end")

        # Bracket matching
        for match in re.finditer(r'[{}\[\]]', text):
            start = f"1.0+{match.start()}c"
            end = f"1.0+{match.end()}c"
            self._json_editor.tag_add("bracket", start, end)

        # Keys
        for match in re.finditer(r'"(?P<key>[^"]+)"\s*:', text):
            start = f"1.0+{match.start()}c"
            end = f"1.0+{match.end()}c"
            self._json_editor.tag_add("key", start, end)

        # Strings
        for match in re.finditer(r'"[^"]*"', text):
            start = f"1.0+{match.start()}c"
            end = f"1.0+{match.end()}c"
            self._json_editor.tag_add("string", start, end)

        # Numbers
        for match in re.finditer(r'\b-?\d+\.?\d*\b', text):
            start = f"1.0+{match.start()}c"
            end = f"1.0+{match.end()}c"
            self._json_editor.tag_add("number", start, end)

        # Booleans and null
        for match in re.finditer(r'\b(true|false|null)\b', text):
            start = f"1.0+{match.start()}c"
            end = f"1.0+{match.end()}c"
            self._json_editor.tag_add("bool_null", start, end)

        self._json_editor.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _set_status(self, text: str, colour: str | None = None) -> None:
        if colour is None:
            colour = self._p.disabled_fg
        self._lbl_status.config(text=text, fg=colour)
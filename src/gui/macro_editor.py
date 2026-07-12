"""
Phase 5.4 — Macro JSON Editor.

Provides a Toplevel window with:
- Left sidebar listing all macro names
- Right panel showing the selected macro's JSON with syntax highlighting
- Toolbar buttons: Add, Delete, Save, Refresh
- JSON validation on save
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from tkinter import messagebox
from typing import Any

import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------------------
# Colour palette (matches main_window.py dark theme)
# ---------------------------------------------------------------------------

_BG = "#1E1E1E"
_FG = "#D4D4D4"
_ACCENT = "#0078D4"
_SUCCESS = "#50C878"
_DANGER = "#E04040"
_WARNING = "#E8A317"
_DISABLED_BG = "#3C3C3C"
_DISABLED_FG = "#808080"
_EDITOR_BG = "#252526"
_SIDEBAR_BG = "#1A1A1A"

# Syntax highlight tag colours
_TAG_KEY = "#9CDCFE"          # light blue — JSON keys
_TAG_STRING = "#CE9178"       # orange — string values
_TAG_NUMBER = "#B5CEA8"       # green — numbers
_TAG_BOOL_NULL = "#569CD6"    # blue — booleans & null
_TAG_BRACKET = "#FFD700"      # gold — brackets / braces
_TAG_COMMENT = "#6A9955"      # not used for JSON, kept for symmetry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MACROS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "macros.json"

# Regex for syntax highlighting tokens
_RE_NUMBER = re.compile(r"\b-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")
_RE_BOOL_NULL = re.compile(r"\b(?:true|false|null)\b")
_RE_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
# Match a JSON key: start of line, optional whitespace, capture the quoted key, then colon.
# Group 1 captures just the quoted key string (without the surrounding whitespace/colon).
_RE_KEY = re.compile(r'^\s*("(?:[^"\\]|\\.)*")\s*:', re.MULTILINE)
_RE_BRACKET = re.compile(r'[\[\]{}]')


# ---------------------------------------------------------------------------
# Macro JSON validation helpers
# ---------------------------------------------------------------------------

def _validate_macro_json(text: str) -> tuple[bool, str]:
    """Validate that *text* is a well‑formed macro object (name, description, actions).

    Returns ``(True, "")`` on success or ``(False, error_message)`` on failure.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON: {exc}"

    if not isinstance(obj, dict):
        return False, "Top-level value must be a JSON object (dictionary)."

    # Required keys
    if "name" not in obj:
        return False, 'Missing required field: "name"'
    if "actions" not in obj:
        return False, 'Missing required field: "actions"'

    if not isinstance(obj.get("actions"), list):
        return False, '"actions" must be an array.'

    # Validate each action
    valid_types = {"key", "delay", "mouse_move", "click", "type_string",
                   "dynamic_click", "dynamic_move", "dynamic_attack"}
    for i, action in enumerate(obj["actions"]):
        if not isinstance(action, dict):
            return False, f"Action[{i}] is not an object."
        atype = action.get("type")
        if atype is None:
            return False, f"Action[{i}] missing 'type'."
        if atype not in valid_types:
            return False, f"Action[{i}] has unknown type '{atype}'. Valid types: {', '.join(sorted(valid_types))}"

    return True, ""


# ---------------------------------------------------------------------------
# MacroEditor
# ---------------------------------------------------------------------------


class MacroEditor(tk.Toplevel):
    """Modal window for editing macros.json with syntax highlighting.

    Layout::

        ┌────────────────────────────────────────────────────┐
        │  Toolbar  [+ Add] [− Delete] [💾 Save] [↻ Refresh] │
        ├───────────┬────────────────────────────────────────┤
        │           │                                         │
        │  Macro    │  JSON Editor (syntax‑highlighted Text)  │
        │  list    │                                         │
        │  (Listbox) │                                         │
        │           │                                         │
        ├───────────┴────────────────────────────────────────┤
        │  Status bar                                          │
        └────────────────────────────────────────────────────┘
    """

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)

        self.title("Macro Editor")
        self.configure(bg=_BG)
        self.geometry("900x600")
        self.minsize(700, 400)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Data
        self._macros: list[dict[str, Any]] = []
        self._selected_index: int = -1
        self._macros_path: Path = _MACROS_PATH
        self._dirty: bool = False

        # Logger
        from src.logging_config import get_logger
        self._logger = get_logger(__name__)

        # Build UI
        self._build_toolbar()
        self._build_body()
        self._build_status_bar()

        # Load data
        self._refresh_from_disk()

        # Handle close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._logger.info("Macro Editor opened.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        """Create the top toolbar."""
        toolbar = tk.Frame(self, bg=_BG, padx=6, pady=4)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Editor.TButton",
            background=_ACCENT,
            foreground="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            borderwidth=1,
            padding=(10, 2),
        )
        style.map("Editor.TButton", background=[("active", "#005A9E")])
        style.configure(
            "Danger.TButton",
            background=_DANGER,
            foreground="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            borderwidth=1,
            padding=(10, 2),
        )
        style.map("Danger.TButton", background=[("active", "#C03030")])

        ttk.Button(toolbar, text="＋ Add", style="Editor.TButton", command=self._on_add).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="− Delete", style="Danger.TButton", command=self._on_delete).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Button(toolbar, text="💾 Save", style="Editor.TButton", command=self._on_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="↻ Refresh", style="Editor.TButton", command=self._on_refresh).pack(side=tk.LEFT, padx=2)

        # Spacer
        tk.Frame(toolbar, bg=_BG).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # File path label
        tk.Label(
            toolbar,
            text=str(self._macros_path.name),
            bg=_BG,
            fg=_DISABLED_FG,
            font=("Segoe UI", 8),
        ).pack(side=tk.RIGHT, padx=8)

    def _build_body(self) -> None:
        """Build the left sidebar + right editor panel."""
        container = tk.Frame(self, bg=_BG)
        container.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        # ----- Left sidebar (macro list) -------------------------------------
        sidebar = tk.Frame(container, bg=_SIDEBAR_BG, width=200)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="Macros", bg=_SIDEBAR_BG, fg=_FG,
            font=("Segoe UI", 10, "bold"), anchor=tk.W, padx=8, pady=4,
        ).pack(fill=tk.X)

        list_frame = tk.Frame(sidebar, bg=_SIDEBAR_BG)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._macro_listbox = tk.Listbox(
            list_frame,
            bg=_EDITOR_BG,
            fg=_FG,
            selectbackground=_ACCENT,
            selectforeground="#FFFFFF",
            font=("Consolas", 10),
            yscrollcommand=scrollbar.set,
            activestyle="none",
            borderwidth=0,
            highlightthickness=0,
        )
        scrollbar.config(command=self._macro_listbox.yview)
        self._macro_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._macro_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # ----- Right panel (JSON editor) -------------------------------------
        editor_frame = tk.Frame(container, bg=_BG)
        editor_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._editor = tk.Text(
            editor_frame,
            bg=_EDITOR_BG,
            fg=_FG,
            insertbackground=_FG,  # cursor colour
            font=("Consolas", 11),
            wrap=tk.NONE,
            undo=True,
            padx=8,
            pady=8,
            borderwidth=0,
            highlightthickness=0,
        )
        self._editor.pack(fill=tk.BOTH, expand=True)

        # Configure syntax highlight tags
        self._editor.tag_configure("key", foreground=_TAG_KEY)
        self._editor.tag_configure("string", foreground=_TAG_STRING)
        self._editor.tag_configure("number", foreground=_TAG_NUMBER)
        self._editor.tag_configure("bool_null", foreground=_TAG_BOOL_NULL)
        self._editor.tag_configure("bracket", foreground=_TAG_BRACKET)

        # Bind text change event for live highlighting
        self._editor.bind("<KeyRelease>", self._schedule_highlight)
        self._editor.bind("<ButtonRelease-1>", self._schedule_highlight)
        # Also handle paste and cut
        self._editor.bind("<<Paste>>", self._schedule_highlight)
        self._editor.bind("<<Cut>>", self._schedule_highlight)

        self._highlight_after_id: str | None = None

    def _build_status_bar(self) -> None:
        """Create the bottom status bar."""
        status_frame = tk.Frame(self, bg="#2D2D2D", height=26)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self._lbl_status = tk.Label(
            status_frame,
            text="Ready",
            bg="#2D2D2D",
            fg=_DISABLED_FG,
            font=("Segoe UI", 9),
            padx=10,
        )
        self._lbl_status.pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Highlighting
    # ------------------------------------------------------------------

    def _schedule_highlight(self, event: tk.Event | None = None) -> None:  # noqa: ARG002
        """Schedule a syntax highlight pass after a short debounce."""
        if self._highlight_after_id is not None:
            self.after_cancel(str(self._highlight_after_id))
        self._highlight_after_id = self.after(200, self._apply_highlighting)

    def _apply_highlighting(self) -> None:
        """Apply syntax‑highlighting tags to the editor text."""
        self._highlight_after_id = None

        text = self._editor.get("1.0", "end-1c")
        if not text.strip():
            return

        # Remove all existing tags
        for tag in ("key", "string", "number", "bool_null", "bracket"):
            self._editor.tag_remove(tag, "1.0", "end")

        # Apply tags in priority order (keys first so they aren't overpainted by strings)
        # Keys use capture group 1 so we only highlight the quoted key, not the surrounding whitespace/colon
        self._highlight_regex(_RE_KEY, "key", text, group=1)
        self._highlight_regex(_RE_STRING, "string", text)
        self._highlight_regex(_RE_NUMBER, "number", text)
        self._highlight_regex(_RE_BOOL_NULL, "bool_null", text)
        self._highlight_regex(_RE_BRACKET, "bracket", text)

    @staticmethod
    def _pos_to_tkindex(pos: int) -> str:
        """Convert a character offset to a tkinter Text index string."""
        return f"1.0+{pos}c"

    def _highlight_regex(self, pattern: re.Pattern[str], tag: str, text: str, group: int = 0) -> None:
        """Find all matches of *pattern* in *text* and apply *tag*.

        Uses *group* to select which capture group span to highlight (default 0 = full match).
        """
        for m in pattern.finditer(text):
            start = self._pos_to_tkindex(m.start(group))
            end = self._pos_to_tkindex(m.end(group))
            self._editor.tag_add(tag, start, end)

    # ------------------------------------------------------------------
    # Macro list operations
    # ------------------------------------------------------------------

    def _refresh_from_disk(self) -> None:
        """Load macros from disk and repopulate the listbox."""
        try:
            if self._macros_path.exists():
                data = json.loads(self._macros_path.read_text(encoding="utf-8"))
                self._macros = data.get("macros", [])
            else:
                self._macros = []
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            self._logger.warning(f"Failed to load macros: {exc}")
            self._macros = []
            self._set_status(f"Error loading macros: {exc}", _DANGER)
            return

        self._macro_listbox.delete(0, tk.END)
        for m in self._macros:
            name = m.get("name", "(unnamed)")
            self._macro_listbox.insert(tk.END, name)

        self._dirty = False
        self._set_status(f"Loaded {len(self._macros)} macro(s).", _SUCCESS)

    def _on_listbox_select(self, event: tk.Event | None = None) -> None:  # noqa: ARG002
        """Handle macro selection in the listbox."""
        selection = self._macro_listbox.curselection()
        if not selection:
            return

        # Save current selection if dirty
        self._commit_current_selection()

        idx = selection[0]
        self._selected_index = idx
        self._display_macro(idx)

    def _display_macro(self, index: int) -> None:
        """Display the macro at *index* in the editor."""
        if index < 0 or index >= len(self._macros):
            self._editor.delete("1.0", "end")
            return

        macro = self._macros[index]
        text = json.dumps(macro, indent=2, ensure_ascii=False)
        self._editor.delete("1.0", "end")
        self._editor.insert("1.0", text)
        self._apply_highlighting()
        self._dirty = False

    def _commit_current_selection(self) -> None:
        """If there's a current selection with unsaved edits, update self._macros."""
        if self._selected_index < 0 or self._selected_index >= len(self._macros):
            return

        text = self._editor.get("1.0", "end-1c").strip()
        if not text:
            return

        try:
            obj = json.loads(text)
            self._macros[self._selected_index] = obj
            self._dirty = False
            # Update listbox entry name
            name = obj.get("name", "(unnamed)")
            self._macro_listbox.delete(self._selected_index)
            self._macro_listbox.insert(self._selected_index, name)
        except json.JSONDecodeError:
            # Don't commit invalid JSON; user will see validation error on save
            pass

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        """Add a new macro."""
        self._commit_current_selection()

        new_macro: dict[str, Any] = {
            "name": "new_macro",
            "description": "Describe this macro",
            "actions": [
                {"type": "key", "key": "a", "hold_ms": 100},
            ],
        }

        self._macros.append(new_macro)
        idx = len(self._macros) - 1
        self._macro_listbox.insert(tk.END, new_macro["name"])
        self._macro_listbox.selection_clear(0, tk.END)
        self._macro_listbox.selection_set(idx)
        self._macro_listbox.see(idx)
        self._selected_index = idx
        self._display_macro(idx)
        self._set_status("New macro added. Remember to Save.", _WARNING)

    def _on_delete(self) -> None:
        """Delete the currently selected macro."""
        selection = self._macro_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        name = self._macros[idx].get("name", "this macro")

        if not messagebox.askyesno(
            "Delete Macro",
            f'Delete "{name}"? This cannot be undone until you Refresh.',
            parent=self,
        ):
            return

        del self._macros[idx]
        self._macro_listbox.delete(idx)

        self._editor.delete("1.0", "end")
        self._selected_index = -1

        # Reselect the previous item if available
        if idx > 0:
            new_idx = idx - 1
        elif len(self._macros) > 0:
            new_idx = 0
        else:
            new_idx = -1

        if new_idx >= 0:
            self._macro_listbox.selection_set(new_idx)
            self._selected_index = new_idx
            self._display_macro(new_idx)

        self._dirty = True
        self._set_status(f'Deleted "{name}". Save to persist.', _WARNING)

    def _on_save(self) -> None:
        """Validate and save macros to disk."""
        self._commit_current_selection()

        # Validate all macros
        for i, macro in enumerate(self._macros):
            name = macro.get("name", f"Macro #{i}")
            text = json.dumps(macro, indent=2, ensure_ascii=False)
            ok, err = _validate_macro_json(text)
            if not ok:
                self._set_status(f"Validation error in '{name}': {err}", _DANGER)
                self._macro_listbox.selection_clear(0, tk.END)
                self._macro_listbox.selection_set(i)
                self._macro_listbox.see(i)
                self._selected_index = i
                self._display_macro(i)
                return

        # Write to disk
        data = {
            "version": "1.0.0",
            "macros": self._macros,
        }

        try:
            self._macros_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write
            tmp_path = self._macros_path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(self._macros_path)
            self._dirty = False
            self._set_status(f"Saved {len(self._macros)} macro(s) to {self._macros_path.name}.", _SUCCESS)
            self._logger.info(f"Saved {len(self._macros)} macros to {self._macros_path}")
        except OSError as exc:
            self._set_status(f"Failed to save: {exc}", _DANGER)
            self._logger.error(f"Failed to save macros: {exc}")

    def _on_refresh(self) -> None:
        """Reload macros from disk, discarding unsaved changes."""
        if self._dirty:
            if not messagebox.askyesno(
                "Discard Changes?",
                "You have unsaved changes. Refresh will discard them. Continue?",
                parent=self,
            ):
                return

        self._editor.delete("1.0", "end")
        self._selected_index = -1
        self._refresh_from_disk()

    def _on_close(self) -> None:
        """Handle window close — warn about unsaved changes."""
        self._commit_current_selection()
        if self._dirty:
            if messagebox.askyesno(
                "Unsaved Changes",
                "You have unsaved changes. Close without saving?",
                parent=self,
            ):
                self._logger.info("Macro Editor closed without saving.")
            else:
                return
        else:
            self._logger.info("Macro Editor closed.")
        self.destroy()

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str, colour: str = _DISABLED_FG) -> None:
        """Update the status bar text and colour."""
        self._lbl_status.config(text=text, fg=colour)
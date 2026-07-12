"""
Phase 5.7 — Profile Manager Dialog.

Provides a modal Toplevel dialog with two modes:
- **Import**: Select a .gameai_profile file, view validation results, and import.
- **Export**: Fill in profile metadata, choose optional inclusions, and export.

Matches the dark theme palette from main_window.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from tkinter import filedialog, messagebox
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
_FRAME_BG = "#252526"
_ENTRY_BG = "#2D2D2D"
_ENTRY_FG = "#D4D4D4"


# ---------------------------------------------------------------------------
# ProfileManagerDialog
# ---------------------------------------------------------------------------


class ProfileManagerDialog(tk.Toplevel):
    """Modal dialog for importing or exporting .gameai_profile archives.

    Parameters
    ----------
    parent : tk.Widget
        Parent window.
    mode : str
        Either ``"import"`` or ``"export"``.
    profile_name : str or None
        Pre‑filled profile name for export mode.
    on_imported : callable or None
        Called after a successful import with ``(profile_name, profile_path)``.
    """

    def __init__(
        self,
        parent: tk.Widget,
        mode: str = "import",
        *,
        profile_name: str | None = None,
        on_imported: Any = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._profile_name = profile_name
        self._on_imported = on_imported

        self.configure(bg=_BG)
        if mode == "import":
            self.title("Import Profile")
        else:
            self.title("Export Profile")

        self.resizable(False, False)
        self.geometry("")

        # Build UI
        self._build_ui()

        # Centre on parent
        self.transient(parent)
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_reqwidth()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_reqheight()) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        if self._mode == "import":
            self._build_import_ui()
        else:
            self._build_export_ui()

    # ------------------------------------------------------------------
    # Import UI
    # ------------------------------------------------------------------

    def _build_import_ui(self) -> None:
        """Build the import‑mode dialog."""
        main = tk.Frame(self, bg=_BG, padx=16, pady=12)
        main.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(
            main,
            text="Import a .gameai_profile Archive",
            bg=_BG,
            fg=_FG,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor=tk.W, pady=(0, 12))

        # File picker row
        file_row = tk.Frame(main, bg=_BG)
        file_row.pack(fill=tk.X, pady=(0, 8))

        self._zip_path_var = tk.StringVar()
        tk.Entry(
            file_row,
            textvariable=self._zip_path_var,
            bg=_ENTRY_BG,
            fg=_ENTRY_FG,
            insertbackground=_ENTRY_FG,
            relief=tk.FLAT,
            font=("Segoe UI", 10),
            width=42,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        ttk.Button(
            file_row,
            text="Browse …",
            command=self._on_browse_import,
        ).pack(side=tk.LEFT)

        # Validate button
        btn_row = tk.Frame(main, bg=_BG)
        btn_row.pack(fill=tk.X, pady=(0, 8))

        self._btn_validate = ttk.Button(
            btn_row, text="🔍  Validate", command=self._on_validate
        )
        self._btn_validate.pack(side=tk.LEFT, padx=(0, 6))

        self._btn_import = ttk.Button(
            btn_row, text="📥  Import", command=self._on_import, state=tk.DISABLED
        )
        self._btn_import.pack(side=tk.LEFT)

        # Validation results area
        results_frame = tk.LabelFrame(
            main,
            text="Validation Results",
            bg=_BG,
            fg=_FG,
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=6,
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self._results_text = tk.Text(
            results_frame,
            bg=_ENTRY_BG,
            fg=_FG,
            insertbackground=_FG,
            font=("Consolas", 9),
            height=12,
            width=60,
            state=tk.DISABLED,
            relief=tk.FLAT,
            wrap=tk.WORD,
        )
        self._results_text.pack(fill=tk.BOTH, expand=True)

        # Scrollbar for results
        scroll = ttk.Scrollbar(
            results_frame, orient=tk.VERTICAL, command=self._results_text.yview
        )
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._results_text.configure(yscrollcommand=scroll.set)

        # Close button
        ttk.Button(main, text="Close", command=self.destroy).pack(
            anchor=tk.E, pady=(10, 0)
        )

    def _on_browse_import(self) -> None:
        """Open file dialog to select a .gameai_profile zip."""
        path = filedialog.askopenfilename(
            parent=self,
            title="Select .gameai_profile archive",
            filetypes=[
                ("Game AI Profile", "*.gameai_profile"),
                ("Zip files", "*.zip"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._zip_path_var.set(path)

    def _on_validate(self) -> None:
        """Run validation and display results."""
        zip_path = self._zip_path_var.get().strip()
        if not zip_path:
            messagebox.showwarning("No file", "Please select a .gameai_profile file first.")
            return

        from src.profile_manager import validate_profile_zip

        result = validate_profile_zip(zip_path)

        # Clear and show results
        self._results_text.configure(state=tk.NORMAL)
        self._results_text.delete("1.0", tk.END)

        if result.is_valid:
            self._results_text.insert(tk.END, "✅  VALIDATION PASSED\n\n", "success")
            # Show manifest info if available
            if result.manifest:
                self._results_text.insert(tk.END, "Manifest:\n", "header")
                for key in ("profile_name", "target_game", "author", "version", "created"):
                    val = result.manifest.get(key)
                    if val:
                        self._results_text.insert(tk.END, f"  {key}: {val}\n")
                tags = result.manifest.get("tags", [])
                if tags:
                    self._results_text.insert(tk.END, f"  tags: {', '.join(tags)}\n")
            self._btn_import.configure(state=tk.NORMAL)
        else:
            self._results_text.insert(tk.END, "❌  VALIDATION FAILED\n\n", "error")
            for err in result.errors:
                self._results_text.insert(tk.END, f"  • {err}\n", "error")
            self._btn_import.configure(state=tk.DISABLED)

        if result.warnings:
            self._results_text.insert(tk.END, "\nWarnings:\n", "warning")
            for warn in result.warnings:
                self._results_text.insert(tk.END, f"  ⚠ {warn}\n", "warning")

        self._results_text.configure(state=tk.DISABLED)

        # Tag configuration for coloured output
        self._results_text.tag_configure("success", foreground=_SUCCESS)
        self._results_text.tag_configure("error", foreground=_DANGER)
        self._results_text.tag_configure("warning", foreground=_WARNING)
        self._results_text.tag_configure("header", foreground=_ACCENT)

    def _on_import(self) -> None:
        """Run the import and notify the caller."""
        zip_path = self._zip_path_var.get().strip()
        if not zip_path:
            return

        from src.profile_manager import import_profile

        try:
            profile_name, profile_path = import_profile(zip_path)
            messagebox.showinfo(
                "Import Successful",
                f"Profile '{profile_name}' imported successfully.\n\n"
                f"Location: {profile_path}",
            )
            if self._on_imported is not None:
                self._on_imported(profile_name, profile_path)
            self.destroy()
        except ValueError as exc:
            messagebox.showerror("Import Failed", str(exc))
        except Exception as exc:
            messagebox.showerror("Import Error", f"An unexpected error occurred:\n{exc}")

    # ------------------------------------------------------------------
    # Export UI
    # ------------------------------------------------------------------

    def _build_export_ui(self) -> None:
        """Build the export‑mode dialog."""
        main = tk.Frame(self, bg=_BG, padx=16, pady=12)
        main.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(
            main,
            text="Export Profile as .gameai_profile",
            bg=_BG,
            fg=_FG,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor=tk.W, pady=(0, 12))

        # Form fields
        fields: list[tuple[str, str, str | None]] = [
            ("Profile Name *", "profile_name", self._profile_name or ""),
            ("Version *", "version", "1.0.0"),
            ("Target Game", "target_game", ""),
            ("Author", "author", ""),
            ("Description", "description", ""),
        ]

        self._field_vars: dict[str, tk.StringVar] = {}

        for label_text, key, default in fields:
            row = tk.Frame(main, bg=_BG)
            row.pack(fill=tk.X, pady=2)

            tk.Label(
                row,
                text=label_text,
                bg=_BG,
                fg=_FG,
                font=("Segoe UI", 10),
                width=16,
                anchor=tk.W,
            ).pack(side=tk.LEFT)

            var = tk.StringVar(value=default)
            self._field_vars[key] = var
            tk.Entry(
                row,
                textvariable=var,
                bg=_ENTRY_BG,
                fg=_ENTRY_FG,
                insertbackground=_ENTRY_FG,
                relief=tk.FLAT,
                font=("Segoe UI", 10),
                width=30,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Tags
        tags_row = tk.Frame(main, bg=_BG)
        tags_row.pack(fill=tk.X, pady=2)
        tk.Label(
            tags_row,
            text="Tags (comma-sep.)",
            bg=_BG,
            fg=_FG,
            font=("Segoe UI", 10),
            width=16,
            anchor=tk.W,
        ).pack(side=tk.LEFT)
        self._tags_var = tk.StringVar()
        tk.Entry(
            tags_row,
            textvariable=self._tags_var,
            bg=_ENTRY_BG,
            fg=_ENTRY_FG,
            insertbackground=_ENTRY_FG,
            relief=tk.FLAT,
            font=("Segoe UI", 10),
            width=30,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Optional inclusions (checkboxes)
        opt_frame = tk.LabelFrame(
            main,
            text="Optional Inclusions",
            bg=_BG,
            fg=_FG,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=6,
        )
        opt_frame.pack(fill=tk.X, pady=(12, 8))

        self._include_state = tk.BooleanVar(value=True)
        self._include_corrections = tk.BooleanVar(value=True)
        self._include_readme = tk.BooleanVar(value=True)
        self._include_icon = tk.BooleanVar(value=False)

        for text, var in [
            ("Include state.json (runtime state)", self._include_state),
            ("Include corrections.json (OCR corrections)", self._include_corrections),
            ("Include README.md", self._include_readme),
            ("Include res/icon.png", self._include_icon),
        ]:
            cb = tk.Checkbutton(
                opt_frame,
                text=text,
                variable=var,
                bg=_BG,
                fg=_FG,
                selectcolor=_BG,
                activebackground=_BG,
                activeforeground=_FG,
                font=("Segoe UI", 9),
            )
            cb.pack(anchor=tk.W)

        # Output file
        out_row = tk.Frame(main, bg=_BG)
        out_row.pack(fill=tk.X, pady=(8, 4))
        tk.Label(
            out_row,
            text="Output File",
            bg=_BG,
            fg=_FG,
            font=("Segoe UI", 10),
            width=16,
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        self._output_var = tk.StringVar()
        tk.Entry(
            out_row,
            textvariable=self._output_var,
            bg=_ENTRY_BG,
            fg=_ENTRY_FG,
            insertbackground=_ENTRY_FG,
            relief=tk.FLAT,
            font=("Segoe UI", 10),
            width=30,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        ttk.Button(out_row, text="Browse …", command=self._on_browse_export).pack(
            side=tk.LEFT
        )

        # Action buttons
        btn_row = tk.Frame(main, bg=_BG)
        btn_row.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(btn_row, text="📤  Export", command=self._on_export).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(btn_row, text="Close", command=self.destroy).pack(side=tk.LEFT)

    def _on_browse_export(self) -> None:
        """Open save dialog for the output archive."""
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save .gameai_profile archive",
            defaultextension=".gameai_profile",
            filetypes=[
                ("Game AI Profile", "*.gameai_profile"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._output_var.set(path)

    def _on_export(self) -> None:
        """Run the export."""
        profile_name = self._field_vars["profile_name"].get().strip()
        version = self._field_vars["version"].get().strip()
        output_path = self._output_var.get().strip()

        if not profile_name:
            messagebox.showwarning("Missing Info", "Profile Name is required.")
            return
        if not version:
            messagebox.showwarning("Missing Info", "Version is required.")
            return

        from src.profile_manager import ExportOptions, export_profile

        # Parse tags
        raw_tags = self._tags_var.get().strip()
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()] if raw_tags else []

        opts = ExportOptions(
            profile_name=profile_name,
            version=version,
            author=self._field_vars["author"].get().strip(),
            description=self._field_vars["description"].get().strip(),
            target_game=self._field_vars["target_game"].get().strip(),
            tags=tags,
            include_state=self._include_state.get(),
            include_corrections=self._include_corrections.get(),
            include_readme=self._include_readme.get(),
            include_icon=self._include_icon.get(),
        )

        # Determine output directory vs file
        if output_path:
            output = Path(output_path)
        else:
            output = Path.cwd()

        try:
            result_path = export_profile(opts, output)
            messagebox.showinfo(
                "Export Successful",
                f"Profile exported to:\n{result_path}",
            )
            self.destroy()
        except FileNotFoundError as exc:
            messagebox.showerror("Export Failed", str(exc))
        except Exception as exc:
            messagebox.showerror("Export Error", f"An unexpected error occurred:\n{exc}")
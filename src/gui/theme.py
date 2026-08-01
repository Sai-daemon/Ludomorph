"""
Centralised theme system for the Ludomorph desktop app.

Provides:
- ``ThemePalette`` — dataclass holding all colour / font data for one theme
- Built‑in ``"dark"`` and ``"light"`` palettes
- ``ThemeManager`` — singleton that reads config, detects the system theme,
  and applies styles globally
- ``apply_ttk_styles()`` — registers all ttk widget styles from the active palette
- System‑theme detection on Linux (GNOME/KDE), Windows, and macOS

All other GUI modules should import their colours and fonts from here
rather than hard‑coding palette constants.

Usage::

    from src.gui.theme import ThemeManager

    tm = ThemeManager()
    palette = tm.get_palette()
    frame = tk.Frame(parent, bg=palette.bg)
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

# ---------------------------------------------------------------------------
# ThemePalette — single source of truth for all colour & font values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThemePalette:
    """Holds every colour, font, and visual parameter for one theme.

    Colours are stored as hex strings (``"#RRGGBB"``).  Font stacks are
    comma‑separated strings suitable for tkinter's ``font`` parameter.
    """

    # -- Metadata -------------------------------------------------------------
    name: str
    display_name: str  # human‑readable for the settings UI
    is_dark: bool

    # -- Core palette ---------------------------------------------------------
    bg: str                    # main background
    fg: str                    # default foreground / text
    accent: str                # accent blue — buttons, highlights, tabs
    success: str               # green  — running / healthy
    danger: str                # red    — stopped / unhealthy
    warning: str               # amber  — degraded / paused
    disabled_bg: str           # disabled button / inactive tab background
    disabled_fg: str           # disabled / subtle foreground

    # -- Derived / secondary colours ------------------------------------------
    frame_bg: str = ""         # slightly lighter than bg (section frames)
    card_bg: str = ""          # card / entry / editor background
    entry_bg: str = ""         # text entry background
    entry_fg: str = ""         # text entry foreground
    entry_insert: str = ""     # text entry cursor colour
    status_bar_bg: str = ""    # status bar background
    header_bg: str = ""        # section header background (health panel)
    sidebar_bg: str = ""       # list sidebar background
    separator: str = ""        # separator / border colour
    subtle: str = ""           # subtle / hint text colour

    # -- Semantic overrides (independent of dark/light) -----------------------
    syntax_key: str = "#9CDCFE"     # JSON key highlight
    syntax_string: str = "#CE9178"  # JSON string highlight
    syntax_number: str = "#B5CEA8"  # JSON number highlight
    syntax_bool_null: str = "#569CD6"  # JSON bool/null highlight
    syntax_bracket: str = "#FFD700"  # bracket highlight

    # -- Calibration overlay colours ------------------------------------------
    selection: str = "#00FF00"       # selected region rectangle
    ocr_colour: str = "#50C878"      # OCR region border
    colour_bar_colour: str = "#4FC3F7"  # colour‑bar region border
    detection_colour: str = "#FF7043"   # YOLO detection border
    detection_label_bg: str = "#3C1414"  # detection label background

    # -- Font stacks (comma‑separated for tkinter fallback) -------------------
    ui_font: str = '"Segoe UI", "DejaVu Sans", "Noto Sans", "TkDefaultFont"'
    mono_font: str = '"Cascadia Code", "Consolas", "DejaVu Sans Mono", "TkFixedFont"'

    # -- Derived field defaults are filled in post‑init -----------------------

    def __post_init__(self) -> None:
        """Populate any empty derived fields with sensible defaults."""
        # We can't use self.__class__ because the dataclass is frozen after init.
        # Instead we compute defaults and object.__setattr__ them.
        if not self.frame_bg:
            object.__setattr__(self, "frame_bg", self._lighten(self.bg, 8))
        if not self.card_bg:
            object.__setattr__(self, "card_bg", self._lighten(self.bg, 12))
        if not self.entry_bg:
            object.__setattr__(self, "entry_bg", self._lighten(self.bg, 14))
        if not self.entry_fg:
            object.__setattr__(self, "entry_fg", self.fg)
        if not self.entry_insert:
            object.__setattr__(self, "entry_insert", self.fg)
        if not self.status_bar_bg:
            object.__setattr__(self, "status_bar_bg", self._lighten(self.bg, 6))
        if not self.header_bg:
            object.__setattr__(self, "header_bg", self._lighten(self.bg, 10))
        if not self.sidebar_bg:
            object.__setattr__(self, "sidebar_bg", self._lighten(self.bg, 4))
        if not self.separator:
            object.__setattr__(self, "separator", self.disabled_bg)
        if not self.subtle:
            object.__setattr__(self, "subtle", self.disabled_fg)

    @staticmethod
    def _lighten(hex_colour: str, amount: int) -> str:
        """Lighten a hex colour by *amount* (0–255 added to each channel)."""
        h = hex_colour.lstrip("#")
        r = max(0, min(255, int(h[0:2], 16) + amount))
        g = max(0, min(255, int(h[2:4], 16) + amount))
        b = max(0, min(255, int(h[4:6], 16) + amount))
        return f"#{r:02X}{g:02X}{b:02X}"

    def to_dict(self) -> dict[str, Any]:
        """Return all palette values as a plain dict (for serialisation / debugging)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---------------------------------------------------------------------------
# Built‑in palettes
# ---------------------------------------------------------------------------

# Dark theme — anchored on the user‑requested #0F141E background
DARK_PALETTE = ThemePalette(
    name="dark",
    display_name="Dark",
    is_dark=True,
    bg="#0F141E",            # deep navy‑black
    fg="#D4D4D4",            # soft white
    accent="#0078D4",        # Windows 10 blue
    success="#50C878",       # mint green
    danger="#E04040",        # soft red
    warning="#E8A317",       # amber
    disabled_bg="#283040",   # muted navy
    disabled_fg="#808080",   # medium grey
    frame_bg="#151C2A",      # 6 up from bg
    card_bg="#192237",       # 10 up
    entry_bg="#1C2740",      # 14 up
    entry_fg="#D4D4D4",
    entry_insert="#D4D4D4",
    status_bar_bg="#131B28",     # 4 up
    header_bg="#162030",         # 8 up
    sidebar_bg="#101725",        # 2 up
    separator="#283040",
    subtle="#808080",
)

# Light theme — a clean, high‑contrast professional light scheme
LIGHT_PALETTE = ThemePalette(
    name="light",
    display_name="Light",
    is_dark=False,
    bg="#F3F3F3",            # near‑white
    fg="#1E1E1E",            # near‑black
    accent="#0078D4",        # same blue works well on light
    success="#2E7D32",       # darker green for readability on white
    danger="#C62828",        # slightly darker red
    warning="#EF6C00",       # deeper amber
    disabled_bg="#E0E0E0",   # light grey
    disabled_fg="#616161",   # darker grey for WCAG AA contrast (~5.3:1)
    frame_bg="#FFFFFF",      # card white
    card_bg="#FAFAFA",       # off‑white
    entry_bg="#FFFFFF",      # white entry
    entry_fg="#1E1E1E",
    entry_insert="#1E1E1E",
    status_bar_bg="#E8E8E8",
    header_bg="#EBEBEB",
    sidebar_bg="#EEEEEE",
    separator="#D0D0D0",
    subtle="#4F4F4F",        # darker subtle text (~5.8:1 contrast on F3F3F3)
    syntax_key="#0451A5",         # darker blue for light bg
    syntax_string="#A31515",      # darker red
    syntax_number="#098658",      # darker green
    syntax_bool_null="#0000FF",   # blue
    syntax_bracket="#795E26",     # brown
)


BUILT_IN_THEMES: dict[str, ThemePalette] = {
    "dark": DARK_PALETTE,
    "light": LIGHT_PALETTE,
}


# ---------------------------------------------------------------------------
# System theme detection
# ---------------------------------------------------------------------------


def _detect_linux_theme() -> str | None:
    """Detect system dark/light preference on Linux.

    Checks (in order):
    1. GNOME ``gsettings`` colour‑scheme
    2. KDE ``kreadconfig5`` colourscheme
    3. GTK theme name heuristic
    Returns ``"dark"``, ``"light"``, or ``None`` if undetectable.
    """
    # 1. GNOME colour-scheme
    try:
        result = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            capture_output=True, text=True, timeout=3,
        )
        if "prefer-dark" in result.stdout:
            return "dark"
        if "default" in result.stdout:
            return "light"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 2. KDE Plasma
    try:
        result = subprocess.run(
            ["kreadconfig5", "--group", "General", "--key", "ColorScheme"],
            capture_output=True, text=True, timeout=3,
        )
        scheme = result.stdout.strip().lower()
        if scheme and "dark" in scheme:
            return "dark"
        if scheme:
            return "light"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. GTK theme name heuristic
    try:
        result = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"],
            capture_output=True, text=True, timeout=3,
        )
        theme = result.stdout.strip().lower()
        if "dark" in theme:
            return "dark"
        return "light"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _detect_windows_theme() -> str | None:
    """Detect system dark/light preference on Windows via registry."""
    if sys.platform != "win32":
        return None
    try:
        import winreg  # type: ignore[import-not-found]

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        )
        value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        winreg.CloseKey(key)
        return "light" if value == 1 else "dark"
    except Exception:
        return None


def _detect_macos_theme() -> str | None:
    """Detect system dark/light preference on macOS."""
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["defaults", "read", "-g", "AppleInterfaceStyle"],
            capture_output=True, text=True, timeout=3,
        )
        style = result.stdout.strip()
        if style == "Dark":
            return "dark"
        return "light"
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return "light"  # default is light on macOS


def detect_system_theme() -> str:
    """Return ``"dark"`` or ``"light"`` based on the OS preference.

    Falls back to ``"dark"`` if detection is impossible.
    """
    detectors = [
        ("linux", _detect_linux_theme),
        ("darwin", _detect_macos_theme),
        ("win32", _detect_windows_theme),
    ]
    for platform_name, detector in detectors:
        if sys.platform.startswith(platform_name):
            result = detector()
            if result is not None:
                return result
    # Fallback: check environment variables that various DEs set
    env_hints = [
        "GTK_THEME",
        "QT_STYLE_OVERRIDE",
        "DESKTOP_SESSION",
    ]
    for var in env_hints:
        value = os.environ.get(var, "").lower()
        if "dark" in value:
            return "dark"
    return "dark"


# ---------------------------------------------------------------------------
# Font fallback helpers
# ---------------------------------------------------------------------------


def resolve_font_stack(font_stack: str) -> str:
    """Return the first available font from a comma‑separated stack.

    Falls back to ``"TkDefaultFont"`` or ``"TkFixedFont"`` if none are found.
    The returned string is suitable as a tkinter ``font`` parameter.
    """
    import tkinter.font as tkfont

    available = set(tkfont.families())
    for name in (f.strip().strip('"\'') for f in font_stack.split(",")):
        if name in available:
            return name
    # Ultimate fallback
    return "TkDefaultFont" if "Mono" not in font_stack else "TkFixedFont"


# ---------------------------------------------------------------------------
# ThemeManager — singleton that owns the active palette
# ---------------------------------------------------------------------------


class ThemeManager:
    """Singleton manager for the active theme palette.

    Reads ``config.json`` (via ``ConfigManager`` if available) to determine
    the active theme on construction.  Supports:

    * ``theme: "auto"`` → auto‑detect system preference
    * ``theme: "dark"`` → force dark
    * ``theme: "light"`` → force light
    * ``theme.custom_bg: "#RRGGBB"`` → override the main background colour

    Call :meth:`apply_ttk_styles` after the root tk window is created to
    register all themed ttk styles.
    """

    _instance: ClassVar[ThemeManager | None] = None
    _palette: ThemePalette
    _config: dict[str, Any]
    _save_config: Any  # callable that persists config

    def __new__(cls) -> ThemeManager:
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialised = False
            cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        self._initialised = True
        self._load_config()
        self._resolve_palette()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load the global config to read theme settings."""
        self._config = {}
        self._save_config = None
        try:
            from src.config_manager import load_global_config, save_global_config
            self._config = load_global_config()
            self._save_config = save_global_config
        except Exception:
            pass

    def _resolve_palette(self) -> None:
        """Determine the active palette from config + system detection."""
        theme_mode = self._config.get("theme", "auto")
        if theme_mode not in ("auto", "dark", "light"):
            theme_mode = "auto"

        if theme_mode == "auto":
            base_name = detect_system_theme()
        else:
            base_name = theme_mode

        self._base_name = base_name
        palette = BUILT_IN_THEMES.get(base_name, DARK_PALETTE)

        # Apply custom background override if configured
        custom_bg = self._config.get("theme_custom_bg")
        if custom_bg and isinstance(custom_bg, str) and custom_bg.startswith("#"):
            palette = self._override_bg(palette, custom_bg)

        self._palette = palette

    @staticmethod
    def _override_bg(base: ThemePalette, new_bg: str) -> ThemePalette:
        """Create a new palette with *new_bg* as the main background,
        re‑deriving all dependent colours from it.

        Uses the same lighten‑step arithmetic as ``ThemePalette.__post_init__``
        to maintain visual coherence.
        """
        # Compute dependent colours from the new base
        def _lt(h: str, a: int) -> str:
            return ThemePalette._lighten(h, a)

        # We can't construct a *new* ThemePalette normally because it's frozen,
        # but we *can* use object.__setattr__ on a fresh instance.
        # A simpler path: create a shallow copy dictionary and build a new one.
        raw: dict[str, Any] = {}
        for f in fields(base):
            raw[f.name] = getattr(base, f.name)
        # Override bg + derived
        raw["bg"] = new_bg
        raw["frame_bg"] = _lt(new_bg, 8)
        raw["card_bg"] = _lt(new_bg, 12)
        raw["entry_bg"] = _lt(new_bg, 14)
        raw["status_bar_bg"] = _lt(new_bg, 4)
        raw["header_bg"] = _lt(new_bg, 10)
        raw["sidebar_bg"] = _lt(new_bg, 2)
        raw["separator"] = _lt(new_bg, 16)
        return ThemePalette(**raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_palette(self) -> ThemePalette:
        """Return the active ``ThemePalette``."""
        return self._palette

    @property
    def palette(self) -> ThemePalette:
        """Shorthand for ``get_palette()``."""
        return self._palette

    @property
    def theme_mode(self) -> str:
        """Return the configured theme mode (``"auto"``, ``"dark"``, ``"light"``)."""
        return self._config.get("theme", "auto")

    @property
    def base_theme_name(self) -> str:
        """Return the resolved base theme name (``"dark"`` or ``"light"``)."""
        return self._base_name

    @property
    def custom_bg(self) -> str | None:
        """Return the custom background override colour, if set."""
        val = self._config.get("theme_custom_bg")
        if val and isinstance(val, str) and val.startswith("#") and len(val) == 7:
            return val
        return None

    def set_theme(self, mode: str, custom_bg: str | None = None) -> None:
        """Change the active theme at runtime.

        Args:
            mode: ``"auto"``, ``"dark"``, or ``"light"``.
            custom_bg: Optional custom background hex colour (``"#RRGGBB"``).
        """
        if mode not in ("auto", "dark", "light"):
            return
        self._config["theme"] = mode
        if custom_bg is not None:
            if custom_bg.startswith("#") and len(custom_bg) == 7:
                self._config["theme_custom_bg"] = custom_bg
            else:
                self._config.pop("theme_custom_bg", None)
        else:
            self._config.pop("theme_custom_bg", None)

        # Persist
        if self._save_config is not None:
            try:
                self._save_config(self._config)
            except Exception:
                pass

        # Re‑resolve
        self._resolve_palette()

        # Emit a virtual event on all tk widgets so they can reconfigure
        try:
            self._broadcast_theme_change()
        except Exception:
            pass

    def _broadcast_theme_change(self) -> None:
        """Generate a ``<<ThemeChanged>>`` virtual event on all existing tkinter widgets."""
        import tkinter as tk

        try:
            root = tk._default_root  # type: ignore[attr-defined]
            if root is not None:
                root.event_generate("<<ThemeChanged>>", when="tail")
        except Exception:
            pass

    def apply_ttk_styles(self, root: Any = None) -> None:
        """Configure all ttk widget styles from the active palette.

        Call this after the root tk window is created.  Registers:

        * Base ``TNotebook``, ``TFrame``, ``TButton``, ``TLabel`` styles
        * Application‑specific styles (``Start.TButton``, ``Stop.TButton``, etc.)
        * Health panel styles
        * Editor / settings styles

        Args:
            root: Optional root window (unused; kept for future extension).
        """
        import tkinter as tk
        from tkinter import ttk

        p = self._palette

        style = ttk.Style()
        # Use 'clam' on Linux/Windows, 'aqua' is forced on macOS automatically
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass  # some platforms may not have clam

        # -- Base styles -------------------------------------------------------
        style.configure("TNotebook", background=p.bg, borderwidth=0, tabmargins=(2, 2, 2, 0))
        style.configure(
            "TNotebook.Tab",
            background=p.disabled_bg,
            foreground=p.fg,
            padding=(12, 4),
            font=(p.ui_font, 10),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", p.accent)],
            foreground=[("selected", "#FFFFFF")],
        )

        style.configure("TFrame", background=p.bg)
        style.configure("TLabel", background=p.bg, foreground=p.fg)
        style.configure("TButton", background=p.accent, foreground="#FFFFFF",
                        font=(p.ui_font, 9, "bold"), borderwidth=1, padding=(12, 3))
        style.map("TButton", background=[("active", ThemePalette._lighten(p.accent, -20))])

        # -- Toolbar / action button styles ------------------------------------
        style.configure(
            "Start.TButton",
            background=p.success,
            foreground="#FFFFFF" if p.is_dark else "#1E1E1E",
            font=(p.ui_font, 10, "bold"),
            borderwidth=1,
            padding=(16, 4),
        )
        style.map("Start.TButton",
                  background=[("active", ThemePalette._lighten(p.success, -15))])

        style.configure(
            "Stop.TButton",
            background=p.danger,
            foreground="#FFFFFF",
            font=(p.ui_font, 10, "bold"),
            borderwidth=1,
            padding=(16, 4),
        )
        style.map("Stop.TButton",
                  background=[("active", ThemePalette._lighten(p.danger, -20))])

        style.configure(
            "Pause.TButton",
            background=p.warning,
            foreground="#1E1E1E" if not p.is_dark else "#1E1E1E",
            font=(p.ui_font, 10, "bold"),
            borderwidth=1,
            padding=(16, 4),
        )
        style.map("Pause.TButton",
                  background=[("active", ThemePalette._lighten(p.warning, -15))])

        style.configure(
            "Calibrate.TButton",
            background=p.accent,
            foreground="#FFFFFF",
            font=(p.ui_font, 10, "bold"),
            borderwidth=1,
            padding=(16, 4),
        )
        style.map("Calibrate.TButton",
                  background=[("active", ThemePalette._lighten(p.accent, -20))])

        style.configure(
            "Capture.TButton",
            background=p.accent,
            foreground="#FFFFFF",
            font=(p.ui_font, 9, "bold"),
            borderwidth=1,
            padding=(10, 3),
        )

        # -- Settings panel styles ---------------------------------------------
        style.configure(
            "Settings.TButton",
            background=p.accent,
            foreground="#FFFFFF",
            font=(p.ui_font, 9, "bold"),
            borderwidth=1,
            padding=(12, 3),
        )

        style.configure(
            "SettingsDanger.TButton",
            background=p.danger,
            foreground="#FFFFFF",
            font=(p.ui_font, 9, "bold"),
            borderwidth=1,
            padding=(12, 3),
        )

        # -- Editor styles -----------------------------------------------------
        style.configure(
            "Editor.TButton",
            background=p.accent,
            foreground="#FFFFFF",
            font=(p.ui_font, 9, "bold"),
            borderwidth=1,
            padding=(10, 2),
        )

        style.configure(
            "Danger.TButton",
            background=p.danger,
            foreground="#FFFFFF",
            font=(p.ui_font, 9, "bold"),
            borderwidth=1,
            padding=(10, 2),
        )

        style.configure(
            "Mode.TButton",
            background=p.warning,
            foreground="#1E1E1E" if not p.is_dark else "#1E1E1E",
            font=(p.ui_font, 9, "bold"),
            borderwidth=1,
            padding=(10, 2),
        )

        # -- Scrollbar styles --------------------------------------------------
        style.configure(
            "TScrollbar",
            background=p.bg,
            troughcolor=p.disabled_bg,
            bordercolor=p.bg,
            arrowcolor=p.fg,
            arrowsize=12,
        )
        style.map(
            "TScrollbar",
            background=[("active", p.accent)],
        )

        # Vertical and horizontal overrides if needed
        style.configure(
            "Vertical.TScrollbar",
            background=p.disabled_bg,
            troughcolor=p.bg,
            arrowsize=14,
        )

        # -- Health panel styles -----------------------------------------------
        style.configure("HealthPanel.TFrame", background=p.bg)
        style.configure("HealthCard.TFrame", background=p.card_bg)
        style.configure("HealthHeader.TFrame", background=p.header_bg)
        style.configure("HealthStatus.TLabel", background=p.card_bg, foreground=p.fg,
                        font=(p.ui_font, 9))
        style.configure("HealthService.TLabel", background=p.card_bg, foreground=p.fg,
                        font=(p.ui_font, 10, "bold"))
        style.configure("HealthReason.TLabel", background=p.card_bg,
                        foreground=p.disabled_fg, font=(p.ui_font, 8), wraplength=380)


# ---------------------------------------------------------------------------
# Convenience module‑level access
# ---------------------------------------------------------------------------


def get_active_palette() -> ThemePalette:
    """Return the currently active ``ThemePalette`` (convenience function)."""
    return ThemeManager().palette


def get_colour(name: str) -> str:
    """Return a single colour value from the active palette by attribute name."""
    return getattr(ThemeManager().palette, name, "")


# Refresh on import
_applied_styles: bool = False


def _ensure_styles() -> None:
    """Lazily apply ttk styles on first access (called by theme consumers)."""
    global _applied_styles
    if not _applied_styles:
        try:
            ThemeManager().apply_ttk_styles()
        except Exception:
            pass
        _applied_styles = True
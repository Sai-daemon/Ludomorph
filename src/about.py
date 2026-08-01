"""
About section for Ludomorph.

Provides programmatically-accessible constants for the application name,
version, author, license, and a human-readable ABOUT_TEXT string suitable
for display in the Help → About dialog.

The version is imported from ``src.__init__`` — the single source of truth —
so the about dialog, status bar, and CLI banner always stay in sync.
"""

from src import __version__

APP_NAME: str = "Ludomorph"
APP_AUTHOR: str = "Sarah Schmidt / SAI-DAEMON"
APP_LICENSE: str = "See LICENSE file for full license text."
APP_COPYRIGHT: str = "Copyright (c) 2026 Sarah Schmidt"
APP_DESCRIPTION: str = (
    "A universal, external application that injects an autonomous LLM agent "
    "into any PC game by capturing the screen and simulating keyboard/mouse input."
)

ABOUT_TEXT: str = (
    f"{APP_NAME} — v{__version__}\n\n"
    f"{APP_DESCRIPTION}\n\n"
    f"{APP_COPYRIGHT}\n"
    f"Author: {APP_AUTHOR}\n\n"
    f"License: {APP_LICENSE}\n\n"
    "This software includes third-party components.\n"
    "See the notices.md file for full attribution."
)
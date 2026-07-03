"""
AI Game Master - Core package.

A universal, external application that injects an autonomous LLM agent
into any PC game by capturing the screen and simulating keyboard/mouse input.
"""

__version__ = "0.1.0"

from src.macro_executor import (
    CancellationToken,
    MacroCancelledError,
    MacroError,
    MacroExecutor,
    MacroPriority,
    MacroRejectedError,
    MacroRequest,
    accurate_hold,
)
from src.ollama_health import (
    OllamaHealthError,
    OllamaHealthResult,
    ollama_health_check,
    ollama_health_check_or_raise,
)

__all__ = [
    "MacroExecutor",
    "MacroPriority",
    "MacroRequest",
    "CancellationToken",
    "MacroError",
    "MacroCancelledError",
    "MacroRejectedError",
    "accurate_hold",
    "ollama_health_check",
    "ollama_health_check_or_raise",
    "OllamaHealthResult",
    "OllamaHealthError",
]

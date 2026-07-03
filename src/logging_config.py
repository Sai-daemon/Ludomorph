"""
Logging configuration for AI Game Master.

Uses loguru for structured, coloured console output and rotating file logs.
"""

import sys
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_is_initialized: bool = False
_sink_ids: list[int] = []


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(
    log_level: str = "INFO",
    log_dir: Path | None = None,
    retention: str = "7 days",
    rotation: str = "10 MB",
) -> None:
    """
    Configure loguru with console and file sinks.

    On first call this creates the sinks.  Subsequent calls with a different
    *log_level* reconfigure the console sink without touching the file sink.
    This allows ``main.py`` to apply the user's configured log level after
    startup.

    Parameters
    ----------
    log_level : str
        Minimum log level for console output ("DEBUG", "INFO", etc.).
    log_dir : Path or None
        Directory for log files. If None, defaults to ~/.gameai/logs/.
    retention : str
        How long to keep log files (loguru format).
    rotation : str
        When to rotate log files (loguru format).
    """
    global _is_initialized, _sink_ids

    if _is_initialized:
        # Already initialised — only update the console sink's level.
        # Remove the old console sink (should be the first non-file sink).
        for sid in _sink_ids[:1]:
            try:
                logger.remove(sid)
            except ValueError:
                pass
        sid = logger.add(
            sys.stderr,
            level=log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=True,
        )
        _sink_ids[0] = sid
        logger.debug(f"Log level updated to {log_level}")
        return

    # First-time initialisation

    # Remove default sink
    logger.remove()

    # Console sink with colours (sink ID 0)
    sid = logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    _sink_ids.append(sid)

    # File sink
    if log_dir is None:
        import appdirs

        log_dir = Path(appdirs.user_config_dir("gameai", appauthor=False)) / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    sid = logger.add(
        log_dir / "gameai_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,  # thread-safe file writes
    )
    _sink_ids.append(sid)

    _is_initialized = True
    logger.debug(f"Logging initialized. Level={log_level}, log_dir={log_dir}")


def get_logger(name: str = __name__):
    """
    Return a bound logger with the given module name.

    Usage::

        from src.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Hello")
    """
    return logger.bind(name=name)
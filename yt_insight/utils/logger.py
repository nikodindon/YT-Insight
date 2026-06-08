"""
Centralized logging setup for YT-Insight.

We don't want every module to call ``logging.basicConfig`` independently,
so this module provides a single :func:`setup_logging` function that
the CLI calls once at startup. Modules just do ``logger = logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal

# Rich is the canonical console handler — it gives us colored output
# and a clean format that works well in a TTY. If Rich isn't installed
# (shouldn't happen in practice — it's a hard dep of the CLI), we fall
# back to plain stderr.
try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False


_LEVELS = {
    "DEBUG":    logging.DEBUG,
    "INFO":     logging.INFO,
    "WARNING":  logging.WARNING,
    "WARN":     logging.WARNING,
    "ERROR":    logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_LOG_FORMAT = "%(message)s"
_LOG_DATEFMT = "[%X]"


def setup_logging(
    level: str | int | None = None,
    *,
    show_time: bool = True,
    show_path: bool = False,
    rich_tracebacks: bool = True,
) -> logging.Logger:
    """
    Configure the root logger for YT-Insight.

    Parameters
    ----------
    level:
        Either a string (``"DEBUG"``, ``"INFO"``, …) or an int level.
        If ``None``, falls back to the ``YT_INSIGHT_LOG_LEVEL`` env var
        or ``"INFO"``.
    show_time, show_path, rich_tracebacks:
        Forwarded to Rich's handler for nicer console output.
    """
    if level is None:
        level = os.getenv("YT_INSIGHT_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = _LEVELS.get(level.upper(), logging.INFO)

    # Reset any previously-installed handlers on the root logger so this
    # function is idempotent (e.g. in tests).
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    if _RICH_AVAILABLE:
        handler: logging.Handler = RichHandler(
            rich_tracebacks=rich_tracebacks,
            show_time=show_time,
            show_path=show_path,
            markup=False,
        )
    else:  # pragma: no cover
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    root.addHandler(handler)

    # Tame noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper around ``logging.getLogger``."""
    return logging.getLogger(name)

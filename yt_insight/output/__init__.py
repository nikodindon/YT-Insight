"""Output backends: Rich terminal rendering + Markdown / JSON file writers.

Each module is independent and exposes a single class:

- :class:`ConsoleRenderer` — pretty-prints an :class:`AnalysisResult` to
  the terminal (with boxes, panels, tables).
- :class:`FileWriter`     — saves the same result as Markdown and/or JSON.

Both classes are pure functions of the input — no network, no I/O
side-effects beyond writing the file. The CLI is responsible for
chaining them.
"""

from .console import ConsoleRenderer, RenderConfig, render_to_console, render_to_string
from .file_writer import FileWriter, write_outputs

__all__ = [
    "ConsoleRenderer",
    "RenderConfig",
    "render_to_console",
    "render_to_string",
    "FileWriter",
    "write_outputs",
]

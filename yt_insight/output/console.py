"""
Pretty-printing of an :class:`AnalysisResult` to the terminal.

Uses Rich (https://rich.readthedocs.io) for boxes, colors and tables.
The renderer is **declarative**: it takes a result, builds the
structure once, and prints it. No streaming, no live updates — that's
the CLI's job.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    from rich.box import ROUNDED
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False

if TYPE_CHECKING:
    from yt_insight.analyzer.base import AnalysisResult
    from yt_insight.transcriber import TranscriptionResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RenderConfig:
    """Toggle individual sections on/off for the console output."""
    show_header: bool = True
    show_metadata: bool = True
    show_topic_banner: bool = True
    show_summary: bool = True
    show_key_points: bool = True
    show_analysis: bool = True
    show_quotes: bool = True
    show_transcript: bool = False        # off by default — too verbose
    show_stats_footer: bool = True
    max_transcript_chars: int = 4_000    # truncate transcript preview


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class ConsoleRenderer:
    """
    Render an :class:`AnalysisResult` (and optionally a
    :class:`TranscriptionResult`) to a Rich console.

    Parameters
    ----------
    config:
        Toggle individual sections.
    console:
        Optional :class:`rich.console.Console` to print to. Defaults to
        the current stdout. Pass a custom one for testing or for
        capturing output to a string.
    """

    def __init__(
        self,
        config: RenderConfig | None = None,
        console: "Console | None" = None,
    ):
        if not _RICH_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "rich is required for ConsoleRenderer. Run: pip install rich"
            )
        self.config = config or RenderConfig()
        self.console = console or Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        analysis: "AnalysisResult",
        transcription: "TranscriptionResult | None" = None,
        video_url: str = "",
        video_title: str = "",
    ) -> None:
        """Print *analysis* (+ optional *transcription*) to the console."""
        if self.config.show_header:
            self._render_header(analysis, video_url)

        if self.config.show_metadata and self.config.show_topic_banner:
            self._render_topic_banner(analysis)

        if self.config.show_summary and analysis.summary:
            self._render_section("📝 Résumé détaillé", analysis.summary)

        if self.config.show_key_points and analysis.key_points:
            self._render_key_points(analysis.key_points)

        if self.config.show_analysis and analysis.analysis:
            self._render_analysis(analysis.analysis)

        if self.config.show_quotes and analysis.quotes:
            self._render_quotes(analysis.quotes)

        if self.config.show_transcript and transcription is not None:
            self._render_transcript_preview(transcription)

        if self.config.show_stats_footer:
            self._render_footer(analysis, transcription)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _render_header(
        self, analysis: "AnalysisResult", video_url: str,
    ) -> None:
        title = f"YT-Insight — {analysis.model_name or 'analyse'}"
        body = f"Backend : {analysis.backend or '?'}"
        if video_url:
            body += f"\nURL     : {video_url}"
        self.console.print()
        self.console.print(Panel(body, title=title, border_style="cyan", box=ROUNDED))
        self.console.print()

    def _render_topic_banner(self, analysis: "AnalysisResult") -> None:
        bits: list[str] = []
        if analysis.topic:
            bits.append(f"[bold]Sujet :[/] {analysis.topic}")
        if analysis.tone:
            bits.append(f"[bold]Ton  :[/] {analysis.tone}")
        if analysis.audience:
            bits.append(f"[bold]Public :[/] {analysis.audience}")
        if bits:
            self.console.print("  ".join(bits))
            self.console.print()

    def _render_section(self, title: str, body: str) -> None:
        self.console.print(Panel(
            Markdown(body) if _looks_like_markdown(body) else body,
            title=title,
            border_style="green",
            box=ROUNDED,
        ))
        self.console.print()

    def _render_key_points(self, points: list[str]) -> None:
        numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(points))
        self._render_section("🎯 Points clés", numbered)

    def _render_analysis(self, body: str) -> None:
        self._render_section("🔍 Analyse approfondie", body)

    def _render_quotes(self, quotes) -> None:
        # `quotes` is a list[Quote] — import lazily to avoid a cycle.
        from yt_insight.analyzer.base import Quote
        assert all(isinstance(q, Quote) for q in quotes)

        table = Table(
            title="💬 Citations notables",
            box=ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Timestamp", style="cyan", no_wrap=True)
        table.add_column("Citation", style="white")
        for q in quotes:
            ts = q.timestamp_str or "—"
            table.add_row(ts, q.text)
        self.console.print()
        self.console.print(table)
        self.console.print()

    def _render_transcript_preview(self, transcription: "TranscriptionResult") -> None:
        text = transcription.formatted_transcript(with_timestamps=True)
        if not text:
            return
        snippet = text[: self.config.max_transcript_chars]
        if len(text) > self.config.max_transcript_chars:
            snippet += "\n[…truncated…]"
        self._render_section("📄 Transcription (aperçu)", snippet)

    def _render_footer(
        self,
        analysis: "AnalysisResult",
        transcription: "TranscriptionResult | None",
    ) -> None:
        parts: list[str] = []
        if transcription is not None:
            parts.append(f"Langue : {transcription.language}")
            parts.append(f"Durée  : {transcription.duration_str}")
            parts.append(f"Tokens ≈ {transcription.estimated_tokens:,}")
        if analysis.key_points:
            parts.append(f"{len(analysis.key_points)} points clés")
        if analysis.quotes:
            parts.append(f"{len(analysis.quotes)} citations")
        if parts:
            self.console.print(
                f"[dim]{' · '.join(parts)}[/dim]"
            )
            self.console.print()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def render_to_console(
    analysis: "AnalysisResult",
    transcription: "TranscriptionResult | None" = None,
    *,
    video_url: str = "",
    video_title: str = "",
    config: RenderConfig | None = None,
) -> None:
    """One-shot render to stdout."""
    ConsoleRenderer(config=config).render(
        analysis, transcription, video_url=video_url, video_title=video_title,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _looks_like_markdown(text: str) -> bool:
    """Heuristic: does *text* contain any Markdown-y markup?"""
    if not text:
        return False
    markers = ("**", "##", "###", "- ", "* ", "1. ", "> ")
    return any(m in text for m in markers)


# ---------------------------------------------------------------------------
# Exposed for testing — capture Rich output to a string
# ---------------------------------------------------------------------------

def render_to_string(
    analysis: "AnalysisResult",
    transcription: "TranscriptionResult | None" = None,
    *,
    config: RenderConfig | None = None,
) -> str:
    """Same as :func:`render_to_console` but returns the rendered string.

    Useful for unit tests — no need to mock Rich.
    """
    if not _RICH_AVAILABLE:  # pragma: no cover
        raise ImportError("rich is required")
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100, color_system=None)
    ConsoleRenderer(config=config, console=console).render(analysis, transcription)
    return buf.getvalue()

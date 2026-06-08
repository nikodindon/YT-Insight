"""
File writers for :class:`AnalysisResult` (and optional
:class:`TranscriptionResult`).

Two formats are supported out of the box:

- **Markdown** — human-readable, with YAML front matter, sections,
  timestamped transcript.
- **JSON**     — strict serialization of the dataclasses for
  downstream tooling (Notebook, DB import, etc.).

The :class:`FileWriter` is idempotent: re-running the analysis on the
same input will overwrite the existing files (filenames are
deterministic from the video title + date).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yt_insight.analyzer.base import AnalysisResult, Quote
    from yt_insight.transcriber import TranscriptionResult
    from yt_insight.downloader import VideoMetadata


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class FileWriter:
    """
    Save an analysis to disk as Markdown and/or JSON.

    Parameters
    ----------
    output_dir:
        Root directory for outputs. Created if missing.
    include_transcript:
        Whether to embed the full transcript in the Markdown file.
    include_timestamps:
        Whether to include ``[HH:MM:SS]`` prefixes in the embedded
        transcript.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        include_transcript: bool = True,
        include_timestamps: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_transcript = include_transcript
        self.include_timestamps = include_timestamps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_path(self, title: str, ext: str, tag: str | None = None) -> Path:
        """
        Compute the output path for a given title and extension.

        If ``tag`` is provided, it is always appended to the
        filename. This guarantees that re-running the pipeline
        with a different config (model, max_prompt_tokens, …)
        produces a distinct output and never overwrites an
        earlier one.
        """
        slug = _slugify(title or "untitled")
        date = _today()
        base = self.output_dir / f"{date}_{slug}"
        if tag:
            return Path(f"{base}-{tag}.{ext}")
        return Path(f"{base}.{ext}")

    def write_markdown(
        self,
        analysis: "AnalysisResult",
        *,
        title: str = "",
        video_url: str = "",
        metadata: "VideoMetadata | None" = None,
        transcription: "TranscriptionResult | None" = None,
        tag: str | None = None,
    ) -> Path:
        """Write a Markdown report. Returns the path of the file written."""
        path = self._build_path(title, "md", tag=tag)
        path.write_text(
            self._render_markdown(
                analysis,
                title=title,
                video_url=video_url,
                metadata=metadata,
                transcription=transcription,
            ),
            encoding="utf-8",
        )
        return path

    def write_json(
        self,
        analysis: "AnalysisResult",
        *,
        title: str = "",
        video_url: str = "",
        metadata: "VideoMetadata | None" = None,
        transcription: "TranscriptionResult | None" = None,
        tag: str | None = None,
    ) -> Path:
        """Write a strict-JSON dump. Returns the path of the file written."""
        path = self._build_path(title, "json", tag=tag)
        payload = self._build_json_payload(
            analysis,
            title=title,
            video_url=video_url,
            metadata=metadata,
            transcription=transcription,
        )
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    def write_both(
        self,
        analysis: "AnalysisResult",
        **kwargs: Any,
    ) -> dict[str, Path]:
        """Write Markdown + JSON. Returns a dict of ``{"markdown": ..., "json": ...}``."""
        return {
            "markdown": self.write_markdown(analysis, **kwargs),
            "json": self.write_json(analysis, **kwargs),
        }

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------

    def _render_markdown(
        self,
        analysis: "AnalysisResult",
        *,
        title: str,
        video_url: str,
        metadata: "VideoMetadata | None",
        transcription: "TranscriptionResult | None",
    ) -> str:
        lines: list[str] = []
        lines.append(f"# {title or 'Analyse YT-Insight'}")
        lines.append("")

        # --- Front matter ------------------------------------------------
        front_matter = {
            "title": title or "(non renseigné)",
            "url": video_url or "(non renseignée)",
            "date": _today_iso(),
            "model": analysis.model_name or "?",
            "backend": analysis.backend or "?",
        }
        if transcription is not None:
            front_matter["language"] = transcription.language
            front_matter["duration"] = transcription.duration_str
        if analysis.topic:
            front_matter["topic"] = analysis.topic
        if analysis.tone:
            front_matter["tone"] = analysis.tone
        if analysis.audience:
            front_matter["audience"] = analysis.audience
        if metadata is not None:
            front_matter["channel"] = metadata.channel
            front_matter["video_id"] = metadata.video_id
        lines.append("```yaml")
        for k, v in front_matter.items():
            lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append("---")
        lines.append("")

        # --- Sections ----------------------------------------------------
        if analysis.summary:
            lines.append("## 📝 Résumé détaillé")
            lines.append("")
            lines.append(analysis.summary)
            lines.append("")
            lines.append("---")
            lines.append("")

        if analysis.key_points:
            lines.append("## 🎯 Points clés")
            lines.append("")
            for i, p in enumerate(analysis.key_points, start=1):
                lines.append(f"{i}. **{p}**")
            lines.append("")
            lines.append("---")
            lines.append("")

        if analysis.analysis:
            lines.append("## 🔍 Analyse approfondie")
            lines.append("")
            lines.append(analysis.analysis)
            lines.append("")
            lines.append("---")
            lines.append("")

        if analysis.quotes:
            lines.append("## 💬 Citations notables")
            lines.append("")
            for q in analysis.quotes:
                ts = q.timestamp_str or "?"
                lines.append(f"> {q.text}")
                lines.append(f">")
                lines.append(f"> — *{ts}*")
                lines.append("")
            lines.append("---")
            lines.append("")

        if self.include_transcript and transcription is not None and transcription.text:
            lines.append("## 📄 Transcription complète")
            lines.append("")
            formatted = transcription.formatted_transcript(
                with_timestamps=self.include_timestamps,
            )
            lines.append("```text")
            lines.append(formatted)
            lines.append("```")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _build_json_payload(
        self,
        analysis: "AnalysisResult",
        *,
        title: str,
        video_url: str,
        metadata: "VideoMetadata | None",
        transcription: "TranscriptionResult | None",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "title": title,
            "url": video_url,
            "date": _today_iso(),
            "analysis": analysis.to_dict(),
        }
        if metadata is not None:
            payload["metadata"] = {
                "video_id": metadata.video_id,
                "title": metadata.title,
                "channel": metadata.channel,
                "duration_seconds": metadata.duration_seconds,
                "duration_str": metadata.duration_str,
                "upload_date": metadata.upload_date,
                "view_count": metadata.view_count,
                "url": metadata.url,
            }
        if transcription is not None:
            payload["transcription"] = {
                "language": transcription.language,
                "language_probability": transcription.language_probability,
                "duration_seconds": transcription.duration_seconds,
                "duration_str": transcription.duration_str,
                "estimated_tokens": transcription.estimated_tokens,
                "model_name": transcription.model_name,
                "text": transcription.text if self.include_transcript else None,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in transcription.segments
                ] if self.include_transcript else [],
            }
        return payload


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def write_outputs(
    analysis: "AnalysisResult",
    output_dir: Path,
    *,
    formats: list[str] | None = None,
    title: str = "",
    video_url: str = "",
    metadata: "VideoMetadata | None" = None,
    transcription: "TranscriptionResult | None" = None,
    include_transcript: bool = True,
    include_timestamps: bool = True,
    tag: str | None = None,
) -> dict[str, Path]:
    """
    One-shot helper: write the requested formats and return the paths.

    Parameters
    ----------
    formats:
        Any combination of ``"markdown"`` and ``"json"``.
        ``None`` defaults to ``["markdown", "json"]``.
    tag:
        Optional disambiguation tag appended to the filename when a
        file with the same base name already exists. Use this to keep
        multiple analyses of the same video side-by-side (e.g. one
        per model / config).
    """
    if formats is None:
        formats = ["markdown", "json"]

    writer = FileWriter(
        output_dir,
        include_transcript=include_transcript,
        include_timestamps=include_timestamps,
    )
    paths: dict[str, Path] = {}
    for fmt in formats:
        if fmt == "markdown":
            paths["markdown"] = writer.write_markdown(
                analysis, title=title, video_url=video_url,
                metadata=metadata, transcription=transcription,
                tag=tag,
            )
        elif fmt == "json":
            paths["json"] = writer.write_json(
                analysis, title=title, video_url=video_url,
                metadata=metadata, transcription=transcription,
                tag=tag,
            )
        else:
            raise ValueError(
                f"Unknown output format: {fmt!r}. Expected 'markdown' or 'json'."
            )
    return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str, max_len: int = 60) -> str:
    """Make *text* filesystem-safe (lowercase, no special chars, hyphens)."""
    import re
    s = text.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return (s[:max_len].strip("-") or "untitled")


def _today() -> str:
    """Date slug like ``2026-06-08``."""
    return datetime.now().strftime("%Y-%m-%d")


def _today_iso() -> str:
    """ISO 8601 timestamp for the YAML front matter."""
    return datetime.now().isoformat(timespec="seconds")

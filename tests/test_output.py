"""
Tests for ``yt_insight.output`` (console + file_writer).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yt_insight.analyzer import AnalysisResult, Quote
from yt_insight.output import (
    ConsoleRenderer,
    FileWriter,
    RenderConfig,
    render_to_string,
    write_outputs,
)
from yt_insight.transcriber import Segment, TranscriptionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analysis() -> AnalysisResult:
    return AnalysisResult(
        summary="Une vidéo passionnante sur l'IA et les transformers.",
        key_points=[
            "Les transformers ont révolutionné le NLP depuis 2017.",
            "Le mécanisme d'attention est au cœur du modèle.",
            "L'auto-attention permet de capturer les dépendances longue distance.",
            "Les modèles pré-entraînés permettent le transfer learning.",
            "Les LLMs généralistes émergent à partir du scale.",
            "L'alignement par RLHF rend les modèles plus utiles.",
            "Les hallucinations restent un défi majeur.",
            "L'avenir passe par le multimodal et les agents.",
        ],
        analysis=(
            "**Forces**\n\n"
            "- Clarté des explications.\n- Exemples concrets.\n\n"
            "**Concepts centraux**\n\n"
            "- Attention, embeddings, pré-entraînement.\n\n"
            "**Implications**\n\n"
            "- Évolution rapide du domaine."
        ),
        quotes=[
            Quote(text="L'attention est le mécanisme clé.", timestamp_seconds=120.0),
            Quote(text="Le scale a tout changé.", timestamp_seconds=600.0),
        ],
        topic="Intelligence Artificielle",
        tone="pédagogique",
        audience="développeurs Python",
        model_name="Qwen3.6-35B-A3B",
        backend="llamacpp-local",
    )


@pytest.fixture
def transcription() -> TranscriptionResult:
    return TranscriptionResult(
        text="Bonjour à tous. Aujourd'hui on parle d'IA. Les transformers sont partout.",
        segments=[
            Segment(start=0.0,  end=2.0,  text="Bonjour à tous."),
            Segment(start=2.0,  end=5.0,  text="Aujourd'hui on parle d'IA."),
            Segment(start=5.0,  end=10.0, text="Les transformers sont partout."),
        ],
        language="fr",
        language_probability=0.99,
        duration_seconds=10.0,
        model_name="large-v3",
    )


# ---------------------------------------------------------------------------
# ConsoleRenderer
# ---------------------------------------------------------------------------

class TestConsoleRenderer:
    def test_render_to_string_contains_sections(self, analysis):
        out = render_to_string(analysis)
        # Header has the model name and backend.
        assert "Qwen3.6-35B-A3B" in out
        assert "llamacpp-local" in out
        # Topic banner
        assert "Intelligence Artificielle" in out
        # Sections (titles appear in panel borders)
        assert "Résumé" in out
        assert "Points clés" in out
        assert "Analyse approfondie" in out
        assert "Citations" in out

    def test_render_to_string_quotes_table(self, analysis):
        out = render_to_string(analysis)
        # Both quote texts should appear, with their timestamps.
        assert "L'attention est le mécanisme clé." in out
        assert "Le scale a tout changé." in out
        assert "2:00" in out
        assert "10:00" in out

    def test_render_with_transcript_preview(self, analysis, transcription):
        cfg = RenderConfig(show_transcript=True, max_transcript_chars=200)
        out = render_to_string(analysis, transcription, config=cfg)
        assert "Transcription" in out
        assert "Bonjour à tous" in out

    def test_render_skips_disabled_sections(self, analysis):
        cfg = RenderConfig(
            show_summary=False,
            show_key_points=False,
            show_analysis=False,
            show_quotes=False,
            show_topic_banner=False,
        )
        out = render_to_string(analysis, config=cfg)
        assert "Résumé" not in out
        assert "Points clés" not in out
        assert "Analyse" not in out
        assert "Citations" not in out

    def test_renderer_uses_custom_console(self, analysis):
        from rich.console import Console
        import io

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120, color_system=None)
        renderer = ConsoleRenderer(console=console)
        renderer.render(analysis)
        assert "Qwen3.6-35B-A3B" in buf.getvalue()

    def test_render_with_metadata_block(self, analysis):
        cfg = RenderConfig(show_metadata=False)
        out = render_to_string(analysis, config=cfg)
        # Topic banner is gated on show_metadata — should be absent.
        assert "Sujet" not in out


# ---------------------------------------------------------------------------
# FileWriter — Markdown
# ---------------------------------------------------------------------------

class TestMarkdownWriter:
    def test_writes_markdown(self, analysis, tmp_path):
        writer = FileWriter(tmp_path, include_transcript=False)
        path = writer.write_markdown(analysis, title="L'IA en 2026", video_url="https://x")
        assert path.exists()
        assert path.suffix == ".md"
        content = path.read_text(encoding="utf-8")
        # Title heading
        assert "# L'IA en 2026" in content
        # YAML front matter
        assert "```yaml" in content
        assert "model: Qwen3.6-35B-A3B" in content
        # Sections
        assert "## 📝 Résumé détaillé" in content
        assert "## 🎯 Points clés" in content
        assert "## 🔍 Analyse approfondie" in content
        assert "## 💬 Citations notables" in content
        # Numbered key points
        assert "1. **Les transformers" in content
        # Quotes as blockquotes
        assert "> L'attention est le mécanisme clé." in content

    def test_slug_appears_in_filename(self, analysis, tmp_path):
        writer = FileWriter(tmp_path)
        path = writer.write_markdown(analysis, title="Comment fonctionne l'attention ?")
        # Slug is "comment-fonctionne-lattention" or similar.
        assert "comment-fonctionne" in path.name
        assert path.name.endswith(".md")

    def test_untitled_slug(self, analysis, tmp_path):
        writer = FileWriter(tmp_path)
        path = writer.write_markdown(analysis, title="")
        assert "untitled" in path.name

    def test_includes_transcript_section(self, analysis, transcription, tmp_path):
        writer = FileWriter(tmp_path, include_transcript=True)
        path = writer.write_markdown(
            analysis, title="X", video_url="", transcription=transcription,
        )
        content = path.read_text(encoding="utf-8")
        assert "## 📄 Transcription complète" in content
        assert "Bonjour à tous" in content
        # Timestamps are on by default
        assert "[0:00]" in content

    def test_no_transcript_when_disabled(self, analysis, transcription, tmp_path):
        writer = FileWriter(tmp_path, include_transcript=False)
        path = writer.write_markdown(
            analysis, title="X", video_url="", transcription=transcription,
        )
        content = path.read_text(encoding="utf-8")
        assert "## 📄 Transcription complète" not in content

    def test_timestamps_off(self, analysis, transcription, tmp_path):
        writer = FileWriter(tmp_path, include_transcript=True, include_timestamps=False)
        path = writer.write_markdown(
            analysis, title="X", transcription=transcription,
        )
        content = path.read_text(encoding="utf-8")
        assert "[0:00]" not in content
        assert "Bonjour à tous" in content  # text still there


# ---------------------------------------------------------------------------
# FileWriter — JSON
# ---------------------------------------------------------------------------

class TestJsonWriter:
    def test_writes_json(self, analysis, tmp_path):
        writer = FileWriter(tmp_path)
        path = writer.write_json(analysis, title="X", video_url="https://x")
        assert path.exists()
        assert path.suffix == ".json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["title"] == "X"
        assert data["url"] == "https://x"
        assert data["analysis"]["summary"] == analysis.summary
        assert len(data["analysis"]["key_points"]) == 8
        assert len(data["analysis"]["quotes"]) == 2
        assert data["analysis"]["quotes"][0]["text"] == "L'attention est le mécanisme clé."

    def test_json_with_transcription(self, analysis, transcription, tmp_path):
        writer = FileWriter(tmp_path, include_transcript=True)
        path = writer.write_json(
            analysis, title="X", transcription=transcription,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["transcription"]["language"] == "fr"
        assert data["transcription"]["duration_str"] == "0:10"
        assert "Bonjour à tous" in data["transcription"]["text"]
        assert len(data["transcription"]["segments"]) == 3

    def test_json_without_transcript(self, analysis, transcription, tmp_path):
        writer = FileWriter(tmp_path, include_transcript=False)
        path = writer.write_json(
            analysis, title="X", transcription=transcription,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["transcription"]["text"] is None
        assert data["transcription"]["segments"] == []


# ---------------------------------------------------------------------------
# write_outputs (one-shot)
# ---------------------------------------------------------------------------

class TestWriteOutputs:
    def test_default_formats(self, analysis, tmp_path):
        paths = write_outputs(analysis, tmp_path, title="X")
        assert "markdown" in paths
        assert "json" in paths
        assert paths["markdown"].exists()
        assert paths["json"].exists()

    def test_only_markdown(self, analysis, tmp_path):
        paths = write_outputs(
            analysis, tmp_path, title="X", formats=["markdown"],
        )
        assert "markdown" in paths
        assert "json" not in paths

    def test_only_json(self, analysis, tmp_path):
        paths = write_outputs(
            analysis, tmp_path, title="X", formats=["json"],
        )
        assert "json" in paths
        assert "markdown" not in paths

    def test_invalid_format_raises(self, analysis, tmp_path):
        with pytest.raises(ValueError, match="Unknown output format"):
            write_outputs(analysis, tmp_path, title="X", formats=["pdf"])

    def test_creates_output_dir(self, analysis, tmp_path):
        new_dir = tmp_path / "deep" / "nested" / "out"
        assert not new_dir.exists()
        write_outputs(analysis, new_dir, title="X")
        assert new_dir.exists()

    def test_with_video_metadata(self, analysis, tmp_path):
        # Build a minimal VideoMetadata-like object via duck typing
        # (to avoid a hard dep on downloader for the output tests).
        class _V:
            video_id = "abc"
            title = "T"
            channel = "C"
            duration_seconds = 100.0
            duration_str = "1:40"
            upload_date = "20260101"
            view_count = 42
            url = "https://x"

        paths = write_outputs(analysis, tmp_path, title="X", metadata=_V())
        md = paths["markdown"].read_text(encoding="utf-8")
        assert "channel: C" in md
        assert "video_id: abc" in md

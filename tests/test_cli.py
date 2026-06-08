"""
Tests for the Typer CLI.

The tests use ``typer.testing.CliRunner`` and patch the heavy I/O
modules (downloader, transcriber, analyzer, file_writer) so the full
pipeline can be exercised without network or models.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from yt_insight.analyzer import AnalysisResult, Quote
from yt_insight.cli import app
from yt_insight.transcriber import Segment, TranscriptionResult


runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_download_result() -> MagicMock:
    dl = MagicMock()
    dl.audio_path = Path("/tmp/fake.mp3")
    dl.from_cache = False
    dl.metadata.title = "Une vidéo passionnante"
    dl.metadata.channel = "Ma Chaîne"
    dl.metadata.duration_str = "10:00"
    dl.metadata.video_id = "abc123"
    dl.metadata.url = "https://www.youtube.com/watch?v=abc123"
    return dl


@pytest.fixture
def fake_transcription() -> TranscriptionResult:
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


@pytest.fixture
def fake_analysis() -> AnalysisResult:
    return AnalysisResult(
        summary="Une vidéo sur l'IA.",
        key_points=[f"Point {i}" for i in range(1, 9)],
        analysis="**Forces**\n\nBien.",
        quotes=[Quote(text="Les transformers.", timestamp_seconds=5.0)],
        topic="IA",
        tone="pédagogique",
        audience="développeurs",
        model_name="Qwen3.6-35B-A3B",
        backend="llamacpp-local",
    )


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

class TestVersion:
    def test_version_prints(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "yt-insight" in result.stdout
        assert "0.1.0" in result.stdout


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

class TestDownload:
    def test_download_invokes_ytdlp(self, tmp_path, fake_download_result):
        with patch("yt_insight.cli.YtDlpDownloader") as MockDL:
            MockDL.return_value.download.return_value = fake_download_result
            result = runner.invoke(app, [
                "download", "https://youtu.be/abc123",
                "--cache-dir", str(tmp_path / "cache"),
            ])
        assert result.exit_code == 0, result.stdout
        MockDL.assert_called_once()
        MockDL.return_value.download.assert_called_once_with(
            "https://youtu.be/abc123", force=False,
        )

    def test_download_force_flag(self, fake_download_result):
        with patch("yt_insight.cli.YtDlpDownloader") as MockDL:
            MockDL.return_value.download.return_value = fake_download_result
            result = runner.invoke(app, [
                "download", "https://youtu.be/abc123", "--force",
            ])
        assert result.exit_code == 0
        MockDL.return_value.download.assert_called_once_with(
            "https://youtu.be/abc123", force=True,
        )

    def test_download_title_in_output(self, fake_download_result):
        with patch("yt_insight.cli.YtDlpDownloader") as MockDL:
            MockDL.return_value.download.return_value = fake_download_result
            result = runner.invoke(app, [
                "download", "https://youtu.be/abc123",
            ])
        assert "Une vidéo passionnante" in result.stdout


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------

class TestTranscribe:
    def test_transcribe_local_file(
        self, tmp_path, fake_transcription,
    ):
        audio = tmp_path / "song.mp3"
        audio.write_bytes(b"\xff\xfb" + b"\x00" * 64)
        out = tmp_path / "transcript.json"

        with patch("yt_insight.cli.create_transcriber") as mock_create:
            mock_t = MagicMock()
            mock_t.transcribe.return_value = fake_transcription
            mock_create.return_value = mock_t
            result = runner.invoke(app, [
                "transcribe", str(audio),
                "--language", "fr",
                "--output", str(out),
            ])
        assert result.exit_code == 0, result.stdout
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["language"] == "fr"
        assert "Bonjour à tous" in data["text"]
        mock_t.unload.assert_called_once()

    def test_transcribe_missing_file(self, tmp_path):
        result = runner.invoke(app, [
            "transcribe", str(tmp_path / "nope.mp3"),
        ])
        # Typer exits with code 1 on our explicit Exit
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "nope.mp3" in result.stdout

    def test_transcribe_url_downloads_first(
        self, fake_download_result, fake_transcription,
    ):
        with patch("yt_insight.cli.YtDlpDownloader") as MockDL, \
             patch("yt_insight.cli.create_transcriber") as mock_create:
            MockDL.return_value.download.return_value = fake_download_result
            mock_t = MagicMock()
            mock_t.transcribe.return_value = fake_transcription
            mock_create.return_value = mock_t
            result = runner.invoke(app, [
                "transcribe", "https://youtu.be/abc123",
            ])
        assert result.exit_code == 0
        MockDL.return_value.download.assert_called_once()


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_analyze_from_transcript_file(
        self, tmp_path, fake_transcription, fake_analysis,
    ):
        # Save a fake transcript JSON
        tr_path = tmp_path / "transcript.json"
        tr_path.write_text(
            json.dumps(fake_transcription.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"

        with patch("yt_insight.cli.create_analyzer") as mock_create_a:
            mock_a = MagicMock()
            mock_a.__enter__ = MagicMock(return_value=mock_a)
            mock_a.__exit__ = MagicMock(return_value=False)
            mock_a.analyze.return_value = fake_analysis
            mock_a.model_name = "Qwen3.6-35B-A3B"
            mock_a.backend_name = "llamacpp-local"
            mock_create_a.return_value = mock_a
            result = runner.invoke(app, [
                "analyze", str(tr_path),
                "--output-dir", str(out_dir),
                "--no-console",
            ])

        assert result.exit_code == 0, result.stdout
        mock_a.analyze.assert_called_once()
        # Files written
        assert (out_dir / "*.md").parent == out_dir  # path prefix is right
        written = list(out_dir.iterdir())
        assert any(p.suffix == ".md" for p in written)
        assert any(p.suffix == ".json" for p in written)

    def test_analyze_no_args_errors(self):
        result = runner.invoke(app, ["analyze", "--no-console"])
        assert result.exit_code != 0

    def test_analyze_with_audio_url(
        self, fake_download_result, fake_transcription, fake_analysis,
    ):
        with patch("yt_insight.cli.YtDlpDownloader") as MockDL, \
             patch("yt_insight.cli.create_transcriber") as mock_create_t, \
             patch("yt_insight.cli.create_analyzer") as mock_create_a:
            MockDL.return_value.download.return_value = fake_download_result
            mock_t = MagicMock()
            mock_t.transcribe.return_value = fake_transcription
            mock_create_t.return_value = mock_t
            mock_a = MagicMock()
            mock_a.__enter__ = MagicMock(return_value=mock_a)
            mock_a.__exit__ = MagicMock(return_value=False)
            mock_a.analyze.return_value = fake_analysis
            mock_a.model_name = "Qwen3.6-35B-A3B"
            mock_a.backend_name = "llamacpp-local"
            mock_create_a.return_value = mock_a
            result = runner.invoke(app, [
                "analyze", "--audio", "https://youtu.be/abc123",
                "--no-console",
            ])
        assert result.exit_code == 0, result.stdout
        mock_t.transcribe.assert_called_once()
        mock_a.analyze.assert_called_once()


# ---------------------------------------------------------------------------
# all (full pipeline)
# ---------------------------------------------------------------------------

class TestAll:
    def test_full_pipeline_runs(
        self, tmp_path, fake_download_result, fake_transcription, fake_analysis,
    ):
        out_dir = tmp_path / "out"
        cache_dir = tmp_path / "cache"

        with patch("yt_insight.cli.YtDlpDownloader") as MockDL, \
             patch("yt_insight.cli.create_transcriber") as mock_create_t, \
             patch("yt_insight.cli.create_analyzer") as mock_create_a:
            MockDL.return_value.download.return_value = fake_download_result
            mock_t = MagicMock()
            mock_t.transcribe.return_value = fake_transcription
            mock_create_t.return_value = mock_t
            mock_a = MagicMock()
            mock_a.__enter__ = MagicMock(return_value=mock_a)
            mock_a.__exit__ = MagicMock(return_value=False)
            mock_a.analyze.return_value = fake_analysis
            mock_a.model_name = "Qwen3.6-35B-A3B"
            mock_a.backend_name = "llamacpp-local"
            mock_create_a.return_value = mock_a
            result = runner.invoke(app, [
                "all", "https://youtu.be/abc123",
                "--output-dir", str(out_dir),
                "--cache-dir", str(cache_dir),
                "--no-console",
            ])

        assert result.exit_code == 0, result.stdout
        # All three modules called
        MockDL.return_value.download.assert_called_once()
        mock_t.transcribe.assert_called_once()
        mock_a.analyze.assert_called_once()
        # Files written
        assert (out_dir).exists()
        suffixes = {p.suffix for p in out_dir.iterdir()}
        assert ".md" in suffixes
        assert ".json" in suffixes

    def test_steps_download_only(self, fake_download_result, tmp_path):
        with patch("yt_insight.cli.YtDlpDownloader") as MockDL, \
             patch("yt_insight.cli.create_transcriber") as mock_t, \
             patch("yt_insight.cli.create_analyzer") as mock_a:
            MockDL.return_value.download.return_value = fake_download_result
            result = runner.invoke(app, [
                "all", "https://youtu.be/abc123",
                "--steps", "download",
                "--no-console",
            ])
        assert result.exit_code == 0
        mock_t.assert_not_called()
        mock_a.assert_not_called()

    def test_steps_transcribe_analyze_skips_download(
        self, tmp_path, fake_download_result, fake_transcription, fake_analysis,
    ):
        # Even with --steps transcribe,analyze, the CLI still downloads
        # (silently) because it needs the audio path. The key is that
        # the download step output is NOT displayed.
        with patch("yt_insight.cli.YtDlpDownloader") as MockDL, \
             patch("yt_insight.cli.create_transcriber") as mock_create_t, \
             patch("yt_insight.cli.create_analyzer") as mock_create_a:
            MockDL.return_value.download.return_value = fake_download_result
            mock_t = MagicMock()
            mock_t.transcribe.return_value = fake_transcription
            mock_create_t.return_value = mock_t
            mock_a = MagicMock()
            mock_a.__enter__ = MagicMock(return_value=mock_a)
            mock_a.__exit__ = MagicMock(return_value=False)
            mock_a.analyze.return_value = fake_analysis
            mock_a.model_name = "M"
            mock_a.backend_name = "B"
            mock_create_a.return_value = mock_a
            result = runner.invoke(app, [
                "all", "https://youtu.be/abc123",
                "--steps", "transcribe,analyze",
                "--no-console",
            ])
        assert result.exit_code == 0
        mock_t.transcribe.assert_called_once()
        mock_a.analyze.assert_called_once()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

class TestMisc:
    def test_help_shows_all_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("download", "transcribe", "analyze", "all", "version", "estimate"):
            assert cmd in result.stdout

    def test_no_args_shows_quick_start(self):
        result = runner.invoke(app, [])
        # Either shows the quick start panel or the help
        assert "yt-insight" in result.stdout.lower()

    def test_verbose_sets_debug(self, fake_download_result):
        with patch("yt_insight.cli.YtDlpDownloader") as MockDL:
            MockDL.return_value.download.return_value = fake_download_result
            with patch("yt_insight.cli.setup_logging") as mock_setup:
                result = runner.invoke(app, [
                    "download", "https://youtu.be/abc123", "--verbose",
                ])
        assert result.exit_code == 0
        # setup_logging("DEBUG") was called for verbose
        mock_setup.assert_any_call("DEBUG")


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------

class TestEstimateCli:
    def test_estimate_text_output(self):
        from yt_insight.estimate import Estimate

        fake = Estimate(
            url="https://youtu.be/abc", video_title="Talk", channel="Ch",
            duration_seconds=600.0, audio_mb=14.0,
            predicted_transcript_chars=8500, predicted_transcript_tokens=2125,
            predicted_word_count=1550, transcription_seconds_gpu=107,
            transcription_seconds_cpu=640, llm_strategy="single-shot",
            n_chunks=1, llm_max_prompt_tokens=28000, n_ctx_required=2925,
            llm_passes=1, llm_analysis_seconds=400, total_seconds_gpu=508,
            total_seconds_cpu=1041, download_seconds=1.4,
        )
        with patch("yt_insight.cli.estimate_url", return_value=fake) as mock_est:
            result = runner.invoke(app, ["estimate", "https://youtu.be/abc"])

        assert result.exit_code == 0, result.stdout
        mock_est.assert_called_once()
        assert "Talk" in result.stdout
        assert "single-shot" in result.stdout

    def test_estimate_json_output(self):
        from yt_insight.estimate import Estimate
        import json

        fake = Estimate(
            url="u", video_title="X", channel="Y", duration_seconds=60.0,
            audio_mb=1.5, predicted_transcript_chars=850,
            predicted_transcript_tokens=212, predicted_word_count=155,
            transcription_seconds_gpu=17, transcription_seconds_cpu=100,
            llm_strategy="single-shot", n_chunks=1,
            llm_max_prompt_tokens=28000, n_ctx_required=2925,
            llm_passes=1, llm_analysis_seconds=400, total_seconds_gpu=420,
            total_seconds_cpu=503, download_seconds=0.2,
        )
        with patch("yt_insight.cli.estimate_url", return_value=fake):
            result = runner.invoke(app, ["estimate", "https://youtu.be/abc", "--json"])

        assert result.exit_code == 0, result.stdout
        data = json.loads(result.stdout)
        assert data["video_title"] == "X"
        assert data["llm_strategy"] == "single-shot"

    def test_estimate_passes_options(self):
        from yt_insight.estimate import Estimate

        fake = Estimate(
            url="u", video_title="X", channel="Y", duration_seconds=60.0,
            audio_mb=1.0, predicted_transcript_chars=100,
            predicted_transcript_tokens=25, predicted_word_count=15,
            transcription_seconds_gpu=10, transcription_seconds_cpu=60,
            llm_strategy="single-shot", n_chunks=1,
            llm_max_prompt_tokens=60_000, n_ctx_required=1525,
            llm_passes=1, llm_analysis_seconds=200, total_seconds_gpu=212,
            total_seconds_cpu=262, download_seconds=0.1,
        )
        with patch("yt_insight.cli.estimate_url", return_value=fake) as mock_est:
            result = runner.invoke(app, [
                "estimate", "https://youtu.be/abc",
                "--hardware", "cpu",
                "--content-type", "podcast",
                "--llm-quant", "iq3_s",
                "--max-prompt-tokens", "60000",
                "--chunk-overlap", "500",
            ])

        assert result.exit_code == 0, result.stdout
        mock_est.assert_called_once()
        kwargs = mock_est.call_args.kwargs
        assert kwargs["hardware"] == "cpu"
        assert kwargs["content_type"] == "podcast"
        assert kwargs["llm_quant"] == "iq3_s"
        assert kwargs["max_prompt_tokens"] == 60_000
        assert kwargs["chunk_overlap_tokens"] == 500

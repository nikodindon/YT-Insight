"""
Tests for ``yt_insight.utils.config`` and ``yt_insight.utils.logger``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from yt_insight.utils.config import (
    AnalysisConfig,
    AppConfig,
    OutputConfig,
    PathsConfig,
    PipelineConfig,
    TranscriptionConfig,
    load_config,
    transcript_fits_in_window,
)
from yt_insight.utils.logger import get_logger, setup_logging


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_paths_defaults(self):
        p = PathsConfig()
        assert p.output_dir == Path("./outputs")
        assert p.cache_dir == Path("./cache")
        assert p.keep_audio is False

    def test_pipeline_defaults(self):
        pl = PipelineConfig()
        assert "download" in pl.steps
        assert "transcribe" in pl.steps
        assert "analyze" in pl.steps
        assert pl.fail_fast is True

    def test_transcription_defaults(self):
        t = TranscriptionConfig()
        assert t.model == "large-v3"
        assert t.device == "auto"
        assert t.compute_type == "int8"
        assert t.beam_size == 5
        assert t.language is None

    def test_analysis_defaults(self):
        a = AnalysisConfig()
        assert a.max_transcript_tokens > 0
        assert a.overflow_strategy in ("chunk", "truncate", "summarize_chunks")

    def test_output_defaults(self):
        o = OutputConfig()
        assert "console" in o.formats
        assert o.include_transcript is True


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

class TestYamlLoading:
    def test_load_from_yaml(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "paths:\n"
            "  output_dir: ./my_outputs\n"
            "  cache_dir: ./my_cache\n"
            "  keep_audio: true\n"
            "transcription:\n"
            "  model: medium\n"
            "  language: fr\n"
            "  beam_size: 3\n"
            "analysis:\n"
            "  max_transcript_tokens: 50000\n"
            "output:\n"
            "  formats: [markdown, json]\n"
            "  include_timestamps: false\n",
            encoding="utf-8",
        )
        cfg = load_config(yaml_path, create_paths=False)
        assert cfg.paths.output_dir == Path("./my_outputs")
        assert cfg.paths.cache_dir == Path("./my_cache")
        assert cfg.paths.keep_audio is True
        assert cfg.transcription.model == "medium"
        assert cfg.transcription.language == "fr"
        assert cfg.transcription.beam_size == 3
        assert cfg.analysis.max_transcript_tokens == 50_000
        assert cfg.output.formats == ["markdown", "json"]
        assert cfg.output.include_timestamps is False

    def test_steps_as_csv_string(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "pipeline:\n  steps: download,transcribe\n", encoding="utf-8"
        )
        cfg = load_config(yaml_path, create_paths=False)
        assert cfg.pipeline.steps == ["download", "transcribe"]

    def test_invalid_yaml_returns_defaults(self, tmp_path, caplog):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(":\n  - this is not a mapping\n  - but a list\n",
                             encoding="utf-8")
        with caplog.at_level("WARNING"):
            cfg = load_config(yaml_path, create_paths=False)
        assert isinstance(cfg, AppConfig)
        # Falls back to defaults
        assert cfg.transcription.model == "large-v3"

    def test_missing_file_uses_defaults(self, tmp_path):
        # Point at a path that doesn't exist — should not error.
        cfg = load_config(tmp_path / "absent.yaml", create_paths=False)
        assert isinstance(cfg, AppConfig)
        assert cfg.transcription.model == "large-v3"

    def test_partial_yaml_keeps_other_defaults(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("transcription:\n  model: tiny\n", encoding="utf-8")
        cfg = load_config(yaml_path, create_paths=False)
        # Only model changed, everything else is default.
        assert cfg.transcription.model == "tiny"
        assert cfg.transcription.beam_size == 5


# ---------------------------------------------------------------------------
# Env overrides
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    def test_env_beats_yaml(self, tmp_path, monkeypatch):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("transcription:\n  model: tiny\n", encoding="utf-8")
        monkeypatch.setenv("WHISPER_MODEL", "large-v3")
        cfg = load_config(yaml_path, create_paths=False)
        assert cfg.transcription.model == "large-v3"

    def test_paths_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YT_INSIGHT_OUTPUT_DIR", str(tmp_path / "out"))
        monkeypatch.setenv("YT_INSIGHT_CACHE_DIR", str(tmp_path / "cache"))
        cfg = load_config(create_paths=True)
        assert cfg.paths.output_dir == tmp_path / "out"
        assert cfg.paths.cache_dir == tmp_path / "cache"
        assert (tmp_path / "out").exists()
        assert (tmp_path / "cache").exists()

    def test_boolean_envs(self, monkeypatch):
        monkeypatch.setenv("YT_INSIGHT_KEEP_AUDIO", "1")
        cfg = load_config(create_paths=False)
        assert cfg.paths.keep_audio is True

        monkeypatch.setenv("YT_INSIGHT_KEEP_AUDIO", "false")
        cfg = load_config(create_paths=False)
        assert cfg.paths.keep_audio is False

    def test_output_formats_csv(self, monkeypatch):
        monkeypatch.setenv("YT_INSIGHT_OUTPUT_FORMATS", "markdown,json")
        cfg = load_config(create_paths=False)
        assert cfg.output.formats == ["markdown", "json"]


# ---------------------------------------------------------------------------
# transcript_fits_in_window
# ---------------------------------------------------------------------------

class TestFitsInWindow:
    def test_fits(self):
        assert transcript_fits_in_window("Bonjour le monde", 100) is True

    def test_does_not_fit(self):
        text = "a" * 1000
        assert transcript_fits_in_window(text, 10) is False

    def test_empty_fits(self):
        assert transcript_fits_in_window("", 10) is True


# ---------------------------------------------------------------------------
# logger
# ---------------------------------------------------------------------------

class TestLogger:
    def test_setup_changes_level(self):
        setup_logging("DEBUG")
        # Add a memory handler so we can capture without going through
        # pytest's caplog (which doesn't intercept RichHandler output).
        log = get_logger("yt_insight.test_level")
        records = []

        class _ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        h = _ListHandler()
        log.addHandler(h)
        try:
            log.debug("hello-debug")
            log.info("hello-info")
        finally:
            log.removeHandler(h)
        msgs = [r.getMessage() for r in records]
        assert "hello-debug" in msgs
        assert "hello-info" in msgs

    def test_setup_respects_string_level(self):
        setup_logging("WARNING")
        log = get_logger("yt_insight.test_warn")
        records = []

        class _ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        h = _ListHandler()
        log.addHandler(h)
        # Note: we do NOT call log.setLevel(DEBUG) — the effective level
        # is governed by the root logger that setup_logging configured.
        try:
            log.debug("should-be-filtered")
            log.warning("should-pass")
        finally:
            log.removeHandler(h)
        msgs = [r.getMessage() for r in records]
        assert "should-be-filtered" not in msgs
        assert "should-pass" in msgs

    def test_get_logger_returns_named(self):
        log = get_logger("yt_insight.foo.bar")
        assert log.name == "yt_insight.foo.bar"

    def test_httpx_is_tamed(self):
        setup_logging("DEBUG")
        # httpx must be at WARNING regardless of root level
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("yt_dlp").level == logging.WARNING

    def test_idempotent(self):
        # Calling setup twice should not duplicate handlers.
        setup_logging("INFO")
        n1 = len(logging.getLogger().handlers)
        setup_logging("INFO")
        n2 = len(logging.getLogger().handlers)
        assert n1 == n2

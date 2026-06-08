"""
Centralized configuration loading for YT-Insight.

Sources, in increasing order of priority (later overrides earlier):
1. Built-in defaults (sane, no-op values).
2. A YAML file (default: ``./config.yaml``, override via env ``YT_INSIGHT_CONFIG``).
3. Environment variables (``YT_INSIGHT_*`` + module-specific like ``WHISPER_*``,
   ``LLAMACPP_*``).

The :func:`load_config` function is intentionally lightweight — it does
NOT bind to any specific module. Instead, callers pull the dataclass
they need (:class:`TranscriptionConfig`, :class:`AnalysisConfig`, etc.).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .text_utils import estimate_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    output_dir: Path = Path("./outputs")
    cache_dir: Path = Path("./cache")
    keep_audio: bool = False


@dataclass
class PipelineConfig:
    steps: list[str] = field(default_factory=lambda: ["download", "transcribe", "analyze"])
    fail_fast: bool = True


@dataclass
class TranscriptionConfig:
    model: str = "large-v3"
    device: str = "auto"            # "auto" | "cuda" | "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    language: str | None = None     # None = auto-detect
    vad_filter: bool = True
    word_timestamps: bool = False


@dataclass
class AnalysisConfig:
    max_transcript_tokens: int = 100_000
    overflow_strategy: str = "chunk"            # chunk | truncate | summarize_chunks
    outputs: list[str] = field(default_factory=lambda: [
        "summary", "key_points", "analysis", "quotes", "metadata",
    ])


@dataclass
class OutputConfig:
    formats: list[str] = field(default_factory=lambda: ["console", "markdown"])
    include_transcript: bool = True
    include_timestamps: bool = True


@dataclass
class AppConfig:
    """The full configuration tree."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def ensure_paths(self) -> None:
        """Create output_dir and cache_dir if they don't exist."""
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.cache_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# Map of dotted YAML keys to (dataclass, key) pairs. Kept explicit
# (no clever auto-binding) so the schema is documented in one place.
_KEY_MAP: list[tuple[str, dataclass, str]] = [
    # (yaml key, dataclass instance, attribute name)
    ("paths.output_dir",       PathsConfig,        "output_dir"),
    ("paths.cache_dir",        PathsConfig,        "cache_dir"),
    ("paths.keep_audio",       PathsConfig,        "keep_audio"),
    ("pipeline.steps",         PipelineConfig,     "steps"),
    ("pipeline.fail_fast",     PipelineConfig,     "fail_fast"),
    ("transcription.model",          TranscriptionConfig, "model"),
    ("transcription.device",         TranscriptionConfig, "device"),
    ("transcription.compute_type",   TranscriptionConfig, "compute_type"),
    ("transcription.beam_size",      TranscriptionConfig, "beam_size"),
    ("transcription.language",       TranscriptionConfig, "language"),
    ("transcription.vad_filter",     TranscriptionConfig, "vad_filter"),
    ("transcription.word_timestamps", TranscriptionConfig, "word_timestamps"),
    ("analysis.max_transcript_tokens", AnalysisConfig,    "max_transcript_tokens"),
    ("analysis.overflow_strategy",     AnalysisConfig,    "overflow_strategy"),
    ("analysis.outputs",               AnalysisConfig,    "outputs"),
    ("output.formats",            OutputConfig, "formats"),
    ("output.include_transcript", OutputConfig, "include_transcript"),
    ("output.include_timestamps", OutputConfig, "include_timestamps"),
]


def _cast_value(dc_field_name: str, raw: Any) -> Any:
    """Best-effort casting of a YAML/env value to the dataclass field type."""
    if raw is None:
        return None
    if dc_field_name == "output_dir" or dc_field_name == "cache_dir":
        return Path(str(raw))
    if dc_field_name == "steps" or dc_field_name == "outputs" or dc_field_name == "formats":
        if isinstance(raw, list):
            return [str(x) for x in raw]
        return [s.strip() for s in str(raw).split(",") if s.strip()]
    if dc_field_name in ("keep_audio", "fail_fast", "vad_filter", "word_timestamps",
                          "include_transcript", "include_timestamps"):
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if dc_field_name in ("beam_size", "max_transcript_tokens"):
        return int(raw)
    if dc_field_name == "language":
        s = str(raw).strip()
        return s or None
    return raw


def _apply_yaml(config: AppConfig, raw: dict[str, Any]) -> None:
    """Recursively merge *raw* YAML data into *config*."""
    if not isinstance(raw, dict):
        return
    for dotted, dc_cls, attr in _KEY_MAP:
        if "." in dotted:
            section, key = dotted.split(".", 1)
            section_dict = raw.get(section) or {}
            if not isinstance(section_dict, dict):
                continue
            if key in section_dict:
                value = _cast_value(attr, section_dict[key])
                if value is not None:
                    setattr(_get_section(config, section), attr, value)
        else:
            if dotted in raw:
                value = _cast_value(attr, raw[dotted])
                if value is not None:
                    setattr(config, attr, value)


def _get_section(config: AppConfig, name: str) -> Any:
    return {
        "paths": config.paths,
        "pipeline": config.pipeline,
        "transcription": config.transcription,
        "analysis": config.analysis,
        "output": config.output,
    }[name]


def _apply_env(config: AppConfig) -> None:
    """Override config values from environment variables (highest priority)."""
    env_mappings: list[tuple[str, str, str]] = [
        # (env var, section, attr)
        ("YT_INSIGHT_OUTPUT_DIR",        "paths",         "output_dir"),
        ("YT_INSIGHT_CACHE_DIR",         "paths",         "cache_dir"),
        ("YT_INSIGHT_KEEP_AUDIO",        "paths",         "keep_audio"),
        ("YT_INSIGHT_FAIL_FAST",         "pipeline",      "fail_fast"),
        ("WHISPER_MODEL",                "transcription", "model"),
        ("WHISPER_DEVICE",               "transcription", "device"),
        ("WHISPER_COMPUTE_TYPE",         "transcription", "compute_type"),
        ("WHISPER_BEAM_SIZE",            "transcription", "beam_size"),
        ("WHISPER_LANGUAGE",             "transcription", "language"),
        ("WHISPER_VAD_FILTER",           "transcription", "vad_filter"),
        ("YT_INSIGHT_MAX_TOKENS",        "analysis",      "max_transcript_tokens"),
        ("YT_INSIGHT_OVERFLOW",          "analysis",      "overflow_strategy"),
        ("YT_INSIGHT_OUTPUT_FORMATS",    "output",        "formats"),
    ]
    for env_var, section, attr in env_mappings:
        raw = os.getenv(env_var)
        if raw is None:
            continue
        section_obj = _get_section(config, section)
        value = _cast_value(attr, raw)
        if value is not None:
            setattr(section_obj, attr, value)


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse %s: %s — using defaults", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("%s did not contain a YAML mapping — using defaults", path)
        return {}
    return data


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    *,
    create_paths: bool = True,
) -> AppConfig:
    """
    Build an :class:`AppConfig` from defaults + YAML + env vars.

    Parameters
    ----------
    config_path:
        Path to a YAML file. If ``None``, uses ``YT_INSIGHT_CONFIG`` env var
        or falls back to ``./config.yaml``. Missing files are not an error.
    create_paths:
        If ``True`` (default), call :meth:`AppConfig.ensure_paths` to
        create output/cache directories.
    """
    config = AppConfig()

    if config_path is None:
        config_path = os.getenv("YT_INSIGHT_CONFIG", "./config.yaml")
    yaml_data = _load_yaml_file(Path(config_path))
    if yaml_data:
        _apply_yaml(config, yaml_data)
    _apply_env(config)

    if create_paths:
        config.ensure_paths()

    logger.debug("Loaded config: %s", config)
    return config


# ---------------------------------------------------------------------------
# Helpers used by other modules
# ---------------------------------------------------------------------------

def transcript_fits_in_window(
    text: str,
    max_tokens: int,
) -> bool:
    """Return True if *text* estimated-tokens count is within *max_tokens*."""
    return estimate_tokens(text) <= max_tokens

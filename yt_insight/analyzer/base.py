"""
Abstract base class for LLM analysis backends.

Any LLM provider (llama.cpp local, OpenAI, Anthropic Claude, …) must
subclass :class:`BaseAnalyzer` and implement :meth:`analyze`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yt_insight.transcriber import TranscriptionResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    """A notable quote extracted from the transcript.

    ``timestamp_seconds`` is ``None`` when the analyzer cannot tie the
    quote to a specific moment in the video (e.g. short cloud-only
    backends that don't see Whisper segments).
    """
    text: str
    timestamp_seconds: float | None = None
    speaker: str | None = None   # best-effort; often unknown

    @property
    def timestamp_str(self) -> str | None:
        if self.timestamp_seconds is None:
            return None
        return _seconds_to_str(self.timestamp_seconds)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "timestamp_seconds": self.timestamp_seconds,
            "timestamp_str": self.timestamp_str,
            "speaker": self.speaker,
        }


@dataclass
class AnalysisResult:
    """Structured output produced by an LLM analyzer."""
    summary: str = ""                                       # 500-1000 words
    key_points: list[str] = field(default_factory=list)
    analysis: str = ""                                    # deeper analysis
    quotes: list[Quote] = field(default_factory=list)
    topic: str = ""                                       # short topic tag
    tone: str = ""                                        # e.g. "informatif"
    audience: str = ""                                    # target audience
    model_name: str = ""                                  # e.g. "Qwen3.6-35B-A3B"
    backend: str = ""                                     # e.g. "llamacpp-local"

    # --- Derived helpers ---------------------------------------------------

    @property
    def has_content(self) -> bool:
        return bool(self.summary or self.key_points or self.analysis)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "key_points": list(self.key_points),
            "analysis": self.analysis,
            "quotes": [q.to_dict() for q in self.quotes],
            "topic": self.topic,
            "tone": self.tone,
            "audience": self.audience,
            "model_name": self.model_name,
            "backend": self.backend,
        }


# ---------------------------------------------------------------------------
# Abstract analyzer
# ---------------------------------------------------------------------------

class BaseAnalyzer(ABC):
    """
    Contract that every analysis backend must fulfil.

    Subclasses must implement :meth:`analyze`. They should also:
    - Manage their own HTTP client lifecycle (``close()`` is provided)
    - Expose ``model_name`` so downstream code can label outputs
    """

    @abstractmethod
    def analyze(
        self,
        transcription: "TranscriptionResult",
        *,
        title: str = "",
        language: str | None = None,
    ) -> AnalysisResult:
        """
        Run the full analysis pipeline on *transcription* and return an
        :class:`AnalysisResult`.

        Parameters
        ----------
        transcription:
            The Whisper output to analyze.
        title:
            Video title, if known. Helps the LLM contextualize.
        language:
            ISO 639-1 code of the transcript language. If provided,
            prompts are written in that language; otherwise the analyzer
            falls back to French.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier of the model behind the backend."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short slug identifying the backend ('llamacpp-local', …)."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release any resources held by the backend (HTTP clients, etc.)."""


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _seconds_to_str(seconds: float) -> str:
    """Format a duration in seconds as ``H:MM:SS`` or ``M:SS``."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"

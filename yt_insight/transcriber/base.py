"""
Abstract base class for all transcriber backends.

Any new transcription engine (whisper.cpp, Vosk, cloud API…) must
subclass BaseTranscriber and implement `transcribe()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One timed chunk of speech, as returned by Whisper."""
    start: float          # seconds
    end: float            # seconds
    text: str

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    @property
    def start_str(self) -> str:
        """Human-readable timestamp, e.g. '1:23:45'."""
        return _seconds_to_str(self.start)

    @property
    def end_str(self) -> str:
        return _seconds_to_str(self.end)


@dataclass
class TranscriptionResult:
    """Everything the rest of the pipeline needs after transcription."""
    text: str                          # Full plain-text transcript
    segments: list[Segment]            # Timestamped segments
    language: str                      # ISO 639-1 code, e.g. "fr"
    language_probability: float        # 0.0 – 1.0
    duration_seconds: float
    model_name: str                    # e.g. "large-v3"

    # --- Derived helpers ---------------------------------------------------

    @property
    def estimated_tokens(self) -> int:
        """Rough token count (chars / 4). Good enough for chunking decisions."""
        return len(self.text) // 4

    @property
    def duration_str(self) -> str:
        return _seconds_to_str(self.duration_seconds)

    def formatted_transcript(self, with_timestamps: bool = True) -> str:
        """
        Return the transcript as a single string, optionally with per-segment
        timestamps in [HH:MM:SS] format.
        """
        if not with_timestamps:
            return self.text

        lines = []
        for seg in self.segments:
            lines.append(f"[{seg.start_str}] {seg.text.strip()}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration_seconds": self.duration_seconds,
            "duration_str": self.duration_str,
            "estimated_tokens": self.estimated_tokens,
            "model_name": self.model_name,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in self.segments
            ],
        }


# ---------------------------------------------------------------------------
# Abstract transcriber
# ---------------------------------------------------------------------------

class BaseTranscriber(ABC):
    """
    Contract that every transcription backend must fulfil.

    Subclasses must implement :meth:`transcribe`.
    They should also:
    - Load their model lazily (first call) or in ``__init__``
    - Free GPU memory after transcription via :meth:`unload`
    """

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe *audio_path* and return a :class:`TranscriptionResult`.

        Parameters
        ----------
        audio_path:
            Path to a local audio file (mp3, wav, m4a…).
        language:
            ISO 639-1 language code to skip auto-detection (faster).
            Pass ``None`` for automatic detection.
        """

    def unload(self) -> None:
        """
        Release any GPU/CPU memory held by the model.

        Called automatically by the pipeline after transcription so that
        the LLM backend can use the freed VRAM.
        Override in subclasses that hold a model in memory.
        """


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _seconds_to_str(seconds: float) -> str:
    """Convert a float number of seconds to 'H:MM:SS' or 'M:SS'."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"

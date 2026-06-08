"""Transcription backends (faster-whisper, future: whisper.cpp, cloud APIs)."""

from .base import BaseTranscriber, Segment, TranscriptionResult
from .faster_whisper_transcriber import (
    FasterWhisperTranscriber,
    TranscriptionError,
    create_transcriber,
)

__all__ = [
    "BaseTranscriber",
    "Segment",
    "TranscriptionResult",
    "FasterWhisperTranscriber",
    "TranscriptionError",
    "create_transcriber",
]

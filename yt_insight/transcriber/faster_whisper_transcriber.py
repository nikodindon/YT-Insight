"""
Transcriber backend using faster-whisper (CTranslate2-based Whisper).

Key design decisions
--------------------
- Model is loaded once and kept in memory until ``unload()`` is called.
- After transcription the model is explicitly deleted and ``torch.cuda``
  cache is cleared so the LLM backend (Qwen3 35B) can use the freed VRAM.
- CUDA int8 is the default: large-v3 fits in ~2–3 GB on a GTX 1650 Super.
- Automatic CPU fallback if CUDA is unavailable or VRAM is insufficient.
- ``beam_size`` is tunable: 5 gives the best quality/speed trade-off.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Literal

from .base import BaseTranscriber, Segment, TranscriptionResult

# ``WhisperModel`` is exposed at module level so tests can patch
# ``yt_insight.transcriber.faster_whisper_transcriber.WhisperModel``.
# If faster-whisper is not installed we fall back to ``None``; the real
# import is attempted lazily inside :meth:`_load_model`.
WhisperModel = None  # type: ignore
try:
    from faster_whisper import WhisperModel as _WhisperModel
    WhisperModel = _WhisperModel
except ImportError:  # pragma: no cover - exercised only without faster-whisper
    pass

logger = logging.getLogger(__name__)


# Supported compute types per device
_CUDA_COMPUTE_TYPES = ("int8", "int8_float16", "float16", "float32")
_CPU_COMPUTE_TYPES  = ("int8", "float32")

ModelSize = Literal[
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
    "distil-large-v2", "distil-large-v3",
]


class FasterWhisperTranscriber(BaseTranscriber):
    """
    Transcriber backed by ``faster-whisper``.

    Parameters
    ----------
    model_size:
        Whisper model variant. ``large-v3`` gives the best accuracy;
        ``medium`` is a good compromise on limited VRAM.
    device:
        ``"cuda"`` (recommended) or ``"cpu"``.
        ``"auto"`` (default) picks CUDA if available, else CPU.
    compute_type:
        Quantisation level. ``"int8"`` on CUDA is strongly recommended for
        GTX 1650 Super (4 GB VRAM) — large-v3 uses ~2.5 GB in that mode.
    beam_size:
        Beam search width. 5 is the default; lower = faster but less accurate.
    language:
        Default language (ISO 639-1). Can be overridden per call.
        ``None`` = auto-detect (adds ~1 s overhead).
    vad_filter:
        Enable voice-activity detection to skip silent regions. Speeds up
        transcription on videos with long pauses / music.

    Usage::

        t = FasterWhisperTranscriber(model_size="large-v3", device="auto")
        result = t.transcribe(Path("audio.mp3"))
        print(result.text)
        print(result.duration_str)   # "42:17"
        t.unload()                   # free VRAM before loading LLM
    """

    def __init__(
        self,
        model_size: ModelSize = "large-v3",
        device: Literal["auto", "cuda", "cpu"] = "auto",
        compute_type: str = "int8",
        beam_size: int = 3,
        language: str | None = None,
        vad_filter: bool = True,
        chunk_length: int | None = None,
    ):
        """
        Parameters
        ----------
        chunk_length:
            Max audio chunk length in seconds. ``None`` = let faster-whisper
            decide (default 30s). Lower this (e.g. 20) on GPUs with tight
            VRAM to avoid CUDA OOM during generation.
        """
        self.model_size   = model_size
        self.compute_type = compute_type
        self.beam_size    = beam_size
        self.language     = language
        self.vad_filter   = vad_filter
        self.chunk_length = chunk_length

        self._device = self._resolve_device(device)
        self._model  = None   # lazy-loaded on first transcribe()

        logger.info(
            "FasterWhisperTranscriber configured — model=%s device=%s compute=%s "
            "beam=%d chunk_length=%s",
            model_size, self._device, compute_type, beam_size,
            f"{chunk_length}s" if chunk_length else "auto",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,   # overrides instance default
    ) -> TranscriptionResult:
        """
        Transcribe *audio_path* and return a :class:`TranscriptionResult`.

        The model is loaded on the first call (lazy init).
        Subsequent calls reuse the loaded model.

        Parameters
        ----------
        audio_path:
            Path to the audio file. Any format supported by ffmpeg works
            (mp3, wav, m4a, ogg, flac…).
        language:
            Override the instance-level language for this call only.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        lang = language or self.language   # call-level > instance-level

        # Try CUDA first (or CPU if device=cpu), with OOM → CPU fallback
        return self._transcribe_with_oom_fallback(audio_path, lang)

    # ------------------------------------------------------------------
    # OOM-safe transcription core
    # ------------------------------------------------------------------

    def _transcribe_with_oom_fallback(
        self,
        audio_path: Path,
        lang: str | None,
    ) -> TranscriptionResult:
        """
        Run transcription, with automatic CUDA → CPU fallback on OOM.

        OOM can happen in two places:
        1. During model load (already handled in :meth:`_load_model`).
        2. During segment generation — this method handles that case by
           catching the error, unloading the CUDA model, and reloading
           on CPU. The audio file is re-transcribed from scratch (this
           is unavoidable since faster-whisper doesn't support resuming
           mid-stream).
        """
        # Make sure the right model is loaded for the current device.
        self._load_model()

        try:
            return self._run_transcription(audio_path, lang)
        except _CUDA_OOM_ERRORS as exc:
            if self._device != "cuda":
                # Already on CPU — nothing to fall back to.
                raise TranscriptionError(
                    f"CUDA OOM on CPU device (unexpected): {exc}"
                ) from exc
            logger.warning(
                "CUDA OOM during transcription (%s) — falling back to CPU. "
                "Re-transcribing the whole file (slower but reliable).",
                exc,
            )
            # Unload the CUDA model, reload on CPU, retry.
            self.unload()
            self._device = "cpu"
            self._load_model()
            return self._run_transcription(audio_path, lang)

    def _run_transcription(
        self,
        audio_path: Path,
        lang: str | None,
    ) -> TranscriptionResult:
        """Single transcription pass on the currently-loaded model."""
        t0 = time.time()
        logger.info(
            "Transcribing %s on %s (lang=%s, beam=%d, chunk_length=%s)…",
            audio_path.name, self._device, lang or "auto",
            self.beam_size,
            f"{self.chunk_length}s" if self.chunk_length else "auto",
        )

        segments_raw, info = self._model.transcribe(
            str(audio_path),
            language=lang,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            chunk_length=self.chunk_length,
            word_timestamps=False,   # segment-level is enough for our use case
        )

        # faster-whisper returns a generator — consume it inside the
        # try/except so we can catch CUDA OOM mid-stream.
        segments: list[Segment] = []
        full_text_parts: list[str] = []

        for seg in segments_raw:
            text = seg.text.strip()
            if text:
                segments.append(Segment(start=seg.start, end=seg.end, text=text))
                full_text_parts.append(text)

        full_text = " ".join(full_text_parts)
        elapsed = time.time() - t0

        logger.info(
            "Transcription done in %.1fs on %s — lang=%s (p=%.2f) "
            "segments=%d tokens≈%d",
            elapsed, self._device,
            info.language, info.language_probability,
            len(segments), len(full_text) // 4,
        )

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration_seconds=info.duration,
            model_name=self.model_size,
        )

    def unload(self) -> None:
        """
        Delete the model from memory and flush the CUDA cache.

        Call this after transcription so Qwen3 / any LLM backend can use
        the freed VRAM without OOM errors.
        """
        if self._model is not None:
            logger.info("Unloading Whisper model from memory…")
            del self._model
            self._model = None
            gc.collect()
            self._flush_cuda_cache()
            logger.info("Whisper model unloaded.")

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the faster-whisper model (only on first call)."""
        if self._model is not None:
            return

        logger.info(
            "Loading Whisper model '%s' on %s (%s)…",
            self.model_size, self._device, self.compute_type,
        )
        t0 = time.time()

        try:
            self._model = WhisperModel(
                self.model_size,
                device=self._device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            # VRAM too small → retry on CPU
            if self._device == "cuda":
                logger.warning(
                    "Failed to load on CUDA (%s) — falling back to CPU.", exc
                )
                self._device = "cpu"
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                )
            else:
                raise TranscriptionError(
                    f"Could not load Whisper model '{self.model_size}': {exc}"
                ) from exc

        elapsed = time.time() - t0
        logger.info("Model loaded in %.1fs.", elapsed)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu' based on availability."""
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA available — using GPU for transcription.")
                return "cuda"
        except ImportError:
            pass
        logger.info("CUDA not available — using CPU for transcription.")
        return "cpu"

    @staticmethod
    def _flush_cuda_cache() -> None:
        """Best-effort CUDA cache flush. No-op if torch is not installed."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache flushed.")
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_transcriber(
    model_size: ModelSize = "large-v3",
    device: str = "auto",
    compute_type: str = "int8",
    beam_size: int = 3,
    language: str | None = None,
    vad_filter: bool = True,
    chunk_length: int | None = None,
) -> FasterWhisperTranscriber:
    """
    Convenience factory — reads environment variables as defaults.

    Environment variables (all optional, override by keyword arg):
    - ``WHISPER_MODEL``         → model_size
    - ``WHISPER_DEVICE``        → device
    - ``WHISPER_COMPUTE_TYPE``  → compute_type
    - ``WHISPER_LANGUAGE``      → language (empty string = None)
    - ``WHISPER_CHUNK_LENGTH``  → chunk_length (seconds, empty = auto)
    """
    import os
    model_size   = os.getenv("WHISPER_MODEL",        model_size)
    device       = os.getenv("WHISPER_DEVICE",       device)
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", compute_type)
    lang_env     = os.getenv("WHISPER_LANGUAGE",     "")
    language     = lang_env if lang_env else language
    chunk_env    = os.getenv("WHISPER_CHUNK_LENGTH", "")
    if chunk_env:
        try:
            chunk_length = int(chunk_env)
        except ValueError:
            chunk_length = None

    return FasterWhisperTranscriber(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        language=language,
        vad_filter=vad_filter,
        chunk_length=chunk_length,
    )


# ---------------------------------------------------------------------------
# CUDA OOM detection (best-effort, optional torch)
# ---------------------------------------------------------------------------

def _build_cuda_oom_errors() -> tuple[type[BaseException], ...]:
    """
    Return a tuple of exception classes that signal a CUDA OOM.

    Tries to import torch and grab ``torch.cuda.OutOfMemoryError``. If torch
    is not installed, returns an empty tuple — the OOM fallback path
    simply won't be triggered (which is fine, since OOMs are impossible
    without CUDA).
    """
    errs: list[type[BaseException]] = [RuntimeError]   # fallback: any RuntimeError
    try:
        import torch
        # torch.cuda.OutOfMemoryError exists in modern torch (>=1.13).
        if hasattr(torch.cuda, "OutOfMemoryError"):
            oom = torch.cuda.OutOfMemoryError
            if isinstance(oom, type) and issubclass(oom, BaseException):
                errs.insert(0, oom)
    except ImportError:
        pass
    return tuple(errs)


#: Exceptions that trigger the CUDA → CPU fallback.
_CUDA_OOM_ERRORS: tuple[type[BaseException], ...] = _build_cuda_oom_errors()


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class TranscriptionError(Exception):
    """Raised when transcription fails for a non-recoverable reason."""

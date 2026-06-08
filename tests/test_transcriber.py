"""
Tests for yt_insight.transcriber

faster-whisper is mocked throughout — no model download required.

Run with:  pytest tests/test_transcriber.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from yt_insight.transcriber.base import (
    Segment,
    TranscriptionResult,
    _seconds_to_str,
)
from yt_insight.transcriber.faster_whisper_transcriber import (
    FasterWhisperTranscriber,
    TranscriptionError,
    create_transcriber,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def make_fake_segment(start=0.0, end=5.0, text="Hello world") -> SimpleNamespace:
    """Mimics a faster-whisper Segment namedtuple."""
    return SimpleNamespace(start=start, end=end, text=text)


def make_fake_info(language="fr", language_probability=0.99, duration=120.0):
    """Mimics faster-whisper's TranscriptionInfo."""
    return SimpleNamespace(
        language=language,
        language_probability=language_probability,
        duration=duration,
    )


def fake_model_transcribe(audio_path, **kwargs):
    """Return a generator of fake segments + fake info."""
    segments = [
        make_fake_segment(0.0,  4.5,  "Bonjour tout le monde."),
        make_fake_segment(5.0,  9.8,  "Aujourd'hui on parle de transformers."),
        make_fake_segment(10.2, 15.0, "C'est un sujet fascinant."),
    ]
    info = make_fake_info()
    return iter(segments), info


@pytest.fixture
def tmp_audio(tmp_path) -> Path:
    """Create a dummy audio file so path-existence checks pass."""
    p = tmp_path / "audio.mp3"
    p.write_bytes(b"\xff\xfb" + b"\x00" * 128)   # minimal fake mp3 header
    return p


@pytest.fixture
def transcriber():
    """Return a transcriber instance with device pre-set to cpu."""
    return FasterWhisperTranscriber(
        model_size="medium",
        device="cpu",
        compute_type="int8",
    )


# ---------------------------------------------------------------------------
# _seconds_to_str  (utility function)
# ---------------------------------------------------------------------------

class TestSecondsToStr:
    @pytest.mark.parametrize("secs, expected", [
        (0,      "0:00"),
        (59,     "0:59"),
        (60,     "1:00"),
        (75,     "1:15"),
        (3600,   "1:00:00"),
        (3661,   "1:01:01"),
        (3723,   "1:02:03"),
        (7384,   "2:03:04"),
    ])
    def test_formatting(self, secs, expected):
        assert _seconds_to_str(secs) == expected


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------

class TestSegment:
    def test_duration(self):
        s = Segment(start=10.0, end=13.5, text="Hi")
        assert s.duration == 3.5

    def test_start_str(self):
        s = Segment(start=125.0, end=130.0, text="Hi")
        assert s.start_str == "2:05"

    def test_end_str_with_hours(self):
        s = Segment(start=3600.0, end=3661.0, text="Hi")
        assert s.end_str == "1:01:01"


# ---------------------------------------------------------------------------
# TranscriptionResult
# ---------------------------------------------------------------------------

class TestTranscriptionResult:
    def _make_result(self) -> TranscriptionResult:
        return TranscriptionResult(
            text="Bonjour tout le monde. Aujourd'hui on parle de transformers.",
            segments=[
                Segment(0.0, 4.5, "Bonjour tout le monde."),
                Segment(5.0, 9.8, "Aujourd'hui on parle de transformers."),
            ],
            language="fr",
            language_probability=0.99,
            duration_seconds=120.0,
            model_name="large-v3",
        )

    def test_estimated_tokens(self):
        r = self._make_result()
        assert r.estimated_tokens == len(r.text) // 4

    def test_duration_str(self):
        r = self._make_result()
        assert r.duration_str == "2:00"

    def test_formatted_transcript_with_timestamps(self):
        r = self._make_result()
        out = r.formatted_transcript(with_timestamps=True)
        assert "[0:00]" in out
        assert "[0:05]" in out
        assert "Bonjour" in out

    def test_formatted_transcript_without_timestamps(self):
        r = self._make_result()
        out = r.formatted_transcript(with_timestamps=False)
        assert "[" not in out
        assert "Bonjour" in out

    def test_to_dict_keys(self):
        r = self._make_result()
        d = r.to_dict()
        for key in ("text", "language", "language_probability",
                    "duration_seconds", "duration_str",
                    "estimated_tokens", "model_name", "segments"):
            assert key in d

    def test_to_dict_segments_shape(self):
        r = self._make_result()
        segs = r.to_dict()["segments"]
        assert len(segs) == 2
        assert {"start", "end", "text"} == set(segs[0].keys())


# ---------------------------------------------------------------------------
# FasterWhisperTranscriber — device resolution
# ---------------------------------------------------------------------------

class TestDeviceResolution:
    def test_explicit_cpu(self):
        t = FasterWhisperTranscriber(device="cpu")
        assert t.device == "cpu"

    def test_explicit_cuda(self):
        t = FasterWhisperTranscriber(device="cuda")
        assert t.device == "cuda"

    def test_auto_resolves_to_cuda_when_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            t = FasterWhisperTranscriber(device="auto")
        assert t.device == "cuda"

    def test_auto_resolves_to_cpu_when_cuda_unavailable(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            t = FasterWhisperTranscriber(device="auto")
        assert t.device == "cpu"

    def test_auto_resolves_to_cpu_when_torch_missing(self):
        with patch.dict("sys.modules", {"torch": None}):
            t = FasterWhisperTranscriber(device="auto")
        assert t.device == "cpu"


# ---------------------------------------------------------------------------
# FasterWhisperTranscriber — lazy loading
# ---------------------------------------------------------------------------

class TestLazyLoading:
    def test_model_not_loaded_at_init(self, transcriber):
        assert transcriber.is_loaded is False

    def test_model_loaded_after_transcribe(self, transcriber, tmp_audio):
        mock_whisper_model = MagicMock()
        mock_whisper_model.transcribe = MagicMock(side_effect=fake_model_transcribe)

        with patch("yt_insight.transcriber.faster_whisper_transcriber.WhisperModel",
                   return_value=mock_whisper_model):
            transcriber.transcribe(tmp_audio)

        assert transcriber.is_loaded is True

    def test_model_reused_on_second_call(self, transcriber, tmp_audio):
        mock_whisper_model = MagicMock()
        mock_whisper_model.transcribe = MagicMock(side_effect=fake_model_transcribe)

        with patch("yt_insight.transcriber.faster_whisper_transcriber.WhisperModel",
                   return_value=mock_whisper_model) as MockClass:
            transcriber.transcribe(tmp_audio)
            transcriber.transcribe(tmp_audio)

        # WhisperModel() constructor called only ONCE
        assert MockClass.call_count == 1


# ---------------------------------------------------------------------------
# FasterWhisperTranscriber — transcribe()
# ---------------------------------------------------------------------------

class TestTranscribe:
    def _run(self, transcriber, tmp_audio, side_effect=None):
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(
            side_effect=side_effect or fake_model_transcribe
        )
        with patch(
            "yt_insight.transcriber.faster_whisper_transcriber.WhisperModel",
            return_value=mock_model,
        ):
            return transcriber.transcribe(tmp_audio)

    def test_returns_transcription_result(self, transcriber, tmp_audio):
        result = self._run(transcriber, tmp_audio)
        assert isinstance(result, TranscriptionResult)

    def test_text_is_joined_segments(self, transcriber, tmp_audio):
        result = self._run(transcriber, tmp_audio)
        assert "Bonjour" in result.text
        assert "transformers" in result.text

    def test_segments_count(self, transcriber, tmp_audio):
        result = self._run(transcriber, tmp_audio)
        assert len(result.segments) == 3

    def test_language_detected(self, transcriber, tmp_audio):
        result = self._run(transcriber, tmp_audio)
        assert result.language == "fr"
        assert result.language_probability == pytest.approx(0.99)

    def test_duration(self, transcriber, tmp_audio):
        result = self._run(transcriber, tmp_audio)
        assert result.duration_seconds == 120.0

    def test_model_name_recorded(self, transcriber, tmp_audio):
        result = self._run(transcriber, tmp_audio)
        assert result.model_name == "medium"

    def test_language_override_per_call(self, transcriber, tmp_audio):
        """Language passed at call level must be forwarded to the model."""
        mock_model = MagicMock()

        def _transcribe_check(audio_path, language=None, **kwargs):
            assert language == "en"
            return fake_model_transcribe(audio_path)

        mock_model.transcribe = MagicMock(side_effect=_transcribe_check)
        with patch(
            "yt_insight.transcriber.faster_whisper_transcriber.WhisperModel",
            return_value=mock_model,
        ):
            transcriber.transcribe(tmp_audio, language="en")

    def test_raises_on_missing_audio_file(self, transcriber):
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe(Path("/nonexistent/file.mp3"))

    def test_empty_segments_are_skipped(self, transcriber, tmp_audio):
        """Segments with empty/whitespace text must not appear in result."""
        def _with_empty(audio_path, **kwargs):
            segs = [
                make_fake_segment(0.0, 2.0, ""),
                make_fake_segment(2.0, 4.0, "   "),
                make_fake_segment(4.0, 6.0, "Valid text."),
            ]
            return iter(segs), make_fake_info()

        result = self._run(transcriber, tmp_audio, side_effect=_with_empty)
        assert len(result.segments) == 1
        assert result.segments[0].text == "Valid text."


# ---------------------------------------------------------------------------
# FasterWhisperTranscriber — unload()
# ---------------------------------------------------------------------------

class TestUnload:
    def test_unload_clears_model(self, transcriber, tmp_audio):
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(side_effect=fake_model_transcribe)

        with patch(
            "yt_insight.transcriber.faster_whisper_transcriber.WhisperModel",
            return_value=mock_model,
        ):
            transcriber.transcribe(tmp_audio)

        assert transcriber.is_loaded is True
        transcriber.unload()
        assert transcriber.is_loaded is False

    def test_unload_is_idempotent(self, transcriber):
        """Calling unload() when no model is loaded must not raise."""
        transcriber.unload()   # should be a no-op
        transcriber.unload()   # second call also fine

    def test_unload_flushes_cuda_cache(self, transcriber, tmp_audio):
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(side_effect=fake_model_transcribe)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch(
            "yt_insight.transcriber.faster_whisper_transcriber.WhisperModel",
            return_value=mock_model,
        ):
            transcriber.transcribe(tmp_audio)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            transcriber.unload()

        mock_torch.cuda.empty_cache.assert_called_once()


# ---------------------------------------------------------------------------
# FasterWhisperTranscriber — CUDA fallback
# ---------------------------------------------------------------------------

class TestCudaFallback:
    def test_falls_back_to_cpu_on_cuda_oom(self, tmp_audio):
        """If CUDA loading fails, must retry on CPU automatically."""
        transcriber = FasterWhisperTranscriber(device="cuda", compute_type="int8")

        call_count = {"n": 0}

        def _model_factory(model_size, device, compute_type):
            call_count["n"] += 1
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            m = MagicMock()
            m.transcribe = MagicMock(side_effect=fake_model_transcribe)
            return m

        with patch(
            "yt_insight.transcriber.faster_whisper_transcriber.WhisperModel",
            side_effect=_model_factory,
        ):
            result = transcriber.transcribe(tmp_audio)

        assert transcriber.device == "cpu"
        assert call_count["n"] == 2        # first CUDA attempt + CPU retry
        assert isinstance(result, TranscriptionResult)


# ---------------------------------------------------------------------------
# create_transcriber factory
# ---------------------------------------------------------------------------

class TestCreateTranscriber:
    def test_defaults(self):
        t = create_transcriber()
        assert isinstance(t, FasterWhisperTranscriber)

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("WHISPER_MODEL",        "medium")
        monkeypatch.setenv("WHISPER_DEVICE",       "cpu")
        monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float32")
        monkeypatch.setenv("WHISPER_LANGUAGE",     "fr")

        t = create_transcriber()
        assert t.model_size   == "medium"
        assert t.device       == "cpu"
        assert t.compute_type == "float32"
        assert t.language     == "fr"

    def test_empty_language_env_is_none(self, monkeypatch):
        monkeypatch.setenv("WHISPER_LANGUAGE", "")
        t = create_transcriber(language=None)
        assert t.language is None


# ---------------------------------------------------------------------------
# OOM fallback during transcription
# ---------------------------------------------------------------------------

class TestOomFallback:
    """Verify that a RuntimeError (CUDA OOM) during the generator
    consumption triggers an automatic CUDA → CPU fallback."""

    def test_oom_in_segment_consumption_falls_back_to_cpu(self, transcriber, tmp_audio):
        from yt_insight.transcriber.faster_whisper_transcriber import (
            TranscriptionError,
        )

        # Force device to cuda (we never actually use a real GPU).
        transcriber._device = "cuda"

        call_count = {"n": 0}

        def oom_then_ok(*args, **kwargs):
            """First call OOMs, second call returns the fake segments."""
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("CUDA failed with error out of memory")
            return iter([
                SimpleNamespace(start=0.0, end=2.0, text="Recovered on CPU"),
            ]), SimpleNamespace(
                language="en", language_probability=0.9, duration=2.0,
            )

        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(side_effect=oom_then_ok)

        with patch(
            "yt_insight.transcriber.faster_whisper_transcriber.WhisperModel",
            return_value=mock_model,
        ):
            with patch.object(transcriber, "unload") as mock_unload:
                result = transcriber.transcribe(tmp_audio)

        # We got 2 transcription calls (cuda fail + cpu ok).
        assert call_count["n"] == 2
        # Device should now be CPU after fallback.
        assert transcriber._device == "cpu"
        # Model was unloaded + reloaded between attempts.
        mock_unload.assert_called_once()
        # Result comes from the second (CPU) call.
        assert result.text == "Recovered on CPU"
        assert result.language == "en"

    def test_oom_on_cpu_raises_transcription_error(self, transcriber, tmp_audio):
        # If we're already on CPU and still OOM, raise cleanly.
        transcriber._device = "cpu"

        def always_oom(*args, **kwargs):
            raise RuntimeError("CUDA failed with error out of memory")

        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(side_effect=always_oom)

        with patch(
            "yt_insight.transcriber.faster_whisper_transcriber.WhisperModel",
            return_value=mock_model,
        ):
            with pytest.raises(Exception) as exc_info:
                transcriber.transcribe(tmp_audio)
        # Should be a TranscriptionError (or RuntimeError wrapping it).
        # We just check that the message mentions CPU and OOM.
        assert "OOM" in str(exc_info.value) or "CPU" in str(exc_info.value)

    def test_no_oom_path_works_as_before(self, transcriber, tmp_audio):
        """The non-OOM path should not trigger any fallback dance."""
        transcriber._device = "cuda"
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(side_effect=fake_model_transcribe)

        with patch(
            "yt_insight.transcriber.faster_whisper_transcriber.WhisperModel",
            return_value=mock_model,
        ):
            result = transcriber.transcribe(tmp_audio)

        # Should still be on cuda (no fallback happened).
        assert transcriber._device == "cuda"
        assert mock_model.transcribe.call_count == 1
        # And we got the normal transcription.
        assert "Bonjour tout le monde" in result.text


# ---------------------------------------------------------------------------
# chunk_length parameter
# ---------------------------------------------------------------------------

class TestChunkLength:
    """Verify that chunk_length is exposed and passed through to faster-whisper."""

    def test_default_chunk_length_is_none(self, transcriber):
        assert transcriber.chunk_length is None

    def test_custom_chunk_length_stored(self):
        t = FasterWhisperTranscriber(
            model_size="medium", device="cpu", compute_type="int8",
            chunk_length=15,
        )
        assert t.chunk_length == 15

    def test_chunk_length_passed_to_model(self, transcriber, tmp_audio):
        transcriber.chunk_length = 20
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(side_effect=fake_model_transcribe)

        with patch(
            "yt_insight.transcriber.faster_whisper_transcriber.WhisperModel",
            return_value=mock_model,
        ):
            transcriber.transcribe(tmp_audio, language="en")

        # Inspect the call kwargs to faster-whisper.
        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert call_kwargs.get("chunk_length") == 20

    def test_chunk_length_env_var(self, monkeypatch):
        monkeypatch.setenv("WHISPER_CHUNK_LENGTH", "25")
        t = create_transcriber()
        assert t.chunk_length == 25

    def test_chunk_length_env_var_invalid_falls_back_to_none(self, monkeypatch):
        monkeypatch.setenv("WHISPER_CHUNK_LENGTH", "not_a_number")
        t = create_transcriber()
        assert t.chunk_length is None

    def test_default_beam_size_is_3(self, transcriber):
        # We lowered the default from 5 to 3 to fit tight-VRAM GPUs.
        assert transcriber.beam_size == 3

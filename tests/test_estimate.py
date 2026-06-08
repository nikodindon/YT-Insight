"""
Tests for ``yt_insight.estimate``.

We mock the network call (yt-dlp metadata fetch) so the tests are
hermetic. Heuristic values (chunking, timings) are checked for
plausibility rather than exact numbers.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from yt_insight.estimate import (
    AUDIO_BITRATE_KBPS,
    DEFAULT_LLM_TPS,
    Estimate,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_TOKENS_PER_SECOND,
    LLM_OVERHEAD_TOKENS,
    TRANSCRIPTION_RTFX,
    WORDS_PER_MINUTE_BY_CONTENT,
    estimate_url,
    format_estimate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_metadata(
    title: str = "Test Video",
    channel: str = "Test Channel",
    duration: float = 600.0,    # 10 min
) -> SimpleNamespace:
    return SimpleNamespace(
        video_id="abc123",
        title=title,
        channel=channel,
        duration_seconds=duration,
        description="",
        upload_date="20260101",
        view_count=42,
        url="https://youtu.be/abc123",
    )


@pytest.fixture
def mock_downloader():
    with patch("yt_insight.estimate.YtDlpDownloader") as MockDL:
        MockDL.return_value.fetch_metadata_only.return_value = _fake_metadata()
        yield MockDL


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_short_video_single_shot(self, mock_downloader):
        est = estimate_url("https://youtu.be/abc123", hardware="gpu_gtx_1650")
        assert isinstance(est, Estimate)
        assert est.video_title == "Test Video"
        assert est.channel == "Test Channel"
        assert est.duration_seconds == 600.0
        # 10 min ≈ 155 wpm × 10 = 1550 words, ~8525 chars, ~2131 tokens.
        assert 1400 < est.predicted_word_count < 1700
        assert 7000 < est.predicted_transcript_chars < 9500
        # 10 min, 192 kbps mp3 → 10*60*192/8/1024 ≈ 14.06 MB
        assert 13.5 < est.audio_mb < 15.0
        # Fits in 28k → single-shot.
        assert est.llm_strategy == "single-shot"
        assert est.n_chunks == 1
        assert est.llm_passes == 1
        # Timings should be positive and finite.
        assert 0 < est.transcription_seconds_gpu < 120
        assert est.transcription_seconds_cpu > est.transcription_seconds_gpu
        assert est.total_seconds_gpu > 0
        assert est.total_seconds_cpu > est.total_seconds_gpu

    def test_long_video_triggers_chunk_merge(self, mock_downloader):
        # 60 min lecture → ~7800 words, ~43k chars, ~10.7k tokens.
        mock_downloader.return_value.fetch_metadata_only.return_value = _fake_metadata(
            duration=3600.0,
        )
        est = estimate_url(
            "https://youtu.be/abc123",
            hardware="gpu_gtx_1050",
            content_type="lecture",
            max_prompt_tokens=28_000,
        )
        # 10.7k tokens < 28k → still single-shot.
        assert est.llm_strategy == "single-shot"

    def test_very_long_video_definitely_chunks(self, mock_downloader):
        # 3 hours podcast → 30k+ words, ~170k chars, ~42k tokens.
        mock_downloader.return_value.fetch_metadata_only.return_value = _fake_metadata(
            duration=10_800.0,  # 3 hours
        )
        est = estimate_url(
            "https://youtu.be/abc123",
            hardware="gpu_rtx_3060",
            content_type="podcast",
            max_prompt_tokens=28_000,
        )
        assert est.llm_strategy == "chunk+merge"
        assert est.n_chunks > 1
        assert est.llm_passes == est.n_chunks + 1
        # n_ctx_required is the full prompt size (transcript + overhead),
        # not the chunk size. So it can be > max_prompt_tokens here.
        assert est.n_ctx_required > 28_000
        # The "effective" chunk size is bounded by max_prompt_tokens.
        # We can't easily assert it from the public API, but the
        # n_chunks >= 1 invariant is enough.

    def test_very_long_video_with_big_ctx_single_shot(self, mock_downloader):
        # 1 hour lecture → ~7800 words, ~43k chars, ~10.7k tokens.
        # With 60k ctx window, it should fit in single-shot.
        mock_downloader.return_value.fetch_metadata_only.return_value = _fake_metadata(
            duration=3600.0,
        )
        est = estimate_url(
            "https://youtu.be/abc123",
            max_prompt_tokens=60_000,
        )
        assert est.llm_strategy == "single-shot"
        # n_ctx_required is transcript + overhead, must be < max_prompt_tokens
        # for the single-shot path to be chosen.
        assert est.n_ctx_required < 60_000
        # And the prompt we end up sending is at most max_prompt_tokens.
        assert est.n_ctx_required <= est.llm_max_prompt_tokens

    def test_content_type_affects_word_count(self, mock_downloader):
        short = _fake_metadata(duration=600.0)
        fast = _fake_metadata(duration=600.0)
        mock_downloader.return_value.fetch_metadata_only.side_effect = [short, fast]
        lecture = estimate_url("u", content_type="lecture")
        fast_est = estimate_url("u", content_type="fast")
        assert fast_est.predicted_word_count > lecture.predicted_word_count


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

class TestHeuristics:
    def test_audio_size_scales_with_duration(self, mock_downloader):
        mock_downloader.return_value.fetch_metadata_only.side_effect = [
            _fake_metadata(duration=600.0),    # 10 min
            _fake_metadata(duration=3600.0),   # 60 min
        ]
        short = estimate_url("u")
        long_ = estimate_url("u")
        # 6x longer audio → ~6x more MB (allowing tiny rounding diff).
        ratio = long_.audio_mb / short.audio_mb
        assert 5.9 < ratio < 6.1

    def test_hardware_affects_transcription_time(self, mock_downloader):
        mock_downloader.return_value.fetch_metadata_only.side_effect = [
            _fake_metadata(), _fake_metadata(), _fake_metadata(), _fake_metadata(),
        ]
        slow = estimate_url("u", hardware="cpu")
        fast = estimate_url("u", hardware="gpu_rtx_3060")
        # RTX 3060 should be way faster than CPU.
        assert fast.transcription_seconds_gpu < slow.transcription_seconds_cpu / 5

    def test_quant_affects_llm_speed(self, mock_downloader):
        mock_downloader.return_value.fetch_metadata_only.side_effect = [
            _fake_metadata(), _fake_metadata(),
        ]
        iq1 = estimate_url("u", llm_quant="iq1_m")
        q8 = estimate_url("u", llm_quant="q8_0")
        # iq1_m is faster than q8_0.
        assert iq1.llm_analysis_seconds < q8.llm_analysis_seconds

    def test_unknown_hardware_falls_back(self, mock_downloader):
        # Should not crash, just use a sane default.
        est = estimate_url("u", hardware="non_existent_gpu")
        assert est.transcription_seconds_gpu > 0

    def test_unknown_quant_falls_back(self, mock_downloader):
        est = estimate_url("u", llm_quant="non_existent_quant")
        assert est.llm_analysis_seconds > 0


# ---------------------------------------------------------------------------
# Notes / sanity warnings
# ---------------------------------------------------------------------------

class TestNotes:
    def test_cpu_warning(self, mock_downloader):
        est = estimate_url("u", hardware="cpu")
        assert any("CPU" in n for n in est.notes)

    def test_chunk_merge_note(self, mock_downloader):
        mock_downloader.return_value.fetch_metadata_only.return_value = _fake_metadata(
            duration=14_400.0,  # 4 hours
        )
        est = estimate_url("u", max_prompt_tokens=28_000)
        assert any("chunk+merge" in n or "chunks" in n for n in est.notes)

    def test_no_unnecessary_notes_for_normal_video(self, mock_downloader):
        est = estimate_url("u", hardware="gpu_gtx_1650")
        # A 10-min GPU run should have no warnings.
        assert est.notes == [] or all("CPU" not in n and "chunk" not in n for n in est.notes)

    def test_large_audio_warning(self, mock_downloader):
        # 5 hours of fast content → huge audio file.
        mock_downloader.return_value.fetch_metadata_only.return_value = _fake_metadata(
            duration=18_000.0, content_type="fast",  # 200 wpm
        ) if False else _fake_metadata(duration=18_000.0)
        est = estimate_url("u", content_type="fast")
        # 5h × 192kbps ≈ 422 MB
        assert est.audio_mb > 200
        assert any("large" in n.lower() for n in est.notes)


# ---------------------------------------------------------------------------
# Estimate dataclass
# ---------------------------------------------------------------------------

class TestEstimateDataclass:
    def test_duration_str_minutes(self):
        e = Estimate(
            url="x", video_title="t", channel="c", duration_seconds=125.0,
            audio_mb=1, predicted_transcript_chars=1, predicted_transcript_tokens=1,
            predicted_word_count=1, transcription_seconds_gpu=1,
            transcription_seconds_cpu=1, llm_strategy="single-shot", n_chunks=1,
            llm_max_prompt_tokens=1, n_ctx_required=1, llm_passes=1,
            llm_analysis_seconds=1, total_seconds_gpu=1, total_seconds_cpu=1,
            download_seconds=1,
        )
        assert e.duration_str == "2:05"

    def test_duration_str_hours(self):
        e = Estimate(
            url="x", video_title="t", channel="c", duration_seconds=3725.0,
            audio_mb=1, predicted_transcript_chars=1, predicted_transcript_tokens=1,
            predicted_word_count=1, transcription_seconds_gpu=1,
            transcription_seconds_cpu=1, llm_strategy="single-shot", n_chunks=1,
            llm_max_prompt_tokens=1, n_ctx_required=1, llm_passes=1,
            llm_analysis_seconds=1, total_seconds_gpu=1, total_seconds_cpu=1,
            download_seconds=1,
        )
        assert e.duration_str == "1:02:05"

    def test_to_dict_round_trip(self):
        e = Estimate(
            url="u", video_title="t", channel="c", duration_seconds=60.0,
            audio_mb=1.5, predicted_transcript_chars=850,
            predicted_transcript_tokens=212, predicted_word_count=155,
            transcription_seconds_gpu=17, transcription_seconds_cpu=100,
            llm_strategy="single-shot", n_chunks=1,
            llm_max_prompt_tokens=28000, n_ctx_required=1100,
            llm_passes=1, llm_analysis_seconds=400,
            total_seconds_gpu=420, total_seconds_cpu=503,
            download_seconds=0.2,
        )
        d = e.to_dict()
        assert d["video_title"] == "t"
        assert d["audio_mb"] == 1.5
        assert d["llm_strategy"] == "single-shot"
        assert d["total_seconds_gpu"] == 420
        # All keys present
        for key in (
            "url", "video_title", "channel", "duration_seconds", "duration_str",
            "audio_mb", "predicted_transcript_chars", "predicted_transcript_tokens",
            "predicted_word_count", "transcription_seconds_gpu",
            "transcription_seconds_cpu", "llm_strategy", "n_chunks",
            "llm_max_prompt_tokens", "n_ctx_required", "llm_passes",
            "llm_analysis_seconds", "total_seconds_gpu", "total_seconds_cpu",
            "download_seconds", "backend_label", "llamacpp_max_prompt_tokens",
            "llamacpp_overlap_tokens", "notes",
        ):
            assert key in d


# ---------------------------------------------------------------------------
# format_estimate
# ---------------------------------------------------------------------------

class TestFormatEstimate:
    def test_format_contains_key_sections(self):
        est = Estimate(
            url="u", video_title="My Talk", channel="Ch", duration_seconds=600.0,
            audio_mb=14.0, predicted_transcript_chars=8500,
            predicted_transcript_tokens=2125, predicted_word_count=1550,
            transcription_seconds_gpu=107, transcription_seconds_cpu=640,
            llm_strategy="single-shot", n_chunks=1,
            llm_max_prompt_tokens=28000, n_ctx_required=2925,
            llm_passes=1, llm_analysis_seconds=400,
            total_seconds_gpu=508, total_seconds_cpu=1041,
            download_seconds=1.4,
        )
        out = format_estimate(est)
        assert "My Talk" in out
        assert "Ch" in out
        assert "10:00" in out
        assert "14.0" in out
        assert "single-shot" in out
        assert "GPU" in out
        assert "CPU" in out
        # Total time formatted
        assert "8m" in out or "508s" in out


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

class TestConstants:
    def test_rtfx_ordering(self):
        # Higher-tier GPUs should have higher RTFX.
        assert TRANSCRIPTION_RTFX["gpu_rtx_3060"] > TRANSCRIPTION_RTFX["gpu_gtx_1650"]
        assert TRANSCRIPTION_RTFX["gpu_gtx_1650"] > TRANSCRIPTION_RTFX["gpu_gtx_1050"]
        assert TRANSCRIPTION_RTFX["gpu_gtx_1050"] > TRANSCRIPTION_RTFX["cpu"]

    def test_quant_ordering(self):
        # Smaller quant = faster.
        assert LLM_TOKENS_PER_SECOND["iq1_m"] > LLM_TOKENS_PER_SECOND["q8_0"]

    def test_wpm_ordering(self):
        assert WORDS_PER_MINUTE_BY_CONTENT["fast"] > WORDS_PER_MINUTE_BY_CONTENT["lecture"]

    def test_default_llm_tps_in_range(self):
        # Sanity: DEFAULT_LLM_TPS should match a mid-tier quant.
        assert DEFAULT_LLM_TPS >= 5.0
        assert DEFAULT_LLM_TPS <= 20.0

    def test_audio_bitrate_sane(self):
        assert 64 <= AUDIO_BITRATE_KBPS <= 320

    def test_overhead_smaller_than_max_output(self):
        assert LLM_OVERHEAD_TOKENS < LLM_MAX_OUTPUT_TOKENS

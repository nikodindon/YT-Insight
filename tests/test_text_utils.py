"""
Tests for ``yt_insight.utils.text_utils``.

No third-party deps required.
"""

from __future__ import annotations

from yt_insight.transcriber.base import Segment
from yt_insight.utils.text_utils import (
    chunk_text,
    clean_transcript,
    estimate_tokens,
    format_transcript_with_timestamps,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 1  # min 1

    def test_short(self):
        # "Bonjour le monde" = 16 chars → 16//4 = 4
        assert estimate_tokens("Bonjour le monde") == 4

    def test_long_text(self):
        text = "a" * 4000
        assert estimate_tokens(text) == 1000

    def test_min_one(self):
        # Even a single character returns at least 1
        assert estimate_tokens("a") >= 1


# ---------------------------------------------------------------------------
# clean_transcript
# ---------------------------------------------------------------------------

class TestCleanTranscript:
    def test_collapses_spaces(self):
        assert clean_transcript("a    b\t\tc") == "a b c"

    def test_collapses_newlines(self):
        assert clean_transcript("a\n\n\n\nb") == "a\n\nb"

    def test_removes_bracket_artifacts(self):
        assert "[Musique]" not in clean_transcript("Salut [Musique] les amis")
        assert "[Applaudissements]" not in clean_transcript("Fin [Applaudissements]")

    def test_strips_lines(self):
        assert clean_transcript("  a  \n  b  \n") == "a\nb"

    def test_empty(self):
        assert clean_transcript("") == ""


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty_returns_empty(self):
        assert chunk_text("", max_tokens=100) == []

    def test_fits_in_one_chunk(self):
        text = "Bonjour le monde."
        assert chunk_text(text, max_tokens=100) == [text]

    def test_splits_long_text(self):
        # 200 sentences, max 5 tokens (≈20 chars) → many chunks
        text = "Bonjour. " * 200
        chunks = chunk_text(text, max_tokens=5)
        assert len(chunks) > 5
        for c in chunks:
            assert estimate_tokens(c) <= 5

    def test_no_overlap_by_default(self):
        text = "Une phrase. " * 100
        chunks = chunk_text(text, max_tokens=20)
        # Reconstructing should approximately equal the input length.
        joined = " ".join(chunks)
        assert "Une phrase" in joined

    def test_with_overlap_shares_content(self):
        text = "Alpha. Bravo. Charlie. Delta. Echo. Foxtrot."
        chunks_no = chunk_text(text, max_tokens=8, overlap_tokens=0)
        chunks_overlap = chunk_text(text, max_tokens=8, overlap_tokens=2)
        # Overlap should produce strictly more text overall.
        total_no = sum(len(c) for c in chunks_no)
        total_overlap = sum(len(c) for c in chunks_overlap)
        assert total_overlap > total_no

    def test_invalid_overlap(self):
        try:
            chunk_text("Bonjour.", max_tokens=10, overlap_tokens=10)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")

        try:
            chunk_text("Bonjour.", max_tokens=10, overlap_tokens=20)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for overlap > max")

    def test_invalid_max_tokens(self):
        try:
            chunk_text("Bonjour.", max_tokens=0)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")

    def test_handles_huge_sentence(self):
        # A single word longer than max_chars should still produce a chunk.
        huge = "x" * 1000
        chunks = chunk_text(huge, max_tokens=10)
        assert chunks
        # Reassembled should still contain all the x's.
        assert "".join(chunks).count("x") == 1000


# ---------------------------------------------------------------------------
# format_transcript_with_timestamps
# ---------------------------------------------------------------------------

class TestFormatTranscript:
    def test_basic(self):
        segments = [
            Segment(start=0.0, end=2.0, text="Bonjour"),
            Segment(start=3.5, end=5.0, text="le monde"),
        ]
        out = format_transcript_with_timestamps(segments)
        assert "[0:00] Bonjour" in out
        assert "[0:03] le monde" in out

    def test_skips_empty(self):
        segments = [
            Segment(start=0.0, end=1.0, text=""),
            Segment(start=2.0, end=3.0, text="Salut"),
        ]
        out = format_transcript_with_timestamps(segments)
        assert "Salut" in out
        assert "\n\n" not in out  # no double newlines

    def test_with_wrapping(self):
        segments = [
            Segment(start=0.0, end=10.0, text="a " * 50),  # 100 chars
        ]
        out = format_transcript_with_timestamps(segments, max_line_chars=20)
        # Should be wrapped to multiple lines.
        assert "\n" in out

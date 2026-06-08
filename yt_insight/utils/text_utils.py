"""
Text utilities for the pipeline.

Right now this module is consumed by the analyzer (chunking, token
estimation) and will be reused by the output module (timestamped
transcript rendering). Kept dependency-free (stdlib only) so it can be
imported anywhere.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yt_insight.transcriber.base import Segment


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

#: Rough heuristic: 1 token ≈ 4 characters for English / French mixed text.
#: Whisper / Qwen tokenizers are close enough to this for budgeting.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Fast, conservative token count for *text* (chars // 4)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Transcript cleanup
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"[ \t]+")
_NEWLINES_RE = re.compile(r"\n{3,}")
_BRACKET_ARTIFACTS_RE = re.compile(r"\[[^\]]*\]")  # e.g. [Musique], [Applaudissements]


def clean_transcript(text: str) -> str:
    """
    Light cleanup of Whisper-style raw transcripts.

    - Collapses runs of spaces/tabs to a single space.
    - Collapses 3+ consecutive newlines to 2.
    - Removes bracket-style artifacts (sound effects, music cues).
    - Strips leading/trailing whitespace on each line.
    """
    text = _BRACKET_ARTIFACTS_RE.sub("", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

#: Sentence-ending punctuation we use to find natural split points.
_SENTENCE_END_RE = re.compile(r"(?<=[\.\!\?\…])\s+")


def chunk_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[str]:
    """
    Split *text* into chunks of at most *max_tokens* estimated tokens.

    Splits are made at sentence boundaries (``.``, ``!``, ``?``, ``…``)
    wherever possible to keep semantic coherence. Whitespace inside a
    chunk is normalized.

    Parameters
    ----------
    text:
        The text to split.
    max_tokens:
        Hard upper bound on the number of tokens per chunk.
    overlap_tokens:
        Number of tokens of overlap between consecutive chunks, useful
        for maintaining context across chunk boundaries. ``0`` = no
        overlap (default).

    Returns
    -------
    list[str]
        A list of chunks. If *text* fits within *max_tokens*, returns
        ``[text]``. Empty input returns ``[]``.

    Raises
    ------
    ValueError
        If ``overlap_tokens >= max_tokens`` (would make chunks empty or
        negative).
    """
    text = text.strip()
    if not text:
        return []

    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
    if overlap_tokens < 0:
        raise ValueError(f"overlap_tokens must be >= 0, got {overlap_tokens}")
    if overlap_tokens >= max_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens})"
        )

    # Quick path: the whole text already fits.
    if estimate_tokens(text) <= max_tokens:
        return [text]

    # Split into sentences, keeping the trailing whitespace on each.
    sentences = _SENTENCE_END_RE.split(text)
    # Re-attach a single space so tokens don't fuse across boundaries.
    sentences = [s.strip() for s in sentences if s and s.strip()]

    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence is longer than the budget, we have to
        # hard-split it on whitespace.
        if sentence_len > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            for piece in _hard_split(sentence, max_chars):
                chunks.append(piece)
            continue

        if current_len + sentence_len + 1 > max_chars and current:
            chunks.append(" ".join(current))

            if overlap_chars > 0:
                # Carry the tail of the just-flushed chunk as overlap.
                tail = _take_tail_chars(" ".join(current), overlap_chars)
                current = [tail] if tail else []
                current_len = len(" ".join(current))
            else:
                current = []
                current_len = 0

        current.append(sentence)
        current_len += sentence_len + 1  # +1 for the joining space

    if current:
        chunks.append(" ".join(current))

    return chunks


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Split a sentence that is longer than ``max_chars`` on whitespace."""
    words = sentence.split()
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        word_len = len(word) + 1  # +1 for the joining space
        if current_len + word_len > max_chars and current:
            pieces.append(" ".join(current))
            current = []
            current_len = 0
        current.append(word)
        current_len += word_len
    if current:
        pieces.append(" ".join(current))
    return pieces


def _take_tail_chars(text: str, max_chars: int) -> str:
    """Return the last ``max_chars`` characters of *text*, aligned to a word."""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    # Snap forward to the next whitespace so we don't start mid-word.
    space = tail.find(" ")
    if 0 < space < len(tail) - 1:
        return tail[space + 1 :]
    return tail


# ---------------------------------------------------------------------------
# Timestamped transcript formatting
# ---------------------------------------------------------------------------

def format_transcript_with_timestamps(
    segments: list["Segment"],
    *,
    max_line_chars: int = 0,
) -> str:
    """
    Render a list of Whisper ``Segment`` objects as a single string,
    one segment per line, prefixed by its timestamp.

    Parameters
    ----------
    segments:
        The Whisper segments (see :class:`Segment` in
        :mod:`yt_insight.transcriber.base`).
    max_line_chars:
        If > 0, wrap individual segment texts at this many characters
        (soft wrap, on whitespace). ``0`` disables wrapping.
    """
    lines: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if max_line_chars > 0 and len(text) > max_line_chars:
            text = _soft_wrap(text, max_line_chars)
        lines.append(f"[{seg.start_str}] {text}")
    return "\n".join(lines)


def _soft_wrap(text: str, width: int) -> str:
    """Wrap *text* at the latest whitespace before *width*."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        word_len = len(word) + 1
        if current_len + word_len > width and current:
            lines.append(" ".join(current))
            current = []
            current_len = 0
        current.append(word)
        current_len += word_len
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)

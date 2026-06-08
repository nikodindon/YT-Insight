"""
Pre-flight cost estimation for a YouTube URL.

Runs the **downloader** just enough to fetch metadata (no audio
download), then uses simple heuristics to predict:

- audio file size on disk
- transcript character / token counts
- transcription time (CPU vs GPU)
- LLM strategy (single-shot vs chunk+merge)
- number of chunks and LLM analysis time
- grand total wall-clock estimate

Nothing heavy is loaded (no Whisper model, no LLM), so this is fast
and safe to run on a URL you're not sure you want to commit to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .downloader import YtDlpDownloader
from .utils.text_utils import estimate_tokens

if TYPE_CHECKING:
    from .analyzer.llamacpp_local import LlamaCppLocalAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------

#: Average spoken words per minute, by content type. Used to predict
#: transcript size from audio duration.
WORDS_PER_MINUTE_BY_CONTENT: dict[str, int] = {
    "lecture":     130,   # slow, clear, technical
    "podcast":     170,   # conversational
    "interview":   180,   # back-and-forth
    "talk":        155,   # tech talk / conference default
    "fast":        200,   # quick YouTube videos, vlogs
}
DEFAULT_WPM = 155   # fallback used when language/content is unknown

#: Average audio bitrate produced by yt-dlp at our default quality (192 kbps mp3).
AUDIO_BITRATE_KBPS = 192

#: Whisper speed heuristics: how many seconds of audio the transcriber
#: can chew per real-time second, on different hardware classes.
#: (Picked from typical faster-whisper benchmarks on these GPUs.)
TRANSCRIPTION_RTFX: dict[str, float] = {
    "gpu_gtx_1050":  6.0,
    "gpu_gtx_1650":  9.0,
    "gpu_rtx_3060": 18.0,
    "cpu":           1.0,   # slow fallback
}

#: LLM analysis speed: tokens generated per second on Qwen3.6 35B-A3B
#: with different quant levels. Conservative numbers.
LLM_TOKENS_PER_SECOND: dict[str, float] = {
    "iq1_m":   18.0,
    "iq3_s":   12.0,
    "iq4_xs":  10.0,
    "q4_k_m":   8.0,
    "q5_k_m":   6.0,
    "q6_k":     5.0,
    "q8_0":     4.0,
}
DEFAULT_LLM_TPS = 10.0  # safe default for IQ3/IQ4 range

#: LLM system prompt + analysis JSON output is roughly this many tokens
#: in addition to the transcript itself.
LLM_OVERHEAD_TOKENS = 1_500

#: LLM typical output size (max_tokens ceiling for the analysis JSON).
LLM_MAX_OUTPUT_TOKENS = 4_000


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Estimate:
    """All the predictions produced by :func:`estimate_url`."""

    # --- Input ---
    url: str
    video_title: str
    channel: str
    duration_seconds: float

    # --- Audio ---
    audio_mb: float                          # predicted mp3 size

    # --- Transcript ---
    predicted_transcript_chars: int
    predicted_transcript_tokens: int
    predicted_word_count: int

    # --- Transcription timing ---
    transcription_seconds_gpu: float         # on a typical GPU
    transcription_seconds_cpu: float         # on CPU

    # --- LLM analysis ---
    llm_strategy: str                       # "single-shot" | "chunk+merge"
    n_chunks: int
    llm_max_prompt_tokens: int
    n_ctx_required: int                      # smallest ctx that fits the prompt
    llm_passes: int                          # = 1 (single) or n_chunks + 1 (chunk+merge)
    llm_analysis_seconds: float             # predicted

    # --- Totals ---
    total_seconds_gpu: float                 # download + transcribe_gpu + analysis
    total_seconds_cpu: float                 # download + transcribe_cpu + analysis
    download_seconds: float                  # raw download time (heuristic)

    # --- Backend config (passed through for clarity) ---
    llamacpp_max_prompt_tokens: int = 28_000
    llamacpp_overlap_tokens: int = 200
    backend_label: str = "llamacpp-local"

    # --- Optional human-readable summary ---
    notes: list[str] = field(default_factory=list)

    @property
    def duration_str(self) -> str:
        total = int(self.duration_seconds)
        h, rem = divmod(total, 3600)
        m, sec = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{sec:02d}"
        return f"{m}:{sec:02d}"

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "video_title": self.video_title,
            "channel": self.channel,
            "duration_seconds": self.duration_seconds,
            "duration_str": self.duration_str,
            "audio_mb": round(self.audio_mb, 1),
            "predicted_transcript_chars": self.predicted_transcript_chars,
            "predicted_transcript_tokens": self.predicted_transcript_tokens,
            "predicted_word_count": self.predicted_word_count,
            "transcription_seconds_gpu": round(self.transcription_seconds_gpu, 1),
            "transcription_seconds_cpu": round(self.transcription_seconds_cpu, 1),
            "llm_strategy": self.llm_strategy,
            "n_chunks": self.n_chunks,
            "llm_max_prompt_tokens": self.llm_max_prompt_tokens,
            "n_ctx_required": self.n_ctx_required,
            "llm_passes": self.llm_passes,
            "llm_analysis_seconds": round(self.llm_analysis_seconds, 1),
            "total_seconds_gpu": round(self.total_seconds_gpu, 1),
            "total_seconds_cpu": round(self.total_seconds_cpu, 1),
            "download_seconds": round(self.download_seconds, 1),
            "backend_label": self.backend_label,
            "llamacpp_max_prompt_tokens": self.llamacpp_max_prompt_tokens,
            "llamacpp_overlap_tokens": self.llamacpp_overlap_tokens,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def estimate_url(
    url: str,
    *,
    max_prompt_tokens: int = 28_000,
    chunk_overlap_tokens: int = 200,
    hardware: str = "gpu_gtx_1050",
    content_type: str = "talk",
    llm_quant: str = "iq4_xs",
    audio_quality_kbps: int = AUDIO_BITRATE_KBPS,
) -> Estimate:
    """
    Predict the cost of running the full pipeline on *url*.

    Parameters
    ----------
    url:
        YouTube URL (any format accepted by yt-dlp).
    max_prompt_tokens:
        LLM window size. Must be < ``n_ctx`` of the server.
    chunk_overlap_tokens:
        Overlap between consecutive chunks when chunk+merge kicks in.
    hardware:
        One of :data:`TRANSCRIPTION_RTFX` keys. Determines the
        transcription time estimate.
    content_type:
        One of :data:`WORDS_PER_MINUTE_BY_CONTENT` keys. Determines the
        transcript length estimate.
    llm_quant:
        One of :data:`LLM_TOKENS_PER_SECOND` keys. Determines the LLM
        generation speed estimate.
    audio_quality_kbps:
        mp3 bitrate produced by yt-dlp.

    Raises
    ------
    yt_insight.downloader.DownloadError
        If yt-dlp can't fetch the metadata (private video, network, …).
    """
    # --- 1. Fetch metadata only (no audio download) --------------------
    downloader = YtDlpDownloader(cache_dir=Path("./cache"))
    metadata = downloader.fetch_metadata_only(url)

    # --- 2. Audio size estimate ---------------------------------------
    duration = metadata.duration_seconds
    audio_mb = (duration * audio_quality_kbps / 8 / 1024)  # kbps → MB

    # --- 3. Transcript prediction -------------------------------------
    wpm = WORDS_PER_MINUTE_BY_CONTENT.get(content_type, DEFAULT_WPM)
    n_words = int(duration / 60 * wpm)
    # Average word length in French/English transcripts: ~5.5 chars
    # (includes spaces and punctuation).
    n_chars = int(n_words * 5.5)
    n_tokens = n_chars // 4   # chars/4 heuristic, same as estimate_tokens

    # --- 4. Transcription time ----------------------------------------
    rtfx_gpu = TRANSCRIPTION_RTFX.get(hardware, 4.0)
    rtfx_cpu = TRANSCRIPTION_RTFX["cpu"]
    # First-call model load: ~5-10s on GPU, ~30-40s on CPU.
    model_load_gpu = 7.0
    model_load_cpu = 40.0
    transcribe_gpu = duration / rtfx_gpu + model_load_gpu
    transcribe_cpu = duration / rtfx_cpu + model_load_cpu

    # --- 5. LLM analysis prediction -----------------------------------
    # The actual prompt = system prompt + transcript + JSON instructions
    # ≈ transcript_tokens + LLM_OVERHEAD.
    effective_prompt_tokens = n_tokens + LLM_OVERHEAD_TOKENS
    n_ctx_required = effective_prompt_tokens

    if effective_prompt_tokens <= max_prompt_tokens:
        strategy = "single-shot"
        n_chunks = 1
        llm_passes = 1
        # Output = the full AnalysisResult JSON, ~LLM_MAX_OUTPUT_TOKENS tokens.
        output_tokens = LLM_MAX_OUTPUT_TOKENS
    else:
        strategy = "chunk+merge"
        # Each chunk is max_prompt_tokens, minus the chunk prompt overhead.
        # Overlap is added on top of the raw chunking.
        usable_per_chunk = max(1, max_prompt_tokens - 500)  # 500 tok of chunk prompt
        # Effective unique content per chunk: usable - overlap
        unique_per_chunk = max(1, usable_per_chunk - chunk_overlap_tokens)
        n_chunks = max(1, -(-n_tokens // unique_per_chunk))   # ceil division
        llm_passes = n_chunks + 1   # per-chunk + final merge
        # Per-chunk outputs are smaller (just summary + key_points + quotes)
        output_tokens = (n_chunks * 1_200) + LLM_MAX_OUTPUT_TOKENS

    tps = LLM_TOKENS_PER_SECOND.get(llm_quant, DEFAULT_LLM_TPS)
    llm_analysis_seconds = output_tokens / tps
    # Add a small fixed HTTP overhead per pass (TCP/HTTP roundtrip).
    llm_analysis_seconds += llm_passes * 0.5

    # --- 6. Download time (heuristic) ---------------------------------
    # Assume a typical home connection: 10 MB/s download.
    download_seconds = audio_mb / 10.0

    # --- 7. Totals -----------------------------------------------------
    total_gpu = download_seconds + transcribe_gpu + llm_analysis_seconds
    total_cpu = download_seconds + transcribe_cpu + llm_analysis_seconds

    # --- 8. Sanity notes ----------------------------------------------
    notes: list[str] = []
    if n_ctx_required > 65_536:
        notes.append(
            f"⚠️  n_ctx_required={n_ctx_required} > 65,536 — even at max llama-server "
            f"context, the analysis will need chunk+merge (already enabled)."
        )
    if effective_prompt_tokens > max_prompt_tokens:
        notes.append(
            f"Transcript ({n_tokens} tok) exceeds max_prompt_tokens ({max_prompt_tokens}) "
            f"→ chunk+merge will fire ({n_chunks} chunks + 1 merge)."
        )
    if hardware == "cpu":
        notes.append("CPU transcription is ~6x slower than GPU — consider CUDA setup.")
    if audio_mb > 200:
        notes.append(
            f"Audio is large ({audio_mb:.0f} MB) — download alone will take "
            f"~{download_seconds:.0f}s on a typical connection."
        )

    return Estimate(
        url=url,
        video_title=metadata.title,
        channel=metadata.channel,
        duration_seconds=duration,
        audio_mb=audio_mb,
        predicted_transcript_chars=n_chars,
        predicted_transcript_tokens=n_tokens,
        predicted_word_count=n_words,
        transcription_seconds_gpu=transcribe_gpu,
        transcription_seconds_cpu=transcribe_cpu,
        llm_strategy=strategy,
        n_chunks=n_chunks,
        llm_max_prompt_tokens=max_prompt_tokens,
        n_ctx_required=n_ctx_required,
        llm_passes=llm_passes,
        llm_analysis_seconds=llm_analysis_seconds,
        total_seconds_gpu=total_gpu,
        total_seconds_cpu=total_cpu,
        download_seconds=download_seconds,
        llamacpp_max_prompt_tokens=max_prompt_tokens,
        llamacpp_overlap_tokens=chunk_overlap_tokens,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Pretty-printer (used by the CLI)
# ---------------------------------------------------------------------------

def format_estimate(est: Estimate) -> str:
    """Return a human-friendly multi-line summary of *est*."""
    def _fmt_time(s: float) -> str:
        s = int(round(s))
        if s < 60:
            return f"{s}s"
        m, sec = divmod(s, 60)
        if m < 60:
            return f"{m}m{sec:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"

    lines = [
        f"📺 {est.video_title}",
        f"   Chaîne       : {est.channel}",
        f"   Durée vidéo  : {est.duration_str} ({est.duration_seconds:.0f}s)",
        f"   URL          : {est.url}",
        "",
        f"💾 Audio estimé : {est.audio_mb:.1f} Mo (mp3 192kbps)",
        f"   Télécharg.   : ~{_fmt_time(est.download_seconds)}",
        "",
        f"📝 Transcript prévu (heuristique) :",
        f"   Mots         : ~{est.predicted_word_count:,}",
        f"   Caractères   : ~{est.predicted_transcript_chars:,}",
        f"   Tokens       : ~{est.predicted_transcript_tokens:,}",
        "",
        f"🎙️ Transcription (faster-whisper large-v3) :",
        f"   Sur GPU      : ~{_fmt_time(est.transcription_seconds_gpu)}",
        f"   Sur CPU      : ~{_fmt_time(est.transcription_seconds_cpu)}",
        "",
        f"🤖 Analyse LLM ({est.backend_label}) :",
        f"   Stratégie    : {est.llm_strategy}",
        f"   Passes LLM   : {est.llm_passes}",
        f"   n_ctx requis : {est.n_ctx_required:,} tokens",
        f"   Temps estimé : ~{_fmt_time(est.llm_analysis_seconds)}",
        "",
        f"⏱️  TOTAL estimé :",
        f"   Avec GPU     : ~{_fmt_time(est.total_seconds_gpu)}",
        f"   Avec CPU     : ~{_fmt_time(est.total_seconds_cpu)}",
    ]
    if est.notes:
        lines.append("")
        lines.append("📌 Notes :")
        for n in est.notes:
            lines.append(f"   {n}")
    return "\n".join(lines)

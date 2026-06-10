"""
YT-Insight command-line interface.

Built with Typer + Rich. Single command, multiple subcommands:

- ``yt-insight all URL`` (default) — full pipeline: download → transcribe
  → analyze → render + write.
- ``yt-insight download URL``     — only download the audio.
- ``yt-insight transcribe PATH`` — transcribe a local audio file.
- ``yt-insight analyze FILE``    — analyze an existing transcript file
  (or pair ``--audio`` with a URL to transcribe + analyze in one shot).
- ``yt-insight version``         — print the installed version.

Most options are common across subcommands and can be supplied on the
command line OR via environment variables / ``config.yaml``.
"""

from __future__ import annotations

import json
import re
import logging
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .analyzer import create_analyzer
from .estimate import estimate_url, format_estimate
from .downloader import DownloadResult, YtDlpDownloader
from .output import ConsoleRenderer, write_outputs
from .transcriber import create_transcriber
from .transcriber.base import Segment, TranscriptionResult
from .utils import (
    AppConfig,
    load_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App + global options
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="yt-insight",
    help="YouTube transcription & analysis pipeline (yt-dlp + faster-whisper + llama.cpp).",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

# Reusable option callbacks -----------------------------------------------

def _verbose_callback(value: bool) -> None:
    if value:
        setup_logging("DEBUG")
    else:
        setup_logging("INFO")


def _config_callback(value: Optional[Path]) -> Optional[Path]:
    return value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_app_config(
    config_path: Optional[Path],
    verbose: bool,
) -> AppConfig:
    """Setup logging + load config in one go."""
    setup_logging("DEBUG" if verbose else "INFO")
    return load_config(config_path, create_paths=True)


def _build_console() -> Console:
    return Console()


def _print_banner(console: Console, title: str) -> None:
    console.print(
        Panel(
            f"[bold]YT-Insight[/] v{__version__}\n{title}",
            border_style="cyan",
        )
    )


def _validate_depth_sections(depth: str | None, sections: str | None) -> tuple:
    """
    Validate --depth and --sections CLI values early so the user
    gets a clean error before any network call.

    Returns ``(depth_obj, sections_or_sentinel)`` where
    ``sections_or_sentinel`` is either:
    - ``None`` if the user did not pass ``--sections`` (so the analyzer
      applies the **depth-specific** default in ``__init__``)
    - a tuple of names if the user passed ``--sections`` (any string
      is coerced and validated here)
    """
    from .analyzer.depth import coerce_depth
    depth_obj = coerce_depth(depth)
    # If user didn't pass --sections, we MUST leave it as None so that
    # ``LlamaCppLocalAnalyzer.__init__`` picks the right default per
    # depth (e.g. 8 sections for extreme, 1 for shallow, ...).
    # Calling ``coerce_sections(None)`` would incorrectly return the
    # NORMAL default (3 sections), which is what used to happen.
    if sections is None or sections == "":
        sections_final = None
    else:
        # Validate now (coerce_sections raises ValueError on bad names).
        from .analyzer.depth import coerce_sections
        sections_final = coerce_sections(sections)
    return depth_obj, sections_final


def _build_output_tag(analyzer, max_prompt_tokens: int) -> str:
    """
    Build a short disambiguation tag for output filenames.

    Format: ``{model_short}-p{max_prompt_tokens//1000}k``,
    e.g. ``qwen3-50k`` for the Qwen3 model with a 50k ctx cap.

    This is appended to the filename when an output with the same
    base name already exists, so the user can keep multiple
    analyses of the same video side-by-side.

    Defensive against mocks / partial analyzer objects: falls back
    to ``"model"`` if the attribute is not a real string.
    """
    raw_model = getattr(analyzer, "_model", None) if analyzer else None
    if not isinstance(raw_model, str):
        model = "model"
    else:
        model = raw_model.lower()
    # Strip common GGUF suffixes and qualifiers.
    for suffix in (".gguf", "-ud", "-instruct", "-chat", "-base"):
        if suffix in model:
            model = model.split(suffix)[0]
    # Keep just the first 12 chars of the model name.
    model_short = re.sub(r"[^a-z0-9]+", "-", model).strip("-")[:12] or "model"
    tok_k = max_prompt_tokens // 1000
    return f"{model_short}-p{tok_k}k"


def _print_kv(console: Console, items: list[tuple[str, str]]) -> None:
    """Pretty-print a list of (key, value) tuples."""
    width = max(len(k) for k, _ in items) if items else 0
    for k, v in items:
        console.print(f"  [cyan]{k.ljust(width)}[/]  {v}")


def _run_analyze_with_live(
    console: Console,
    analyzer,                                  # LlamaCppLocalAnalyzer (forward ref)
    transcription: "TranscriptionResult",
    *,
    title: str,
    language: str | None,
    show_live: bool = True,
):
    """
    Run ``analyzer.analyze()`` and stream the LLM output live to the terminal.

    Uses Rich's ``Live`` with ``transient=True`` so the panel is erased
    at the end and a clean console is left behind. We render the
    *current preview* (capped at ~1000 chars) inside a Panel; tokens are
    appended live via the ``on_token`` callback.

    If the console is not interactive (CI, pipe, redirect), we fall
    back to a non-live call so the user still gets the analysis.
    """
    if not show_live or not console.is_interactive:
        with analyzer:
            return analyzer.analyze(
                transcription, title=title, language=language,
            )

    # Live state — mutated by the on_token callback.
    state: dict[str, object] = {
        "chars": 0,
        "tokens": 0,
        "t0": time.time(),
        "buffer": "",
    }

    def on_token(delta: str) -> None:
        state["chars"] = int(state["chars"]) + len(delta)  # type: ignore[arg-type]
        state["tokens"] = int(state["tokens"]) + 1  # type: ignore[arg-type]
        state["buffer"] = str(state["buffer"]) + delta  # type: ignore[assignment]
        # Trim the preview so the panel doesn't grow forever.
        if len(str(state["buffer"])) > 1200:  # type: ignore[arg-type]
            state["buffer"] = "…" + str(state["buffer"])[-1000:]  # type: ignore[assignment]

    def make_panel() -> Panel | Text:
        """
        Build the live panel, or an empty Text if no tokens have arrived
        yet (Rich's ``Live`` draws an empty panel on entry otherwise,
        producing the visible "ghost" duplicate the user complained
        about).
        """
        buffer = str(state["buffer"])  # type: ignore[arg-type]
        elapsed = time.time() - float(state["t0"])  # type: ignore[arg-type]
        footer = Text(
            f" {int(state['chars'])} chars · {int(state['tokens'])} tokens · "  # type: ignore[arg-type]
            f"{elapsed:.1f}s ",
            style="dim cyan",
        )
        if not buffer:
            # Empty buffer → return a vertical-fill Text that the Live
            # block sizes to the right dimensions but draws nothing
            # visible. This avoids the "empty frame" ghost panel that
            # appears on the first render.
            return Text("\n\n  …waiting for first token…\n", style="dim")
        body = Text(buffer, style="bright_white")
        return Panel(
            body,
            title="[bold magenta]LLM generation[/]",
            subtitle=footer,
            border_style="magenta",
            padding=(0, 1),
        )

    from .analyzer.llamacpp_local import AnalysisError

    # ``transient=True`` makes Rich erase the panel on exit, leaving
    # a clean console. ``redirect_stdout`` is critical: it prevents
    # child output (httpx logs, etc.) from racing the Live redraw and
    # leaving stray "ghost" panels behind.
    import contextlib
    import sys

    result_holder: dict = {}

    with contextlib.redirect_stdout(sys.stderr):
        with Live(
            make_panel(),
            console=console,
            refresh_per_second=6,
            transient=True,
            redirect_stdout=False,  # we already redirected above
        ) as live:
            def live_on_token(delta: str) -> None:
                on_token(delta)
                live.update(make_panel())

            try:
                with analyzer:
                    result_holder["result"] = analyzer.analyze(
                        transcription, title=title, language=language,
                        on_token=live_on_token,
                    )
            except AnalysisError:
                # Final panel shows the error before re-raising.
                state["buffer"] = (
                    str(state["buffer"]) + "\n\n[ERROR] "  # type: ignore[assignment]
                )
                live.update(make_panel())
                raise
            # One last refresh so the user sees the last tokens
            # before the panel is erased (transient=True).
            live.update(make_panel())

    return result_holder["result"]


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

@app.command()
def download(
    url: str = typer.Argument(..., help="YouTube URL (watch, short, embed…)."),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Where to cache audio files."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Where to put output reports."),
    force: bool = typer.Option(False, "--force", help="Re-download even if cached."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Download the audio of a YouTube video to the local cache."""
    app_cfg = _load_app_config(config, verbose)
    console = _build_console()
    _print_banner(console, f"Téléchargement — {url}")

    paths = app_cfg.paths
    if cache_dir:
        paths.cache_dir = Path(cache_dir)
        paths.cache_dir.mkdir(parents=True, exist_ok=True)
    if output_dir:
        paths.output_dir = Path(output_dir)
        paths.output_dir.mkdir(parents=True, exist_ok=True)

    downloader = YtDlpDownloader(cache_dir=paths.cache_dir)
    t0 = time.time()
    result = downloader.download(url, force=force)
    elapsed = time.time() - t0

    _print_kv(console, [
        ("Titre",    result.metadata.title),
        ("Chaîne",   result.metadata.channel),
        ("Durée",    result.metadata.duration_str),
        ("Fichier",  str(result.audio_path)),
        ("Cache",    "oui (hit)" if result.from_cache else f"non ({elapsed:.1f}s)"),
    ])


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------

@app.command()
def estimate(
    url: str = typer.Argument(..., help="YouTube URL to estimate."),
    hardware: str = typer.Option(
        "gpu_gtx_1050", "--hardware",
        help="Transcription hardware: gpu_gtx_1050, gpu_gtx_1650, gpu_rtx_3060, cpu.",
    ),
    content_type: str = typer.Option(
        "talk", "--content-type",
        help="talk | lecture | podcast | interview | fast.",
    ),
    llm_quant: str = typer.Option(
        "iq4_xs", "--llm-quant",
        help="Model quantization: iq1_m, iq3_s, iq4_xs, q4_k_m, q5_k_m, q6_k, q8_0.",
    ),
    max_prompt_tokens: int = typer.Option(
        28_000, "--max-prompt-tokens", min=1_000,
        help="LLM window size (must be < n_ctx of the server).",
    ),
    chunk_overlap: int = typer.Option(
        200, "--chunk-overlap", min=0,
        help="Overlap between consecutive chunks when chunk+merge fires.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output the estimate as JSON instead of formatted text.",
    ),
) -> None:
    """
    Estimate the cost (time, transcript size, LLM strategy) of running the
    full pipeline on a YouTube URL — without actually downloading or
    transcribing anything (just metadata via yt-dlp).
    """
    est = estimate_url(
        url,
        max_prompt_tokens=max_prompt_tokens,
        chunk_overlap_tokens=chunk_overlap,
        hardware=hardware,
        content_type=content_type,
        llm_quant=llm_quant,
    )
    if json_output:
        typer.echo(json.dumps(est.to_dict(), ensure_ascii=False, indent=2))
    else:
        typer.echo(format_estimate(est))


@app.command()
def transcribe(
    source: str = typer.Argument(..., help="Local audio path OR YouTube URL."),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Force ISO 639-1 language."),
    whisper_model: Optional[str] = typer.Option(
        None, "--whisper-model",
        help="Whisper model size: tiny, base, small, medium, large-v1, large-v2, large-v3, "
             "distil-large-v2, distil-large-v3. Default from config (large-v3).",
    ),
    whisper_chunk_length: Optional[int] = typer.Option(
        None, "--whisper-chunk-length",
        help="Max audio chunk length in seconds (default: faster-whisper's 30s). "
             "Lower this (e.g. 20) on GPUs with tight VRAM to avoid OOM.",
    ),
    output_json: Optional[Path] = typer.Option(None, "--output", "-o", help="Write the transcript to this JSON file."),
    config: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Transcribe a local audio file (or download + transcribe a YouTube URL)."""
    app_cfg = _load_app_config(config, verbose)
    console = _build_console()
    _print_banner(console, f"Transcription — {source}")

    # Resolve source: URL → download, path → use directly
    if source.startswith("http://") or source.startswith("https://"):
        downloader = YtDlpDownloader(cache_dir=app_cfg.paths.cache_dir)
        dl: DownloadResult = downloader.download(source)
        audio_path = dl.audio_path
        title = dl.metadata.title
    else:
        audio_path = Path(source)
        if not audio_path.exists():
            console.print(f"[red]Audio file not found: {audio_path}[/red]")
            raise typer.Exit(code=1)
        title = audio_path.stem

    transcriber = create_transcriber(
        model_size=whisper_model or app_cfg.transcription.model,
        device=app_cfg.transcription.device,
        compute_type=app_cfg.transcription.compute_type,
        language=language or app_cfg.transcription.language,
        chunk_length=whisper_chunk_length,
    )

    t0 = time.time()
    result: TranscriptionResult = transcriber.transcribe(audio_path, language=language)
    elapsed = time.time() - t0

    _print_kv(console, [
        ("Titre",          title),
        ("Langue",         f"{result.language} (p={result.language_probability:.2f})"),
        ("Durée",          result.duration_str),
        ("Modèle",         result.model_name),
        ("Segments",       str(len(result.segments))),
        ("Tokens estimés", f"{result.estimated_tokens:,}"),
        ("Temps",          f"{elapsed:.1f}s"),
    ])

    # Save transcript to disk if requested
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        console.print(f"  [green]Transcription écrite dans {output_json}[/green]")

    transcriber.unload()


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@app.command()
def analyze(
    transcript_file: Optional[Path] = typer.Argument(None, help="JSON file produced by `transcribe --output`."),
    audio: Optional[str] = typer.Option(None, "--audio", help="Local audio path OR YouTube URL (will transcribe first)."),
    language: Optional[str] = typer.Option(None, "--language", "-l"),
    whisper_model: Optional[str] = typer.Option(
        None, "--whisper-model",
        help="Whisper model size (only used if --audio is set).",
    ),
    whisper_chunk_length: Optional[int] = typer.Option(
        None, "--whisper-chunk-length",
        help="Max audio chunk length in seconds (default: 30).",
    ),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    format: list[str] = typer.Option(["markdown", "json"], "--format", help="Output formats: markdown, json."),
    no_console: bool = typer.Option(False, "--no-console", help="Skip the Rich console rendering."),
    llamacpp_url: Optional[str] = typer.Option(None, "--llamacpp-url"),
    llamacpp_model: Optional[str] = typer.Option(None, "--llamacpp-model"),
    llamacpp_timeout: Optional[float] = typer.Option(
        None, "--llamacpp-timeout",
        help="HTTP timeout in seconds for the LLM request (default 7200 = 2h). "
             "Increase for very long single-shot analyses on big transcripts.",
    ),
    llamacpp_idle_timeout: Optional[float] = typer.Option(
        None, "--llamacpp-idle-timeout",
        help="Idle timeout between LLM tokens in seconds (default 600 = 10 min). "
             "Must be > prompt-processing time for big chunks.",
    ),
    llamacpp_max_prompt_tokens: Optional[int] = typer.Option(
        None, "--llamacpp-max-prompt-tokens",
        help="Soft cap on prompt tokens (default 50000). Above this we switch to "
             "chunk+merge. Must stay under your server's n_ctx.",
    ),
    llamacpp_repetition_penalty: Optional[float] = typer.Option(
        None, "--llamacpp-repetition-penalty",
        help="Penalty for already-generated tokens (default 1.1). "
             "Values > 1.0 discourage repetition and break infinite loops. "
             "1.0 = no penalty. Recommended: 1.05–1.2 for long generations.",
    ),
    depth: Optional[str] = typer.Option(
        None, "--depth",
        help="Analysis depth preset: shallow | normal | deep | extreme. "
             "Controls max_tokens, num_key_points, num_quotes, temperature. "
             "Default: normal.",
    ),
    sections: Optional[str] = typer.Option(
        None, "--sections",
        help="Comma-separated analysis rubrics to include. Valid: forces, "
             "concepts, implications, weaknesses, contradictions, biases, "
             "limitations, context_gaps. Default: depends on --depth.",
    ),
    config: Optional[Path] = typer.Option(None, "--config"),

    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Analyze a transcript (or transcribe + analyze) using the local LLM."""
    app_cfg = _load_app_config(config, verbose)
    console = _build_console()

    if transcript_file is None and audio is None:
        console.print("[red]Provide either a transcript file or --audio[/red]")
        raise typer.Exit(code=1)

    # --- Step 1: get a TranscriptionResult -----------------------------
    title = ""
    metadata = None
    if transcript_file is not None:
        _print_banner(console, f"Analyse — {transcript_file.name}")
        data = json.loads(transcript_file.read_text(encoding="utf-8"))
        transcription = TranscriptionResult(
            text=data.get("text", ""),
            segments=[Segment(**s) for s in data.get("segments", [])],
            language=data.get("language", "fr"),
            language_probability=data.get("language_probability", 0.0),
            duration_seconds=data.get("duration_seconds", 0.0),
            model_name=data.get("model_name", "?"),
        )
    else:
        _print_banner(console, f"Transcription + Analyse — {audio}")
        # Transcribe first
        if audio.startswith("http://") or audio.startswith("https://"):
            downloader = YtDlpDownloader(cache_dir=app_cfg.paths.cache_dir)
            dl = downloader.download(audio)
            audio_path = dl.audio_path
            title = dl.metadata.title
            metadata = dl.metadata
        else:
            audio_path = Path(audio)
            title = audio_path.stem
        transcriber = create_transcriber(
            model_size=whisper_model or app_cfg.transcription.model,
            device=app_cfg.transcription.device,
            compute_type=app_cfg.transcription.compute_type,
            language=language or app_cfg.transcription.language,
            chunk_length=whisper_chunk_length,
        )
        transcription = transcriber.transcribe(audio_path, language=language)
        transcriber.unload()

    # --- Step 2: analyze -----------------------------------------------
    # Validate depth + sections early (before any network call).
    try:
        depth_obj, sections_tuple = _validate_depth_sections(depth, sections)
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    analyzer = create_analyzer(
        base_url=llamacpp_url,
        model=llamacpp_model,
        timeout_s=llamacpp_timeout,
        idle_timeout_s=llamacpp_idle_timeout,
        max_prompt_tokens=llamacpp_max_prompt_tokens,
        depth=depth_obj,
        sections=sections_tuple,
        repetition_penalty=llamacpp_repetition_penalty,
    )
    t0 = time.time()
    analysis = _run_analyze_with_live(
        console, analyzer, transcription,
        title=title, language=language, show_live=True,
    )
    elapsed = time.time() - t0

    console.print(f"  [green]✓[/green] Analyse terminée en {elapsed:.1f}s")
    console.print(f"  [green]✓[/green] Modèle : {analysis.model_name}")
    console.print(f"  [green]✓[/green] Backend : {analysis.backend}")

    # --- Step 3: render -----------------------------------------------
    if not no_console:
        ConsoleRenderer().render(analysis, transcription, video_url=audio or "")

    # --- Step 4: write -------------------------------------------------
    if output_dir is None:
        output_dir = app_cfg.paths.output_dir
    out_tag = _build_output_tag(analyzer, app_cfg.analysis.max_transcript_tokens)
    paths = write_outputs(
        analysis,
        output_dir,
        formats=format,
        title=title or "untitled",
        video_url=audio or "",
        metadata=metadata,
        transcription=transcription if app_cfg.output.include_transcript else None,
        include_transcript=app_cfg.output.include_transcript,
        include_timestamps=app_cfg.output.include_timestamps,
        tag=out_tag,
    )
    for kind, p in paths.items():
        console.print(f"  [green]✓[/green] {kind:<8} → {p}")


# ---------------------------------------------------------------------------
# all (default)
# ---------------------------------------------------------------------------

@app.command()
def all(  # noqa: A001 — `all` is the command name users will type
    url: str = typer.Argument(..., help="YouTube URL."),
    language: Optional[str] = typer.Option(None, "--language", "-l"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    format: list[str] = typer.Option(["markdown", "json"], "--format"),
    steps: list[str] = typer.Option(
        ["download", "transcribe", "analyze"],
        "--steps",
        help="Subset of: download, transcribe, analyze. Repeat the flag or "
             "pass a comma-separated list.",
    ),
    whisper_model: Optional[str] = typer.Option(
        None, "--whisper-model",
        help="Whisper model size: tiny, base, small, medium, large-v1/v2/v3, "
             "distil-large-v2/v3. Default from config (large-v3). "
             "Try 'medium' (1.5 Go VRAM) or 'distil-large-v3' (~1.5 Go, 1.5x faster than large-v3) "
             "on tight-VRAM GPUs.",
    ),
    whisper_chunk_length: Optional[int] = typer.Option(
        None, "--whisper-chunk-length",
        help="Max audio chunk length in seconds (default: 30). "
             "Lower this (e.g. 20) on tight-VRAM GPUs to avoid OOM.",
    ),
    llamacpp_url: Optional[str] = typer.Option(None, "--llamacpp-url"),
    llamacpp_model: Optional[str] = typer.Option(None, "--llamacpp-model"),
    llamacpp_timeout: Optional[float] = typer.Option(
        None, "--llamacpp-timeout",
        help="HTTP timeout in seconds for the LLM request (default 7200 = 2h).",
    ),
    llamacpp_idle_timeout: Optional[float] = typer.Option(
        None, "--llamacpp-idle-timeout",
        help="Idle timeout between LLM tokens in seconds (default 600 = 10 min).",
    ),
    llamacpp_max_prompt_tokens: Optional[int] = typer.Option(
        None, "--llamacpp-max-prompt-tokens",
        help="Soft cap on prompt tokens (default 50000).",
    ),
    llamacpp_repetition_penalty: Optional[float] = typer.Option(
        None, "--llamacpp-repetition-penalty",
        help="Repetition penalty (default 1.1). 1.0 = no penalty. "
             "Recommended: 1.05–1.2 to break infinite loops in long gens.",
    ),
    depth: Optional[str] = typer.Option(
        None, "--depth",
        help="Analysis depth preset: shallow | normal | deep | extreme.",
    ),
    sections: Optional[str] = typer.Option(
        None, "--sections",
        help="Comma-separated analysis rubrics. Valid: forces, concepts, "
             "implications, weaknesses, contradictions, biases, limitations, "
             "context_gaps.",
    ),
    no_console: bool = typer.Option(False, "--no-console"),
    config: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    cookies: Optional[Path] = typer.Option(
        None, "--cookies",
        help="Path to a Netscape-format cookies.txt file. Required for YouTube "
             "videos that gate downloads behind a 'Sign in to confirm you're "
             "not a bot' check. Export from your browser while logged in to "
             "youtube.com (e.g. via the 'Get cookies.txt LOCALLY' extension).",
    ),
    js_runtime: Optional[str] = typer.Option(
        None, "--js-runtime",
        help="JavaScript runtime for yt-dlp's n-challenge solver. "
             "Format: 'name' or 'name:/path/to/binary'. "
             "Common: 'node' (needs Node.js 18+), 'deno', 'bun'. "
             "Default: 'node' if found on PATH. Pass an absolute path "
             "if yt-dlp cannot find your runtime (e.g. node installed in "
             "~/.local/bin).",
    ),
) -> None:
    """Run the full pipeline: download → transcribe → analyze → render + write."""
    app_cfg = _load_app_config(config, verbose)
    console = _build_console()
    _print_banner(console, f"Pipeline complet — {url}")

    paths = app_cfg.paths
    if cache_dir:
        paths.cache_dir = Path(cache_dir)
        paths.cache_dir.mkdir(parents=True, exist_ok=True)
    if output_dir:
        paths.output_dir = Path(output_dir)
        paths.output_dir.mkdir(parents=True, exist_ok=True)

    steps_set: set[str] = set()
    for s in steps:
        # Accept both `--steps transcribe --steps analyze` and `--steps transcribe,analyze`.
        steps_set.update(p.strip() for p in s.split(",") if p.strip())
    title = ""
    metadata = None
    transcription: TranscriptionResult | None = None

    # --- Step 1: download ---------------------------------------------
    # Resolve JS runtime default: yt-dlp looks on PATH but our user
    # has node in ~/.local/bin which it may miss. Pre-detect and pass
    # explicit path so the n-challenge solver works.
    resolved_js_runtime = js_runtime
    if resolved_js_runtime is None:
        import shutil
        node_path = shutil.which("node")
        if node_path:
            resolved_js_runtime = f"node:{node_path}"

    downloader = YtDlpDownloader(
        cache_dir=paths.cache_dir,
        cookies_file=str(cookies) if cookies else None,
        js_runtimes=resolved_js_runtime,
    )

    if "download" in steps_set:
        dl = downloader.download(url)
        title = dl.metadata.title
        metadata = dl.metadata
        _print_kv(console, [
            ("📥 Titre",   dl.metadata.title),
            ("   Chaîne",  dl.metadata.channel),
            ("   Durée",   dl.metadata.duration_str),
            ("   Fichier", str(dl.audio_path)),
            ("   Cache",   "oui" if dl.from_cache else "non"),
        ])
    else:
        # We still need the audio path to transcribe. Re-download silently.
        dl = downloader.download(url)
        title = dl.metadata.title
        metadata = dl.metadata

    # --- Step 2: transcribe -------------------------------------------
    if "transcribe" in steps_set:
        # Cache check: reuse an existing transcript if present.
        transcript_cache = paths.cache_dir / f"{dl.metadata.video_id}.transcript.json"
        if transcript_cache.exists():
            console.print(
                f"  [green]✓[/green] Transcript en cache : {transcript_cache} "
                "[dim](transcription skipped)[/dim]"
            )
            transcription = TranscriptionResult.from_dict(
                json.loads(transcript_cache.read_text(encoding="utf-8"))
            )
        else:
            transcriber = create_transcriber(
                model_size=whisper_model or app_cfg.transcription.model,
                device=app_cfg.transcription.device,
                compute_type=app_cfg.transcription.compute_type,
                language=language or app_cfg.transcription.language,
                chunk_length=whisper_chunk_length,
            )
            t0 = time.time()
            transcription = transcriber.transcribe(dl.audio_path, language=language)
            elapsed = time.time() - t0
            transcriber.unload()
            _print_kv(console, [
                ("🎙️ Langue",         f"{transcription.language} (p={transcription.language_probability:.2f})"),
                ("    Segments",      str(len(transcription.segments))),
                ("    Tokens estimés", f"{transcription.estimated_tokens:,}"),
                ("    Temps",         f"{elapsed:.1f}s"),
            ])
            # Save transcript to cache so the next run skips re-transcription.
            transcript_cache.write_text(
                json.dumps(transcription.to_dict(), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            console.print(
                f"  [dim]Transcript sauvé dans {transcript_cache}[/dim]"
            )

    if "analyze" in steps_set and transcription is None:
        # If the user asked for analyze but skipped transcribe, the
        # transcription may legitimately be None — that's a user error,
        # not a pipeline failure worth exit(1) over.
        console.print(
            "[red]No transcription available for analysis. "
            "Add 'transcribe' to --steps, or provide --audio / "
            "--transcript-file.[/red]"
        )
        raise typer.Exit(code=1)

    # If we got here with transcription still None (e.g. --steps
    # 'download' only), there's nothing more to do — we're done.
    if transcription is None:
        console.print(
            "[green]✓[/green] Pipeline terminé (étape download uniquement)."
        )
        return

    # --- Step 3: analyze ----------------------------------------------
    if "analyze" in steps_set:
        # Validate depth + sections early (before any network call).
        try:
            depth_obj, sections_tuple = _validate_depth_sections(depth, sections)
        except ValueError as e:
            console.print(f"[red]✗[/red] {e}")
            raise typer.Exit(1)
        analyzer = create_analyzer(
            base_url=llamacpp_url,
            model=llamacpp_model,
            timeout_s=llamacpp_timeout,
            idle_timeout_s=llamacpp_idle_timeout,
            max_prompt_tokens=llamacpp_max_prompt_tokens,
            depth=depth_obj,
            sections=sections_tuple,
            repetition_penalty=llamacpp_repetition_penalty,
        )
        t0 = time.time()
        analysis = _run_analyze_with_live(
            console, analyzer, transcription,
            title=title, language=language, show_live=not no_console,
        )
        elapsed = time.time() - t0
        _print_kv(console, [
            ("🤖 Modèle",         analysis.model_name),
            ("    Backend",       analysis.backend),
            ("    Points clés",   str(len(analysis.key_points))),
            ("    Citations",     str(len(analysis.quotes))),
            ("    Temps",         f"{elapsed:.1f}s"),
        ])

        # --- Step 4: render -------------------------------------------
        if not no_console:
            ConsoleRenderer().render(analysis, transcription, video_url=url)

        # --- Step 5: write --------------------------------------------
        out_tag = _build_output_tag(analyzer, app_cfg.analysis.max_transcript_tokens)
        paths_out = write_outputs(
            analysis,
            paths.output_dir,
            formats=format,
            title=title,
            video_url=url,
            metadata=metadata,
            transcription=transcription if app_cfg.output.include_transcript else None,
            include_transcript=app_cfg.output.include_transcript,
            include_timestamps=app_cfg.output.include_timestamps,
            tag=out_tag,
        )
        console.print()
        console.print("[bold green]💾 Fichiers écrits :[/bold green]")
        for kind, p in paths_out.items():
            console.print(f"   {kind:<8} → {p}")


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

@app.command()
def version() -> None:
    """Print the installed YT-Insight version."""
    typer.echo(f"yt-insight {__version__}")


# ---------------------------------------------------------------------------
# Default command routing
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """YT-Insight — YouTube transcription & analysis pipeline."""
    if ctx.invoked_subcommand is None:
        # No subcommand → show help
        console = _build_console()
        console.print(
            Panel(
                "[bold]YT-Insight[/] v{}\n\n"
                "Quick start:\n"
                "  [cyan]yt-insight all[/] [green]<youtube-url>[/]\n"
                "  [cyan]yt-insight analyze[/] [green]--audio <url>[/]\n"
                "  [cyan]yt-insight transcribe[/] [green]<local-audio.mp3>[/]\n"
                "  [cyan]yt-insight version[/]\n\n"
                "Run [cyan]yt-insight --help[/] for the full list.".format(__version__),
                border_style="cyan",
            )
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover - exercised through `yt-insight` script
    app()


if __name__ == "__main__":
    main()

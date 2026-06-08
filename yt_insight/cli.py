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
import logging
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

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


def _print_kv(console: Console, items: list[tuple[str, str]]) -> None:
    """Pretty-print a list of (key, value) tuples."""
    width = max(len(k) for k, _ in items) if items else 0
    for k, v in items:
        console.print(f"  [cyan]{k.ljust(width)}[/]  {v}")


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
        model_size=app_cfg.transcription.model,
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
    whisper_chunk_length: Optional[int] = typer.Option(
        None, "--whisper-chunk-length",
        help="Max audio chunk length in seconds (default: 30).",
    ),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    format: list[str] = typer.Option(["markdown", "json"], "--format", help="Output formats: markdown, json."),
    no_console: bool = typer.Option(False, "--no-console", help="Skip the Rich console rendering."),
    llamacpp_url: Optional[str] = typer.Option(None, "--llamacpp-url"),
    llamacpp_model: Optional[str] = typer.Option(None, "--llamacpp-model"),
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
            model_size=app_cfg.transcription.model,
            device=app_cfg.transcription.device,
            compute_type=app_cfg.transcription.compute_type,
            language=language or app_cfg.transcription.language,
            chunk_length=whisper_chunk_length,
        )
        transcription = transcriber.transcribe(audio_path, language=language)
        transcriber.unload()

    # --- Step 2: analyze -----------------------------------------------
    analyzer = create_analyzer(
        base_url=llamacpp_url,
        model=llamacpp_model,
    )
    t0 = time.time()
    with analyzer:
        analysis = analyzer.analyze(transcription, title=title, language=language)
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
    whisper_chunk_length: Optional[int] = typer.Option(
        None, "--whisper-chunk-length",
        help="Max audio chunk length in seconds (default: 30). "
             "Lower this (e.g. 20) on tight-VRAM GPUs to avoid OOM.",
    ),
    llamacpp_url: Optional[str] = typer.Option(None, "--llamacpp-url"),
    llamacpp_model: Optional[str] = typer.Option(None, "--llamacpp-model"),
    no_console: bool = typer.Option(False, "--no-console"),
    config: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
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
    if "download" in steps_set:
        downloader = YtDlpDownloader(cache_dir=paths.cache_dir)
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
        downloader = YtDlpDownloader(cache_dir=paths.cache_dir)
        dl = downloader.download(url)
        title = dl.metadata.title
        metadata = dl.metadata

    # --- Step 2: transcribe -------------------------------------------
    if "transcribe" in steps_set:
        transcriber = create_transcriber(
            model_size=app_cfg.transcription.model,
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
        analyzer = create_analyzer(
            base_url=llamacpp_url,
            model=llamacpp_model,
        )
        t0 = time.time()
        with analyzer:
            analysis = analyzer.analyze(transcription, title=title, language=language)
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

"""YT-Insight command-line interface (Typer + Rich).

This is a placeholder — the full CLI (download / transcribe / analyze
subcommands, progress bars, formatted output) will be implemented in
Phase 1 once the analyzer and output modules are in place.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="yt-insight",
    help="YouTube transcription & analysis pipeline.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the installed YT-Insight version."""
    from yt_insight import __version__

    typer.echo(f"yt-insight {__version__}")


if __name__ == "__main__":
    app()

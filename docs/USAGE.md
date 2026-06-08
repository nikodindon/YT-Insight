# YT-Insight — Usage

Complete reference for the `yt-insight` command-line interface.

> For installation, see the [main README](../README.md#installation).
> For output formats (Markdown / JSON), see [OUTPUT.md](OUTPUT.md).
> For analysis depth & sections, see [ANALYSIS_DEPTH.md](ANALYSIS_DEPTH.md).

## Table of contents

- [Prerequisites](#prerequisites)
- [Subcommands](#subcommands)
  - [`all`](#all---full-pipeline)
  - [`download`](#download---just-the-audio)
  - [`transcribe`](#transcribe---just-the-transcript)
  - [`analyze`](#analyze---just-the-analysis)
  - [`estimate`](#estimate---dry-run-no-network)
- [Global options](#global-options)
- [Common recipes](#common-recipes)
- [Caching behavior](#caching-behavior)

---

## Prerequisites

`llama-server` must be running on `:8080` (or whatever `--llamacpp-url`
points to). In a separate terminal:

```bash
# Example: 35B-A3B model, 65k ctx, GPU offload
llama-server -m ~/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
    -c 65536 -ngl 99 --n-cpu-moe 36 -t 10 \
    -ctk q8_0 -ctv q8_0 --host 0.0.0.0 \
    -b 2048 -ub 2048 --timeout 60000 --jinja
```

Verify with:

```bash
curl -s http://localhost:8080/health
# {"status":"ok"}
```

---

## Subcommands

### `all` — full pipeline

```
yt-insight all URL [OPTIONS]
```

Download → transcribe → analyze → write. This is the command you'll
use 95% of the time.

**Required:**
- `URL` — YouTube URL, YouTube Shorts, or local audio file path.

**Most useful flags:**

| Flag | Default | What it does |
|------|---------|--------------|
| `--language`, `-l` | auto-detect | Force ISO 639-1 code (e.g. `en`, `fr`) |
| `--output-dir` | `./outputs` | Where to write Markdown / JSON |
| `--format` | `markdown json` | Output formats: `markdown`, `json`, or both |
| `--steps` | `download,transcribe,analyze` | Subset of steps to run |
| `--no-console` | off | Skip the live Rich rendering (pipe-friendly) |
| `--verbose`, `-v` | off | Debug-level logging |
| `--config` | `config.yaml` | Path to a custom config file |

**Whisper flags:**

| Flag | Default | What it does |
|------|---------|--------------|
| `--whisper-model` | `large-v3` | `tiny` / `base` / `small` / `medium` / `large-v3` / `distil-large-v3` |
| `--whisper-chunk-length` | `auto` | Chunk length in seconds (`30` for 4GB VRAM) |

**llama.cpp flags:**

| Flag | Default | What it does |
|------|---------|--------------|
| `--llamacpp-url` | `http://localhost:8080` | OpenAI-compat server URL |
| `--llamacpp-model` | Qwen3.6 35B-A3B | Model name as the server reports it |
| `--llamacpp-timeout` | `7200` (2h) | Wall-clock HTTP timeout in seconds |
| `--llamacpp-idle-timeout` | `1800` (30 min) | Per-token idle timeout. Must be > your server's prompt processing time. Observed 14 min for 30k tokens on a GTX 1650S. |
| `--llamacpp-max-prompt-tokens` | `50000` | Soft cap; above this we switch to chunk+merge |

**Analysis depth & sections (see [ANALYSIS_DEPTH.md](ANALYSIS_DEPTH.md)):**

| Flag | Default | What it does |
|------|---------|--------------|
| `--depth` | `normal` | `shallow` / `normal` / `deep` / `extreme` |
| `--sections` | depth's defaults | Comma-separated rubrics from the 8 available |

### `download` — just the audio

```
yt-insight download URL [OPTIONS]
```

Downloads to `cache/{video_id}.mp3` and saves metadata to
`cache/{video_id}.meta.json`. Useful as a preview step.

### `transcribe` — just the transcript

```
yt-insight transcribe PATH_OR_URL [OPTIONS]
```

Transcribes a local audio file or a YouTube URL. Writes JSON to
`./transcript.json` by default (or `--output PATH`).

### `analyze` — just the analysis

```
yt-insight analyze [TRANSCRIPT_FILE] [OPTIONS]
```

Two modes:

1. **From an existing transcript JSON** (skip download + transcription):
   ```bash
   yt-insight analyze outputs/foo.transcript.json
   ```

2. **Transcribe + analyze in one shot** (from a URL or local audio):
   ```bash
   yt-insight analyze --audio "https://youtube.com/watch?v=..."
   ```

### `estimate` — dry run, no network

```
yt-insight estimate URL [OPTIONS]
```

Fetches YouTube metadata (1 HTTP call) and predicts:
- audio duration
- transcript size (words / tokens)
- whisper model + compute time
- LLM strategy (single-shot vs chunk+merge)
- estimated LLM analysis time

Add `--json` to get machine-readable output, or `--hardware gpu_rtx_3060`
to test against different GPUs.

---

## Global options

These flags work on every subcommand:

| Flag | Default | What it does |
|------|---------|--------------|
| `--config PATH` | `config.yaml` | Use a custom config file |
| `--verbose`, `-v` | off | Enable debug logging |
| `--no-console` | off | Disable Rich console rendering |
| `--help` | — | Show help for any subcommand |

---

## Common recipes

### 1. Standard analysis

```bash
yt-insight all "https://youtube.com/watch?v=VIDEO_ID" --language fr
```

### 2. Use a remote LLM server (laptop → desktop)

```bash
yt-insight all "URL" --llamacpp-url http://192.168.1.42:8080
```

### 3. Run only certain steps

```bash
# Just download + transcribe (no analysis)
yt-insight all "URL" --steps download,transcribe

# Just analysis (you already have the transcript)
yt-insight all "URL" --steps analyze
```

### 4. Re-run without re-transcribing (cache hit)

The 1st run takes ~10-20 min (transcription is the slow part).
The 2nd run is ~1-5 min (analysis only):

```bash
# 1st run: download + transcribe + analyze
yt-insight all "URL" --language fr

# 2nd run: same command — automatically skips download + transcribe
yt-insight all "URL" --language fr
```

The transcript is cached in `cache/{video_id}.transcript.json`.

### 5. Lightweight Whisper for 4GB VRAM

```bash
yt-insight all "URL" --whisper-model medium --whisper-chunk-length 30
```

`medium` uses 1.5GB VRAM (vs 2.5GB for `large-v3`), so combined with
`--whisper-chunk-length 30` you stay comfortably under 4GB.

### 6. Force CPU transcription

```bash
WHISPER_DEVICE=cpu yt-insight all "URL"
```

Useful if your GPU is busy, or for guaranteed non-OOM runs.

### 7. Long videos (3h+)

```bash
yt-insight all "URL" \
    --llamacpp-max-prompt-tokens 80000 \
    --llamacpp-timeout 14400 \
    --llamacpp-idle-timeout 1800
```

The 30-min idle timeout covers even the slowest prompt processing
on consumer GPUs. `-c 100000` on the server side is recommended
for very long videos.

### 8. Shallow analysis (TL;DR mode)

```bash
yt-insight all "URL" --depth shallow
```

3 quotes, 5 key points, ~1024 tokens output. Fast.

### 9. Critical-rhetoric analysis

```bash
yt-insight all "URL" \
    --depth deep \
    --sections forces,weaknesses,contradictions,biases
```

Skips the "Concepts" and "Implications" rubrics to focus on critique.

See [ANALYSIS_DEPTH.md](ANALYSIS_DEPTH.md) for the full depth/section
matrix and tuning guide.

### 10. Pipe-friendly / cron

```bash
yt-insight all "URL" --no-console
```

Suppresses Rich colors and live panel. Use in cron jobs, log files,
or when piping to `grep` / `less`.

---

## Caching behavior

| Artifact | Path | Reused on re-run? |
|----------|------|-------------------|
| Audio (mp3) | `cache/{video_id}.mp3` | ✅ yes, if file exists |
| Video metadata | `cache/{video_id}.meta.json` | ✅ yes |
| **Transcript** | `cache/{video_id}.transcript.json` | ✅ **yes** — biggest time saver |
| Markdown analysis | `outputs/...md` | ⚠️ overwritten unless a depth tag is appended |
| JSON analysis | `outputs/...json` | ⚠️ overwritten unless a depth tag is appended |

Output filenames are auto-suffixed with the depth (e.g.
`...-deep-qwen3-50k.md`), so different depths on the same video
**do not overwrite each other** — you can keep all variants.

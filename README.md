# 🎙️ YT-Insight

**YouTube transcription + LLM analysis pipeline, runs locally.**

Point it at a YouTube URL, get back a structured Markdown / JSON
analysis (summary, key points, deep analysis, timestamped quotes) —
all on your own hardware, no cloud APIs.

```
yt-insight all "https://youtube.com/watch?v=VIDEO_ID" --language fr
```

## What it does

```
URL ──> download (yt-dlp) ──> transcribe (faster-whisper) ──> analyze (llama.cpp) ──> outputs/{md, json}
         cache/{id}.mp3         cache/{id}.transcript.json                       auto-suffixed by depth
```

- 🎙️ **Transcription** via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CUDA-accelerated Whisper, runs on your GPU)
- 🤖 **Analysis** via [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server` (any OpenAI-compat model works)
- 📊 **Output** in both Markdown (human) and JSON (machine), with auto-suffixed filenames to keep multiple variants
- 🔍 **Configurable analysis depth**: 4 presets × 8 rubrics, fully orthogonal

## Typical use cases

- 📝 **Notes & summaries** of long podcasts and lectures (1h+ videos become 1-page summaries)
- 🎓 **Course material** — turn recorded classes into searchable study notes
- 🗣️ **Speech / talk analysis** — extract arguments, identify weaknesses, find contradictions
- 📚 **Research corpus** — bulk-process hundreds of videos, then grep / query the JSON outputs
- 🔎 **YouTube as a research tool** — search inside transcripts, jump to specific timestamps

## Quickstart

### 1. Install

```bash
git clone https://github.com/nikodindon/YT-Insight.git
cd YT-Insight
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

For GPU-accelerated Whisper, see the [CUDA setup section](#gpu-accelerated-whisper).

### 2. Start your LLM server

In a separate terminal:

```bash
# Example: Qwen3.6 35B-A3B MoE, 65k ctx, GPU offload
llama-server -m ~/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
    -c 65536 -ngl 99 --n-cpu-moe 36 -t 10 \
    -ctk q8_0 -ctv q8_0 --host 0.0.0.0 \
    -b 2048 -ub 2048 --timeout 60000 --jinja
```

Any OpenAI-compat server works (Ollama, vLLM, LM Studio). Verify:

```bash
curl -s http://localhost:8080/health
# {"status":"ok"}
```

### 3. Run

```bash
yt-insight all "https://youtube.com/watch?v=VIDEO_ID" --language fr
```

That's it. Markdown + JSON will land in `./outputs/`, audio + transcript
cached in `./cache/`.

## State of the project

**257/257 tests pass** ✅

| Phase | Status |
|-------|--------|
| Phase 1 — MVP (download, transcribe, analyze, write) | ✅ Done |
| Phase 2 — Backends supplémentaires (Ollama, vLLM, OpenAI) | ⏳ Backlog |
| Phase 3 — Features avancées (playlists, PDF, search) | ⏳ Backlog |
| Phase 4 — Qualité (CI/CD, Docker, PyPI) | ⏳ Backlog |
| Phase 5 — Streaming, progress, interactivity | 🟡 Streaming SSE done, others open |

See [docs/ROADMAP.md](docs/ROADMAP.md) for full ticket list and priorities.

## Architecture overview

```
                ┌─────────────────┐
                │   yt-insight    │  CLI (Typer)
                │   (Typer)       │
                └────────┬────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  downloader  │  │  transcriber │  │   analyzer   │
│  (yt-dlp)    │─>│ (faster-     │─>│ (llama.cpp   │
│              │  │  whisper)    │  │  via OpenAI  │
│              │  │              │  │  API)        │
└──────────────┘  └──────────────┘  └──────────────┘
       │                 │                  │
       ▼                 ▼                  ▼
   cache/           cache/            outputs/
   {id}.mp3        {id}.transcript    {date}_{slug}.{md,json}
                   .json
```

### Why faster-whisper?

CTranslate2-based Whisper reimplementation. **~4x faster than
openai-whisper** on the same hardware, with **lower VRAM usage**
(int8 quantization), and a clean Python API. Pure local.

### Why llama.cpp in local mode?

- **Privacy**: video transcripts and analyses never leave your network.
- **Cost**: no per-token charges.
- **Performance**: a 35B-A3B MoE on a 16GB GPU runs at ~60 tok/s,
  faster than most cloud APIs from a wall-clock perspective.
- **Compatibility**: llama-server speaks the OpenAI chat completions
  API, so any tool that supports OpenAI can use it.

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
# === Whisper ===
WHISPER_MODEL=large-v3       # tiny/base/small/medium/large-v3/distil-large-v3
WHISPER_DEVICE=auto          # auto/cuda/cpu
WHISPER_COMPUTE_TYPE=int8    # int8/float16/float32

# === llama.cpp ===
LLAMACPP_BASE_URL=http://localhost:8080
LLAMACPP_MODEL=Qwen3.6-35B-A3B-UD-IQ3_S.gguf
LLAMACPP_TIMEOUT_S=7200      # 2h wall-clock
LLAMACPP_IDLE_TIMEOUT_S=600  # 10min per-token
LLAMACPP_MAX_PROMPT_TOKENS=*** # 50k, soft cap → chunk+merge above
```

A `config.yaml` is also supported for structured config (see
[docs/USAGE.md](docs/USAGE.md#subcommands)).

## Documentation

| Doc | What's in it |
|-----|--------------|
| [docs/USAGE.md](docs/USAGE.md) | Full CLI reference, all flags, common recipes, caching behavior |
| [docs/OUTPUT.md](docs/OUTPUT.md) | Markdown vs JSON, when to use which, structure, programmatic access |
| [docs/ANALYSIS_DEPTH.md](docs/ANALYSIS_DEPTH.md) | The 4 depth presets × 8 section rubrics, matrix, recommended recipes |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Backlog, completed tickets, design notes |
| [docs/session-handoff.md](docs/session-handoff.md) | End-of-session summary, state of the project, bug history. **Read this on resume.** |

## Project structure

```
YT-Insight/
├── yt_insight/
│   ├── cli.py                # Typer CLI (5 subcommands)
│   ├── downloader/           # yt-dlp wrapper
│   ├── transcriber/          # faster-whisper wrapper
│   ├── analyzer/             # llama.cpp + prompts + depth
│   │   ├── depth.py          # Depth + Section enums, presets
│   │   ├── prompts.py        # Prompt builders (depth-aware)
│   │   └── llamacpp_local.py # Streaming SSE + idle timeout
│   ├── output/               # Markdown + JSON writers
│   ├── utils/                # config, logger, text_utils
│   └── estimate.py           # Dry-run cost estimator
├── tests/                    # 234 tests, pytest
├── docs/                     # USAGE, OUTPUT, ANALYSIS_DEPTH, ROADMAP
├── cache/                    # Audio + transcripts (gitignored)
├── outputs/                  # Markdown + JSON analyses
├── config.yaml               # Optional structured config
├── .env.example
├── pyproject.toml
└── README.md                 # ← you are here
```

## Performance

On a GTX 1650 Super 4GB + 16GB RAM (the dev box this was built on):

| Phase | 1h video | 3h video |
|-------|----------|----------|
| Download (yt-dlp) | ~10s | ~30s |
| Transcription (Whisper medium, CUDA) | ~5-8 min | ~25-35 min |
| Analysis (Qwen3.6 35B-A3B, single-shot) | ~10-20 min | chunk+merge, ~30-50 min |

For 4GB VRAM GPUs, **use `--whisper-model medium`** (1.5GB VRAM vs
2.5GB for large-v3). See [docs/USAGE.md](docs/USAGE.md#lightweight-whisper-for-4gb-vram).

## GPU-accelerated Whisper

For NVIDIA GPUs, faster-whisper needs CTranslate2 with CUDA support.
This is **not** installed by default. See the project's
[CUDA setup notes](https://github.com/SYSTRAN/faster-whisper#cuda)
for your CUDA version. Quick check:

```bash
python3 -c "import ctranslate2; print(ctranslate2.__version__); print('CUDA:', ctranslate2.get_cuda_device_count())"
# Should print CUDA: 1 (or more)
```

## License

Personal project. No license specified yet — treat as proprietary until
a `LICENSE` file is added.

# YT-Insight — Session Handoff

> **This document is the canonical handoff** for resuming work on the
> YT-Insight project. Read this at the start of any session, then
> load the `yt-insight` skill for full context.

## TL;DR

YT-Insight is a CLI that downloads a YouTube video, transcribes it
with `faster-whisper`, and produces a structured Markdown + JSON
analysis with a local LLM (`llama.cpp`). **All local, no cloud APIs.**

- Repo: `/home/niko/YT-Insight`
- Remote: `github.com/nikodindon/YT-Insight`
- Branch: `main` (always), 21 commits ahead of initial commit
- Tests: **252/252 passing**
- Working tree: clean
- All outputs already committed

## How to resume

1. Load the `yt-insight` skill: `skill_view(name="yt-insight")` or
   ask "load the yt-insight skill".
2. Read this file: `docs/session-handoff.md` (in the repo).
3. Read `docs/ROADMAP.md` for the current backlog.
4. Read `README.md` for the project overview.
5. Read `docs/USAGE.md`, `docs/OUTPUT.md`, `docs/ANALYSIS_DEPTH.md`
   for the detailed docs.

The skill points to this file. The file points to the skill.
Together they bootstrap a new session.

## Project state (as of end of session 2026-06-08)

### Architecture

```
URL ─> downloader (yt-dlp) ─> transcriber (faster-whisper) ─> analyzer (llama.cpp) ─> outputs/{md, json}
        cache/{id}.mp3          cache/{id}.transcript.json
```

### Code layout

```
yt_insight/
├── cli.py                      # Typer CLI (5 subcommands)
├── downloader/ytdlp_downloader.py
├── transcriber/{base, faster_whisper_transcriber}.py
├── analyzer/
│   ├── base.py                 # BaseAnalyzer abstract
│   ├── depth.py                # Depth + Section enums, presets
│   ├── prompts.py              # depth-aware prompt builders
│   └── llamacpp_local.py       # streaming SSE, idle timeout, depth wiring
├── output/{console, file_writer}.py
├── utils/{config, logger, text_utils}.py
└── estimate.py
tests/                          # 252 tests, pytest
docs/                           # USAGE, OUTPUT, ANALYSIS_DEPTH, ROADMAP, session-handoff
```

### Key features (all working, all tested)

| Feature | Status | Where |
|---------|--------|-------|
| Download audio via yt-dlp | ✅ | `downloader/` |
| Transcribe with faster-whisper (CUDA + CPU fallback) | ✅ | `transcriber/` |
| Cache transcript in `cache/{video_id}.transcript.json` | ✅ | `transcriber/` |
| Analyze with llama.cpp (OpenAI-compat API) | ✅ | `analyzer/` |
| Streaming SSE with idle timeout | ✅ | `analyzer/llamacpp_local.py` |
| Rich Live panel for token streaming | ✅ | `cli.py:_run_analyze_with_live` |
| Auto-suffixed output filenames (no overwrite) | ✅ | `output/file_writer.py` |
| `--depth shallow\|normal\|deep\|extreme` | ✅ | `analyzer/depth.py` |
| `--sections` (8 rubrics) | ✅ | `analyzer/depth.py` |
| Markdown + JSON output | ✅ | `output/file_writer.py` |
| Estimate cost before running (`yt-insight estimate`) | ✅ | `estimate.py` |
| OOM auto-fallback (CUDA → CPU) | ✅ | `transcriber/` |
| 4-strategy tolerant JSON parsing | ✅ | `analyzer/llamacpp_local.py` |
| Debug-on-failure (raw model response saved) | ✅ | `analyzer/llamacpp_local.py` |

### Defaults (the "vibe" of the project)

| Setting | Default | Why |
|---------|---------|-----|
| Whisper model | `large-v3` | Best quality, ~2.5GB VRAM |
| Whisper chunk length | `auto` (30s for medium, 20s for large-v3) | Memory-safe |
| Whisper beam size | `3` | Good accuracy/speed tradeoff |
| LLM timeout | 7200s (2h) | Long enough for 3h videos |
| LLM idle timeout | 1800s (30 min) | Covers worst-case prefill on consumer GPUs |
| LLM max prompt tokens | 50000 | Safe under llama.cpp's 65k default ctx |
| Chunk overlap | 500 tokens | Avoid context loss at chunk boundaries |
| Temperature (normal) | 0.2 | Low for structured JSON |
| LLM disable_thinking | `True` | Qwen3 thinking mode off (cleaner JSON) |

### Depth presets

| Depth | max_tokens | key_points | quotes | temp | default sections |
|-------|-----------|------------|--------|------|------------------|
| shallow | 1024 | 5 | 3 | 0.4 | forces |
| normal | 4096 | 15 | 5 | 0.2 | forces, concepts, implications |
| deep | 8192 | 25 | 10 | 0.1 | forces, concepts, implications, weaknesses, contradictions, biases |
| extreme | 16384 | 40 | 15 | 0.0 | forces, concepts, implications, weaknesses, contradictions, biases, limitations, context_gaps |

### Sections (8 rubrics, in display order)

1. `forces` — Forces du contenu
2. `concepts` — Concepts centraux
3. `implications` — Implications et perspectives
4. `weaknesses` — Faiblesses et limites
5. `contradictions` — Contradictions internes
6. `biases` — Biais cognitifs et rhétoriques
7. `limitations` — Limites de cette analyse
8. `context_gaps` — Contexte manquant

## What was done in this session (2026-06-08)

This is the chronological narrative. **Skim this on resume** to
remember the arc, then dive into the docs.

### Phase 1 — Streaming SSE + Rich Live panel (morning)

- Implemented `LlamaCppLocalAnalyzer.stream_chat()` with
  `_stream_deltas()` helper
- Idempotent idle check on EVERY line read (not just data lines)
- Rich Live panel with preview + stats (chars/tokens/s)
- `transient=True` to avoid ghost panels
- 6 streaming tests in `TestStreamingChat`

### Phase 2 — Asselineau 3h+ runs (mid-morning)

- Hit 4 bugs back-to-back: timeout, idle timeout, prefill cancel
- Fixed: `MAX_PROMPT_TOKENS 28k→50k`, `TIMEOUT_S 30min→2h`,
  `IDLE_TIMEOUT_S 2min→10min` (then 30min)
- Critical fix: `last_token_at` resets on every `data:` line, not
  on entry — otherwise prefill counts as idle and aborts
- Also: `disable_thinking=True` default for Qwen3 to get clean JSON

### Phase 3 — Auto-suffix output filenames (mid-morning)

- Outputs now include depth tag (e.g. `...-extreme.md`)
- `output/file_writer.py:_build_path(tag)` adds the suffix
- 2 tests in `tests/test_output.py`

### Phase 4 — `--depth` and `--sections` (noon)

- New module `analyzer/depth.py` with `Depth` + `Section` enums,
  `DEPTH_PRESETS`, `coerce_depth`, `coerce_sections`
- Refactored `prompts.py` to take `depth` + `sections`
- 25 tests in `tests/test_depth.py`

### Phase 5 — Docs split (afternoon)

- `README.md` shrunk from 1292 → 209 lines
- `docs/USAGE.md` (274 lines) — CLI reference
- `docs/OUTPUT.md` (194 lines) — md vs json
- `docs/ANALYSIS_DEPTH.md` (207 lines) — depth/section guide
- All cross-links verified

### Phase 6 — Jancovici (10 min interview) test + 4 bugs (afternoon)

Tested with a small video to validate the full pipeline. Hit 4
bugs in 2 attempts:

1. `create_analyzer` didn't have `depth`/`sections` kwargs
   (added in commit `7514f29`).
2. `create_analyzer` propagated `None` to `__init__` for scalar
   kwargs from the CLI; logger crashed with `%d None` (fixed in
   `968e6d2` with `_env_int_or_default` / `_env_float_or_default`).
3. `_extract_json_object` couldn't parse truncated/invalid JSON
   output. The model hit `max_tokens=1024` and was cut mid-JSON
   (fixed in `2e2e0e9` with balanced-brace matching + `json_repair`
   + debug-on-failure saving to `cache/debug/{ts}.txt`).
4. `_validate_depth_sections` was calling `coerce_sections(None)`
   which returns the NORMAL default (3 sections), so
   `--depth extreme --sections ""` gave 3 sections, not 8 (fixed
   in `c47cc35`).

### Phase 7 — Ghost panel fix (late afternoon)

- Initial panel rendered an empty Panel border, producing a visible
  "ghost" duplicate.
- Fix: `make_panel()` returns a `Text` placeholder while buffer
  is empty, switching to the full `Panel` once tokens arrive.
- 1 commit (`70786d9`).

## Recent bugs that were painful (don't repeat them)

These are the gotchas we hit in this session. **Read this list
before touching the related code.**

1. **`logger.info("max_prompt=%d", self.max_prompt_tokens)` crashes
   if value is `None`**. Either convert None to default in the
   factory, or use `%s` everywhere.

2. **Typer passes `None` for every flag the user didn't set**. The
   factory must convert None → default. The `__init__` must handle
   None for kwargs that have a "use the preset" semantics.

3. **LLM JSON output is often truncated or invalid**. The model
   hits `max_tokens` and stops mid-JSON. Use balanced-brace
   matching + `json_repair`, not just regex.

4. **`coerce_sections(None)` returns the NORMAL default**, not
   `None`. If you want depth-specific defaults, leave `None` as
   `None` and let the analyzer pick.

5. **Idle timeout during prefill**. The prefill of a 30k-token
   prompt on a 1650S takes 8-14 minutes. Our `idle_timeout_s`
   must be > that. Default is now 1800s (30 min). Don't lower it
   without re-validating on the actual hardware.

6. **LLM temperature 0.0 with low max_tokens is brittle**. If the
   model needs more tokens than allowed, you get truncated JSON.
   `extreme` depth has `max_tokens=16384` and `temperature=0.0`
   (deterministic), which is fine for the use case.

## Roadmap (current)

See `docs/ROADMAP.md` for the full list. Key tickets:

### Backlog (priority order)
- [ ] **Fix chunked timestamps** (low priority, 2-3h):
  the chunk+merge mode loses citation timestamps because the merge
  step only sees the partial summaries. Need to pass timestamps
  through the merge prompt.
- [ ] **CI/CD GitHub Actions** (Phase 4): run pytest on every push.
- [ ] **Dockerisation** (Phase 4): single-image build for CI/local.
- [ ] **PyPI packaging** (Phase 4): `pip install yt-insight`.
- [ ] **Backend Ollama / vLLM / OpenAI** (Phase 2): only llama.cpp
  is implemented.
- [ ] **Playlists** (Phase 3): analyze a whole YouTube playlist.
- [ ] **PDF export** (Phase 3): convert the Markdown to a styled
  PDF.
- [ ] **Full-text search in transcripts** (Phase 3): index N
  transcripts, then grep.
- [ ] **Compare subcommand** (Phase 3): side-by-side comparison of
  multiple depths on the same video. (User asked for this; small
  ticket, ~30 min.)

### Done in this session
- [x] Streaming SSE
- [x] Rich Live panel (no ghost panels)
- [x] 3h+ video support (long timeouts)
- [x] Auto-suffixed output filenames
- [x] `--depth` and `--sections`
- [x] Tolerant JSON extraction
- [x] Docs split (USAGE / OUTPUT / ANALYSIS_DEPTH)
- [x] Debug-on-failure (`cache/debug/`)
- [x] `json_repair` for last-resort JSON repair

## What the user is currently doing

The user has 3 analysis runs in progress on Asselineau 3h15
(`ktuubyVVE_M`):
- normal
- deep
- extreme

**IMPORTANT**: those runs were started BEFORE the `c47cc35` fix.
They will have used the wrong sections (3 NORMAL sections)
because the bug applied to them. The user will likely want to
cancel them and re-run with the fix.

**The fix is committed and pushed** (`c47cc35` on `main`).

The Asselineau transcript is cached in
`cache/ktuubyVVE_M.transcript.json` (~43k tokens, 3h15). All
re-runs skip the download + transcription phases.

## User profile and preferences

- Hardware: Linux Mint, GTX 1050 (dev PC) + GTX 1650 Super (LLM
  server on the LAN at `http://100.118.85.70:8080`).
- Prefers local LLMs over cloud APIs.
- Wants autonomous agents that figure things out independently.
- Prefers pastel visuals for games, multiple-options presentations.
- Likes iterative development with regular progress checkpoints.
- Hates manual corrections — expects self-correcting agents.
- For benchmark/optimization tasks: wants ALL viable options
  tested, then a single "ultimate command" as final deliverable.
- Mid-task redirects are OK if grounded in data.
- Use `clarify` before any op > 20 min.
- Commits with `git -c user.name='Niko' -c user.email='niko@local'`.

## Test command

```bash
cd /home/niko/YT-Insight
python3 -m pytest tests/ -q
```

Expected: `252 passed in ~1s`.

If you add a new test file or new test class, **also add it to
this count** in this file.

## Outputs inventory

The `outputs/` directory has all analyses run so far:

| File | Video | Depth |
|------|-------|-------|
| `2026-06-08_stanford-mse435-economics-of-the-ai-supercycle-spring-2026-a.md` (and .json) | Stanford AI supercycle, 49:15 | normal (manual) |
| `2026-06-08_le-discours-de-francois-hollande-au-bourget.md` (and .json) | Hollande Bourget 5:20 | normal (manual) |
| `2026-06-08_francois-asselineau-lhistoire-de-france-...-qwen3-6-35b--p100k.md` (and .json) | Asselineau 3h15 | shallow |
| `2026-06-08_climat-jean-marc-jancovici-face-a-thomas-sotto-sur-rtl-integ-...-qwen3-6-35b--p100k.md` (and .json) | Jancovici RTL 10:43 | shallow + extreme |

The user has been using the `qwen3-6-35b--p100k` suffix, which
suggests they have `LLAMACPP_MAX_PROMPT_TOKENS=*** or
`--llamacpp-max-prompt-tokens 100000` in their `.env` (or it
leaked from a previous command).

## Where to find what

| If you want to know… | Read… |
|----------------------|-------|
| What the project does | `README.md` |
| How to use the CLI | `docs/USAGE.md` |
| What output files look like | `docs/OUTPUT.md` |
| All depth × section options | `docs/ANALYSIS_DEPTH.md` |
| What's planned next | `docs/ROADMAP.md` |
| This session's work | this file (`docs/session-handoff.md`) |
| A function's API | the docstring in the source file |
| Why a design decision was made | the test or commit message that introduced it |

## Conventions to follow

1. **No markdown in chat** — the user prefers plain text.
2. **Always `git -c user.name='Niko' -c user.email='niko@local' commit`**
   with a clear message. Push to `main`.
3. **Test before commit**: `python3 -m pytest tests/ -q` must
   show all green.
4. **Update the docs** when changing CLI behavior:
   `docs/USAGE.md` for flags, `docs/OUTPUT.md` for outputs,
   `docs/ANALYSIS_DEPTH.md` for depth/section changes,
   `docs/ROADMAP.md` for backlog changes.
5. **Update this file** (`docs/session-handoff.md`) at the END of
   each session so the next one has a starting point.
6. **Use `clarify`** for non-trivial decisions.
7. **Don't fabricate** — if a tool or run failed, say so.

## Quick run command (the "ultimate command")

```bash
# Standard analysis (most common case)
yt-insight all "https://www.youtube.com/watch?v=VIDEO_ID" \
    --language fr \
    --llamacpp-url http://100.118.85.70:8080 \
    --llamacpp-timeout 14400 \
    --depth normal
```

## Bumping the test count

When tests are added, the count in this file's "Project state"
section must be updated. The test command is:

```bash
python3 -m pytest tests/ -q
```

The expected count is documented in 2 places:
1. `README.md` line: `- [x] Suite de tests complète (N/N ✅)`
2. This file: `Tests: **252/252 passing**`

Both must be updated together.

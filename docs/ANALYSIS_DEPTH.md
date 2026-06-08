# Analysis depth & sections

YT-Insight exposes **two orthogonal axes** to control the analysis
output: **depth** (numeric levers) and **sections** (semantic content).
This page explains both, the relationship between them, and the
recommended recipes for common use cases.

## TL;DR

```bash
# Default (recommended for most cases)
yt-insight all "URL"

# Quick TL;DR
yt-insight all "URL" --depth shallow

# Critical / rhetorical analysis
yt-insight all "URL" --depth deep --sections forces,weaknesses,contradictions,biases

# Research-grade, no holds barred
yt-insight all "URL" --depth extreme
```

## Why two axes?

Because **what to ask** and **how much to ask** are different questions.

- **Depth** answers "how much?" — how many key points, how many
  quotes, how much prose, how deterministic the model is.
- **Sections** answers "what?" — which rubrics appear in the analysis
  markdown (Forces? Weaknesses? Biais cognitifs?).

A political speech needs `weaknesses` and `biases`. A research talk
needs `limitations` and `context_gaps`. A tutorial doesn't need
either. By decoupling the two, you can keep the depth preset and
just swap rubrics.

## Depth — 4 opinionated presets

| Depth | max_tokens | num_key_points | num_quotes | temperature | default sections |
|-------|------------|----------------|------------|-------------|------------------|
| `shallow` | 1024 | 5 | 3 | 0.4 | forces |
| `normal`  | 4096 | 15 | 5 | 0.2 | forces, concepts, implications |
| `deep`    | 8192 | 25 | 10 | 0.1 | forces, concepts, implications, weaknesses, contradictions, biases |
| `extreme` | 16384 | 40 | 15 | 0.0 | forces, concepts, implications, weaknesses, contradictions, biases, limitations, context_gaps |

**Temperature** decreases with depth:
- `shallow` (0.4) is more **creative** — the model has freedom to
  pick interesting angles even if they're not perfectly faithful.
- `extreme` (0.0) is **deterministic** — the same prompt always
  produces the same output, useful for reproducibility.

**max_tokens** is the hard ceiling on output length. Beyond that,
the model is cut off mid-sentence. Default presets are sized so the
model has room to produce the requested counts comfortably.

**num_key_points / num_quotes** are surfaced as count ranges in
the prompt (e.g. "13 to 17 key points"), so the model has leeway.

## Sections — 8 rubrics

| Section key | Prompt title | What it does |
|-------------|--------------|--------------|
| `forces` | Forces du contenu | The strong points: well-argued, concrete examples, effective structuring. |
| `concepts` | Concepts centraux | Key ideas explained or defined by the speaker. |
| `implications` | Implications et perspectives | Consequences, projections, and lessons the author draws (or that one can draw). |
| `weaknesses` | Faiblesses et limites | Badly-argued points, missing data, sophisms, approximations, fuzzy zones. |
| `contradictions` | Contradictions internes | Elements of the discourse that contradict or invalidate each other. |
| `biases` | Biais cognitifs et rhétoriques | Reasoning biases: confirmation bias, cherry-picking, unfounded emotional appeals, abusive generalisations, etc. |
| `limitations` | Limites de cette analyse | What this analysis cannot conclude: lack of context, scope, partiality, etc. |
| `context_gaps` | Contexte manquant | Context that the speaker deliberately omits, and that would change the interpretation. |

**Order matters.** The rubrics appear in the prompt in the order you
specify them via `--sections`, so use a logical order:

```bash
# Good: descriptive → critical
yt-insight all "URL" --sections forces,weaknesses,contradictions

# Confusing: same thing reversed
yt-insight all "URL" --sections contradictions,weaknesses,forces
```

## How the two interact

`--depth` controls the **numeric levers**. `--sections` controls
**which rubrics** are emitted. If both are given:

- `--depth`'s numeric levers are used (max_tokens, num_key_points,
  num_quotes, temperature).
- `--sections` **overrides** the depth's default rubrics.

Example:

```bash
# Numeric levers of "extreme" (40 pts, 15 quotes, max 16k tokens)
# but only the "forces" rubric in the analysis block
yt-insight all "URL" --depth extreme --sections forces
```

If you only specify `--depth`, the depth's default sections apply.
If you only specify `--sections`, depth defaults to `normal`.

## Recommended recipes

### Podcast / casual talk

```bash
yt-insight all "URL" --depth normal
# 15 key points, 5 quotes, full Forces/Concepts/Implications
```

### Long lecture / course

```bash
yt-insight all "URL" --depth deep
# 25 key points, 10 quotes, 6 rubrics
```

### Political speech / rhetoric analysis

```bash
yt-insight all "URL" \
    --depth deep \
    --sections forces,weaknesses,contradictions,biases
# 25 key points, 10 quotes, 4 critical rubrics
```

### Academic talk / research

```bash
yt-insight all "URL" --depth extreme
# 40 key points, 15 quotes, 8 rubrics (all)
# Includes Limitations and Context Gaps
```

### Quick summary (mobile, on the go)

```bash
yt-insight all "URL" --depth shallow
# 5 key points, 3 quotes, only Forces. ~10x faster than normal.
```

### Just want the speaker's positions + critique

```bash
yt-insight all "URL" \
    --depth normal \
    --sections forces,weaknesses,contradictions
# Standard 15 key points, 5 quotes, 3 critical rubrics.
```

## How the LLM prompt is built

The numeric levers (e.g. "8 to 12 key points") and the rubrics
(e.g. "**Forces** — Les points forts : ...") are inserted into
the prompt as plain text, in French by default. The model is then
asked to produce a single JSON object with the requested structure.

This is a **prompt engineering** approach rather than a tool-call
approach. Pros: simpler, works with any chat-compat server (llama.cpp,
Ollama, vLLM, LM Studio, OpenAI). Cons: less strict than JSON Schema
tool calls.

## Invalid values

Both flags are validated **before** any network call:

```bash
$ yt-insight all "URL" --depth bogus
✗ Unknown depth 'bogus'. Valid options: shallow, normal, deep, extreme

$ yt-insight all "URL" --sections forces,bogus
✗ Unknown section(s): ['bogus']. Valid options: biases, concepts,
  context_gaps, contradictions, forces, implications, limitations,
  weaknesses
```

## Performance impact

Higher depth = more output tokens = longer analysis time:

| Depth | Wall-clock time (1h video, normal server) |
|-------|--------------------------------------------|
| `shallow` | ~5 min |
| `normal` | ~10-20 min |
| `deep` | ~20-30 min |
| `extreme` | ~30-45 min |

Transcription time is the same regardless of depth (only the LLM
analysis changes). These estimates assume a typical LLM on a
single consumer GPU.

## Implementation

The depth/sections logic lives in:
- `yt_insight/analyzer/depth.py` — `Depth`, `Section`, `DEPTH_PRESETS`,
  `coerce_depth()`, `coerce_sections()`.
- `yt_insight/analyzer/prompts.py` — `build_analysis_prompt()`,
  `build_chunk_prompt()`, `build_merge_prompt()` take `depth` and
  `sections` and render the appropriate prompt.
- `yt_insight/analyzer/llamacpp_local.py` — `LlamaCppLocalAnalyzer.__init__`
  accepts `depth` and `sections`; `temperature` and `max_tokens`
  are auto-overridden by the depth preset unless explicitly set.

See the source for details, or `tests/test_depth.py` for unit tests
covering every preset and section combo.

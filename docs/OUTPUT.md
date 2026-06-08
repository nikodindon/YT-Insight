# YT-Insight — Output formats

By default, `yt-insight all` produces **two output files** for the
same analysis: one Markdown (`.md`) and one JSON (`.json`). They
contain the same data, optimized for different consumers.

## Why two formats?

| Format | Optimized for | Consumer |
|--------|---------------|----------|
| **Markdown** (`.md`) | **Reading** — human eyes, prose, formatted sections, clickable timestamps | You, in your terminal, in Obsidian, in Notion |
| **JSON** (`.json`) | **Processing** — strict schema, parseable, machine timestamps, structured arrays | Scripts, notebooks, dashboards, downstream tools |

They're not redundant — they have **different shapes**:

| Aspect | Markdown | JSON |
|--------|----------|------|
| Timestamp format | `[1:02:03]` or `[12:34]` | `3723.0` (seconds, float) |
| Sections | `## 📝 Résumé` etc. | `analysis.summary` (string) |
| Quotes | Table with `[mm:ss]` column | `quotes: [{text, timestamp_seconds}]` |
| Key points | Numbered markdown list | `key_points: ["...", "..."]` |
| Embeds full transcript | Yes (if `--include-transcript`) | Yes (in `transcription` field) |

## Why both?

Because they answer **different questions**:

- "What did the speaker actually say at 18:45?" → Markdown
  (readable, clickable).
- "Build me a YouTube player that auto-jumps to the citations." → JSON
  (machine-readable timestamps).
- "What were the 15 main points?" → both, but the JSON is easier
  to grep through 100 analyses.
- "Save the analysis in my Obsidian vault." → Markdown
  (renders natively).
- "Aggregate stats across 50 videos: avg num_key_points per category
  of content." → JSON (programmatic).

You can also **produce only one**:

```bash
# Only Markdown
yt-insight all "URL" --format markdown

# Only JSON (smaller, parseable)
yt-insight all "URL" --format json
```

## Markdown format

```
outputs/2026-06-08_<video-slug>[-depth][-qwen3-50k].md
```

Top of the file (YAML front matter):

```markdown
---
title: <video title>
url: <url>
date: 2026-06-08
model: Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
backend: llamacpp-local
language: fr
duration: 3:15:45
topic: Souveraineté française et histoire
tone: Analytique et engagé
---
```

Then sections:

- `## 📝 Résumé détaillé` — prose continue, depth-driven word count
- `## 🎯 Points clés` — numbered list, depth-driven count
- `## 🔍 Analyse approfondie` — sub-rubrics (Forces / Concepts /
  Implications / Weaknesses / etc.) depending on `--sections`
- `## 💬 Citations notables` — markdown table with `[mm:ss]` column
- `## 📄 Transcription complète` — embedded if
  `--include-transcript` (default on)

## JSON format

```
outputs/2026-06-08_<video-slug>[-depth][-qwen3-50k].json
```

Strict top-level shape:

```json
{
  "title": "<video title>",
  "url": "https://...",
  "date": "2026-06-08T15:30:00",
  "metadata": {
    "video_id": "...",
    "title": "...",
    "channel": "...",
    "duration_seconds": 11745.0,
    "duration_str": "3:15:45",
    "upload_date": "20140626",
    "view_count": 79484
  },
  "transcription": {
    "language": "fr",
    "language_probability": 0.99,
    "duration_seconds": 11745.0,
    "duration_str": "3:15:45",
    "num_segments": 1526,
    "num_tokens_est": 43503,
    "model_name": "medium",
    "text": "<full transcript, possibly truncated>",
    "segments": [
      {"start": 0.0, "end": 30.5, "text": "..."},
      ...
    ]
  },
  "analysis": {
    "summary": "...",
    "key_points": ["...", "..."],
    "analysis": "**Forces** ...\n\n**Concepts** ...",
    "quotes": [
      {
        "text": "...",
        "timestamp_seconds": 3723.0,
        "timestamp_str": "1:02:03"
      }
    ],
    "topic": "...",
    "tone": "...",
    "audience": "...",
    "model_name": "...",
    "backend": "..."
  }
}
```

## Programmatic access (Python)

```python
import json
from pathlib import Path

p = Path("outputs/2026-06-08_asselineau-qwen3-50k-extreme.json")
data = json.loads(p.read_text())

# Top-level fields
title = data["title"]
url = data["url"]
duration_s = data["metadata"]["duration_seconds"]

# Analysis
analysis = data["analysis"]
summary = analysis["summary"]
key_points = analysis["key_points"]
quotes = analysis["quotes"]

# Jump to citation in a video player
for q in quotes:
    print(f"At {q['timestamp_seconds']}s: {q['text'][:80]}…")

# Cross-video grep
for md_file in Path("outputs").glob("*.json"):
    d = json.loads(md_file.read_text())
    if "souveraineté" in d["analysis"]["summary"].lower():
        print(md_file.name)
```

## Output filename auto-tagging

When you specify `--depth` or a non-default model / context size, the
output filename is auto-suffixed so re-runs with different settings
**don't overwrite each other**:

```bash
# These three runs produce three distinct files:
yt-insight all "URL" --depth shallow
# → 2026-06-08_video-shallow.md

yt-insight all "URL" --depth deep
# → 2026-06-08_video-deep.md

yt-insight all "URL"  # normal default → no depth suffix
# → 2026-06-08_video.md
```

The tag format is: `{model_short}-p{max_prompt_tokens//1000}k`,
e.g. `qwen3-50k` for a Qwen3 model with a 50k ctx cap.

## Cache vs. output

| Directory | Purpose | Persistent? |
|-----------|---------|-------------|
| `cache/` | Audio + transcript + video metadata | ✅ yes, reused on re-run |
| `outputs/` | Markdown + JSON analyses | ✅ yes, but each `(depth, model, ctx)` combo has its own file |

# YT-Insight — Roadmap

Last updated: 2026-06-08.

This file tracks work beyond the MVP (Phase 1, completed in commit `853b9cc`).
It is intentionally short — every item should be a self-contained ticket with a
clear "Definition of Done".

---

## Phase 2 — Backends supplémentaires
- [ ] **Backend Ollama** — same OpenAI-compat API, just swap `LLAMACPP_BASE_URL`.
- [ ] **Backend OpenAI-compat cloud** (OpenRouter, MiniMax, Anthropic via gateway).
- [ ] **Backend Claude via Anthropic SDK** — separate transport (HTTP/2 + auth headers).

## Phase 3 — Productivity features
- [ ] **YouTube playlists** — accept a playlist URL, batch-process all videos.
- [ ] **YouTube channel mode** — process the N most-recent uploads.
- [ ] **SQLite cache layer** — store transcripts + analyses, search by date / channel / topic.
- [ ] **PDF export** — Markdown → PDF via `weasyprint` or `pandoc`.
- [ ] **CLI history** — `yt-insight history` shows past runs, with re-run button.
- [ ] **Markdown backlinks** — auto-link between related videos via topic tags.

## Phase 4 — Quality
- [x] Suite de tests complète (200/200 ✅)
- [ ] CI/CD GitHub Actions (lint + test on every PR)
- [ ] Dockerisation (multi-stage image, ~500 Mo final)
- [ ] Packaging PyPI (`pip install yt-insight`)
- [ ] Type checking with Pyright in CI (currently local-only)

## Phase 5 — Streaming, progress, interactivity
- [x] **✅ Streaming SSE for LLM analysis** — completed 2026-06-08 (commit `TBD`)
- [ ] **Live progress bar** for chunk+merge analyses (Rich Progress) — not started
- [ ] **ETA estimate** during long runs — not started
- [ ] **Cancellation** — Ctrl-C should cleanly stop chunk+merge mid-flight — not started
- [ ] **Reactive TUI** — show chunks being summarized in real time — not started

---

## Ticket: Streaming SSE for LLM analysis

**Status:** ✅ Done (completed 2026-06-08, commit `57e732e`)
**Priority:** Medium
**Estimated effort:** 3-4h
**Owner:** Niko

### Problem

The current analyzer (`yt_insight/analyzer/llamacpp_local.py`) uses httpx's
**blocking** mode for the chat completion call. The httpx `Timeout` is split
into `connect=10s` / `read=1800s` / `write=10s` / `pool=10s`, but `read` is a
**wall-clock** timeout on the full response. For a 1h Stanford-class
transcript (11k tokens), a single chunk can take 5-10 minutes to generate,
and chunk+merge runs 10+ calls in sequence. Any hiccup on the server side
(network blip, token stalls, etc.) kills the entire pipeline.

Worst case: 25-minute analysis aborts at minute 24 with a `ReadTimeout`
and we have nothing.

### Goal

Replace the blocking `client.post("/v1/chat/completions")` with an SSE
streaming call (`client.stream("POST", ...)` + iterate over `data: ` lines),
and detect "stuck" generations via an **idle timeout** instead of a
wall-clock timeout. The user should see the model generate token by token
in the terminal, and a stuck generation should fail loudly within ~2
minutes (not 30).

### Why it matters

1. **Visibility**: watching the analysis generate is a strong UX win.
   The user knows the pipeline is alive and can see the model "thinking".
2. **Robustness**: idle-timeout catches the failure modes wall-clock
   timeout misses (token stalls, network drops, server GC pauses).
3. **Shorter max wait**: with idle=120s, a stuck chunk aborts in 2 min
   instead of 30 min. The user can retry or switch to a smaller model.

### Design sketch

```python
with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
    resp.raise_for_status()
    last_token_at = time.time()
    full_text_parts = []
    for line in resp.iter_lines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ").strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        delta = chunk["choices"][0].get("delta", {}).get("content")
        if delta:
            full_text_parts.append(delta)
            last_token_at = time.time()
            if on_token:
                on_token(delta)  # callback for live display
        # Idle check
        if time.time() - last_token_at > idle_timeout_s:
            raise AnalysisError(
                f"No token received for {idle_timeout_s}s — model stuck"
            )
return "".join(full_text_parts)
```

### Acceptance criteria

- [ ] `llamacpp_local._chat` uses httpx `client.stream(...)` instead of
      `client.post(...)`.
- [ ] Default `idle_timeout_s=120`, configurable via
      `LLAMACPP_IDLE_TIMEOUT_S` env var and `--llamacpp-idle-timeout` CLI flag.
- [ ] A callback `on_token: Callable[[str], None]` is plumbed through
      `analyze()` so the CLI can pipe tokens to a Rich `Live` display.
- [ ] At least one test that simulates a stalled stream (server emits
      `[DONE]` after a 3-second pause) and verifies the idle timeout
      raises a clean `AnalysisError`.
- [ ] The `--verbose` flag triggers live token display by default.
- [ ] All 200 existing tests still pass.

### Out of scope

- Token-by-token rate display (req/s, tok/s) — nice-to-have, separate ticket.
- Tool-use / function-calling streaming — not yet a use case.
- Streaming for the downloader/transcriber (Whisper is already streamed
  via faster-whisper's generator, not relevant).

### References

- httpx streaming docs: https://www.python-httpx.org/advanced/timeouts/#streaming
- llama-server OpenAI API: https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md
- Current blocking implementation: `yt_insight/analyzer/llamacpp_local.py`
  lines ~430-450 (`_chat` method).

---

## Ticket: Fix missing timestamps on quotes in chunked mode

**Status:** Backlog (not started)
**Priority:** Medium (not urgent, low-volume use case)
**Estimated effort:** 2-3h
**Owner:** Niko

### Problem

When the transcript is too large for single-shot (e.g. 3h+ Asselineau
conference, 43 503 tokens), the analyzer switches to chunk+merge.
After analysis, the user notices that all quotes from the **merged
final pass** are missing their timestamps — every `quote.timestamp_seconds`
is `null`.

The 80k single-shot run on the same video preserved 7/7 timestamps.
The 25k chunked run preserved 0/6.

### Root cause

In `_chunk_and_merge()`, the per-chunk summaries each come with their
own timestamped quotes (those work fine). But the **final merge** asks
the LLM to combine the partial summaries into a global one, and the
LLM has lost access to the per-chunk timestamps. It can only see the
text of the partial quotes, not their position. So it either omits
timestamps or hallucinates them.

### Goal

Preserve accurate quote timestamps in chunked mode.

### Possible approaches

1. **Carry timestamps into the merge prompt explicitly.** When
   building the merge user prompt, list every per-chunk quote with
   its `[mm:ss] ` prefix (or `(start-end)` range) so the LLM can
   copy them through. Then in the merge response, instruct the LLM
   to preserve those exact timestamps verbatim.

2. **Programmatic merge of timestamps.** After the LLM produces
   the merged quotes, re-attach timestamps by string-matching each
   final quote back to the closest per-chunk quote (Levenshtein on
   the first 30 chars, e.g.).

3. **Skip the final-merge quotes.** Only keep the per-chunk quotes
   in the output. The user can see them in the markdown but the
   JSON `quotes` field is "chunk quotes" not "global quotes".

### Acceptance criteria

- [ ] Quotes produced by chunk+merge have valid `timestamp_seconds`
      for at least 80% of them.
- [ ] The first 30 chars of each quote match a per-chunk quote
      exactly (no LLM-hallucinated text).
- [ ] All existing tests still pass.

### Out of scope

- Sub-second precision. Minute-level is fine for human consumption.
- Re-deriving timestamps from the audio (would require replaying).

---

## Ticket: Analysis depth & configurable sections

**Status:** Backlog (not started)
**Priority:** High (very high user value)
**Estimated effort:** 6-8h
**Owner:** Niko

### Problem

Today's analyzer produces a single fixed output:
- 15 key points
- 5 quotes with timestamps
- An analysis section with Forces + Concepts + Implications
- ~3500 char summary

This is a good "default" but doesn't serve all use cases:
- A user who just wants a TL;DR of a 3h video doesn't need 15 points.
- A user analyzing a political speech wants Faiblesses and
  Contradictions, not just Forces.
- A researcher wants the model to flag limitations and context
  gaps explicitly.

### Goal

Expose **two orthogonal axes** to the user:

1. **Depth (numeric levers)** — preset opinionated modes
2. **Sections (semantic content)** — which analysis rubrics to include

### Design

```bash
# Depth: controls max_tokens, num_key_points, num_quotes, temperature
yt-insight all "URL" --depth shallow   # 1024 tok, 5 pts, 3 quotes
yt-insight all "URL" --depth normal    # 4096 tok, 15 pts, 5 quotes  (default)
yt-insight all "URL" --depth deep      # 8192 tok, 25 pts, 10 quotes
yt-insight all "URL" --depth extreme  # 16384 tok, 40 pts, 15 quotes

# Sections: which rubrics in the analysis section
yt-insight all "URL" --sections forces,concepts,implications   # default
yt-insight all "URL" --sections forces,weaknesses,contradictions  # rhetorical
yt-insight all "URL" --sections forces,concepts,implications,weaknesses,contradictions,biases,limitations,context_gaps  # all 8

# Override: --sections wins over the depth's default sections
yt-insight all "URL" --depth extreme --sections forces   # numeric levers of extreme, only "forces"
```

### Depth → numeric levers table

| Depth    | max_tokens | num_key_points | num_quotes | temperature |
|----------|------------|----------------|------------|-------------|
| shallow  | 1024       |  5             |  3         | 0.4         |
| normal   | 4096       | 15             |  5         | 0.2         |
| deep     | 8192       | 25             | 10         | 0.1         |
| extreme  | 16384      | 40             | 15         | 0.0         |

### Sections (8 rubrics)

| Section         | Description                                         |
|-----------------|----------------------------------------------------|
| `forces`        | Points forts du contenu                            |
| `concepts`      | Concepts centraux expliqués                        |
| `implications`  | Implications et perspectives                       |
| `weaknesses`    | Faiblesses / arguments mal étayés                  |
| `contradictions`| Contradictions internes du propos                  |
| `biases`        | Biais cognitifs / rhétoriques détectés             |
| `limitations`   | Limites de l'analyse elle-même                     |
| `context_gaps`  | Contexte manquant / omissions volontaires          |

### Implementation sketch

- Add a `Depth` enum (shallow/normal/deep/extreme) in `analyzer/config.py`.
- Add a `Section` enum (8 values) in `analyzer/config.py`.
- Extend `AnalysisConfig` with `depth: Depth` and `sections: list[Section]`.
- Refactor `prompts.build_analysis_prompt()` to accept a `sections` list
  and emit only the requested rubric instructions.
- Extend `_single_shot` / `_chunk_and_merge` to pass the section list
  to the prompt and the numeric levers to the LLM payload.
- Validate the input sections on the CLI side (typo → clean error).
- Tag the output filename with the depth (e.g. `...-deep-qwen3-50k.md`).

### Acceptance criteria

- [ ] `--depth shallow|normal|deep|extreme` works on both `analyze` and `all`.
- [ ] `--sections a,b,c` accepts the 8 valid names and rejects typos
      with a clean error message listing valid options.
- [ ] Prompt includes **only** the requested rubrics (verified by a
      unit test that snapshots the prompt for each depth/section combo).
- [ ] max_tokens and num_key_points/num_quotes are forwarded to the
      LLM payload correctly.
- [ ] Default behavior is unchanged (back-compat): no --depth flag
      produces the same output as today.
- [ ] Output filename includes the depth tag when explicitly set.

### Out of scope

- Auto-detection of appropriate depth (user must opt in).
- Per-section token budgets (one global max_tokens for the whole output).
- User-defined custom rubrics (only the 8 built-ins for v1).

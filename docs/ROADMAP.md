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

**Status:** Backlog (not started)
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

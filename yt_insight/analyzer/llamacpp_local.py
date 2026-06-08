"""
LLM analyzer backend that talks to a locally running ``llama-server``.

The server is the one shipped with llama.cpp (``llama-server``). It
exposes an OpenAI-compatible HTTP API, so we talk to it via
``httpx.AsyncClient`` using the standard ``/v1/chat/completions``
endpoint.

Why not Ollama?
---------------
The user runs ``llama-server`` directly (with a custom Qwen3.6 35B-A3B
UD-IQ3_S quant on port 8080) rather than going through Ollama. The
HTTP surface is identical though, so this backend could trivially
target Ollama as well — just point ``base_url`` at its ``/v1`` mount.

Key design decisions
--------------------
- **Thinking is disabled.** Qwen3 chat templates default to thinking
  mode, which burns a lot of tokens for structured analysis tasks.
  We pass ``chat_template_kwargs={"enable_thinking": false}`` to keep
  responses concise and predictable.
- **Single-shot when possible, chunk+merge otherwise.** If the
  transcript fits in ``max_prompt_tokens`` (default 28k, leaving
  headroom under llama.cpp's 32k ctx), we send it as one prompt.
  Otherwise we chunk, summarize per chunk, then run a merge prompt
  to fuse everything into a single ``AnalysisResult``.
- **JSON contract.** We always ask the model for a JSON object and
  parse it leniently. Failures raise :class:`AnalysisError`.
- **Streaming is optional.** ``stream=True`` lets us yield tokens as
  they arrive, useful for a Rich live display in the CLI.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

# httpx is the only third-party dep of this module. Expose it at
# module level so tests can patch
# ``yt_insight.analyzer.llamacpp_local.httpx``. If the package isn't
# installed we leave a sentinel — the rest of the module still parses.
httpx = None  # type: ignore
try:
    import httpx as _httpx
    httpx = _httpx
except ImportError:  # pragma: no cover - exercised only without httpx
    pass

from ..utils.text_utils import (
    chunk_text,
    clean_transcript,
    estimate_tokens,
)
from . import prompts
from .base import AnalysisResult, BaseAnalyzer, Quote

if TYPE_CHECKING:
    from yt_insight.transcriber import TranscriptionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL = "Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
DEFAULT_TIMEOUT_S = 7200.0         # 2 h — long enough for 3h videos
DEFAULT_IDLE_TIMEOUT_S = 1800.0   # 30 min — covers prompt eval of very large chunks
                                 # (observed 842s for ~30k tokens on GTX 1650S; 1800s gives 2x headroom)
DEFAULT_MAX_PROMPT_TOKENS = 50_000 # 50k — safe under llama.cpp's 65k default ctx
DEFAULT_CHUNK_OVERLAP_TOKENS = 500
DEFAULT_TEMPERATURE = 0.2          # low for structured JSON
DEFAULT_MAX_TOKENS = 4096          # big enough for the analysis JSON


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class AnalysisError(Exception):
    """Raised when the analyzer fails for any reason (HTTP, JSON, etc.)."""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

#: Match a fenced ```json ... ``` block first, then a bare JSON object.
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _balanced_json_object(text: str) -> str | None:
    """
    Return the first balanced ``{...}`` substring, correctly handling
    nested braces **and** strings (with escapes). Returns ``None`` if
    no balanced object is found.

    This is the most reliable extractor when the model emits
    multiline JSON with literal newlines or unescaped quotes inside
    string values.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _write_debug_response(text: str, err: Exception) -> Path | None:
    """
    Save the raw model response to ``cache/debug/{timestamp}.txt`` for
    post-mortem analysis when JSON extraction fails. Returns the path
    or ``None`` if writing is impossible.
    """
    try:
        from datetime import datetime
        debug_dir = Path("cache") / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = debug_dir / f"{ts}.txt"
        path.write_text(
            f"--- ERROR ---\n{type(err).__name__}: {err}\n"
            f"--- LENGTH ---\n{len(text)} chars\n"
            f"--- FIRST 200 ---\n{text[:200]!r}\n"
            f"--- LAST 200 ---\n{text[-200:]!r}\n"
            f"--- FULL ---\n{text}\n",
            encoding="utf-8",
        )
        return path
    except Exception:
        return None


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    Find the first JSON object in *text* and parse it.

    The model is asked to reply with a single JSON object only, but
    sometimes it wraps it in a Markdown code fence or adds a stray
    sentence. This helper is tolerant.
    """
    text = text.strip()

    # 1. Try the full string first (fast path).
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Look for a ```json ... ``` block.
    fenced = _FENCED_JSON_RE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Fall back to balanced-brace matching: handles nested objects
    #    and strings (with escapes) correctly. Most reliable for
    #    multi-line JSON with literal newlines or unescaped quotes.
    candidate = _balanced_json_object(text)
    if candidate is not None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 4. Last resort: try ``json_repair`` (if available) which fixes
    #    common issues like trailing commas, single quotes, unescaped
    #    newlines in strings, missing closing braces, etc.
    try:
        from json_repair import repair_json
        repaired = repair_json(
            json_str=text,
            return_objects=True,
            ensure_ascii=False,
        )
        if isinstance(repaired, dict):
            return repaired
    except ImportError:
        pass
    except Exception:
        pass

    # All extraction strategies failed. Save the raw response for
    # post-mortem and raise a clear error.
    err = AnalysisError(
        f"Could not extract a JSON object from the model response "
        f"({len(text)} chars). First 200 chars: {text[:200]!r}"
    )
    debug_path = _write_debug_response(text, err)
    if debug_path is not None:
        err = AnalysisError(
            f"{err}. Raw response saved to {debug_path}."
        )
    raise err


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class LlamaCppLocalAnalyzer(BaseAnalyzer):
    """
    Analyzer that talks to a local ``llama-server`` instance.

    Parameters
    ----------
    base_url:
        Root URL of the llama-server (e.g. ``http://localhost:8080``).
        The OpenAI-compatible mount is expected at ``{base_url}/v1``.
    model:
        Model name as reported by ``GET /v1/models``. Used to populate
        :attr:`model_name` and to address the right slot on the server.
    timeout_s:
        HTTP timeout in seconds. Set generously: a chunk+merge run on
        a 1-hour video can easily take several minutes.
    max_prompt_tokens:
        Soft cap on the number of tokens in the user prompt. If the
        transcript exceeds this, we switch to chunk+merge.
    chunk_overlap_tokens:
        Number of tokens of overlap between consecutive chunks.
    temperature:
        Sampling temperature for the model. Low values (0.1-0.3) are
        recommended for structured JSON outputs.
    max_tokens:
        Maximum number of tokens the model is allowed to generate per
        call.
    disable_thinking:
        Whether to pass ``chat_template_kwargs={"enable_thinking":
        false}``. Strongly recommended for Qwen3 to get concise,
        direct answers.
    idle_timeout_s:
        Maximum number of seconds we wait between two tokens before
        aborting the stream with an :class:`AnalysisError`. Catches
        "stuck" generations (network blips, server GC pauses, token
        stalls) that wall-clock timeouts miss.
    system_prompt:
        Override the default system prompt. Must remain JSON-strict.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        disable_thinking: bool = True,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        system_prompt: str | None = None,
        depth: "Depth" = None,                  # see ``yt_insight.analyzer.depth``
        sections: tuple[str, ...] | None = None,
    ):
        from .depth import Depth, DEPTH_DEFAULT_SECTIONS, DEPTH_PRESETS, coerce_sections
        if depth is None:
            depth = Depth.NORMAL
        if not isinstance(depth, Depth):
            depth = Depth(str(depth).lower())

        # When ``depth`` is given, the depth preset takes precedence
        # over explicit temperature/max_tokens unless the user passed
        # non-default values (we keep the user's choice if explicit).
        preset = DEPTH_PRESETS[depth]
        # If user kept the default temperature (or didn't pass any),
        # override with preset. ``None`` from the CLI is treated as
        # "use the preset" so the depth default is always applied.
        if temperature is None or temperature == DEFAULT_TEMPERATURE:
            temperature = float(preset["temperature"])  # type: ignore[arg-type]
        # If user kept the default max_tokens (or didn't pass any),
        # override with preset.
        if max_tokens is None or max_tokens == DEFAULT_MAX_TOKENS:
            max_tokens = int(preset["max_tokens"])  # type: ignore[arg-type]
        if sections is None:
            sections = DEPTH_DEFAULT_SECTIONS[depth]
        else:
            sections = coerce_sections(sections)

        self.base_url = base_url.rstrip("/")
        self._model = model
        self.timeout_s = timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.max_prompt_tokens = max_prompt_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        self.system_prompt = system_prompt or prompts.SYSTEM_PROMPT
        self.depth = depth
        self.sections = sections

        self._client: httpx.Client | None = None
        # Resolved at first call (after server handshake) — may differ
        # from ``self._model`` if the user passed a partial alias.
        self._resolved_model: str | None = None

        logger.info(
            "LlamaCppLocalAnalyzer configured — url=%s model=%s max_prompt=%d tok "
            "depth=%s sections=%s",
            self.base_url, self._model, self.max_prompt_tokens,
            self.depth.value, ",".join(self.sections),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.Client:
        if httpx is None:  # pragma: no cover - guarded by factory
            raise AnalysisError(
                "httpx is not installed. Run: pip install httpx"
            )
        if self._client is None:
            # Connect quickly (10s), read generously (timeout_s).
            # write=10s is fine since we never upload big payloads.
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.timeout_s,
                    write=10.0,
                    pool=10.0,
                ),
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "LlamaCppLocalAnalyzer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._resolved_model or self._model

    @property
    def backend_name(self) -> str:
        return "llamacpp-local"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        transcription: "TranscriptionResult",
        *,
        title: str = "",
        language: str | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> AnalysisResult:
        """
        Run the full analysis pipeline on *transcription*.

        See :meth:`BaseAnalyzer.analyze` for the contract.

        Parameters
        ----------
        on_token:
            Optional callback invoked with every text delta as it
            arrives from the LLM. Useful for live console display
            (e.g. Rich ``Live`` in the CLI). Has no effect on the
            returned :class:`AnalysisResult`.
        """
        # Resolve the actual model name once (and check the server is up).
        self._ensure_server_ready()

        lang_label = _language_label(language or transcription.language)
        clean_text = clean_transcript(transcription.text)
        if not clean_text:
            raise AnalysisError("Transcript is empty after cleanup.")

        t0 = time.time()
        logger.info(
            "Analyzing transcript: %d chars, ~%d tokens, lang=%s",
            len(clean_text), estimate_tokens(clean_text), lang_label,
        )

        if estimate_tokens(clean_text) <= self.max_prompt_tokens:
            payload = self._single_shot(
                clean_text, title=title, language=lang_label,
                duration=transcription.duration_str,
                depth=self.depth, sections=self.sections,
                on_token=on_token,
            )
        else:
            logger.info(
                "Transcript exceeds max_prompt_tokens — switching to chunk+merge"
            )
            payload = self._chunk_and_merge(
                clean_text,
                title=title,
                duration=transcription.duration_str,
                language=lang_label,
                depth=self.depth, sections=self.sections,
                on_token=on_token,
            )

        result = self._build_result(payload)
        # Tag the result with the backend + model identity so downstream
        # code (output writers, console) can label it.
        result.model_name = self.model_name
        result.backend = self.backend_name
        elapsed = time.time() - t0
        logger.info("Analysis done in %.1fs.", elapsed)
        return result

    # ------------------------------------------------------------------
    # Server handshake
    # ------------------------------------------------------------------

    def _ensure_server_ready(self) -> None:
        """Check that the server is up and resolve the model name."""
        if httpx is None:
            raise AnalysisError(
                "httpx is not installed. Run: pip install httpx"
            )
        client = self._get_client()
        try:
            resp = client.get("/v1/models")
            resp.raise_for_status()
        except Exception as exc:
            # httpx.HTTPError covers all transport-level errors. We catch
            # ``Exception`` so the analyzer works even if httpx is later
            # swapped for another HTTP library at this seam.
            if httpx is not None and isinstance(exc, httpx.HTTPError):
                raise AnalysisError(
                    f"Could not reach llama-server at {self.base_url}: {exc}"
                ) from exc
            raise AnalysisError(
                f"Could not reach llama-server at {self.base_url}: {exc}"
            ) from exc

        data = resp.json()
        models = data.get("data") or data.get("models") or []
        if not models:
            raise AnalysisError(
                f"llama-server at {self.base_url} reports no models. "
                f"Start it with: llama-server -m your-model.gguf"
            )

        # Resolve model name: exact match, else first available, else keep as-is.
        names = [m.get("id") or m.get("name") or "" for m in models]
        if self._model in names:
            self._resolved_model = self._model
        else:
            # Try a basename match (e.g. user passed "Qwen3.6-35B-A3B" without .gguf).
            basename = self._model
            for n in names:
                if n == basename or n.split(".")[0] == basename.split(".")[0]:
                    self._resolved_model = n
                    break
            if self._resolved_model is None:
                # Fall back to the first model the server has loaded.
                self._resolved_model = names[0]
                logger.warning(
                    "Model %r not found on server; falling back to %r",
                    self._model, self._resolved_model,
                )

    # ------------------------------------------------------------------
    # Single-shot path
    # ------------------------------------------------------------------

    def _single_shot(
        self,
        text: str,
        *,
        title: str,
        duration: str,
        language: str,
        depth,
        sections: tuple[str, ...],
        on_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        user_prompt = prompts.build_analysis_prompt(
            text, title=title, duration=duration, language=language,
            depth=depth, sections=sections,
        )
        content = self._chat(user_prompt, on_token=on_token)
        return _extract_json_object(content)

    # ------------------------------------------------------------------
    # Chunk + merge path
    # ------------------------------------------------------------------

    def _chunk_and_merge(
        self,
        text: str,
        *,
        title: str,
        duration: str,
        language: str,
        depth,
        sections: tuple[str, ...],
        on_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        # Use overlap = max(overlap, 5% of max_prompt_tokens) for long
        # transcripts, so the model still has cross-chunk context.
        overlap = self.chunk_overlap_tokens
        if estimate_tokens(text) > 2 * self.max_prompt_tokens:
            overlap = max(overlap, self.max_prompt_tokens // 20)

        chunks = chunk_text(
            text,
            max_tokens=self.max_prompt_tokens,
            overlap_tokens=overlap,
        )
        logger.info("Split transcript into %d chunks (overlap=%d tok).", len(chunks), overlap)

        partial_payloads: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            user_prompt = prompts.build_chunk_prompt(
                chunk,
                chunk_index=i,
                chunk_total=len(chunks),
                title=title,
                duration=duration,
                language=language,
                depth=depth,
                sections=sections,
            )
            content = self._chat(user_prompt, on_token=on_token)
            # Keep the raw JSON blob for the merge prompt; we don't
            # validate it here, only the final merge output.
            partial_payloads.append(content)
            logger.info("Chunk %d/%d summarized (%d chars).", i, len(chunks), len(content))

        merge_prompt = prompts.build_merge_prompt(
            partial_payloads, title=title, duration=duration, language=language,
            depth=depth, sections=sections,
        )
        merged = self._chat(merge_prompt, on_token=on_token)
        return _extract_json_object(merged)

    # ------------------------------------------------------------------
    # HTTP call (streaming with idle timeout)
    # ------------------------------------------------------------------

    def _build_payload(self, user_prompt: str, stream: bool) -> dict:
        """Build the JSON payload for the chat completion endpoint."""
        payload = {
            "model": self._resolved_model or self._model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _stream_deltas(
        self,
        user_prompt: str,
        on_token: Callable[[str], None] | None = None,
    ) -> Iterator[str]:
        """
        Internal generator: yield every text delta from the server.

        Streams the response (always — even for ``_chat()`` which just
        joins the results), with an **idle timeout** between tokens
        (``self.idle_timeout_s``). Raises :class:`AnalysisError` if the
        server stops sending tokens for too long.

        Parameters
        ----------
        user_prompt:
            The user's message content.
        on_token:
            Optional callback invoked with each delta as it arrives.
            Use this to drive a live console display.
        """
        payload = self._build_payload(user_prompt, stream=True)
        client = self._get_client()
        try:
            with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    # Read body for error message then raise.
                    body = resp.read().decode("utf-8", errors="replace")[:500]
                    raise AnalysisError(
                        f"llama-server returned HTTP {resp.status_code}: {body}"
                    )
                # NOTE: ``last_token_at`` is updated on ANY data: line
                # (even empty content), NOT on the entry. This is
                # because the server can take many minutes to process
                # a big prompt before sending its first content token
                # — we don't want to abort during that "prefill" phase.
                last_token_at = time.time()
                for line in resp.iter_lines():
                    # Idle check on every line read. Resets whenever
                    # we see a data: line (even with empty content),
                    # proving the server is still alive.
                    now = time.time()
                    if now - last_token_at > self.idle_timeout_s:
                        raise AnalysisError(
                            f"No token received for {int(now - last_token_at)}s "
                            f"(idle_timeout_s={self.idle_timeout_s}) — model "
                            f"appears stuck. Aborting."
                        )
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if raw == "[DONE]":
                        break
                    # Server is alive — reset the idle clock.
                    last_token_at = time.time()
                    try:
                        evt = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    delta = (
                        evt.get("choices", [{}])[0]
                           .get("delta", {})
                           .get("content")
                    )
                    if delta:
                        last_token_at = time.time()
                        if on_token is not None:
                            on_token(delta)
                        yield delta
        except AnalysisError:
            raise
        except Exception as exc:
            if httpx is not None and isinstance(exc, httpx.HTTPError):
                raise AnalysisError(
                    f"Streaming HTTP error from {self.base_url}: {exc}"
                ) from exc
            raise AnalysisError(
                f"Streaming HTTP error from {self.base_url}: {exc}"
            ) from exc

    def _chat(
        self,
        user_prompt: str,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Send *user_prompt* to the server and return the assistant text.

        Implementation note: this now goes through the streaming path
        internally so it benefits from the idle-timeout safeguard. The
        public contract is unchanged: it still returns a single string.
        """
        return "".join(self._stream_deltas(user_prompt, on_token=on_token))

    def stream_chat(
        self,
        user_prompt: str,
        on_token: Callable[[str], None] | None = None,
    ) -> Iterator[str]:
        """
        Streaming variant of :meth:`_chat` — yields text chunks as they
        arrive. Useful for the CLI's Rich live display. The caller is
        responsible for reassembling the chunks into a single string
        before calling :func:`_extract_json_object`.
        """
        yield from self._stream_deltas(user_prompt, on_token=on_token)

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(payload: dict[str, Any]) -> AnalysisResult:
        """Turn a parsed JSON *payload* into an :class:`AnalysisResult`."""
        summary = _coerce_str(payload.get("summary"))
        analysis = _coerce_str(payload.get("analysis"))
        topic = _coerce_str(payload.get("topic"))
        tone = _coerce_str(payload.get("tone"))
        audience = _coerce_str(payload.get("audience"))

        key_points_raw = payload.get("key_points") or []
        if not isinstance(key_points_raw, list):
            key_points_raw = [str(key_points_raw)]
        key_points = [str(p).strip() for p in key_points_raw if str(p).strip()]

        quotes_raw = payload.get("quotes") or []
        if not isinstance(quotes_raw, list):
            quotes_raw = []
        quotes: list[Quote] = []
        for q in quotes_raw:
            if isinstance(q, str):
                quotes.append(Quote(text=q.strip()))
                continue
            if not isinstance(q, dict):
                continue
            text = _coerce_str(q.get("text"))
            if not text:
                continue
            ts = q.get("timestamp_seconds")
            ts_f: float | None
            try:
                ts_f = float(ts) if ts is not None else None
            except (TypeError, ValueError):
                ts_f = None
            speaker = q.get("speaker")
            quotes.append(
                Quote(
                    text=text,
                    timestamp_seconds=ts_f,
                    speaker=str(speaker).strip() if speaker else None,
                )
            )

        return AnalysisResult(
            summary=summary,
            key_points=key_points,
            analysis=analysis,
            quotes=quotes,
            topic=topic,
            tone=tone,
            audience=audience,
            # ``model_name`` and ``backend`` are filled in by the caller
            # (analyze) using the analyzer's properties.
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_analyzer(
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout_s: float | None = None,
    idle_timeout_s: float | None = None,
    max_prompt_tokens: int | None = None,
    chunk_overlap_tokens: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    disable_thinking: bool | None = None,
    depth: "Depth | str | None" = None,
    sections: tuple[str, ...] | list[str] | str | None = None,
) -> LlamaCppLocalAnalyzer:
    """
    Build a :class:`LlamaCppLocalAnalyzer` from env vars + kwargs.

    Environment variables (override only if the matching kwarg is None):
    - ``LLAMACPP_BASE_URL``            → ``base_url``
    - ``LLAMACPP_MODEL``               → ``model``
    - ``LLAMACPP_TIMEOUT_S``           → ``timeout_s``
    - ``LLAMACPP_IDLE_TIMEOUT_S``      → ``idle_timeout_s`` (per-token, default 120s)
    - ``LLAMACPP_MAX_PROMPT_TOKENS``   → ``max_prompt_tokens``
    - ``LLAMACPP_CHUNK_OVERLAP_TOKENS``→ ``chunk_overlap_tokens``
    - ``LLAMACPP_TEMPERATURE``         → ``temperature``
    - ``LLAMACPP_MAX_TOKENS``          → ``max_tokens``
    - ``LLAMACPP_DISABLE_THINKING``    → ``disable_thinking`` ("1"/"0")
    - ``LLAMACPP_DEPTH``               → ``depth`` (shallow/normal/deep/extreme)
    - ``LLAMACPP_SECTIONS``            → ``sections`` (comma-separated names)

    Parameters
    ----------
    depth:
        Analysis depth preset. ``None`` → ``Depth.NORMAL`` → from env →
        CLI's ``--depth`` flag. Accepts a :class:`Depth`, a string, or
        ``None``.
    sections:
        Analysis rubrics. ``None`` → depth's defaults. Accepts a tuple
        of names, a comma-separated string, or ``None``.
    """
    import os

    def _env_str(name: str, default: str) -> str:
        v = os.getenv(name)
        return v if v not in (None, "") else default

    def _env_float(name: str, default: float) -> float:
        v = os.getenv(name)
        if v in (None, ""):
            return default
        try:
            return float(v)
        except ValueError:
            return default

    def _env_int(name: str, default: int) -> int:
        v = os.getenv(name)
        if v in (None, ""):
            return default
        try:
            return int(v)
        except ValueError:
            return default

    def _env_bool(name: str, default: bool) -> bool:
        v = os.getenv(name)
        if v in (None, ""):
            return default
        return v.strip().lower() not in ("0", "false", "no", "off")

    def _env_int_or_default(name: str, default: int | None) -> int | None:
        """Like :func:`_env_int` but returns ``default`` unchanged if it's None."""
        v = os.getenv(name)
        if v in (None, ""):
            return default
        try:
            return int(v)
        except ValueError:
            return default

    def _env_float_or_default(name: str, default: float | None) -> float | None:
        v = os.getenv(name)
        if v in (None, ""):
            return default
        try:
            return float(v)
        except ValueError:
            return default

    # ``None`` means "user didn't pass a value, use the kwarg default".
    # We must convert None to the actual default before forwarding, otherwise
    # downstream code (e.g. ``logger.info("max_prompt=%d", self.max_prompt_tokens)``)
    # crashes when the value is None.
    eff_max_prompt = max_prompt_tokens if max_prompt_tokens is not None else DEFAULT_MAX_PROMPT_TOKENS
    eff_idle = idle_timeout_s if idle_timeout_s is not None else DEFAULT_IDLE_TIMEOUT_S
    eff_timeout = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S

    return LlamaCppLocalAnalyzer(
        base_url=base_url or _env_str("LLAMACPP_BASE_URL", DEFAULT_BASE_URL),
        model=model or _env_str("LLAMACPP_MODEL", DEFAULT_MODEL),
        timeout_s=_env_float_or_default("LLAMACPP_TIMEOUT_S", eff_timeout),
        idle_timeout_s=_env_float_or_default("LLAMACPP_IDLE_TIMEOUT_S", eff_idle),
        max_prompt_tokens=_env_int_or_default("LLAMACPP_MAX_PROMPT_TOKENS", eff_max_prompt),
        chunk_overlap_tokens=_env_int_or_default(
            "LLAMACPP_CHUNK_OVERLAP_TOKENS", chunk_overlap_tokens
        ),
        temperature=_env_float_or_default("LLAMACPP_TEMPERATURE", temperature),
        max_tokens=_env_int_or_default("LLAMACPP_MAX_TOKENS", max_tokens),
        disable_thinking=(
            disable_thinking
            if disable_thinking is not None
            else _env_bool("LLAMACPP_DISABLE_THINKING", True)
        ),
        # Depth + sections: env var overrides None, kwarg wins over env.
        depth=depth if depth is not None else (_env_str("LLAMACPP_DEPTH", "") or None),
        sections=sections if sections is not None else (
            _env_str("LLAMACPP_SECTIONS", "") or None
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Mapping from ISO 639-1 codes to human-readable language labels used
#: in the prompts. Extend as needed.
_LANGUAGE_LABELS: dict[str, str] = {
    "fr": "français",
    "en": "english",
    "es": "español",
    "de": "deutsch",
    "it": "italiano",
    "pt": "português",
    "nl": "nederlands",
}


def _language_label(code: str | None) -> str:
    if not code:
        return "français"
    return _LANGUAGE_LABELS.get(code.lower(), code)


def _coerce_str(value: Any) -> str:
    """Return *value* as a stripped string, or ``""`` for None / non-strings."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()

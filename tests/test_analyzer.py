"""
Tests for ``yt_insight.analyzer``.

The HTTP layer is mocked throughout — no llama-server required.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from yt_insight.analyzer import (
    AnalysisResult,
    LlamaCppLocalAnalyzer,
    Quote,
    create_analyzer,
)
from yt_insight.analyzer.llamacpp_local import (
    AnalysisError,
    _extract_json_object,
)
from yt_insight.transcriber import (
    Segment,
    TranscriptionResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_transcript() -> TranscriptionResult:
    return TranscriptionResult(
        text=(
            "Bonjour à tous, aujourd'hui on parle d'intelligence artificielle. "
            "Les transformers ont révolutionné le traitement du langage. "
            "L'attention est le mécanisme clé. "
            "Merci de votre attention."
        ),
        segments=[
            Segment(start=0.0,  end=3.0,  text="Bonjour à tous."),
            Segment(start=3.0,  end=10.0, text="On parle d'IA."),
            Segment(start=10.0, end=20.0, text="Les transformers."),
            Segment(start=20.0, end=30.0, text="L'attention est clé."),
        ],
        language="fr",
        language_probability=0.99,
        duration_seconds=30.0,
        model_name="large-v3",
    )


def _make_analyzer() -> LlamaCppLocalAnalyzer:
    """Build an analyzer without touching the network."""
    return LlamaCppLocalAnalyzer(
        base_url="http://fake:8080",
        model="fake-model",
        timeout_s=5.0,
        max_prompt_tokens=10_000,
        max_tokens=512,
    )


def _fake_models_response(model_id: str = "fake-model") -> SimpleNamespace:
    body = {"data": [{"id": model_id}]}
    resp = MagicMock()
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


def _fake_chat_response(content: str) -> SimpleNamespace:
    body = {"choices": [{"message": {"content": content}}]}
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


def _fake_streaming_response(
    content: str,
    status_code: int = 200,
    error_body: str = "internal error",
) -> MagicMock:
    """
    Build a fake httpx streaming response that yields the given content
    as SSE ``data:`` lines (one chunk), then ``[DONE]``.

    Use as the return value of ``fake_client.stream(...)``:
        fake_client.stream.return_value.__enter__.return_value = <this>
    """
    # Build SSE event(s). We send the whole content in a single delta
    # for simplicity — most tests just check the assembled string.
    payload = {"choices": [{"delta": {"content": content}}]}
    sse_lines = [f"data: {json.dumps(payload)}", "data: [DONE]"]

    resp = MagicMock()
    resp.status_code = status_code
    if status_code >= 400:
        # Body fetch on error
        resp.read.return_value = error_body.encode("utf-8")
    resp.iter_lines.return_value = iter(sse_lines)
    resp.raise_for_status.return_value = None
    return resp


def _install_streaming_post_mock(fake_client: MagicMock, content: str) -> None:
    """
    Wire ``fake_client.stream(...)`` to return a context manager that
    yields a fake SSE response containing *content* as a single delta.
    Also keeps ``fake_client.get(...)`` working (for the /v1/models
    handshake in the analyzer).
    """
    stream_cm = MagicMock()
    stream_cm.__enter__.return_value = _fake_streaming_response(content)
    stream_cm.__exit__.return_value = False
    fake_client.stream.return_value = stream_cm


# ---------------------------------------------------------------------------
# _extract_json_object
# ---------------------------------------------------------------------------

class TestExtractJsonObject:
    def test_pure_json(self):
        assert _extract_json_object('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}

    def test_fenced_json(self):
        text = "Voici:\n```json\n{\"x\": 42}\n```\nFin."
        assert _extract_json_object(text) == {"x": 42}

    def test_bare_json_with_chatter(self):
        text = "Bien sûr, voici la réponse:\n{\"y\": \"ok\"}\nMerci !"
        assert _extract_json_object(text) == {"y": "ok"}

    def test_garbage_raises(self):
        with pytest.raises(AnalysisError):
            _extract_json_object("ceci n'est pas du JSON")

    def test_empty_raises(self):
        with pytest.raises(AnalysisError):
            _extract_json_object("")


# ---------------------------------------------------------------------------
# LlamaCppLocalAnalyzer — handshake
# ---------------------------------------------------------------------------

class TestServerHandshake:
    def test_ensure_server_ready_resolves_model(self):
        analyzer = _make_analyzer()
        fake_client = MagicMock()
        fake_client.get.return_value = _fake_models_response("fake-model")
        with patch.object(analyzer, "_get_client", return_value=fake_client):
            analyzer._ensure_server_ready()
        assert analyzer._resolved_model == "fake-model"

    def test_basename_match(self):
        analyzer = _make_analyzer()
        analyzer._model = "fake"
        fake_client = MagicMock()
        fake_client.get.return_value = _fake_models_response("fake.gguf")
        with patch.object(analyzer, "_get_client", return_value=fake_client):
            analyzer._ensure_server_ready()
        assert analyzer._resolved_model == "fake.gguf"

    def test_fallback_to_first_available(self):
        analyzer = _make_analyzer()
        analyzer._model = "not-listed"
        fake_client = MagicMock()
        fake_client.get.return_value = _fake_models_response("other-model")
        with patch.object(analyzer, "_get_client", return_value=fake_client):
            analyzer._ensure_server_ready()
        assert analyzer._resolved_model == "other-model"

    def test_connection_error(self):
        analyzer = _make_analyzer()
        fake_client = MagicMock()
        # Simulate httpx raising on .get() — we don't need the real
        # httpx module to be installed for this test.
        class _ConnErr(Exception):
            pass
        fake_client.get.side_effect = _ConnErr("nope")
        with patch.object(analyzer, "_get_client", return_value=fake_client):
            with pytest.raises(AnalysisError, match="Could not reach"):
                analyzer._ensure_server_ready()

    def test_no_models_reported(self):
        analyzer = _make_analyzer()
        fake_client = MagicMock()
        fake_client.get.return_value = _fake_models_response("")  # empty
        # Override the fixture to return an actually empty list:
        fake_client.get.return_value.json.return_value = {"data": []}
        with patch.object(analyzer, "_get_client", return_value=fake_client):
            with pytest.raises(AnalysisError, match="no models"):
                analyzer._ensure_server_ready()


# ---------------------------------------------------------------------------
# Single-shot analysis
# ---------------------------------------------------------------------------

class TestSingleShotAnalyze:
    def test_returns_analysis_result(self, small_transcript):
        analyzer = _make_analyzer()
        payload = {
            "summary": "Une vidéo sur l'IA.",
            "key_points": ["Point A", "Point B", "Point C", "Point D",
                           "Point E", "Point F", "Point G", "Point H"],
            "analysis": "**Forces**\nBien structuré.",
            "quotes": [
                {"text": "Les transformers.", "timestamp_seconds": 10.0},
            ],
            "topic": "Intelligence Artificielle",
            "tone": "pédagogique",
            "audience": "développeurs",
        }
        fake_client = MagicMock()
        fake_client.get.return_value = _fake_models_response()
        _install_streaming_post_mock(fake_client, json.dumps(payload))
        with patch.object(analyzer, "_get_client", return_value=fake_client):
            result = analyzer.analyze(small_transcript, title="IA 101", language="fr")

        assert isinstance(result, AnalysisResult)
        assert result.summary == "Une vidéo sur l'IA."
        assert len(result.key_points) == 8
        assert len(result.quotes) == 1
        assert result.quotes[0].text == "Les transformers."
        assert result.quotes[0].timestamp_seconds == 10.0
        assert result.topic == "Intelligence Artificielle"
        assert result.model_name == "fake-model"
        assert result.backend == "llamacpp-local"

    def test_handles_quotes_as_strings(self, small_transcript):
        analyzer = _make_analyzer()
        payload = {
            "summary": "S",
            "key_points": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
            "quotes": ["une citation littérale"],
        }
        fake_client = MagicMock()
        fake_client.get.return_value = _fake_models_response()
        _install_streaming_post_mock(fake_client, json.dumps(payload))
        with patch.object(analyzer, "_get_client", return_value=fake_client):
            result = analyzer.analyze(small_transcript, language="fr")
        assert result.quotes[0].text == "une citation littérale"
        assert result.quotes[0].timestamp_seconds is None

    def test_fenced_json_in_response(self, small_transcript):
        analyzer = _make_analyzer()
        payload = {"summary": "x", "key_points": [], "quotes": []}
        fenced = f"```json\n{json.dumps(payload)}\n```"
        fake_client = MagicMock()
        fake_client.get.return_value = _fake_models_response()
        _install_streaming_post_mock(fake_client, fenced)
        with patch.object(analyzer, "_get_client", return_value=fake_client):
            result = analyzer.analyze(small_transcript, language="fr")
        assert result.summary == "x"

    def test_empty_transcript_raises(self):
        analyzer = _make_analyzer()
        empty = TranscriptionResult(
            text="   \n  ", segments=[], language="fr",
            language_probability=0.0, duration_seconds=0.0, model_name="large-v3",
        )
        fake_client = MagicMock()
        fake_client.get.return_value = _fake_models_response()
        with patch.object(analyzer, "_get_client", return_value=fake_client):
            with pytest.raises(AnalysisError, match="empty"):
                analyzer.analyze(empty)

    def test_http_error_in_chat(self, small_transcript):
        analyzer = _make_analyzer()
        fake_client = MagicMock()
        fake_client.get.return_value = _fake_models_response()
        # Status 500 — body will be read for the error message.
        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = _fake_streaming_response(
            "not used", status_code=500, error_body="internal error",
        )
        stream_cm.__exit__.return_value = False
        fake_client.stream.return_value = stream_cm
        with patch.object(analyzer, "_get_client", return_value=fake_client):
            with pytest.raises(AnalysisError, match="HTTP 500"):
                analyzer.analyze(small_transcript)


# ---------------------------------------------------------------------------
# Chunk + merge path
# ---------------------------------------------------------------------------

class TestChunkAndMerge:
    def test_triggers_when_oversize(self, small_transcript):
        # Build a HUGE transcript so chunking kicks in.
        long_text = ("Une phrase. " * 5000)  # ~65_000 chars → ~16k tokens
        long_transcript = TranscriptionResult(
            text=long_text, segments=[],
            language="fr", language_probability=0.99,
            duration_seconds=1000.0, model_name="large-v3",
        )
        analyzer = LlamaCppLocalAnalyzer(
            base_url="http://fake:8080", model="fake-model",
            max_prompt_tokens=2000, chunk_overlap_tokens=100,
        )

        call_count = {"n": 0}

        def fake_chat(user_prompt: str, **kwargs) -> str:
            call_count["n"] += 1
            # Per-chunk responses
            if "chunk N°" in user_prompt.lower() or "Chunk N°" in user_prompt or \
               ("N°" in user_prompt and "extrait" in user_prompt.lower()):
                return json.dumps({
                    "summary": "résumé partiel",
                    "key_points": ["p1", "p2", "p3"],
                    "quotes": [],
                })
            # Final merge
            return json.dumps({
                "summary": "résumé final",
                "key_points": ["a", "b", "c", "d", "e", "f", "g", "h"],
                "analysis": "**Forces**\nok",
                "quotes": [],
                "topic": "topic",
                "tone": "tone",
                "audience": "audience",
            })

        with patch.object(analyzer, "_chat", side_effect=fake_chat):
            with patch.object(analyzer, "_ensure_server_ready"):
                result = analyzer.analyze(long_transcript, language="fr")

        # 1 merge + N chunks (>= 2)
        assert call_count["n"] >= 3
        assert result.summary == "résumé final"

    def test_single_shot_when_small_enough(self, small_transcript):
        analyzer = _make_analyzer()
        payload = {
            "summary": "ok", "key_points": ["p"] * 8, "quotes": [],
        }
        fake_client = MagicMock()
        fake_client.get.return_value = _fake_models_response()
        _install_streaming_post_mock(fake_client, json.dumps(payload))
        with patch.object(analyzer, "_get_client", return_value=fake_client):
            result = analyzer.analyze(small_transcript, language="fr")
        # Only one HTTP call should have hit /v1/chat/completions.
        assert fake_client.stream.call_count == 1
        assert result.summary == "ok"


# ---------------------------------------------------------------------------
# create_analyzer (factory)
# ---------------------------------------------------------------------------

class TestCreateAnalyzer:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("LLAMACPP_BASE_URL", raising=False)
        monkeypatch.delenv("LLAMACPP_MODEL", raising=False)
        a = create_analyzer()
        assert a.base_url == "http://localhost:8080"
        assert "Qwen3.6" in a._model
        assert a.disable_thinking is True

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("LLAMACPP_BASE_URL", "http://gpu:9090")
        monkeypatch.setenv("LLAMACPP_MODEL", "custom")
        monkeypatch.setenv("LLAMACPP_TEMPERATURE", "0.5")
        monkeypatch.setenv("LLAMACPP_DISABLE_THINKING", "0")
        a = create_analyzer()
        assert a.base_url == "http://gpu:9090"
        assert a._model == "custom"
        assert a.temperature == 0.5
        assert a.disable_thinking is False

    def test_kwarg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LLAMACPP_BASE_URL", "http://env:1234")
        a = create_analyzer(base_url="http://kw:5678")
        assert a.base_url == "http://kw:5678"


# ---------------------------------------------------------------------------
# Quote / AnalysisResult basics
# ---------------------------------------------------------------------------

class TestDataClasses:
    def test_quote_timestamp_str(self):
        q = Quote(text="x", timestamp_seconds=3723.0)
        assert q.timestamp_str == "1:02:03"

    def test_quote_no_timestamp(self):
        q = Quote(text="x")
        assert q.timestamp_str is None

    def test_analysis_result_to_dict(self):
        r = AnalysisResult(
            summary="s", key_points=["a", "b"],
            analysis="x", quotes=[Quote(text="q", timestamp_seconds=5.0)],
            topic="t", tone="tone", audience="aud",
            model_name="m", backend="b",
        )
        d = r.to_dict()
        assert d["summary"] == "s"
        assert d["key_points"] == ["a", "b"]
        assert d["quotes"][0]["timestamp_str"] == "0:05"

    def test_analysis_result_has_content(self):
        assert AnalysisResult(summary="x").has_content is True
        assert AnalysisResult().has_content is False


# ---------------------------------------------------------------------------
# Streaming SSE + idle timeout
# ---------------------------------------------------------------------------

class TestStreamingChat:
    """Verify the streaming + idle-timeout path used in production."""

    def test_stream_deltas_yields_each_token(self, small_transcript):
        from yt_insight.analyzer.llamacpp_local import (
            LlamaCppLocalAnalyzer,
        )

        analyzer = LlamaCppLocalAnalyzer(
            base_url="http://fake:8080", model="fake",
            idle_timeout_s=5.0,
        )

        # 3 SSE events, each with a small delta.
        sse_lines = [
            'data: {"choices": [{"delta": {"content": "Hello "}}]}',
            'data: {"choices": [{"delta": {"content": "world"}}]}',
            'data: {"choices": [{"delta": {"content": "!"}}]}',
            "data: [DONE]",
        ]
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = iter(sse_lines)
        resp.raise_for_status.return_value = None

        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = resp
        stream_cm.__exit__.return_value = False
        fake_client = MagicMock()
        fake_client.stream.return_value = stream_cm

        with patch.object(analyzer, "_get_client", return_value=fake_client):
            tokens = list(analyzer._stream_deltas("hi"))

        assert tokens == ["Hello ", "world", "!"]

    def test_stream_deltas_invokes_on_token_callback(self, small_transcript):
        analyzer = LlamaCppLocalAnalyzer(
            base_url="http://fake:8080", model="fake", idle_timeout_s=5.0,
        )

        sse_lines = [
            'data: {"choices": [{"delta": {"content": "abc"}}]}',
            'data: {"choices": [{"delta": {"content": "def"}}]}',
            "data: [DONE]",
        ]
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = iter(sse_lines)
        resp.raise_for_status.return_value = None
        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = resp
        stream_cm.__exit__.return_value = False
        fake_client = MagicMock()
        fake_client.stream.return_value = stream_cm

        received: list[str] = []
        with patch.object(analyzer, "_get_client", return_value=fake_client):
            for tok in analyzer._stream_deltas("hi", on_token=received.append):
                pass

        assert received == ["abc", "def"]

    def test_stream_deltas_aborts_on_idle_timeout(self, small_transcript):
        from yt_insight.analyzer.llamacpp_local import (
            AnalysisError, LlamaCppLocalAnalyzer,
        )

        analyzer = LlamaCppLocalAnalyzer(
            base_url="http://fake:8080", model="fake", idle_timeout_s=0.05,
        )

        # Emit ONE token, then yield a few empty lines with a sleep.
        # With idle_timeout_s=50ms, the empty lines (which take >50ms
        # to produce thanks to the sleep) should trigger the abort.
        def slow_iter():
            yield 'data: {"choices": [{"delta": {"content": "first"}}]}'
            # First empty line is fast (still within 50ms).
            yield ""
            # Second empty line is slow (>50ms since last token) → timeout.
            time.sleep(0.1)
            yield ""

        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines.side_effect = lambda: slow_iter()
        resp.raise_for_status.return_value = None
        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = resp
        stream_cm.__exit__.return_value = False
        fake_client = MagicMock()
        fake_client.stream.return_value = stream_cm

        with patch.object(analyzer, "_get_client", return_value=fake_client):
            with pytest.raises(AnalysisError, match="No token received"):
                list(analyzer._stream_deltas("hi"))

    def test_chat_returns_assembled_string(self, small_transcript):
        """`_chat()` should still return a single string (joined deltas)."""
        analyzer = LlamaCppLocalAnalyzer(
            base_url="http://fake:8080", model="fake", idle_timeout_s=5.0,
        )

        sse_lines = [
            'data: {"choices": [{"delta": {"content": "abc"}}]}',
            'data: {"choices": [{"delta": {"content": "def"}}]}',
            "data: [DONE]",
        ]
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = iter(sse_lines)
        resp.raise_for_status.return_value = None
        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = resp
        stream_cm.__exit__.return_value = False
        fake_client = MagicMock()
        fake_client.stream.return_value = stream_cm

        with patch.object(analyzer, "_get_client", return_value=fake_client):
            result = analyzer._chat("hi")
        assert result == "abcdef"

    def test_idle_timeout_default(self):
        # 30 min default — large enough for very long prompt processing
        # on consumer GPUs (observed 14 min for 30k tokens on GTX 1650S).
        from yt_insight.analyzer.llamacpp_local import DEFAULT_IDLE_TIMEOUT_S
        a = create_analyzer()
        assert a.idle_timeout_s == DEFAULT_IDLE_TIMEOUT_S
        assert a.idle_timeout_s >= 1800.0

    def test_default_idle_timeout_is_10min(self):  # legacy name
        from yt_insight.analyzer.llamacpp_local import DEFAULT_IDLE_TIMEOUT_S
        a = create_analyzer()
        assert a.idle_timeout_s == DEFAULT_IDLE_TIMEOUT_S

    def test_max_prompt_tokens_default_is_50k(self):
        from yt_insight.analyzer.llamacpp_local import DEFAULT_MAX_PROMPT_TOKENS
        a = create_analyzer()
        assert a.max_prompt_tokens == DEFAULT_MAX_PROMPT_TOKENS == 50_000

    def test_idle_timeout_env_var(self, monkeypatch):
        monkeypatch.setenv("LLAMACPP_IDLE_TIMEOUT_S", "60")
        a = create_analyzer()
        assert a.idle_timeout_s == 60.0

    def test_create_analyzer_with_depth_and_sections_kwargs(self):
        """Regression: depth + sections kwargs must propagate to LlamaCppLocalAnalyzer."""
        a = create_analyzer(
            depth="shallow",
            sections="forces,biases",
        )
        assert a.depth.value == "shallow"
        assert a.sections == ("forces", "biases")
        # shallow preset forces these numeric levers.
        assert a.temperature == 0.4
        assert a.max_tokens == 1024

    def test_create_analyzer_with_depth_env_var(self, monkeypatch):
        monkeypatch.setenv("LLAMACPP_DEPTH", "deep")
        a = create_analyzer()
        assert a.depth.value == "deep"
        # deep default sections are 6 rubrics.
        assert len(a.sections) == 6

    def test_create_analyzer_with_sections_env_var(self, monkeypatch):
        monkeypatch.setenv("LLAMACPP_SECTIONS", "forces,weaknesses,contradictions")
        a = create_analyzer()
        assert a.sections == ("forces", "weaknesses", "contradictions")

    def test_create_analyzer_kwarg_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("LLAMACPP_DEPTH", "deep")
        a = create_analyzer(depth="shallow")
        assert a.depth.value == "shallow"

    def test_create_analyzer_none_kwargs_fall_back_to_defaults(self):
        """
        Regression: the CLI passes ``max_prompt_tokens=None`` when the
        user didn't set ``--llamacpp-max-prompt-tokens``. The factory
        must convert None to the default, otherwise downstream
        ``logger.info("max_prompt=%d", self.max_prompt_tokens)``
        crashes with TypeError: %d format: a real number is required,
        not NoneType.
        """
        a = create_analyzer(
            base_url="http://x:8080",
            timeout_s=None,           # → DEFAULT_TIMEOUT_S (7200)
            idle_timeout_s=None,      # → DEFAULT_IDLE_TIMEOUT_S (1800)
            max_prompt_tokens=None,   # → DEFAULT_MAX_PROMPT_TOKENS (50000)
            depth="shallow",
            sections="forces",
        )
        assert a.timeout_s == 7200.0
        assert a.idle_timeout_s == 1800.0
        assert a.max_prompt_tokens == 50000


# -------------------------------------------------------------------
# Tolerant JSON extraction
# -------------------------------------------------------------------

class TestExtractJsonObjectTolerant:

    def _extract(self, text):
        from yt_insight.analyzer.llamacpp_local import _extract_json_object
        return _extract_json_object(text)

    def test_well_formed_json(self):
        d = self._extract('{"summary": "x", "key_points": [], "quotes": []}')
        assert d == {"summary": "x", "key_points": [], "quotes": []}

    def test_nested_objects(self):
        d = self._extract(
            '{"summary": "x", "analysis": {"forces": "f", "concepts": ["a"]}}'
        )
        assert d["analysis"]["concepts"] == ["a"]

    def test_fenced_json_block(self):
        text = (
            "Here is the analysis:\n\n"
            "```json\n"
            '{"summary": "x", "key_points": []}\n'
            "```\n"
        )
        d = self._extract(text)
        assert d["summary"] == "x"

    def test_truncated_json_repaired(self):
        """Model hit max_tokens and emitted a partial JSON object."""
        d = self._extract(
            '{"summary": "long text", "key_points": ["a", "b", "c"'
        )
        # json_repair should recover what it can.
        assert d["summary"] == "long text"
        assert "a" in d["key_points"]

    def test_literal_newline_in_string_repaired(self):
        """LLMs often emit literal \\n inside JSON strings."""
        d = self._extract(
            '{"summary": "Line 1\nLine 2\nLine 3", "key_points": []}'
        )
        assert "Line 1" in d["summary"]
        assert "Line 3" in d["summary"]

    def test_trailing_comma_repaired(self):
        d = self._extract(
            '{"summary": "x", "key_points": [], "quotes": [],}'
        )
        assert d["summary"] == "x"
        assert d["quotes"] == []


# -------------------------------------------------------------------
# CLI depth/sections propagation (regression for the Jancovici bug)
# -------------------------------------------------------------------

class TestCliDepthSectionsPropagation:
    """
    Regression: the 'yt-insight all --depth extreme' (no --sections)
    run on the Jancovici video produced only the 3 NORMAL sections
    (forces, concepts, implications) instead of the 8 expected for
    extreme. Root cause: ``_validate_depth_sections`` was calling
    ``coerce_sections(None)`` which returns the NORMAL default, so
    the CLI overrode the depth's per-depth defaults.
    """

    def test_extreme_no_sections_returns_none(self):
        from yt_insight.cli import _validate_depth_sections
        depth_obj, sections = _validate_depth_sections("extreme", None)
        assert depth_obj.value == "extreme"
        assert sections is None  # so __init__ applies EXTREME's 8 sections

    def test_extreme_empty_sections_returns_none(self):
        from yt_insight.cli import _validate_depth_sections
        depth_obj, sections = _validate_depth_sections("extreme", "")
        assert depth_obj.value == "extreme"
        assert sections is None

    def test_extreme_explicit_sections_returns_tuple(self):
        from yt_insight.cli import _validate_depth_sections
        _, sections = _validate_depth_sections(
            "extreme", "forces,weaknesses"
        )
        assert sections == ("forces", "weaknesses")

    def test_invalid_sections_raises(self):
        from yt_insight.cli import _validate_depth_sections
        with pytest.raises(ValueError, match="Unknown section"):
            _validate_depth_sections("extreme", "forces,bogus")

    def test_extreme_no_sections_propagates_to_analyzer(self):
        """
        End-to-end: CLI validation + factory + __init__ must end up
        with 8 sections for extreme when the user didn't pass --sections.
        """
        from yt_insight.cli import _validate_depth_sections
        from yt_insight.analyzer.llamacpp_local import create_analyzer

        depth_obj, sections = _validate_depth_sections("extreme", None)
        a = create_analyzer(depth=depth_obj, sections=sections)
        assert a.depth.value == "extreme"
        assert len(a.sections) == 8

    def test_shallow_no_sections_propagates_to_analyzer(self):
        from yt_insight.cli import _validate_depth_sections
        from yt_insight.analyzer.llamacpp_local import create_analyzer

        depth_obj, sections = _validate_depth_sections("shallow", None)
        a = create_analyzer(depth=depth_obj, sections=sections)
        assert a.depth.value == "shallow"
        assert len(a.sections) == 1  # just 'forces'

    def test_repetition_penalty_default_is_1_1(self):
        """Default repetition_penalty is 1.1, which breaks infinite loops."""
        from yt_insight.analyzer.llamacpp_local import DEFAULT_REPETITION_PENALTY
        a = create_analyzer()
        assert a.repetition_penalty == DEFAULT_REPETITION_PENALTY
        assert a.repetition_penalty == 1.1

    def test_repetition_penalty_override(self):
        a = create_analyzer(repetition_penalty=1.2)
        assert a.repetition_penalty == 1.2

    def test_repetition_penalty_env_var(self, monkeypatch):
        monkeypatch.setenv("LLAMACPP_REPETITION_PENALTY", "1.15")
        a = create_analyzer()
        assert a.repetition_penalty == 1.15

    def test_repetition_penalty_in_payload(self):
        """When > 1.0, repetition_penalty is in the chat completion payload."""
        a = create_analyzer(repetition_penalty=1.1)
        payload = a._build_payload("hello", stream=False)
        assert payload["repetition_penalty"] == 1.1

    def test_repetition_penalty_omitted_when_neutral(self):
        """When == 1.0, the field is omitted (no penalty is the default)."""
        a = create_analyzer(repetition_penalty=1.0)
        payload = a._build_payload("hello", stream=False)
        assert "repetition_penalty" not in payload

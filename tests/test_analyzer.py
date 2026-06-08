"""
Tests for ``yt_insight.analyzer``.

The HTTP layer is mocked throughout — no llama-server required.
"""

from __future__ import annotations

import json
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
        fake_client.post.return_value = _fake_chat_response(json.dumps(payload))
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
        fake_client.post.return_value = _fake_chat_response(json.dumps(payload))
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
        fake_client.post.return_value = _fake_chat_response(fenced)
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
        fake_client.post.return_value = _fake_chat_response("not used")
        fake_client.post.return_value.status_code = 500
        fake_client.post.return_value.json.side_effect = Exception("no json")
        fake_client.post.return_value.text = "internal error"
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

        def fake_chat(user_prompt: str) -> str:
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
        fake_client.post.return_value = _fake_chat_response(json.dumps(payload))
        with patch.object(analyzer, "_get_client", return_value=fake_client):
            result = analyzer.analyze(small_transcript, language="fr")
        # Only one HTTP call should have hit /v1/chat/completions.
        assert fake_client.post.call_count == 1
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

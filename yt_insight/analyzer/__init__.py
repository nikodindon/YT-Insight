"""LLM-based analysis backends for YouTube transcripts.

Each backend takes a ``TranscriptionResult`` (from
:mod:`yt_insight.transcriber`) and produces a structured
``AnalysisResult``: summary, key points, deeper analysis, notable
quotes, and high-level metadata.

Currently supported backends
----------------------------
- :class:`LlamaCppLocalAnalyzer` — talks to a locally running
  ``llama-server`` (the one bundled with llama.cpp) over its
  OpenAI-compatible HTTP API.

Adding a new backend (e.g. OpenAI, Anthropic Claude) only requires
subclassing :class:`BaseAnalyzer` and implementing ``analyze()``.
"""

from .base import AnalysisResult, BaseAnalyzer, Quote
from .llamacpp_local import LlamaCppLocalAnalyzer, create_analyzer

__all__ = [
    "AnalysisResult",
    "BaseAnalyzer",
    "LlamaCppLocalAnalyzer",
    "Quote",
    "create_analyzer",
]

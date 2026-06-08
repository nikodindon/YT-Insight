"""
Tests for the ``Depth`` / ``Section`` enums and the prompt builders.
"""

from __future__ import annotations

import pytest

from yt_insight.analyzer.depth import (
    DEFAULT_SECTIONS, DEPTH_DEFAULT_SECTIONS, DEPTH_PRESETS, SECTION_INFO,
    Depth, Section, coerce_depth, coerce_sections, depth_preset,
)
from yt_insight.analyzer.prompts import (
    build_analysis_prompt, build_chunk_prompt, build_merge_prompt,
)


# -------------------------------------------------------------------
# Depth enum
# -------------------------------------------------------------------

class TestDepth:
    def test_all_4_presets_exist(self):
        assert set(DEPTH_PRESETS.keys()) == {
            Depth.SHALLOW, Depth.NORMAL, Depth.DEEP, Depth.EXTREME,
        }

    def test_each_preset_has_all_levers(self):
        required = {
            "max_tokens", "num_key_points", "num_quotes", "temperature",
            "num_summary_words_min", "num_summary_words_max",
            "num_analysis_words_min", "num_analysis_words_max",
        }
        for depth, preset in DEPTH_PRESETS.items():
            missing = required - preset.keys()
            assert not missing, f"{depth} missing {missing}"

    def test_shallow_is_shortest_and_creative(self):
        s = DEPTH_PRESETS[Depth.SHALLOW]
        n = DEPTH_PRESETS[Depth.NORMAL]
        d = DEPTH_PRESETS[Depth.DEEP]
        assert s["max_tokens"] < n["max_tokens"] < d["max_tokens"]
        assert s["num_key_points"] < n["num_key_points"] < d["num_key_points"]
        assert s["num_quotes"] < n["num_quotes"] < d["num_quotes"]
        # Temperature decreases with depth (more deterministic).
        assert s["temperature"] > n["temperature"] > d["temperature"]

    def test_extreme_is_the_most_detailed(self):
        e = DEPTH_PRESETS[Depth.EXTREME]
        d = DEPTH_PRESETS[Depth.DEEP]
        assert e["max_tokens"] > d["max_tokens"]
        assert e["num_key_points"] > d["num_key_points"]
        assert e["num_quotes"] > d["num_quotes"]

    def test_coerce_depth_from_str(self):
        assert coerce_depth("shallow") is Depth.SHALLOW
        assert coerce_depth("NORMAL") is Depth.NORMAL
        assert coerce_depth(None) is Depth.NORMAL  # default

    def test_coerce_depth_invalid_raises_with_helpful_msg(self):
        with pytest.raises(ValueError, match="Unknown depth"):
            coerce_depth("bogus")

    def test_coerce_depth_passes_through(self):
        assert coerce_depth(Depth.DEEP) is Depth.DEEP

    def test_depth_preset_returns_copy(self):
        p1 = depth_preset(Depth.NORMAL)
        p1["max_tokens"] = 999
        # Mutating the returned dict must not leak into the global table.
        # If it leaked, both would be 999. We want the global to stay 4096.
        assert DEPTH_PRESETS[Depth.NORMAL]["max_tokens"] == 4096
        assert p1["max_tokens"] == 999


# -------------------------------------------------------------------
# Section enum
# -------------------------------------------------------------------

class TestSections:
    def test_all_8_sections_exist(self):
        assert len(SECTION_INFO) == 8
        assert set(Section) == {
            Section.FORCES, Section.CONCEPTS, Section.IMPLICATIONS,
            Section.WEAKNESSES, Section.CONTRADICTIONS, Section.BIASES,
            Section.LIMITATIONS, Section.CONTEXT_GAPS,
        }

    def test_coerce_sections_from_csv_str(self):
        assert coerce_sections("forces,biases") == ("forces", "biases")

    def test_coerce_sections_from_list(self):
        assert coerce_sections(["weaknesses", "contradictions"]) == (
            "weaknesses", "contradictions"
        )

    def test_coerce_sections_dedup_preserves_order(self):
        assert coerce_sections("forces,forces,biases") == ("forces", "biases")

    def test_coerce_sections_strips_whitespace(self):
        assert coerce_sections(" forces , biases ") == ("forces", "biases")

    def test_coerce_sections_none_returns_default(self):
        assert coerce_sections(None) == DEFAULT_SECTIONS

    def test_coerce_sections_invalid_raises_with_list(self):
        with pytest.raises(ValueError, match="Unknown section"):
            coerce_sections("forces,bogus")
        # Error should list valid options.
        with pytest.raises(ValueError, match="forces"):
            coerce_sections("bogus")

    def test_default_sections_match_normal(self):
        assert DEPTH_DEFAULT_SECTIONS[Depth.NORMAL] == DEFAULT_SECTIONS


# -------------------------------------------------------------------
# Prompt builders
# -------------------------------------------------------------------

class TestPromptBuilders:
    def _transcript(self) -> str:
        return "Bonjour à tous, aujourd'hui on parle d'IA."

    def test_analysis_prompt_normal_has_3_sections(self):
        p = build_analysis_prompt(self._transcript())
        assert "Forces" in p
        assert "Concepts centraux" in p
        assert "Implications et perspectives" in p
        assert "Faiblesses" not in p  # not in default normal sections

    def test_analysis_prompt_extreme_has_all_8_sections(self):
        sections = (
            "forces", "concepts", "implications", "weaknesses",
            "contradictions", "biases", "limitations", "context_gaps",
        )
        p = build_analysis_prompt(self._transcript(), depth=Depth.EXTREME, sections=sections)
        for marker in ("Forces", "Concepts", "Implications", "Faiblesses",
                       "Contradictions", "Biais", "Limites", "Contexte manquant"):
            assert marker in p, f"Missing section marker: {marker}"

    def test_analysis_prompt_shallow_has_only_forces(self):
        p = build_analysis_prompt(self._transcript(), depth=Depth.SHALLOW)
        assert "Forces" in p
        assert "Faiblesses" not in p
        assert "Biais" not in p

    def test_analysis_prompt_respects_section_override(self):
        # --depth extreme but --sections just forces
        p = build_analysis_prompt(
            self._transcript(), depth=Depth.EXTREME, sections=("forces",),
        )
        assert "Forces" in p
        assert "Faiblesses" not in p
        assert "Biais" not in p

    def test_analysis_prompt_includes_depth_metadata(self):
        p = build_analysis_prompt(self._transcript(), depth=Depth.DEEP)
        assert "Profondeur d'analyse : deep" in p
        # deep preset has max_tokens=8192, temperature=0.1
        assert "max_tokens=8192" in p
        assert "température=0.1" in p

    def test_analysis_prompt_surfaces_num_key_points_range(self):
        # Normal depth: 15 key points → range is 13-17
        p = build_analysis_prompt(self._transcript(), depth=Depth.NORMAL)
        assert "13 à 17" in p
        # Shallow: 5 key points → 3-7
        p2 = build_analysis_prompt(self._transcript(), depth=Depth.SHALLOW)
        assert "3 à 7" in p2

    def test_chunk_prompt_uses_depth(self):
        p = build_chunk_prompt(
            self._transcript(), chunk_index=1, chunk_total=3,
            depth=Depth.EXTREME,
            sections=("forces", "concepts", "implications",
                      "weaknesses", "contradictions", "biases",
                      "limitations", "context_gaps"),
        )
        # Match the actual literal in the prompt ("chunk N°1 sur 3").
        assert "chunk N" in p
        assert "1/3" in p
        # extreme has 40 key points → ~13-20 per chunk
        assert "40" in p
        # ... and ~7-15 quotes per chunk
        assert "7" in p and "15" in p

    def test_merge_prompt_uses_depth(self):
        p = build_merge_prompt(
            ['{"summary": "x", "key_points": [], "quotes": []}'],
            depth=Depth.DEEP,
            sections=("forces", "concepts", "implications",
                      "weaknesses", "contradictions", "biases"),
        )
        assert "Chunk 1/1" in p
        assert "Profondeur" not in p  # merge doesn't repeat depth
        # deep preset: 25 key points → range 23-27
        assert "23 à 27" in p

    def test_empty_sections_omits_block(self):
        p = build_analysis_prompt(self._transcript(), sections=())
        # No rubric block at all.
        assert "**Forces**" not in p
        assert "doit contenir" not in p or "rubrique imposée" in p

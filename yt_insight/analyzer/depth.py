"""
Depth and section configuration for the analyzer.

Depth controls the **numeric levers** (max_tokens, num_key_points,
num_quotes, temperature) of a single analysis run. Sections control
the **semantic content** (which rubrics appear in the analysis
markdown).

Two orthogonal axes on purpose:
- ``--depth`` is opinionated (4 presets).
- ``--sections`` is fine-grained (8 rubrics, pick what you want).
- If both are given, ``--sections`` overrides the depth's default
  sections, but the depth's numeric levers still apply.

Default behaviour (no flags) is unchanged: ``normal`` depth,
``forces,concepts,implications`` sections, ~15 key points, 5 quotes.
"""

from __future__ import annotations

from enum import Enum
from typing import Final


# -------------------------------------------------------------------
# Depth
# -------------------------------------------------------------------

class Depth(str, Enum):
    """Analysis depth preset."""

    SHALLOW = "shallow"
    NORMAL = "normal"
    DEEP = "deep"
    EXTREME = "extreme"


#: Numeric levers per depth.
#: Keys: max_tokens, num_key_points, num_quotes, temperature
#: num_summary_words is a hint inserted in the prompt.
DEPTH_PRESETS: Final[dict[Depth, dict[str, int | float]]] = {
    Depth.SHALLOW: {
        "max_tokens": 1024,
        "num_key_points": 5,
        "num_quotes": 3,
        "temperature": 0.4,
        "num_summary_words_min": 150,
        "num_summary_words_max": 300,
        "num_analysis_words_min": 100,
        "num_analysis_words_max": 250,
    },
    Depth.NORMAL: {
        "max_tokens": 4096,
        "num_key_points": 15,
        "num_quotes": 5,
        "temperature": 0.2,
        "num_summary_words_min": 500,
        "num_summary_words_max": 1000,
        "num_analysis_words_min": 300,
        "num_analysis_words_max": 600,
    },
    Depth.DEEP: {
        "max_tokens": 8192,
        "num_key_points": 25,
        "num_quotes": 10,
        "temperature": 0.1,
        "num_summary_words_min": 800,
        "num_summary_words_max": 1500,
        "num_analysis_words_min": 600,
        "num_analysis_words_max": 1200,
    },
    Depth.EXTREME: {
        "max_tokens": 16384,
        "num_key_points": 40,
        "num_quotes": 15,
        "temperature": 0.0,
        "num_summary_words_min": 1200,
        "num_summary_words_max": 2500,
        "num_analysis_words_min": 1000,
        "num_analysis_words_max": 2000,
    },
}

#: Default sections per depth (used when --sections is not given).
DEPTH_DEFAULT_SECTIONS: Final[dict[Depth, tuple[str, ...]]] = {
    Depth.SHALLOW: ("forces",),
    Depth.NORMAL: ("forces", "concepts", "implications"),
    Depth.DEEP: (
        "forces", "concepts", "implications",
        "weaknesses", "contradictions", "biases",
    ),
    Depth.EXTREME: (
        "forces", "concepts", "implications",
        "weaknesses", "contradictions", "biases",
        "limitations", "context_gaps",
    ),
}


def coerce_depth(value: str | Depth | None) -> Depth:
    """Coerce a CLI string to a :class:`Depth`. Raises ``ValueError`` on bad input."""
    if value is None:
        return Depth.NORMAL
    if isinstance(value, Depth):
        return value
    s = str(value).strip().lower()
    try:
        return Depth(s)
    except ValueError as exc:
        valid = ", ".join(d.value for d in Depth)
        raise ValueError(
            f"Unknown depth '{value}'. Valid options: {valid}"
        ) from exc


def depth_preset(depth: Depth) -> dict[str, int | float]:
    """Return a **copy** of the depth's numeric levers (caller may mutate)."""
    return dict(DEPTH_PRESETS[depth])


# -------------------------------------------------------------------
# Sections
# -------------------------------------------------------------------

class Section(str, Enum):
    """A rubric that can appear in the analysis markdown."""

    FORCES = "forces"
    CONCEPTS = "concepts"
    IMPLICATIONS = "implications"
    WEAKNESSES = "weaknesses"
    CONTRADICTIONS = "contradictions"
    BIASES = "biases"
    LIMITATIONS = "limitations"
    CONTEXT_GAPS = "context_gaps"


#: Section title (used in the prompt) and a short description.
SECTION_INFO: Final[dict[Section, tuple[str, str]]] = {
    Section.FORCES:         ("Forces du contenu",          "Les points forts : arguments bien étayés, exemples concrets, structuration efficace."),
    Section.CONCEPTS:       ("Concepts centraux",          "Les idées-clés expliquées ou définies par l'auteur."),
    Section.IMPLICATIONS:   ("Implications et perspectives","Les conséquences, projections, et enseignements que l'auteur tire ou que l'on peut tirer."),
    Section.WEAKNESSES:     ("Faiblesses et limites",      "Les arguments mal étayés, données manquantes, sophismes, approximations, zones floues."),
    Section.CONTRADICTIONS: ("Contradictions internes",    "Les éléments du propos qui s'opposent ou s'invalident mutuellement."),
    Section.BIASES:         ("Biais cognitifs et rhétoriques","Les biais de raisonnement détectés : biais de confirmation, cherry-picking, appels émotionnels non fondés, généralisations abusives, etc."),
    Section.LIMITATIONS:    ("Limites de cette analyse",   "Ce que cette analyse ne peut pas conclure : manque de contexte, partialité, scope trop large/étroit, etc."),
    Section.CONTEXT_GAPS:   ("Contexte manquant",          "Les éléments de contexte que l'auteur omet volontairement, et qui changeraient l'interprétation."),
}

#: Default sections (used when --sections is not given AND --depth is not given).
DEFAULT_SECTIONS: Final[tuple[str, ...]] = ("forces", "concepts", "implications")


def coerce_sections(value) -> tuple[str, ...]:
    """
    Coerce a CLI value (str or list[str]) to a tuple of section names.

    Accepts:
    - a comma-separated string: ``"forces,biases"``
    - a list of strings: ``["forces", "biases"]``
    - ``None`` → :data:`DEFAULT_SECTIONS`

    Raises ``ValueError`` on unknown section name (with a list of valid ones).
    """
    if value is None:
        return DEFAULT_SECTIONS
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    else:
        parts = [str(p).strip() for p in value]
    parts = [p for p in parts if p]
    valid = {s.value for s in Section}
    bad = [p for p in parts if p not in valid]
    if bad:
        valid_list = ", ".join(sorted(valid))
        raise ValueError(
            f"Unknown section(s): {bad}. Valid options: {valid_list}"
        )
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)

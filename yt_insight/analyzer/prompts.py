"""
All LLM prompts used by the analyzer.

Keep them in this dedicated module so they can be tweaked without
touching code, versioned, and unit-tested in isolation.

Design notes
------------
- The model is Qwen3.6 35B-A3B (MoE) served by llama.cpp. It's a
  strong instruct model in French *and* English.
- We always disable "thinking" mode in chat_template_kwargs to get
  concise, direct outputs (the Qwen3.6 chat template defaults to
  thinking = on, which would burn tokens and slow us down for
  structured analysis tasks).
- Prompts are in French by default (the user speaks French) but
  accept a ``language`` placeholder for internationalization.
- We ask for a **single JSON object** as the response format. This is
  simpler to validate than multiple separate calls and lets us keep
  the model in a single coherent context window.
"""

from __future__ import annotations

from .depth import (
    DEFAULT_SECTIONS, DEPTH_PRESETS, SECTION_INFO, Depth, Section,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Tu es un assistant expert en analyse de contenus vidéo et \
audio. Tu reçois la transcription complète d'une vidéo YouTube et tu dois \
produire une analyse structurée, précise et utile.

Règles absolues :
- Tu réponds **UNIQUEMENT en JSON valide**, sans texte avant ou après.
- Tu n'inventes aucun fait : si une information n'est pas dans la \
transcription, tu l'omets ou tu écris "non mentionné".
- Tu écris dans la langue de la transcription (par défaut : français).
- Tu ne commences jamais ta réponse par "Voici", "Voici le", "Bien sûr", \
ou toute formule de politesse : le JSON commence directement.
- Les citations doivent être des phrases réellement prononcées dans la \
transcription, mot pour mot ou quasi-mot pour mot.
"""


# ---------------------------------------------------------------------------
# Main analysis prompt (single-shot)
# ---------------------------------------------------------------------------

#: Template variables:
#:   {language}   - "français" / "english" / etc.
#:   {title}      - video title (may be empty)
#:   {duration}   - human-readable duration "42:17"
#:   {transcript} - the (possibly chunked) transcript text
ANALYSIS_PROMPT = """Analyse la transcription de cette vidéo YouTube et \
produis un objet JSON avec **exactement** les clés suivantes :

{{
  "summary":      "<résumé intégral et détaillé en prose, 500-1000 mots, \
qui couvre l'ensemble du contenu en respectant la progression de la vidéo. \
Mentionne les exemples concrets, les arguments principaux et les transitions \
importantes entre les parties.>",
  "key_points":   ["<point clé 1, formulé comme une affirmation complète>", \
"<point clé 2>", "... 8 à 15 points au total."],
  "analysis":     "<analyse approfondie en 3 sous-sections Markdown : \
**Forces** du contenu, **Concepts centraux** expliqués, \
**Implications et perspectives**. 300-600 mots au total.>",
  "quotes":       [{{ "text": "<citation verbatim>", \
"timestamp_seconds": <nombre ou null> }}],
  "topic":        "<sujet principal en 2-6 mots>",
  "tone":         "<ton général : informatif / humoristique / \
argumentatif / pédagogique / promotionnel / etc.>",
  "audience":     "<public visé, en 3-8 mots>"
}}

**Métadonnées vidéo :**
- Titre : {title}
- Durée : {duration}
- Langue : {language}

**Contraintes spécifiques :**
- ``summary`` doit être une prose continue, sans liste à puces.
- ``key_points`` doit contenir entre 8 et 15 éléments, chacun étant une \
affirmation autonome et actionable (pas un simple mot-clé).
- ``quotes`` doit contenir entre 3 et 8 citations, chacune fidèle à la \
transcription. ``timestamp_seconds`` doit être l'instant approximatif en \
secondes depuis le début de la vidéo (entier ou flottant, jamais null \
si la transcription porte des horodatages).
- ``analysis`` peut utiliser du Markdown léger (gras, listes) — il sera \
rendu tel quel dans la sortie.
- Si la transcription est tronquée ou peu claire sur un point, signale-le \
honnêtement dans l'analyse au lieu d'inventer.

**Transcription :**
<<<
{transcript}
>>>

Réponds maintenant avec l'objet JSON uniquement, sans ``` et sans texte \
autour.
"""


# ---------------------------------------------------------------------------
# Per-chunk prompt (used when the transcript doesn't fit in one window)
# ---------------------------------------------------------------------------

CHUNK_PROMPT = """Tu vas recevoir un **extrait** (chunk N°{chunk_index} sur \
{chunk_total}) de la transcription d'une vidéo YouTube. Pour cet extrait \
uniquement, produis un objet JSON avec les clés suivantes :

{{
  "summary":    "<résumé de l'extrait, 100-250 mots, prose continue>",
  "key_points": ["<point clé de l'extrait, 1 phrase>", "..."],
  "quotes":     [{{ "text": "<citation verbatim>", \
"timestamp_seconds": <nombre ou null> }}]
}}

**Métadonnées :**
- Titre global : {title}
- Durée globale : {duration}
- Langue : {language}
- Extrait N°{chunk_index}/{chunk_total}

**Extrait :**
<<<
{transcript}
>>>

JSON uniquement, sans texte autour.
"""


# ---------------------------------------------------------------------------
# Merge prompt (used to fuse per-chunk JSONs into the final AnalysisResult)
# ---------------------------------------------------------------------------

MERGE_PROMPT = """Tu reçois plusieurs résumés partiels d'une même vidéo \
YouTube, produits à partir de tronçons successifs de la transcription. \
Fusionne-les en une analyse finale cohérente et produis l'objet JSON \
final avec **exactement** ces clés :

{{
  "summary":      "<résumé intégral, 500-1000 mots, prose continue, \
qui restitue la progression globale de la vidéo>",
  "key_points":   ["<affirmation complète>", "..."],
  "analysis":     "<analyse approfondie en 3 sous-sections Markdown : \
**Forces** / **Concepts centraux** / **Implications**. 300-600 mots.>",
  "quotes":       [{{ "text": "<citation verbatim>", \
"timestamp_seconds": <nombre ou null> }}],
  "topic":        "<sujet principal en 2-6 mots>",
  "tone":         "<ton général>",
  "audience":     "<public visé, 3-8 mots>"
}}

**Règles de fusion :**
- ``summary`` : recompose une narration fluide qui traverse tous les \
chunks ; ne te contente pas de concaténer.
- ``key_points`` : 8 à 15 points au total, dédupliqués, classés par ordre \
d'importance.
- ``quotes`` : 3 à 8 citations, en privilégiant celles qui ont un vrai \
impact. Garde les timestamps.
- ``analysis`` : écris une analyse de synthèse, pas une simple \
juxtaposition des analyses partielles.
- Si deux chunks disent la même chose, garde la formulation la plus \
claire.

**Métadonnées vidéo :**
- Titre : {title}
- Durée : {duration}
- Langue : {language}

**Résumés partiels (un par chunk, dans l'ordre temporel) :**
{chunks}

JSON uniquement, sans texte autour.
"""


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _render_section_instruction(sections: tuple[str, ...]) -> str:
    """
    Build the ``analysis`` block instruction for the prompt.

    Includes only the requested sections, with a short description
    of each, so the LLM knows exactly what to write under each
    rubric header.
    """
    if not sections:
        # No sections → no analysis block (model can still write
        # one if it wants, but we don't enforce structure).
        return ""

    lines: list[str] = []
    lines.append(
        "L'analyse approfondie doit être rédigée en Markdown et contenir "
        f"les {len(sections)} sous-sections suivantes, dans cet ordre :"
    )
    for i, sec_name in enumerate(sections, 1):
        try:
            sec = Section(sec_name)
        except ValueError:
            # Unknown section name: skip silently (CLI should
            # have validated already).
            continue
        title, desc = SECTION_INFO[sec]
        lines.append(f"{i}. **{title}** — {desc}")
    lines.append("")
    lines.append(
        "Chaque sous-section doit faire au minimum 2-3 phrases. "
        "Le tout doit rester cohérent et bien articulé."
    )
    return "\n".join(lines)


# -------------------------------------------------------------------
# Builders
# -------------------------------------------------------------------

def build_analysis_prompt(
    transcript: str,
    *,
    title: str = "",
    duration: str = "",
    language: str = "français",
    depth: Depth = Depth.NORMAL,
    sections: tuple[str, ...] = DEFAULT_SECTIONS,
) -> str:
    """
    Build the single-shot analysis prompt.

    The numeric levers (max_tokens, num_key_points, num_quotes) are
    surfaced as explicit count ranges in the prompt. ``depth`` and
    ``sections`` drive those numbers and the analysis rubric list.
    """
    preset = DEPTH_PRESETS[depth]
    nk_min = max(3, preset["num_key_points"] - 2)
    nk_max = preset["num_key_points"] + 2
    nq_min = max(1, preset["num_quotes"] - 1)
    nq_max = preset["num_quotes"] + 2
    sum_min = preset["num_summary_words_min"]
    sum_max = preset["num_summary_words_max"]
    ana_min = preset["num_analysis_words_min"]
    ana_max = preset["num_analysis_words_max"]

    analysis_block = _render_section_instruction(sections)
    if analysis_block:
        analysis_value = (
            f"<analyse approfondie structurée en {len(sections)} "
            f"sous-sections, {int(ana_min)}-{int(ana_max)} mots au total.>"
        )
    else:
        analysis_value = (
            f"<analyse approfondie libre en Markdown, "
            f"{int(ana_min)}-{int(ana_max)} mots.>"
        )

    return (
        "Analyse la transcription de cette vidéo YouTube et produis un "
        "objet JSON avec **exactement** les clés suivantes :\n\n"
        "{{\n"
        '  "summary":      "<résumé intégral en prose continue, '
        f"{sum_min}-{sum_max} mots, qui couvre l'ensemble du contenu en "
        "respectant la progression de la vidéo. Mentionne les exemples "
        "concrets, les arguments principaux et les transitions importantes "
        'entre les parties.",\n'
        '  "key_points":   ["<point clé 1, formulé comme une affirmation '
        'complète>", "<point clé 2>", "... '
        f"{nk_min} à {nk_max} points au total."
        '"],\n'
        f'  "analysis":     "{analysis_value}",\n'
        '  "quotes":       [{{ "text": "<citation verbatim>", '
        '"timestamp_seconds": <nombre ou null> }}],\n'
        '  "topic":        "<sujet principal en 2-6 mots>",\n'
        '  "tone":         "<ton général : informatif / humoristique / '
        'argumentatif / pédagogique / promotionnel / etc.>",\n'
        '  "audience":     "<public visé, en 3-8 mots>"\n'
        "}}\n\n"
        "**Métadonnées vidéo :**\n"
        f"- Titre : {title or '(non renseigné)'}\n"
        f"- Durée : {duration or '(non renseignée)'}\n"
        f"- Langue : {language}\n"
        f"- Profondeur d'analyse : {depth.value} "
        f"(max_tokens={int(preset['max_tokens'])}, "
        f"température={preset['temperature']})\n\n"
        "**Instructions pour l'analyse approfondie :**\n"
        f"{analysis_block if analysis_block else '(aucune rubrique imposée — écris librement)'}\n\n"
        "**Contraintes spécifiques :**\n"
        "- ``summary`` doit être une prose continue, sans liste à puces.\n"
        f"- ``key_points`` doit contenir entre {nk_min} et {nk_max} éléments, "
        "chacun étant une affirmation autonome et actionable "
        "(pas un simple mot-clé).\n"
        f"- ``quotes`` doit contenir entre {nq_min} et {nq_max} citations, "
        "chacune fidèle à la transcription. ``timestamp_seconds`` doit être "
        "l'instant approximatif en secondes depuis le début de la vidéo "
        "(entier ou flottant, jamais null si la transcription porte des horodatages).\n"
        "- ``analysis`` peut utiliser du Markdown léger (gras, listes) — "
        "il sera rendu tel quel dans la sortie.\n"
        "- Si la transcription est tronquée ou peu claire sur un point, "
        "signale-le honnêtement dans l'analyse au lieu d'inventer.\n\n"
        "**Transcription :**\n"
        "<<<\n"
        f"{transcript.strip()}\n"
        ">>>\n\n"
        "Réponds maintenant avec l'objet JSON uniquement, sans ``` et "
        "sans texte autour.\n"
    )


def build_chunk_prompt(
    transcript: str,
    *,
    chunk_index: int,
    chunk_total: int,
    title: str = "",
    duration: str = "",
    language: str = "français",
    depth: Depth = Depth.NORMAL,
    sections: tuple[str, ...] = DEFAULT_SECTIONS,
) -> str:
    """
    Build a per-chunk prompt (used when the transcript doesn't fit
    in a single analysis window).

    Chunks use a fixed lightweight schema (no topic/tone/audience —
    those are produced by the final merge).
    """
    preset = DEPTH_PRESETS[depth]
    nk_min = max(2, preset["num_key_points"] // 3)
    nk_max = max(nk_min + 1, preset["num_key_points"] // 2)
    nq_min = max(1, preset["num_quotes"] // 2)
    nq_max = max(nq_min + 1, preset["num_quotes"])
    analysis_block = _render_section_instruction(sections)

    analysis_line = ""
    if analysis_block:
        analysis_line = (
            f',\n  "analysis":     "<mini-analyse du chunk '
            f'({len(sections)} sous-sections : '
            f'{", ".join(sections)})>"'
        )

    return (
        f"Tu vas recevoir un **extrait** (chunk N°{chunk_index} sur "
        f"{chunk_total}) de la transcription d'une vidéo YouTube. Pour "
        "cet extrait uniquement, produis un objet JSON avec les clés "
        "suivantes :\n\n"
        "{{\n"
        '  "summary":    "<résumé de l\'extrait, 100-250 mots, prose continue>",\n'
        f'  "key_points": ["<point clé de l\'extrait, 1 phrase>", "..."],'
        f"\n  // {nk_min} à {nk_max} points par chunk, ~"
        f"{int(preset['num_key_points'])} pour la vidéo entière"
        f'\n  "quotes":     [{{ "text": "<citation verbatim>", '
        f'"timestamp_seconds": <nombre ou null> }}]'
        f"  // {nq_min} à {nq_max} citations par chunk"
        f"{analysis_line}\n"
        "}}\n\n"
        "**Métadonnées :**\n"
        f"- Titre global : {title or '(non renseigné)'}\n"
        f"- Durée globale : {duration or '(non renseignée)'}\n"
        f"- Langue : {language}\n"
        f"- Extrait N°{chunk_index}/{chunk_total}\n\n"
        "**Extrait :**\n"
        "<<<\n"
        f"{transcript.strip()}\n"
        ">>>\n\n"
        "JSON uniquement, sans texte autour.\n"
    )


def build_merge_prompt(
    chunk_payloads: list[str],
    *,
    title: str = "",
    duration: str = "",
    language: str = "français",
    depth: Depth = Depth.NORMAL,
    sections: tuple[str, ...] = DEFAULT_SECTIONS,
) -> str:
    """
    Build the merge prompt that fuses per-chunk JSONs into the
    final :class:`AnalysisResult`.
    """
    preset = DEPTH_PRESETS[depth]
    nk_min = max(3, preset["num_key_points"] - 2)
    nk_max = preset["num_key_points"] + 2
    nq_min = max(1, preset["num_quotes"] - 1)
    nq_max = preset["num_quotes"] + 2
    sum_min = preset["num_summary_words_min"]
    sum_max = preset["num_summary_words_max"]
    ana_min = preset["num_analysis_words_min"]
    ana_max = preset["num_analysis_words_max"]
    analysis_block = _render_section_instruction(sections)

    if analysis_block:
        analysis_value = (
            f"<analyse approfondie de synthèse, structurée en "
            f"{len(sections)} sous-sections, {int(ana_min)}-{int(ana_max)} mots.>"
        )
    else:
        analysis_value = (
            f"<analyse approfondie libre, {int(ana_min)}-{int(ana_max)} mots.>"
        )

    chunks_block = "\n\n---\n\n".join(
        f"### Chunk {i + 1}/{len(chunk_payloads)}\n{c.strip()}"
        for i, c in enumerate(chunk_payloads)
    )

    return (
        "Tu reçois plusieurs résumés partiels d'une même vidéo YouTube, "
        "produits à partir de tronçons successifs de la transcription. "
        "Fusionne-les en une analyse finale cohérente et produis l'objet "
        "JSON final avec **exactement** ces clés :\n\n"
        "{{\n"
        f'  "summary":      "<résumé intégral, {sum_min}-{sum_max} mots, '
        "prose continue, qui restitue la progression globale de la vidéo>\",\n"
        '  "key_points":   ["<affirmation complète>", "..."],\n'
        f'  "analysis":     "{analysis_value}",\n'
        '  "quotes":       [{{ "text": "<citation verbatim>", '
        '"timestamp_seconds": <nombre ou null> }}],\n'
        '  "topic":        "<sujet principal en 2-6 mots>",\n'
        '  "tone":         "<ton général>",\n'
        '  "audience":     "<public visé, 3-8 mots>"\n'
        "}}\n\n"
        "**Règles de fusion :**\n"
        "- ``summary`` : recompose une narration fluide qui traverse tous "
        "les chunks ; ne te contente pas de concaténer.\n"
        f"- ``key_points`` : {nk_min} à {nk_max} points au total, "
        "dédupliqués, classés par ordre d'importance.\n"
        f"- ``quotes`` : {nq_min} à {nq_max} citations, en privilégiant "
        "celles qui ont un vrai impact. Garde les timestamps de la "
        "transcription.\n"
        "- ``analysis`` : écris une analyse de synthèse, pas une simple "
        "juxtaposition des analyses partielles.\n\n"
        "**Instructions pour l'analyse approfondie :**\n"
        f"{analysis_block if analysis_block else '(aucune rubrique imposée — écris librement)'}\n\n"
        "**Métadonnées vidéo :**\n"
        f"- Titre : {title or '(non renseigné)'}\n"
        f"- Durée : {duration or '(non renseignée)'}\n"
        f"- Langue : {language}\n\n"
        "**Résumés partiels (un par chunk, dans l'ordre temporel) :**\n"
        f"{chunks_block}\n\n"
        "JSON uniquement, sans texte autour.\n"
    )

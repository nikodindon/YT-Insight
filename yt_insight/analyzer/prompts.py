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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_analysis_prompt(
    transcript: str,
    *,
    title: str = "",
    duration: str = "",
    language: str = "français",
) -> str:
    """Format :data:`ANALYSIS_PROMPT` with the given context."""
    return ANALYSIS_PROMPT.format(
        language=language,
        title=title or "(non renseigné)",
        duration=duration or "(non renseignée)",
        transcript=transcript.strip(),
    )


def build_chunk_prompt(
    transcript: str,
    *,
    chunk_index: int,
    chunk_total: int,
    title: str = "",
    duration: str = "",
    language: str = "français",
) -> str:
    """Format :data:`CHUNK_PROMPT` for a single chunk."""
    return CHUNK_PROMPT.format(
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        title=title or "(non renseigné)",
        duration=duration or "(non renseignée)",
        language=language,
        transcript=transcript.strip(),
    )


def build_merge_prompt(
    chunk_payloads: list[str],
    *,
    title: str = "",
    duration: str = "",
    language: str = "français",
) -> str:
    """Format :data:`MERGE_PROMPT` with all per-chunk JSON blobs."""
    chunks_block = "\n\n---\n\n".join(
        f"### Chunk {i + 1}/{len(chunk_payloads)}\n{c.strip()}"
        for i, c in enumerate(chunk_payloads)
    )
    return MERGE_PROMPT.format(
        chunks=chunks_block,
        title=title or "(non renseigné)",
        duration=duration or "(non renseignée)",
        language=language,
    )

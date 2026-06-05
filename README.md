# 🎙️ YT-Insight — YouTube Transcription & Analysis Pipeline

> Pipeline local/cloud pour télécharger l'audio d'une vidéo YouTube, le transcrire avec un modèle Whisper local, puis générer un résumé détaillé et une analyse approfondie via LLM (local ou cloud).

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [État d'avancement](#état-davancement)
3. [Architecture du projet](#architecture-du-projet)
4. [Stack technique](#stack-technique)
5. [Structure du projet](#structure-du-projet)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Utilisation](#utilisation)
9. [Modules détaillés](#modules-détaillés)
10. [Flux de données](#flux-de-données)
11. [Performances & optimisations](#performances--optimisations)
12. [Roadmap](#roadmap)

---

## Vue d'ensemble

**YT-Insight** est un pipeline en ligne de commande qui enchaîne automatiquement :

```
URL YouTube  →  Audio .mp3  →  Transcription texte  →  Résumé + Analyse
```

Chaque étape est modulaire, testable indépendamment, et configurable pour tourner entièrement en local ou en hybride local/cloud.

### Cas d'usage typiques

- Analyser une conférence ou un talk technique sans regarder la vidéo entière
- Extraire les points clés d'une interview ou d'un podcast
- Constituer une base de connaissances à partir de vidéos YouTube
- Recherche et veille : résumer des dizaines de vidéos rapidement

---

## État d'avancement

| Module | Fichiers | Tests | Statut |
|--------|----------|-------|--------|
| **Downloader** | `downloader/ytdlp_downloader.py` | 12 ✅ | **Terminé** |
| **Transcriber** | `transcriber/faster_whisper.py` | — | 🔜 En cours |
| **Analyzer** | `analyzer/ollama_local.py` + prompts | — | 🔜 À venir |
| **Output** | `output/console.py`, `file_writer.py` | — | 🔜 À venir |
| **CLI** | `cli.py` | — | 🔜 À venir |
| **Utils** | `utils/config.py`, `text_utils.py` | — | 🔜 À venir |

---

## Architecture du projet

```
┌─────────────────────────────────────────────────────────────────────┐
│                          YT-Insight Pipeline                        │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────┐   │
│  │  CLI /   │    │   Downloader │    │    Transcriber          │   │
│  │  Config  │───▶│  (yt-dlp)   │───▶│  (faster-whisper)       │   │
│  │  (Rich)  │    │             │    │  CUDA int8 / CPU fallback│   │
│  └──────────┘    └──────────────┘    └────────────┬────────────┘   │
│                                                   │                │
│                                                   ▼                │
│                                      ┌────────────────────────┐   │
│                                      │   LLM Analyzer         │   │
│                                      │  ┌──────────────────┐  │   │
│                                      │  │ Résumé détaillé  │  │   │
│                                      │  │ Analyse          │  │   │
│                                      │  │ Points clés      │  │   │
│                                      │  │ Citations        │  │   │
│                                      │  └──────────────────┘  │   │
│                                      │                        │   │
│                                      │  Backends disponibles: │   │
│                                      │  • Ollama local        │   │
│                                      │    (Qwen3 35B)         │   │
│                                      │  • Ollama Cloud        │   │
│                                      │  • MiniMax API         │   │
│                                      │  • Claude API          │   │
│                                      └───────────┬────────────┘   │
│                                                  │                │
│                                                  ▼                │
│                                      ┌────────────────────────┐   │
│                                      │   Output Manager       │   │
│                                      │  • Console (Rich)      │   │
│                                      │  • Fichier Markdown    │   │
│                                      │  • JSON structuré      │   │
│                                      └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Diagramme de séquence

```
User         CLI          Downloader    Transcriber      LLM           Output
 │            │                │              │            │              │
 │──url──────▶│                │              │            │              │
 │            │──download()───▶│              │            │              │
 │            │                │──mp3 file───▶│            │              │
 │            │                │              │──transcribe│              │
 │            │                │              │  (CUDA)    │              │
 │            │                │              │──text─────▶│              │
 │            │                │              │            │──summarize() │
 │            │                │              │            │──analyze()   │
 │            │                │              │            │──result─────▶│
 │            │                │              │            │              │──save()
 │◀─────────────────────────────────────────────────────────────────────│
```

---

## Stack technique

| Composant | Outil retenu | Alternatives | Raison du choix |
|-----------|-------------|--------------|-----------------|
| **Download audio** | `yt-dlp` | pytube, youtube-dl | Actif, fiable, formats flexibles |
| **Transcription** | `faster-whisper` (large-v3) | whisper.cpp, Vosk | CUDA int8, parfait GTX 1650 Super |
| **LLM local** | `Ollama` + Qwen3 35B Q4 | llama.cpp direct | Déjà opérationnel sur ta machine |
| **LLM cloud** | `MiniMax API` / `Ollama Cloud` | Claude API, OpenAI | Contexte 1M tokens (MiniMax), flexibilité |
| **CLI** | `Rich` + `Typer` | Click, argparse | Beau rendu terminal, progress bars |
| **Config** | `python-dotenv` + YAML | TOML, INI | Standard, lisible |
| **Tests** | `pytest` | unittest | Écosystème standard |
| **Audio processing** | `ffmpeg` (via yt-dlp) | pydub | Natif, performances maximales |

### Pourquoi faster-whisper ?

- Utilise **CTranslate2** : jusqu'à 4x plus rapide que Whisper original
- **Quantification int8** : `large-v3` tient dans 2-3 Go de VRAM (ta 1650 Super a 4 Go)
- Support CUDA natif sous Linux
- Sortie avec **timestamps** par segment (utile pour navigation)

### Pourquoi MiniMax pour le cloud ?

- Contexte de **1 million de tokens** → peut ingérer des transcriptions de plusieurs heures
- Bon rapport qualité/prix
- API compatible style OpenAI (facile à intégrer)

---

## Structure du projet

```
yt-insight/
│
├── README.md
├── requirements.txt               ✅ créé
├── requirements-dev.txt           ✅ créé
├── pyproject.toml                 ✅ créé  (entry point: yt-insight = yt_insight.cli:app)
├── .env.example                   ✅ créé
├── config.yaml                    🔜 à créer
│
├── yt_insight/                    # Package principal
│   ├── __init__.py                ✅ créé  (version 0.1.0)
│   ├── cli.py                     🔜 à créer  (Typer + Rich)
│   │
│   ├── downloader/                ✅ MODULE TERMINÉ
│   │   ├── __init__.py            ✅ expose YtDlpDownloader, DownloadResult
│   │   └── ytdlp_downloader.py    ✅ implémentation complète
│   │
│   ├── transcriber/               🔜 module suivant
│   │   ├── __init__.py
│   │   ├── base.py                # Classe abstraite Transcriber
│   │   └── faster_whisper.py      # Implémentation faster-whisper
│   │
│   ├── analyzer/                  🔜 module 3
│   │   ├── __init__.py
│   │   ├── base.py                # Classe abstraite Analyzer
│   │   ├── prompts.py             # Tous les prompts LLM
│   │   ├── ollama_local.py        # Backend Ollama local
│   │   ├── ollama_cloud.py        # Backend Ollama Cloud
│   │   └── minimax.py             # Backend MiniMax API
│   │
│   ├── output/                    🔜 module 4
│   │   ├── __init__.py
│   │   ├── console.py             # Rendu Rich terminal
│   │   └── file_writer.py         # Export Markdown / JSON
│   │
│   └── utils/                     🔜 module 4
│       ├── __init__.py
│       ├── config.py              # Chargement config & env
│       ├── logger.py              # Logging structuré
│       └── text_utils.py          # Chunking, nettoyage texte
│
├── tests/
│   ├── __init__.py                ✅ créé
│   ├── test_downloader.py         ✅ créé  (12 tests)
│   ├── test_transcriber.py        🔜 à créer
│   ├── test_analyzer.py           🔜 à créer
│   └── fixtures/
│       └── sample_transcript.txt  🔜 à créer
│
├── outputs/                       ✅ créé  (gitignored)
└── cache/                         ✅ créé  (gitignored)
```

---

## Installation

### Prérequis système

```bash
# Vérifier CUDA (pour faster-whisper GPU)
nvidia-smi

# Installer ffmpeg (requis par yt-dlp)
sudo apt update && sudo apt install ffmpeg -y

# Vérifier la version Python (3.10+ recommandé)
python3 --version

# Ollama doit être installé et en cours d'exécution
ollama serve  # dans un terminal séparé
ollama pull qwen3:35b  # si pas déjà fait
```

### Installation du projet

```bash
# Cloner le repo
git clone https://github.com/tonuser/yt-insight.git
cd yt-insight

# Créer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Pour le développement
pip install -r requirements-dev.txt

# Installer le package en mode éditable
pip install -e .
```

### Installation de faster-whisper avec support CUDA

```bash
# CUDA 11.x / 12.x — vérifier avec : nvcc --version
pip install faster-whisper

# Dépendances CUDA (si pas déjà présentes)
pip install nvidia-cudnn-cu11  # ou cu12 selon ta version
```

> **Note GPU :** Sur GTX 1650 Super (4 Go VRAM), utiliser `model_size="large-v3"` avec `compute_type="int8"`. Si la VRAM est insuffisante (transcription très longue), le fallback automatique sur CPU est géré.

---

## Configuration

### Fichier `.env`

```bash
cp .env.example .env
```

Contenu de `.env` :

```dotenv
# === LLM Backend (local | ollama_cloud | minimax | claude) ===
LLM_BACKEND=local

# === Ollama local ===
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:35b

# === Ollama Cloud ===
OLLAMA_CLOUD_API_KEY=ta_cle_ici
OLLAMA_CLOUD_MODEL=qwen3:35b

# === MiniMax ===
MINIMAX_API_KEY=ta_cle_ici
MINIMAX_MODEL=MiniMax-Text-01
MINIMAX_GROUP_ID=ton_group_id

# === Claude (optionnel) ===
ANTHROPIC_API_KEY=ta_cle_ici

# === Whisper ===
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda          # cuda | cpu
WHISPER_COMPUTE_TYPE=int8    # int8 | float16 | float32
WHISPER_LANGUAGE=            # Laisser vide pour auto-détection

# === Chemins ===
OUTPUT_DIR=./outputs
CACHE_DIR=./cache
KEEP_AUDIO=false             # Supprimer l'audio après transcription
```

### Fichier `config.yaml`

```yaml
pipeline:
  # Étapes à exécuter
  steps:
    - download
    - transcribe
    - summarize
    - analyze

  # Comportement sur erreur
  fail_fast: true

transcription:
  model: large-v3
  device: cuda
  compute_type: int8
  beam_size: 5
  language: null  # null = auto-détection
  word_timestamps: true

analysis:
  # Longueur max de la transcription envoyée au LLM (en tokens estimés)
  max_transcript_tokens: 100000
  # Stratégie si la transcription est trop longue : chunk | truncate | summarize_chunks
  overflow_strategy: chunk

  outputs:
    - summary       # Résumé intégral détaillé
    - key_points    # Liste des points clés
    - analysis      # Analyse approfondie
    - quotes        # Citations notables
    - metadata      # Durée, langue, titre estimé

output:
  formats:
    - console
    - markdown
  # true = inclure la transcription brute dans le fichier de sortie
  include_transcript: true
  # true = inclure timestamps dans la transcription
  include_timestamps: true
```

---

## Utilisation

### Commande de base

```bash
# Analyser une vidéo avec les paramètres par défaut
yt-insight "https://www.youtube.com/watch?v=VIDEO_ID"

# Spécifier la langue (gain de vitesse Whisper)
yt-insight "https://youtube.com/watch?v=VIDEO_ID" --language fr

# Utiliser le backend cloud MiniMax
yt-insight "https://youtube.com/watch?v=VIDEO_ID" --backend minimax

# Sauvegarder dans un dossier spécifique
yt-insight "https://youtube.com/watch?v=VIDEO_ID" --output-dir ~/analyses/

# Exécuter seulement la transcription (sans analyse LLM)
yt-insight "https://youtube.com/watch?v=VIDEO_ID" --steps download,transcribe

# Analyser une transcription existante (skip download + transcription)
yt-insight --transcript-file ./outputs/ma_transcription.txt

# Mode verbeux
yt-insight "https://youtube.com/watch?v=VIDEO_ID" --verbose
```

### Exemples de sortie console

```
╔══════════════════════════════════════════════════════╗
║              YT-Insight  v0.1.0                     ║
╚══════════════════════════════════════════════════════╝

📥 Téléchargement audio...
  ✓ Titre     : "Comment fonctionne l'attention dans les Transformers"
  ✓ Durée     : 42:17
  ✓ Format    : mp3 / 128kbps
  ✓ Fichier   : cache/abc123.mp3 (38.2 Mo)

🎙️ Transcription (faster-whisper large-v3 · CUDA int8)...
  ████████████████████ 100%  42:17 / 42:17  [00:03:21]
  ✓ Langue détectée : Français (confiance : 0.99)
  ✓ Segments       : 847
  ✓ Tokens estimés : 18,432

🤖 Analyse LLM (Ollama local · qwen3:35b)...
  ✓ Résumé généré
  ✓ Points clés extraits (12 points)
  ✓ Analyse approfondie générée
  ✓ Citations notables extraites (5)

💾 Sauvegarde...
  ✓ outputs/2024-01-15_transformers-attention.md
  ✓ outputs/2024-01-15_transformers-attention.json

⏱️  Temps total : 4m 38s
```

### Format de sortie Markdown

```markdown
# Analyse : "Comment fonctionne l'attention dans les Transformers"

**URL :** https://youtube.com/watch?v=...
**Durée :** 42:17 | **Langue :** Français | **Date d'analyse :** 2024-01-15

---

## 📝 Résumé détaillé

[Résumé intégral de 500-1000 mots couvrant tout le contenu...]

---

## 🎯 Points clés

1. **Le mécanisme d'attention** permet au modèle de pondérer...
2. **Les têtes d'attention multiples** capturent différents types...
...

---

## 🔍 Analyse approfondie

### Forces du contenu
...

### Concepts centraux
...

### Implications et perspectives
...

---

## 💬 Citations notables

> "L'attention est littéralement le mécanisme qui permet au modèle de savoir où regarder" — 12:34

---

## 📄 Transcription complète

[00:00:00] Bonjour à tous, aujourd'hui on va parler...
[00:01:23] Le concept d'attention a été introduit...
```

---

## Modules détaillés

### `downloader/ytdlp_downloader.py` ✅

**Statut : terminé et testé.**

Classes exposées :

`YtDlpDownloader(cache_dir, audio_format, audio_quality, keep_audio)` — le downloader principal.

```python
downloader = YtDlpDownloader(cache_dir=Path("./cache"))
result = downloader.download("https://youtube.com/watch?v=...")
print(result.audio_path)        # Path → cache/dQw4w9WgXcQ.mp3
print(result.metadata.title)    # "Rick Astley - Never Gonna Give You Up"
print(result.metadata.duration_str)  # "3:33"
print(result.from_cache)        # True si déjà téléchargé
```

Méthodes publiques :

- `download(url, force=False) → DownloadResult` — télécharge l'audio et met en cache. Si `force=False` et que le fichier existe déjà, retourne instantanément depuis le cache.
- `fetch_metadata_only(url) → VideoMetadata` — preview sans téléchargement (titre, durée, chaîne).

`VideoMetadata` — dataclass avec les champs `video_id`, `title`, `channel`, `duration_seconds`, `description`, `upload_date`, `view_count`, `url`, et les propriétés calculées :
- `.duration_str` → `"42:17"` ou `"1:02:03"`
- `.slug` → nom de fichier URL-safe depuis le titre (max 60 chars)

`DownloadResult` — dataclass avec `audio_path`, `metadata`, `from_cache`, `download_time_seconds`, et `.to_dict()` pour sérialisation JSON.

`DownloadError` — exception custom avec messages lisibles selon le type d'erreur (vidéo privée, indisponible, âge, URL invalide).

Paramètres yt-dlp utilisés :
```python
{
    'format': 'bestaudio/best',
    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
    'outtmpl': 'cache/{video_id}.%(ext)s',
    'quiet': True,
    'writeinfojson': False,
    'writethumbnail': False,
}
```

Gestion du cache : deux fichiers par vidéo dans `cache/` :
- `{video_id}.mp3` — audio
- `{video_id}.meta.json` — métadonnées persistées

Formats d'URL supportés pour l'extraction du `video_id` :
- `youtube.com/watch?v=VIDEO_ID`
- `youtu.be/VIDEO_ID`
- `youtube.com/shorts/VIDEO_ID`
- `youtube.com/embed/VIDEO_ID`
- Fallback : hash MD5 de l'URL (stable, pour formats non-standards)

**Tests — `tests/test_downloader.py` (12 tests) ✅**

```
TestExtractVideoId         — 5 tests (formats d'URL + fallback hash)
TestVideoMetadata          — 4 tests (duration_str, slug)
TestCacheLogic             — 3 tests (hit, force bypass, persistance meta)
TestErrorHandling          — 2 tests (vidéo privée, indisponible)
TestMetadataSerialization  — 2 tests (round-trip JSON, to_dict)
```

---

### `transcriber/faster_whisper.py`

Responsabilités :
- Charger le modèle Whisper en CUDA int8 (ou CPU si indisponible)
- Transcrire le fichier audio avec timestamps par segment
- Retourner un objet `TranscriptionResult` : texte brut + segments horodatés
- Gérer la détection automatique de langue
- Libérer la VRAM après transcription (important pour laisser de la mémoire au LLM local)

Structure de sortie :
```python
@dataclass
class TranscriptionResult:
    text: str                    # Transcription brute complète
    segments: list[Segment]      # [{start, end, text}, ...]
    language: str                # "fr", "en", etc.
    language_confidence: float
    duration_seconds: float
```

---

### `analyzer/prompts.py`

Contient tous les prompts système et utilisateur, versionnés et modifiables sans toucher au code :

```python
SYSTEM_PROMPT = """
Tu es un expert en analyse de contenu vidéo. Tu reçois la transcription complète
d'une vidéo YouTube et tu dois produire une analyse structurée, précise et utile.
Réponds toujours en JSON valide avec les clés définies.
"""

SUMMARY_PROMPT = """
À partir de cette transcription, génère un résumé intégral et détaillé qui :
- Couvre l'ensemble du contenu sans omission majeure
- Respecte la structure et la progression de la vidéo
- Est rédigé en prose claire et structurée (500-1000 mots)
- Mentionne les exemples concrets donnés
...
"""
```

---

### `analyzer/ollama_local.py`

- Communique avec l'API REST Ollama locale (`http://localhost:11434`)
- Gère le streaming des réponses pour affichage en temps réel
- Implémente la stratégie de **chunking** si la transcription dépasse la fenêtre de contexte :
  - Découpe la transcription en chunks avec overlap
  - Génère un résumé intermédiaire par chunk
  - Fusionne les résumés intermédiaires en une analyse finale
- Gère l'**offload CPU** de Qwen3 35B (déjà configuré sur ta machine)

---

### `utils/text_utils.py`

Fonctions utilitaires :
- `estimate_tokens(text)` : estimation rapide du nombre de tokens (longueur/4)
- `chunk_text(text, max_tokens, overlap)` : découpage intelligent aux frontières de phrases
- `format_transcript_with_timestamps(segments)` : mise en forme `[HH:MM:SS] texte`
- `clean_transcript(text)` : nettoyage basique (espaces, artefacts Whisper)

---

## Flux de données

```
1. INPUT
   └─ URL YouTube (string)

2. DOWNLOAD (yt-dlp)
   └─ Audio file : cache/{video_id}.mp3
   └─ Metadata   : {title, duration, channel, description}

3. TRANSCRIPTION (faster-whisper)
   └─ TranscriptionResult
       ├─ text      : str (transcription brute)
       ├─ segments  : [{start, end, text}]
       ├─ language  : str
       └─ duration  : float

4. PRÉ-TRAITEMENT
   └─ Estimation du nombre de tokens
   └─ Si > max_tokens → chunking en N parties

5. ANALYSE LLM
   ├─ Prompt construction (system + transcript)
   ├─ Appel API (local ou cloud)
   └─ AnalysisResult
       ├─ summary    : str
       ├─ key_points : list[str]
       ├─ analysis   : str
       ├─ quotes     : list[{text, timestamp}]
       └─ metadata   : {topic, tone, audience}

6. OUTPUT
   ├─ Console (Rich)
   ├─ Markdown file : outputs/{date}_{slug}.md
   └─ JSON file     : outputs/{date}_{slug}.json
```

---

## Performances & optimisations

### Benchmarks attendus (ta config : GTX 1650 Super 4 Go, 16 Go RAM)

| Durée vidéo | Téléchargement | Transcription (CUDA) | Transcription (CPU) | Analyse LLM (local) |
|-------------|---------------|----------------------|---------------------|---------------------|
| 10 min      | ~15s          | ~45s                 | ~3min               | ~1-2min             |
| 30 min      | ~30s          | ~2min                | ~9min               | ~3-5min             |
| 1h          | ~60s          | ~4min                | ~18min              | ~5-10min            |
| 2h          | ~90s          | ~8min                | ~35min              | ~10-20min*          |

*avec chunking si transcription > fenêtre de contexte

### Optimisations implémentées

**Mémoire GPU :**
- Transcription Whisper en int8 → libère la VRAM avant le LLM
- Le modèle Whisper est déchargé explicitement après transcription
- Qwen3 35B en offload CPU/GPU hybride (déjà fonctionnel sur ta machine)

**Cache :**
- L'audio téléchargé est mis en cache par `video_id`
- La transcription peut être sauvegardée pour éviter de re-transcrire
- Option `--from-cache` pour relancer uniquement l'analyse

**Chunking intelligent :**
- Découpage aux frontières de paragraphes/silences (via timestamps Whisper)
- Overlap de 10% entre chunks pour éviter les pertes de contexte
- Merge des analyses partielles avec un prompt de fusion dédié

---

## Roadmap

### Phase 1 — MVP (en cours) 🚧
- [x] Architecture & README
- [x] `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.env.example`
- [x] Module downloader — `yt_insight/downloader/ytdlp_downloader.py`
- [x] Tests downloader — `tests/test_downloader.py` (12 tests)
- [ ] Module transcriber (faster-whisper + CUDA)
- [ ] Module analyzer (Ollama local)
- [ ] CLI Rich + Typer
- [ ] Output Markdown + JSON
- [ ] Configuration `config.yaml` + `utils/config.py`

### Phase 2 — Backends cloud
- [ ] Intégration MiniMax API
- [ ] Intégration Ollama Cloud
- [ ] Intégration Claude API (optionnel)
- [ ] Sélection automatique du backend selon longueur

### Phase 3 — Features avancées
- [ ] Support playlists YouTube (batch processing)
- [ ] API REST FastAPI (pour intégration externe)
- [ ] Interface web Gradio (optionnel)
- [ ] Base de données SQLite pour historique des analyses
- [ ] Export PDF
- [ ] Recherche full-text dans les transcriptions

### Phase 4 — Qualité
- [ ] Suite de tests complète (pytest)
- [ ] CI/CD GitHub Actions
- [ ] Dockerisation
- [ ] Packaging PyPI

---

## Dépendances

### `requirements.txt`

```
# Download
yt-dlp>=2024.1.0

# Transcription
faster-whisper>=1.0.0

# LLM backends
ollama>=0.2.0
httpx>=0.27.0          # MiniMax API + HTTP générique
anthropic>=0.25.0      # Optionnel — Claude API

# CLI & UI
typer>=0.12.0
rich>=13.7.0

# Config
python-dotenv>=1.0.0
pyyaml>=6.0.0

# Utils
tqdm>=4.66.0
```

### `requirements-dev.txt`

```
-r requirements.txt

pytest>=8.0.0
pytest-asyncio>=0.23.0
black>=24.0.0
ruff>=0.4.0
mypy>=1.10.0
```

### `pyproject.toml`

Packaging avec `setuptools`. Le projet s'installe en mode éditable via `pip install -e .` et expose la commande `yt-insight` en point d'entrée :

```toml
[project.scripts]
yt-insight = "yt_insight.cli:app"
```

---

## Licence

MIT — voir `LICENSE`

---

*Projet développé pour usage personnel sur Linux Mint · GTX 1650 Super · 16 Go RAM*

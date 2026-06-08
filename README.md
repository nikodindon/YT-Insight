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

| Module      | Fichiers                                            | Tests   | Statut        |
|-------------|-----------------------------------------------------|---------|---------------|
| **Downloader**  | `downloader/ytdlp_downloader.py`                | 16 ✅   | **Terminé**   |
| **Transcriber** | `transcriber/base.py`, `faster_whisper_transcriber.py` | 42 ✅ | **Terminé**   |
| **Analyzer**    | `analyzer/{base,prompts,llamacpp_local}.py`     | 25 ✅   | **Terminé**   |
| **Utils**       | `utils/{text_utils,config,logger}.py`           | 41 ✅   | **Terminé**   |
| **Output**      | `output/{console,file_writer}.py`               | 21 ✅   | **Terminé**   |
| **CLI**         | `cli.py` (5 sous-commandes Typer + Rich Live panel) | 22 ✅   | **Terminé**   |
| **Estimate**    | `estimate.py` + 3 tests CLI                    | 24 ✅   | **Terminé**   |

Total : **206 tests passent** (`pytest tests/ -q`).

Voir aussi [`docs/ROADMAP.md`](docs/ROADMAP.md) pour les tickets à venir (playlists, PDF, etc.).

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
│                                      │  • llama.cpp local      │   │
│                                      │    (Qwen3.6 35B-A3B)    │   │
│                                      │    via API OpenAI-compat │   │
│                                      │  • Futurs: ollama, vLLM │   │
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
| **LLM local** | `llama-server` (llama.cpp) + Qwen3.6 35B-A3B | Ollama, vLLM | Tourne déjà sur ta machine sur :8080, API OpenAI-compat |
| **HTTP client** | `httpx` | requests, aiohttp | Sync + async, timeouts propres, simple |
| **CLI** | `Rich` + `Typer` | Click, argparse | Beau rendu terminal, progress bars |
| **Config** | `python-dotenv` + YAML | TOML, INI | Standard, lisible |
| **Tests** | `pytest` | unittest | Écosystème standard |
| **Audio processing** | `ffmpeg` (via yt-dlp) | pydub | Natif, performances maximales |

### Pourquoi faster-whisper ?

- Utilise **CTranslate2** : jusqu'à 4x plus rapide que Whisper original
- **Quantification int8** : `large-v3` tient dans 2-3 Go de VRAM (ta 1650 Super a 4 Go)
- Support CUDA natif sous Linux
- Sortie avec **timestamps** par segment (utile pour navigation)

### Pourquoi llama.cpp en local ?

- Tourne déjà sur ta machine (`llama-server` sur :8080) avec Qwen3.6 35B-A3B
- **Contrôle total** sur les flags : offload GPU (`-ngl 99`), CPU offload des experts MoE (`--n-cpu-moe`), quantization du KV cache (`-ctk q4_0 -ctv q4_0`)
- **API OpenAI-compatible** : `/v1/chat/completions`, `/v1/models`, streaming natif
- Pas de couche intermédiaire (Ollama) à configurer — le GGUF est chargé directement
- Fonctionne aussi avec Ollama, vLLM, LM Studio : il suffit de changer `LLAMACPP_BASE_URL`

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
│   ├── __init__.py                ✅ version 0.1.0
│   ├── cli.py                     ✅ Typer + Rich : all / download / transcribe / analyze / version
│   │
│   ├── downloader/                ✅ MODULE TERMINÉ
│   │   ├── __init__.py            ✅ expose YtDlpDownloader, DownloadResult, VideoMetadata, DownloadError
│   │   └── ytdlp_downloader.py    ✅ implémentation complète
│   │
│   ├── transcriber/               ✅ MODULE TERMINÉ
│   │   ├── __init__.py            ✅ expose BaseTranscriber, TranscriptionResult, Segment, FasterWhisperTranscriber
│   │   ├── base.py                ✅ dataclasses Segment + TranscriptionResult, classe abstraite BaseTranscriber
│   │   └── faster_whisper_transcriber.py  ✅ implémentation complète + factory create_transcriber()
│   │
│   ├── analyzer/                  ✅ MODULE TERMINÉ
│   │   ├── __init__.py            ✅ expose BaseAnalyzer, AnalysisResult, Quote, LlamaCppLocalAnalyzer, create_analyzer
│   │   ├── base.py                ✅ ABC + dataclasses AnalysisResult / Quote
│   │   ├── prompts.py             ✅ System + analysis + chunk + merge prompts (FR)
│   │   └── llamacpp_local.py      ✅ Backend llama.cpp via /v1/chat/completions + chunk+merge + streaming
│   │
│   ├── output/                    ✅ MODULE TERMINÉ
│   │   ├── __init__.py            ✅ expose ConsoleRenderer, RenderConfig, FileWriter, write_outputs
│   │   ├── console.py             ✅ rendu Rich terminal (panels, tables, Markdown)
│   │   └── file_writer.py         ✅ export Markdown (front matter YAML + sections) et JSON structuré
│   │
│   ├── estimate.py                ✅ prédiction du coût (temps, tokens, stratégie LLM) avant lancement
│   │
│   └── utils/                     ✅ MODULE TERMINÉ
│       ├── __init__.py            ✅ expose load_config, AppConfig + sous-configs, setup_logging, chunk_text, etc.
│       ├── text_utils.py          ✅ chunking intelligent, estimation tokens, nettoyage, formatage timestamps
│       ├── config.py              ✅ dataclasses typées + load_config() (défauts → YAML → env)
│       └── logger.py              ✅ setup_logging() idempotent avec RichHandler
│
├── tests/
│   ├── __init__.py                ✅ créé
│   ├── test_downloader.py         ✅ créé  (16 tests)
│   ├── test_transcriber.py        ✅ créé  (42 tests)
│   ├── test_analyzer.py           ✅ créé  (25 tests)
│   ├── test_text_utils.py         ✅ créé  (19 tests)
│   ├── test_config.py             ✅ créé  (22 tests, config + logger)
│   ├── test_output.py             ✅ créé  (21 tests, console + file_writer)
│   ├── test_cli.py                ✅ créé  (22 tests, Typer CliRunner)
│   ├── test_estimate.py           ✅ créé  (24 tests, prédiction de coût)
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

# llama.cpp doit être compilé (ou installé) et llama-server lancé.
# Exemple (depuis ~/llama.cpp) :
~/llama.cpp/build/bin/llama-server \
    -m /home/niko/models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
    -ngl 99 --n-cpu-moe 34 -t 6 -ctk q4_0 -ctv q4_0 \
    --port 8080 --host 0.0.0.0

# Vérifier que le serveur répond :
curl http://localhost:8080/health
# → {"status":"ok"}
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
# === Whisper ===
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda          # cuda | cpu
WHISPER_COMPUTE_TYPE=int8    # int8 | float16 | float32
WHISPER_LANGUAGE=            # Laisser vide pour auto-détection

# === LLM Backend ===
# Currently only one backend is implemented: llamacpp-local.
LLM_BACKEND=llamacpp_local

# === llama.cpp (llama-server) ===
# Par défaut : llama-server sur :8080, API OpenAI-compat.
# N'importe quel serveur compatible OpenAI fonctionne (Ollama, vLLM, LM Studio…).
LLAMACPP_BASE_URL=http://localhost:8080
LLAMACPP_MODEL=Qwen3.6-35B-A3B-UD-IQ3_S.gguf

# Timeout HTTP en secondes (chunk+merge sur vidéo longue peut prendre plusieurs minutes)
LLAMACPP_TIMEOUT_S=600

# Seuil de tokens au-delà duquel on passe en chunk+merge.
# Doit rester bien sous n_ctx du serveur (32k pour le GGUF par défaut).
LLAMACPP_MAX_PROMPT_TOKENS=28000
LLAMACPP_CHUNK_OVERLAP_TOKENS=200

LLAMACPP_TEMPERATURE=0.2
LLAMACPP_MAX_TOKENS=4000

# Désactive le mode "thinking" de Qwen3 (recommandé pour JSON structuré).
# 0 = activer thinking (plus lent, plus verbeux).
LLAMACPP_DISABLE_THINKING=1

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
  beam_size: 3          # lowered from 5 to fit tight-VRAM GPUs (e.g. GTX 1050 4 Go)
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

> **Prérequis :** `llama-server` doit tourner sur :8080 (cf. section Installation).
> Sinon, lance-le dans un terminal séparé avant d'utiliser `yt-insight`.

### Sous-commandes

| Commande                      | Description                                                |
|-------------------------------|------------------------------------------------------------|
| `yt-insight all URL`          | Pipeline complet : download → transcribe → analyze → write |
| `yt-insight download URL`     | Télécharge l'audio (utile pour pré-cacher)                 |
| `yt-insight transcribe PATH`  | Transcrit un fichier local ou une URL, écrit en JSON       |
| `yt-insight analyze [PATH]`   | Analyse un transcript JSON (ou transcrit+analyse si `--audio`) |
| `yt-insight estimate URL`     | **Prédit** le temps/coût d'analyse sans rien lancer        |
| `yt-insight version`          | Affiche la version installée                               |

Toutes les options peuvent être passées en CLI, via variables d'environnement (`YT_INSIGHT_*`, `WHISPER_*`, `LLAMACPP_*`) ou via `config.yaml`.

### Exemples

```bash
# Pipeline complet avec les paramètres par défaut
yt-insight all "https://www.youtube.com/watch?v=VIDEO_ID"

# Spécifier la langue (gain de vitesse Whisper)
yt-insight all "https://youtube.com/watch?v=VIDEO_ID" --language fr

# Utiliser un autre serveur llama.cpp (ex: GPU distant)
yt-insight all "https://youtube.com/watch?v=VIDEO_ID" --llamacpp-url http://gpu:8080

# Utiliser un modèle différent
yt-insight all "https://youtube.com/watch?v=VIDEO_ID" --llamacpp-model other-model.gguf

# Sauvegarder dans un dossier spécifique
yt-insight all "https://youtube.com/watch?v=VIDEO_ID" --output-dir ~/analyses/

# Exécuter seulement le téléchargement (pas de transcription ni d'analyse)
yt-insight all "https://youtube.com/watch?v=VIDEO_ID" --steps download

# Idem, plusieurs étapes possibles
yt-insight all "https://youtube.com/watch?v=VIDEO_ID" --steps transcribe,analyze

# Transcrire un fichier audio local et sauvegarder le JSON
yt-insight transcribe ./audio.mp3 --language fr --output ./transcript.json

# Analyser un transcript déjà produit (skip download + transcription)
yt-insight analyze ./outputs/transcript.json --no-console

# Transcrire + analyser une URL en un seul shot
yt-insight analyze --audio "https://youtube.com/watch?v=VIDEO_ID"

# Activer le mode verbose (logs détaillés)
yt-insight all "https://youtube.com/watch?v=VIDEO_ID" --verbose

# Désactiver le rendu terminal (utile en cron / pipe)
yt-insight all "URL" --no-console

# Plusieurs formats de sortie
yt-insight all "URL" --format markdown --format json

# Forcer des chunks audio plus petits (utile sur GPU 4 Go type GTX 1050)
yt-insight all "URL" --whisper-chunk-length 20

# Utiliser un modèle Whisper plus léger (medium = 1.5 Go VRAM, distil-large-v3 = 1.5 Go + 1.5x plus rapide)
yt-insight all "URL" --whisper-model medium
yt-insight all "URL" --whisper-model distil-large-v3
# Combiné avec --whisper-chunk-length, ça permet de tenir en VRAM 4 Go sans OOM
yt-insight all "URL" --whisper-model medium --whisper-chunk-length 30

# Le transcriber fallback automatiquement sur CPU si OOM CUDA, mais on peut
# aussi forcer le CPU dès le départ via variable d'env (no GPU dispo / voulu)
WHISPER_DEVICE=cpu yt-insight all "URL"

# Augmenter le timeout HTTP pour les très longues analyses (défaut 2h)
yt-insight all "URL" --llamacpp-timeout 14400   # 4h pour des confs géantes

# Abaisser/augmenter l'idle timeout entre tokens (défaut 10 min).
# Doit rester > temps de prompt processing du 1er chunk
# (~500s sur un 28k-tokens chunk avec ce modèle sur 1650S).
yt-insight all "URL" --llamacpp-idle-timeout 1200   # 20 min

# Élargir la limite de tokens par chunk (défaut 50k).
# Doit rester < n_ctx de votre llama-server.
yt-insight all "URL" --llamacpp-max-prompt-tokens 80000

# === Profondeur d'analyse (4 presets) ===
# shallow  → 5 points clés, 3 citations, ~1024 tokens
# normal   → 15 points clés, 5 citations, ~4096 tokens  (défaut)
# deep     → 25 points clés, 10 citations, ~8192 tokens
# extreme  → 40 points clés, 15 citations, ~16384 tokens
yt-insight all "URL" --depth shallow        # analyse express
yt-insight all "URL" --depth normal         # défaut
yt-insight all "URL" --depth deep           # analyse fouillée
yt-insight all "URL" --depth extreme        # tout, partout, en détail

# === Sections d'analyse (8 rubriques) ===
# Valid: forces, concepts, implications, weaknesses, contradictions,
#        biases, limitations, context_gaps
# Override les sections par défaut du depth.
yt-insight all "URL" --sections forces,weaknesses,contradictions   # vue rhétorique
yt-insight all "URL" --sections biases,contradictions,limitations   # esprit critique
yt-insight all "URL" --depth extreme --sections forces              # extreme mais sobre

# Pendant l'analyse, le panel Rich Live affiche les tokens en temps réel :
#  ╭─ LLM generation ──────────────────────────────────────────╮
#  │ The speaker Duhun, founder of Base 10, opens with his    │
#  │ background transitioning from finance to ML engineering... │
#  │                                                         │
#  ╰──  1847 chars · 312 tokens · 38.2s  ──────────────────╯

# Cache transcript : si on a déjà transcrit une vidéo, le 2e run skip la
# transcription. Le JSON est dans cache/{video_id}.transcript.json
yt-insight all "URL" --language fr
# → 1er run : download + transcribe (lent) + analyze
# → 2e run : download (cache hit) + "Transcript en cache" (instantané) + analyze

# Analyser un transcript déjà sauvegardé (sans re-transcrire)
yt-insight analyze cache/FKqGfecRUkg.transcript.json --llamacpp-url http://100.118.85.70:8080

# Estimer le coût d'une vidéo SANS lancer le pipeline (juste métadonnées yt-dlp)
yt-insight estimate "https://youtube.com/watch?v=VIDEO_ID"
# → affiche durée, taille audio, mots/tokens prédits, stratégie LLM (single-shot
#   vs chunk+merge), temps de transcription et d'analyse estimés, total

# Estimer avec du hardware custom (ex: tu passes sur RTX 3060)
yt-insight estimate "URL" --hardware gpu_rtx_3060

# Estimer pour un podcast (plus rapide) vs une conférence (plus lent)
yt-insight estimate "URL" --content-type podcast

# Sortie JSON de l'estimate (pour scripter / pipeline)
yt-insight estimate "URL" --json
```

### Exemples de sortie console

```
╭──────────────────────────────────────────────────────────────╮
│ YT-Insight v0.1.0                                           │
│ Pipeline complet — https://youtube.com/watch?v=abc123       │
╰──────────────────────────────────────────────────────────────╯
  📥 Titre     Comment fonctionne l'attention dans les Transformers
     Chaîne   DeepLearning.AI
     Durée    42:17
     Fichier  cache/abc123.mp3
     Cache    non
  🎙️ Langue          fr (p=0.99)
     Segments        847
     Tokens estimés  18,432
     Temps           2m 04s
  🤖 Modèle          Qwen3.6-35B-A3B-UD-IQ3_S.gguf
     Backend        llamacpp-local
     Points clés    12
     Citations      5
     Temps          1m 22s

╭─ YT-Insight — Qwen3.6-35B-A3B-UD-IQ3_S.gguf ────────────────╮
│ Backend : llamacpp-local                                    │
│ URL     : https://youtube.com/watch?v=abc123               │
╰─────────────────────────────────────────────────────────────╯

  Sujet : Intelligence Artificielle   Ton : pédagogique   Public : développeurs

╭─ 📝 Résumé détaillé ────────────────────────────────────────╮
│ Cette vidéo explore en profondeur le mécanisme d'attention…│
│ [500-1000 mots de prose continue]                          │
╰─────────────────────────────────────────────────────────────╯

╭─ 🎯 Points clés ────────────────────────────────────────────╮
│ 1. Le mécanisme d'attention permet au modèle de…           │
│ 2. Les têtes d'attention multiples capturent…               │
│ …                                                          │
╰─────────────────────────────────────────────────────────────╯

╭─ 🔍 Analyse approfondie ────────────────────────────────────╮
│ **Forces**                                                 │
│ - Clarté des explications                                  │
│                                                            │
│ **Concepts centraux**                                      │
│ - Attention, embeddings, pré-entraînement                  │
╰─────────────────────────────────────────────────────────────╯

             💬 Citations notables
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Timestamp ┃ Citation                             ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 12:34     │ L'attention est le mécanisme qui…   │
│ 27:15     │ Le scale a tout changé.              │
└───────────┴─────────────────────────────────────┘

  fr · 42:17 · 18,432 tokens · 12 points clés · 5 citations

💾 Fichiers écrits :
   markdown → outputs/2026-06-08_comment-fonctionne-lattention-dans-les-trans.md
   json     → outputs/2026-06-08_comment-fonctionne-lattention-dans-les-trans.json
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

**Tests — `tests/test_downloader.py` (16 tests) ✅**

```
TestExtractVideoId         — 5 tests (formats d'URL + fallback hash)
TestVideoMetadata          — 4 tests (duration_str, slug)
TestCacheLogic             — 3 tests (hit, force bypass, persistance meta)
TestErrorHandling          — 2 tests (vidéo privée, indisponible)
TestMetadataSerialization  — 2 tests (round-trip JSON, to_dict)
```

---

### `transcriber/base.py` + `transcriber/faster_whisper_transcriber.py` ✅

**Statut : terminé et testé.**

#### Classes exposées

**`Segment`** — un segment horodaté retourné par Whisper :
```python
@dataclass
class Segment:
    start: float   # secondes
    end:   float
    text:  str
    # propriétés calculées :
    # .duration   → float
    # .start_str  → "1:23:45"
    # .end_str    → "1:23:50"
```

**`TranscriptionResult`** — tout ce dont le pipeline a besoin après transcription :
```python
result.text                            # transcription brute complète
result.segments                        # list[Segment] avec timestamps
result.language                        # "fr", "en", etc.
result.language_probability            # 0.0 – 1.0
result.duration_seconds / .duration_str
result.estimated_tokens                # len(text) // 4
result.model_name                      # "large-v3"

result.formatted_transcript(with_timestamps=True)
# "[0:00] Bonjour tout le monde.\n[0:05] Aujourd'hui on parle de…"

result.to_dict()   # sérialisable JSON
```

**`FasterWhisperTranscriber`** — le backend principal :
```python
from yt_insight.transcriber import FasterWhisperTranscriber

t = FasterWhisperTranscriber(
    model_size="large-v3",   # ou "medium" si mémoire limitée
    device="auto",           # "auto" → CUDA si dispo, sinon CPU
    compute_type="int8",     # int8 = ~2.5 Go VRAM sur GTX 1650 Super
    beam_size=3,             # 3 = bon compromis qualité/mémoire (5 = trop pour 4 Go)
    language=None,           # None = auto-détection
    chunk_length=None,       # None = auto (30s). Mettre 20 sur GTX 1050 pour éviter OOM.
    vad_filter=True,         # skip les silences → plus rapide
)

result = t.transcribe(Path("cache/abc123.mp3"))
result = t.transcribe(Path("cache/abc123.mp3"), language="fr")  # override

t.unload()   # ⚠️ IMPORTANT — libère la VRAM avant de charger le LLM
```

**`create_transcriber()`** — factory qui lit les variables d'environnement :
```python
# Lit WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_LANGUAGE
t = create_transcriber()
```

#### Comportements clés

**Chargement paresseux (lazy init) :** le modèle Whisper n'est chargé en mémoire qu'au premier appel à `transcribe()`. Les appels suivants réutilisent le modèle déjà chargé.

**Fallback CUDA → CPU automatique :** si le chargement en CUDA échoue (OOM ou driver absent), le transcriber retente automatiquement sur CPU avec `compute_type="int8"`.

**Libération VRAM explicite :** `unload()` supprime le modèle, appelle `gc.collect()` et `torch.cuda.empty_cache()`. À appeler systématiquement avant de démarrer le LLM local pour éviter les conflits mémoire avec Qwen3 35B.

**Filtrage VAD :** les segments vides ou en silence sont ignorés, ce qui évite d'envoyer du bruit au LLM.

**Tests — `tests/test_transcriber.py` (26 tests) ✅**

```
TestSecondsToStr       — 8 tests  (formatage H:MM:SS)
TestSegment            — 3 tests  (duration, start_str, end_str)
TestTranscriptionResult— 6 tests  (tokens, duration, formatted_transcript, to_dict)
TestDeviceResolution   — 5 tests  (auto/cuda/cpu, torch absent)
TestLazyLoading        — 3 tests  (init, premier appel, réutilisation)
TestTranscribe         — 9 tests  (résultat, texte, segments, langue, durée, override, fichier manquant, segments vides)
TestUnload             — 3 tests  (clear, idempotent, flush CUDA)
TestCudaFallback       — 1 test   (OOM → retry CPU)
TestCreateTranscriber  — 3 tests  (defaults, env vars, langue vide)
TestOomFallback        — 3 tests  (CUDA OOM → CPU auto, OOM on CPU raises, no-OOM path)
TestChunkLength        — 6 tests  (default None, custom, passed to model, env var, beam=3 default)
```

> Tous les tests mockent `WhisperModel` — aucun téléchargement de modèle requis pour les faire tourner.

**Gestion robuste des OOM CUDA** : si l'inférence OOM en plein milieu de la génération d'un segment (pas juste au chargement), le transcriber détecte le `RuntimeError`, décharge le modèle, recharge sur CPU, et re-transcrit le fichier. La transparence est totale pour l'utilisateur. C'est particulièrement utile sur les GPU 4 Go comme la GTX 1050 avec large-v3.

---

### `analyzer/base.py`

Classes exposées :

**`Quote`** — une citation notable extraite du transcript :
```python
@dataclass
class Quote:
    text: str
    timestamp_seconds: float | None = None
    speaker: str | None = None
    # propriété calculée : .timestamp_str → "1:23:45" ou None
    # méthode : .to_dict() — sérialisable JSON
```

**`AnalysisResult`** — tout ce que produit l'analyzer :
```python
result.summary              # résumé détaillé, 500-1000 mots
result.key_points           # list[str] de 8 à 15 affirmations
result.analysis             # analyse en Markdown (Forces / Concepts / Implications)
result.quotes               # list[Quote] avec timestamps
result.topic                # "Intelligence Artificielle"
result.tone                 # "pédagogique"
result.audience             # "développeurs Python"
result.model_name           # "Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
result.backend              # "llamacpp-local"

result.has_content          # True si au moins un champ est rempli
result.to_dict()            # sérialisable JSON
```

**`BaseAnalyzer`** — ABC, contrat que tout backend doit respecter (`analyze()`, `model_name`, `backend_name`, `close()`).

---

### `analyzer/prompts.py`

Tous les prompts sont versionnés dans un module dédié, modifiables sans toucher au code :

```python
SYSTEM_PROMPT   # Règles absolues : JSON strict, pas d'invention, langue de la transcription
ANALYSIS_PROMPT # Prompt principal single-shot (résumé + key_points + analysis + quotes + topic/tone/audience)
CHUNK_PROMPT    # Prompt allégé pour un extrait (utilisé en mode chunk+merge)
MERGE_PROMPT    # Prompt de fusion des résumés partiels en un résultat final
```

Trois helpers de formatage : `build_analysis_prompt()`, `build_chunk_prompt()`, `build_merge_prompt()`.

---

### `analyzer/llamacpp_local.py` ✅

**Statut : terminé et testé.**

#### Pourquoi llama.cpp et pas Ollama ?

Tu lances `llama-server` directement avec ton quant Qwen3.6 35B-A3B sur :8080, donc on parle directement à son **API OpenAI-compatible** (`/v1/chat/completions`, `/v1/models`). Avantages :
- Pas de couche intermédiaire (Ollama) à configurer
- Contrôle total sur les flags llama.cpp (`-ngl`, `--n-cpu-moe`, KV cache quant, etc.)
- Le backend marche aussi avec Ollama, vLLM, LM Studio — il suffit de changer `LLAMACPP_BASE_URL`

#### Classes exposées

**`LlamaCppLocalAnalyzer`** — le backend principal :
```python
from yt_insight.analyzer import LlamaCppLocalAnalyzer

a = LlamaCppLocalAnalyzer(
    base_url="http://localhost:8080",
    model="Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
    max_prompt_tokens=28_000,   # sous n_ctx=32k
    temperature=0.2,
    max_tokens=4_000,
    disable_thinking=True,      # crucial pour Qwen3 : réponses concises
    timeout_s=600.0,
)
result = a.analyze(transcription, title="...", language="fr")
```

**`create_analyzer()`** — factory qui lit les variables d'environnement `LLAMACPP_*`.

#### Comportements clés

**Handshake au premier appel :** `_ensure_server_ready()` interroge `GET /v1/models`, vérifie que le serveur répond, et résout le nom du modèle (match exact → basename → fallback au premier modèle chargé).

**Désactivation du thinking Qwen3 :** on passe `chat_template_kwargs={"enable_thinking": False}` à chaque requête. Sans ça, le modèle brûle ~1000 tokens de "réflexion" avant la sortie JSON.

**Stratégie single-shot vs chunk+merge :** si la transcription nettoyée tient dans `max_prompt_tokens` (28k par défaut, confortable sous le 32k de ctx llama.cpp), on envoie tout d'un coup. Sinon :
1. `chunk_text()` découpe la transcription en chunks aux frontières de phrases, avec un overlap configurable
2. Pour chaque chunk, on demande un JSON partiel (résumé + key_points + quotes)
3. On envoie tous les JSON partiels au prompt de fusion
4. Le modèle produit le `AnalysisResult` final

**Extraction JSON tolérante :** `_extract_json_object()` essaie (1) le texte brut, (2) un bloc ```` ```json ... ``` ```, (3) le premier objet `{...}` trouvé. Lève `AnalysisError` si rien ne marche.

**Streaming optionnel :** `analyzer.stream_chat(prompt)` yield les tokens au fur et à mesure, pour affichage Rich live (utilisé par le futur module CLI).

**Tests — `tests/test_analyzer.py` (25 tests) ✅**

```
TestExtractJsonObject       — 5 tests  (JSON pur, fenced, chatter, garbage, empty)
TestServerHandshake         — 5 tests  (resolve exact, basename, fallback, conn error, no models)
TestSingleShotAnalyze       — 5 tests  (result, quotes-as-strings, fenced JSON, empty, HTTP 500)
TestChunkAndMerge           — 2 tests  (chunk+merge path, single-shot when small)
TestCreateAnalyzer          — 3 tests  (defaults, env overrides, kwarg beats env)
TestDataClasses             — 5 tests  (Quote.ts, AnalysisResult.to_dict, has_content)
```

---

### `utils/text_utils.py`

Fonctions utilitaires stateless :
- `estimate_tokens(text)` : estimation rapide du nombre de tokens (longueur/4)
- `chunk_text(text, max_tokens, overlap)` : découpage intelligent aux frontières de phrases
- `format_transcript_with_timestamps(segments)` : mise en forme `[HH:MM:SS] texte`
- `clean_transcript(text)` : nettoyage basique (espaces, artefacts Whisper)

**Tests — `tests/test_text_utils.py` (19 tests) ✅**

---

### `utils/config.py` + `utils/logger.py` ✅

**Statut : terminés et testés.**

#### Configuration typée

Cinq dataclasses immutables composent `AppConfig` :

| Dataclass             | Champs principaux                                              |
|-----------------------|----------------------------------------------------------------|
| `PathsConfig`         | `output_dir`, `cache_dir`, `keep_audio`                        |
| `PipelineConfig`      | `steps: list[str]`, `fail_fast: bool`                          |
| `TranscriptionConfig` | `model`, `device`, `compute_type`, `beam_size`, `language`, `vad_filter`, `chunk_length` |
| `AnalysisConfig`      | `max_transcript_tokens`, `overflow_strategy`, `outputs`        |
| `OutputConfig`        | `formats`, `include_transcript`, `include_timestamps`          |

`load_config(path=None, create_paths=True)` charge la config dans l'ordre de priorité :

1. Défauts sains (intégrés au code)
2. Fichier YAML (défaut `./config.yaml`, override `YT_INSIGHT_CONFIG`)
3. Variables d'environnement (`YT_INSIGHT_*`, `WHISPER_*`, `LLAMACPP_*`)

Exemple de `config.yaml` :

```yaml
paths:
  output_dir: ./outputs
  cache_dir: ./cache
  keep_audio: false

pipeline:
  steps: [download, transcribe, analyze]
  fail_fast: true

transcription:
  model: large-v3
  device: cuda
  compute_type: int8
  beam_size: 3            # lowered from 5 to fit 4 Go VRAM cards
  language: fr          # ou null pour auto-détection
  vad_filter: true

analysis:
  max_transcript_tokens: 100000
  overflow_strategy: chunk    # chunk | truncate | summarize_chunks
  outputs: [summary, key_points, analysis, quotes, metadata]

output:
  formats: [console, markdown, json]
  include_transcript: true
  include_timestamps: true
```

#### Logging structuré

`setup_logging(level)` est idempotent (peut être appelé plusieurs fois sans dupliquer les handlers), utilise `RichHandler` quand rich est dispo, et tame automatiquement les loggers bruyants (`httpx`, `httpcore`, `yt_dlp` → WARNING).

**Tests — `tests/test_config.py` (22 tests) ✅**

```
TestDefaults          — 5 tests
TestYamlLoading       — 5 tests
TestEnvOverrides      — 4 tests
TestFitsInWindow      — 3 tests
TestLogger            — 5 tests
```

---

### `output/console.py` ✅

**Statut : terminé et testé.**

`ConsoleRenderer` rend un `AnalysisResult` (+ éventuellement une `TranscriptionResult`) en sortie terminal avec Rich : Panel cyan pour le header, Panel vert pour chaque section (Résumé, Points clés, Analyse), Table magenta pour les Citations, footer gris avec les stats.

Toggle individuel de chaque section via `RenderConfig` :

```python
from yt_insight.output import ConsoleRenderer, RenderConfig

renderer = ConsoleRenderer(config=RenderConfig(
    show_transcript=False,      # désactive l'aperçu transcription
    show_quotes=True,
    max_transcript_chars=2000,
))
renderer.render(analysis, transcription, video_url="https://…")
```

`render_to_string(analysis, transcription)` retourne le rendu en string (utile pour les tests).

**Tests — `tests/test_output.py` (6 tests console) ✅**

---

### `output/file_writer.py` ✅

**Statut : terminé et testé.**

Deux formats, mêmes options :

```python
from yt_insight.output import FileWriter

writer = FileWriter(
    output_dir=Path("./outputs"),
    include_transcript=True,
    include_timestamps=True,
)

# Markdown
path = writer.write_markdown(analysis, title="…", video_url="…",
                             metadata=metadata, transcription=transcription)
# → outputs/2026-06-08_intitule-de-la-video.md

# JSON
path = writer.write_json(analysis, title="…", video_url="…",
                         metadata=metadata, transcription=transcription)
# → outputs/2026-06-08_intitule-de-la-video.json

# Les deux
paths = writer.write_both(analysis, …)  # {"markdown": Path, "json": Path}
```

Le **Markdown** contient un front matter YAML (titre, URL, date, model, backend, langue, durée, topic, tone, audience, channel, video_id), suivi des sections en français (📝 Résumé, 🎯 Points clés numérotés, 🔍 Analyse, 💬 Citations en blockquote, 📄 Transcription horodatée en bloc de code).

Le **JSON** est une sérialisation stricte des dataclasses, prête pour import Notebook / DB.

Le helper `write_outputs(analysis, output_dir, formats=[…])` fait le one-shot.

**Tests — `tests/test_output.py` (15 tests file_writer) ✅**

---

### `cli.py` ✅

**Statut : terminé et testé.**

Cinq sous-commandes Typer, plus la commande `version` :

| Sous-commande    | Arguments              | Ce qu'elle fait                                    |
|------------------|------------------------|----------------------------------------------------|
| `download`       | `URL`                  | Télécharge l'audio dans le cache                   |
| `transcribe`     | `SOURCE`               | URL ou chemin local → JSON transcript              |
| `analyze`        | `[TRANSCRIPT]` + `--audio` | JSON existant ou transcrit+analyse             |
| `all`            | `URL`                  | Pipeline complet end-to-end                        |
| `estimate`       | `URL`                  | Prédit le coût (temps, tokens, stratégie) sans lancer |
| `version`        | —                      | Affiche la version                                 |

Options communes : `--language`, `--output-dir`, `--cache-dir`, `--llamacpp-url`, `--llamacpp-model`, `--format`, `--steps`, `--no-console`, `--config`, `--verbose`.

L'option `--steps` accepte une liste (répétée) ou une string CSV :
```bash
yt-insight all URL --steps transcribe,analyze
yt-insight all URL --steps transcribe --steps analyze    # équivalent
```

`setup_logging()` est appelé une fois au début (avec `--verbose` on passe en DEBUG), puis `load_config()` construit l'`AppConfig` finale.

**Tests — `tests/test_cli.py` (19 tests) ✅** (avec `typer.testing.CliRunner`, modules I/O mockés)

```
TestDownload        — 3 tests
TestTranscribe      — 3 tests
TestAnalyze         — 3 tests
TestAll             — 3 tests
TestMisc            — 3 tests  (help, no-args, verbose)
TestEstimateCli     — 3 tests  (text, json, options pass-through)
```

---

### `estimate.py` ✅

**Statut : terminé et testé.**

`estimate_url(url, ...)` prédit le coût complet d'un run SANS rien télécharger ni transcrire. Il récupère juste les métadonnées via yt-dlp (`fetch_metadata_only`), puis applique des heuristiques :

- **Transcription** : `audio_duration × WPM (par type de contenu) × 5.5 chars/mot` pour prédire la taille du transcript
- **LLM** : si `n_tokens + 1500 < max_prompt_tokens` → single-shot, sinon chunk+merge avec calcul du nombre de chunks
- **Timings** : table de `RTFX` (real-time factor) par hardware (gtx_1050, gtx_1650, rtx_3060, cpu) et `tokens/sec` par quant (iq1_m, iq3_s, iq4_xs, q4_k_m, q5_k_m, q6_k, q8_0)

```bash
yt-insight estimate "URL" --hardware gpu_gtx_1050 --content-type lecture \
    --llm-quant iq4_xs --max-prompt-tokens 60000
```

Sortie typique :
```
📺 Stanford MS&E435 Economics of the AI Supercycle | Spring 2026
   Chaîne       : Stanford Online
   Durée vidéo  : 49:15 (2955s)
   URL          : https://www.youtube.com/watch?v=Qh7Oxvo5sJI

💾 Audio estimé : 69.3 Mo (mp3 192kbps)
   Télécharg.   : ~7s

📝 Transcript prévu (heuristique) :
   Mots         : ~6,402
   Caractères   : ~35,211
   Tokens       : ~8,802

🎙️ Transcription (faster-whisper large-v3) :
   Sur GPU      : ~8m20s
   Sur CPU      : ~49m55s

🤖 Analyse LLM (llamacpp-local) :
   Stratégie    : single-shot
   Passes LLM   : 1
   n_ctx requis : 10,302 tokens
   Temps estimé : ~6m40s

⏱️  TOTAL estimé :
   Avec GPU     : ~15m07s
   Avec CPU     : ~56m42s
```

`--json` produit la même chose en JSON structuré (pratique pour scripter).

**Tests — `tests/test_estimate.py` (24 tests) ✅** (métadonnées yt-dlp mockées)

```
TestSmoke              — 5 tests  (short single-shot, long chunk+merge, content-type impact)
TestHeuristics         — 5 tests  (audio size, hardware rtfx, quant tps, fallbacks)
TestNotes              — 4 tests  (warnings CPU, chunk+merge, audio > 200 Mo)
TestEstimateDataclass  — 3 tests  (duration_str, to_dict, fields)
TestFormatEstimate     — 1 test
TestConstants          — 6 tests  (ordres RTFX, TPS, WPM, sanity)
```

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

### Phase 1 — MVP ✅ **TERMINÉ**
- [x] Architecture & README
- [x] `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.env.example`
- [x] Module downloader — `yt_insight/downloader/ytdlp_downloader.py` (16 tests)
- [x] Module transcriber — `yt_insight/transcriber/{base,faster_whisper_transcriber}.py` (42 tests)
- [x] Module analyzer — `yt_insight/analyzer/{base,prompts,llamacpp_local}.py` (25 tests)
- [x] Module utils — `yt_insight/utils/{text_utils,config,logger}.py` (41 tests)
- [x] Module output — `yt_insight/output/{console,file_writer}.py` (21 tests)
- [x] Module estimate — `yt_insight/estimate.py` (24 tests, prédiction de coût)
- [x] CLI Typer — `yt_insight/cli.py` avec sous-commandes `all` / `download` / `transcribe` / `analyze` / `estimate` / `version` (19 tests)

### Phase 2 — Backends supplémentaires
- [ ] Backend Ollama (même API OpenAI-compat, juste changer `LLAMACPP_BASE_URL`)
- [ ] Backend vLLM (même chose, déjà compatible)
- [ ] Sélection automatique du backend selon longueur de la transcription
- [ ] Mode "transcript brut" (skip l'analyse, garder la transcription seule)

### Phase 3 — Features avancées
- [ ] Support playlists YouTube (batch processing)
- [ ] API REST FastAPI (pour intégration externe)
- [ ] Interface web Gradio (optionnel)
- [ ] Base de données SQLite pour historique des analyses
- [ ] Export PDF
- [ ] Recherche full-text dans les transcriptions

### Phase 4 — Qualité
- [x] Suite de tests complète (234/234 ✅)
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

# LLM analyzer (parle à llama-server sur :8080 via API OpenAI-compat)
httpx>=0.27.0

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

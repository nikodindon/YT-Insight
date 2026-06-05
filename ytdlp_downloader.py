"""
Downloader module — wraps yt-dlp to extract audio from YouTube URLs.

Responsibilities:
- Download audio-only stream (no video) → saves bandwidth & disk
- Convert to MP3 via ffmpeg post-processor
- Extract video metadata (title, duration, channel, description)
- Local cache: skip download if file already exists
- Clean error handling for private/geo-blocked/invalid videos
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yt_dlp
except ImportError as e:
    raise ImportError("yt-dlp is required. Run: pip install yt-dlp") from e


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VideoMetadata:
    """Raw metadata extracted from YouTube."""
    video_id: str
    title: str
    channel: str
    duration_seconds: float
    description: str
    upload_date: str          # "YYYYMMDD"
    view_count: Optional[int]
    url: str

    @property
    def duration_str(self) -> str:
        """Human-readable duration, e.g. '1:23:45'."""
        total = int(self.duration_seconds)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def slug(self) -> str:
        """URL-safe title slug for file naming."""
        s = self.title.lower()
        s = re.sub(r"[^\w\s-]", "", s)
        s = re.sub(r"[\s_-]+", "-", s)
        return s[:60].strip("-")


@dataclass
class DownloadResult:
    """Everything the rest of the pipeline needs after a download."""
    audio_path: Path
    metadata: VideoMetadata
    from_cache: bool = False
    download_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "audio_path": str(self.audio_path),
            "from_cache": self.from_cache,
            "download_time_seconds": self.download_time_seconds,
            "metadata": {
                "video_id": self.metadata.video_id,
                "title": self.metadata.title,
                "channel": self.metadata.channel,
                "duration_seconds": self.metadata.duration_seconds,
                "duration_str": self.metadata.duration_str,
                "upload_date": self.metadata.upload_date,
                "view_count": self.metadata.view_count,
                "url": self.metadata.url,
            },
        }


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class YtDlpDownloader:
    """
    Downloads audio from a YouTube URL using yt-dlp.

    Usage::

        downloader = YtDlpDownloader(cache_dir=Path("./cache"))
        result = downloader.download("https://youtube.com/watch?v=...")
        print(result.audio_path)       # Path to .mp3
        print(result.metadata.title)   # Video title
    """

    def __init__(
        self,
        cache_dir: Path = Path("./cache"),
        audio_format: str = "mp3",
        audio_quality: str = "192",   # kbps
        keep_audio: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.audio_format = audio_format
        self.audio_quality = audio_quality
        self.keep_audio = keep_audio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, url: str, force: bool = False) -> DownloadResult:
        """
        Download audio from *url* and return a :class:`DownloadResult`.

        Parameters
        ----------
        url:
            Full YouTube URL (watch, short, playlist entry…).
        force:
            If True, re-download even if the file is cached.

        Raises
        ------
        DownloadError
            Wraps any yt-dlp exception with a user-friendly message.
        """
        video_id = self._extract_video_id(url)
        audio_path = self.cache_dir / f"{video_id}.{self.audio_format}"
        meta_path  = self.cache_dir / f"{video_id}.meta.json"

        # --- Cache hit ---
        if not force and audio_path.exists() and meta_path.exists():
            metadata = self._load_metadata(meta_path)
            return DownloadResult(
                audio_path=audio_path,
                metadata=metadata,
                from_cache=True,
            )

        # --- Actual download ---
        t0 = time.time()
        metadata = self._download(url, video_id, audio_path)
        elapsed = time.time() - t0

        # Persist metadata alongside the audio file
        self._save_metadata(meta_path, metadata)

        return DownloadResult(
            audio_path=audio_path,
            metadata=metadata,
            from_cache=False,
            download_time_seconds=round(elapsed, 2),
        )

    def fetch_metadata_only(self, url: str) -> VideoMetadata:
        """
        Fetch video metadata without downloading the audio.
        Useful to preview title/duration before committing.
        """
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
            except yt_dlp.utils.DownloadError as exc:
                raise DownloadError(url, str(exc)) from exc

        return self._parse_metadata(info, url)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download(self, url: str, video_id: str, audio_path: Path) -> VideoMetadata:
        """Run yt-dlp and return parsed metadata."""

        ydl_opts = {
            # Audio only — best available quality
            "format": "bestaudio/best",
            # Convert to target format via ffmpeg
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": self.audio_format,
                    "preferredquality": self.audio_quality,
                }
            ],
            # Output template: fixed path so we know where the file lands
            "outtmpl": str(self.cache_dir / f"{video_id}.%(ext)s"),
            # Silence yt-dlp's own output (we handle progress via Rich)
            "quiet": True,
            "no_warnings": True,
            # Don't write yt-dlp's own info/thumbnail files
            "writeinfojson": False,
            "writethumbnail": False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
            except yt_dlp.utils.DownloadError as exc:
                raise DownloadError(url, str(exc)) from exc

        return self._parse_metadata(info, url)

    def _parse_metadata(self, info: dict, url: str) -> VideoMetadata:
        """Extract the fields we care about from yt-dlp's info dict."""
        return VideoMetadata(
            video_id=info.get("id", "unknown"),
            title=info.get("title", "Unknown Title"),
            channel=info.get("uploader") or info.get("channel") or "Unknown",
            duration_seconds=float(info.get("duration") or 0),
            description=(info.get("description") or "")[:2000],  # cap at 2k chars
            upload_date=info.get("upload_date") or "",
            view_count=info.get("view_count"),
            url=url,
        )

    @staticmethod
    def _extract_video_id(url: str) -> str:
        """
        Extract the YouTube video ID from various URL formats.

        Handles:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://youtube.com/shorts/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID

        Falls back to an MD5 hash of the URL if no ID can be parsed
        (e.g. for non-YouTube URLs or unusual formats).
        """
        patterns = [
            r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # Fallback: hash the URL so the cache key is still stable
        return hashlib.md5(url.encode()).hexdigest()[:11]

    @staticmethod
    def _save_metadata(path: Path, metadata: VideoMetadata) -> None:
        data = {
            "video_id": metadata.video_id,
            "title": metadata.title,
            "channel": metadata.channel,
            "duration_seconds": metadata.duration_seconds,
            "description": metadata.description,
            "upload_date": metadata.upload_date,
            "view_count": metadata.view_count,
            "url": metadata.url,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _load_metadata(path: Path) -> VideoMetadata:
        data = json.loads(path.read_text(encoding="utf-8"))
        return VideoMetadata(**data)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class DownloadError(Exception):
    """Raised when yt-dlp fails to download a video."""

    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(self._format())

    def _format(self) -> str:
        # Try to give a friendly hint based on common yt-dlp error strings
        reason_lower = self.reason.lower()
        if "private" in reason_lower:
            hint = "The video is private."
        elif "unavailable" in reason_lower:
            hint = "The video is unavailable (deleted or region-locked?)."
        elif "sign in" in reason_lower or "age" in reason_lower:
            hint = "The video requires sign-in or age verification."
        elif "not a valid url" in reason_lower:
            hint = "The URL does not appear to be a valid YouTube link."
        else:
            hint = self.reason.split("\n")[0]   # first line only
        return f"Download failed for {self.url!r}: {hint}"

"""Audio downloader backends for YouTube URLs."""

from .ytdlp_downloader import (
    DownloadError,
    DownloadResult,
    VideoMetadata,
    YtDlpDownloader,
)

__all__ = [
    "DownloadError",
    "DownloadResult",
    "VideoMetadata",
    "YtDlpDownloader",
]

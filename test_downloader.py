"""
Tests for yt_insight.downloader

Run with:  pytest tests/test_downloader.py -v
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yt_insight.downloader import YtDlpDownloader, DownloadResult
from yt_insight.downloader.ytdlp_downloader import (
    VideoMetadata,
    DownloadError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_downloader(tmp_path):
    """Return a downloader instance that writes to a temp directory."""
    return YtDlpDownloader(cache_dir=tmp_path)


FAKE_INFO = {
    "id": "dQw4w9WgXcQ",
    "title": "Rick Astley - Never Gonna Give You Up",
    "uploader": "Rick Astley",
    "duration": 213,
    "description": "The classic.",
    "upload_date": "20091025",
    "view_count": 1_400_000_000,
}


def make_fake_metadata() -> VideoMetadata:
    return VideoMetadata(
        video_id="dQw4w9WgXcQ",
        title="Rick Astley - Never Gonna Give You Up",
        channel="Rick Astley",
        duration_seconds=213.0,
        description="The classic.",
        upload_date="20091025",
        view_count=1_400_000_000,
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )


# ---------------------------------------------------------------------------
# _extract_video_id
# ---------------------------------------------------------------------------

class TestExtractVideoId:
    @pytest.mark.parametrize("url, expected", [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42", "dQw4w9WgXcQ"),
    ])
    def test_known_formats(self, url, expected):
        assert YtDlpDownloader._extract_video_id(url) == expected

    def test_fallback_hash_is_stable(self):
        url = "https://example.com/some/video"
        id1 = YtDlpDownloader._extract_video_id(url)
        id2 = YtDlpDownloader._extract_video_id(url)
        assert id1 == id2
        assert len(id1) == 11


# ---------------------------------------------------------------------------
# VideoMetadata helpers
# ---------------------------------------------------------------------------

class TestVideoMetadata:
    def test_duration_str_short(self):
        m = make_fake_metadata()
        m.duration_seconds = 75
        assert m.duration_str == "1:15"

    def test_duration_str_long(self):
        m = make_fake_metadata()
        m.duration_seconds = 3723   # 1h 2m 3s
        assert m.duration_str == "1:02:03"

    def test_slug_sanitizes_special_chars(self):
        m = make_fake_metadata()
        m.title = "Hello, World! (Test) — 2024"
        slug = m.slug
        assert "," not in slug
        assert "!" not in slug
        assert " " not in slug
        assert len(slug) <= 60

    def test_slug_is_lowercase(self):
        m = make_fake_metadata()
        assert m.slug == m.slug.lower()


# ---------------------------------------------------------------------------
# Cache logic
# ---------------------------------------------------------------------------

class TestCacheLogic:
    def test_cache_hit_skips_download(self, tmp_downloader, tmp_path):
        """If audio + meta files exist, download() must NOT call yt-dlp."""
        meta = make_fake_metadata()
        video_id = meta.video_id
        url = meta.url

        # Simulate cached files
        audio_path = tmp_path / f"{video_id}.mp3"
        meta_path  = tmp_path / f"{video_id}.meta.json"
        audio_path.write_bytes(b"fake audio data")
        YtDlpDownloader._save_metadata(meta_path, meta)

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            result = tmp_downloader.download(url)

        mock_ydl.assert_not_called()
        assert result.from_cache is True
        assert result.audio_path == audio_path

    def test_force_flag_bypasses_cache(self, tmp_downloader, tmp_path):
        """force=True must trigger a real download even if cache exists."""
        meta = make_fake_metadata()
        video_id = meta.video_id
        url = meta.url

        audio_path = tmp_path / f"{video_id}.mp3"
        meta_path  = tmp_path / f"{video_id}.meta.json"
        audio_path.write_bytes(b"fake audio data")
        YtDlpDownloader._save_metadata(meta_path, meta)

        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.extract_info = MagicMock(return_value=FAKE_INFO)

        with patch("yt_dlp.YoutubeDL", return_value=mock_instance):
            result = tmp_downloader.download(url, force=True)

        mock_instance.extract_info.assert_called_once()
        assert result.from_cache is False

    def test_metadata_persisted_after_download(self, tmp_downloader, tmp_path):
        """After a download the .meta.json file must exist and be valid."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.extract_info = MagicMock(return_value=FAKE_INFO)

        with patch("yt_dlp.YoutubeDL", return_value=mock_instance):
            tmp_downloader.download(url)

        meta_path = tmp_path / "dQw4w9WgXcQ.meta.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["title"] == FAKE_INFO["title"]
        assert data["video_id"] == FAKE_INFO["id"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_download_error_private_video(self, tmp_downloader):
        import yt_dlp

        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.extract_info = MagicMock(
            side_effect=yt_dlp.utils.DownloadError("Video is private")
        )

        with patch("yt_dlp.YoutubeDL", return_value=mock_instance):
            with pytest.raises(DownloadError) as exc_info:
                tmp_downloader.download("https://youtube.com/watch?v=private123")

        assert "private" in str(exc_info.value).lower()

    def test_download_error_unavailable(self, tmp_downloader):
        import yt_dlp

        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.extract_info = MagicMock(
            side_effect=yt_dlp.utils.DownloadError("Video unavailable")
        )

        with patch("yt_dlp.YoutubeDL", return_value=mock_instance):
            with pytest.raises(DownloadError):
                tmp_downloader.download("https://youtube.com/watch?v=gone123456")


# ---------------------------------------------------------------------------
# Metadata serialisation round-trip
# ---------------------------------------------------------------------------

class TestMetadataSerialization:
    def test_save_and_load_roundtrip(self, tmp_path):
        meta = make_fake_metadata()
        path = tmp_path / "test.meta.json"
        YtDlpDownloader._save_metadata(path, meta)
        loaded = YtDlpDownloader._load_metadata(path)

        assert loaded.video_id       == meta.video_id
        assert loaded.title          == meta.title
        assert loaded.channel        == meta.channel
        assert loaded.duration_seconds == meta.duration_seconds
        assert loaded.upload_date    == meta.upload_date
        assert loaded.view_count     == meta.view_count
        assert loaded.url            == meta.url

    def test_to_dict_has_expected_keys(self):
        meta = make_fake_metadata()
        result = DownloadResult(audio_path=Path("/tmp/x.mp3"), metadata=meta)
        d = result.to_dict()
        assert "audio_path"  in d
        assert "from_cache"  in d
        assert "metadata"    in d
        assert "duration_str" in d["metadata"]

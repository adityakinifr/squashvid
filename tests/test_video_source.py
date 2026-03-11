from __future__ import annotations

from pathlib import Path

import pytest

from squashvid.pipeline.video_source import (
    is_url,
    is_youtube_url,
    prepare_video_source,
)


def test_prepare_video_source_local_path(tmp_path: Path) -> None:
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"fake")

    prepared = prepare_video_source(str(video_file))

    assert prepared.downloaded is False
    assert prepared.local_path == str(video_file.resolve())


def test_prepare_video_source_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    with pytest.raises(FileNotFoundError):
        prepare_video_source(str(missing))


def test_prepare_video_source_non_youtube_url_rejected() -> None:
    with pytest.raises(ValueError):
        prepare_video_source("https://example.com/video.mp4")


def test_url_helpers() -> None:
    assert is_url("https://www.youtube.com/watch?v=abc123")
    assert is_youtube_url("https://www.youtube.com/watch?v=abc123")
    assert is_youtube_url("https://youtu.be/abc123")
    assert not is_youtube_url("https://example.com/abc123")

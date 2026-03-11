from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

YOUTUBE_HOST_KEYWORDS = ("youtube.com", "youtu.be", "m.youtube.com")


@dataclass(slots=True)
class PreparedVideo:
    source: str
    local_path: str
    downloaded: bool = False
    title: str | None = None
    video_id: str | None = None


def is_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return bool(parsed.scheme and parsed.netloc)


def is_youtube_url(value: str) -> bool:
    host = (urlparse(value.strip()).hostname or "").lower()
    return any(keyword in host for keyword in YOUTUBE_HOST_KEYWORDS)


def _download_youtube_video(url: str, cache_dir: str | None = None) -> PreparedVideo:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is required for YouTube URLs. Install dependencies with `pip install -e .`."
        ) from exc

    cache_root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir
        else (Path.cwd() / ".squashvid_cache").resolve()
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    outtmpl = str(cache_root / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        final_path: str | None = None

        requested = info.get("requested_downloads") or []
        for item in requested:
            candidate = item.get("filepath")
            if candidate and Path(candidate).exists():
                final_path = candidate
                break

        if final_path is None:
            prepared = ydl.prepare_filename(info)
            prepared_path = Path(prepared)
            if prepared_path.exists():
                final_path = str(prepared_path)
            else:
                mp4_candidate = prepared_path.with_suffix(".mp4")
                if mp4_candidate.exists():
                    final_path = str(mp4_candidate)

        if final_path is None:
            raise RuntimeError("YouTube download completed but output file was not found.")

        return PreparedVideo(
            source=url,
            local_path=str(Path(final_path).expanduser().resolve()),
            downloaded=True,
            title=str(info.get("title", "")) or None,
            video_id=str(info.get("id", "")) or None,
        )


def prepare_video_source(source: str, cache_dir: str | None = None) -> PreparedVideo:
    trimmed = source.strip()
    if is_url(trimmed):
        if not is_youtube_url(trimmed):
            raise ValueError(
                "Only YouTube URLs are currently supported for remote sources."
            )
        return _download_youtube_video(trimmed, cache_dir=cache_dir)

    path = Path(trimmed).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    return PreparedVideo(source=trimmed, local_path=str(path), downloaded=False)

from __future__ import annotations

import atexit
import base64
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# Module-level temp file for cookies (cleaned up on exit)
_cookies_temp_file: str | None = None


def _get_cookies_content_from_env() -> str | None:
    """Get cookie content from environment variables (base64 or plain text)."""
    # Check for base64 encoded cookies first (YOUTUBE_COOKIES_B64 or YTDLP_COOKIES_B64)
    b64_content = os.environ.get("YOUTUBE_COOKIES_B64") or os.environ.get("YTDLP_COOKIES_B64")
    if b64_content:
        try:
            return base64.b64decode(b64_content).decode("utf-8")
        except Exception:
            pass

    # Fall back to plain text cookies
    return os.environ.get("YTDLP_COOKIES")

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


def setup_youtube_oauth() -> bool:
    """Run interactive OAuth2 setup for YouTube. Returns True on success."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is required for YouTube URLs. Install dependencies with `pip install -e .`."
        ) from exc

    # Use a simple test URL to trigger OAuth flow
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    ydl_opts: dict = {
        "username": "oauth2",
        "password": "",
        "skip_download": True,
        "quiet": False,  # Show OAuth instructions
    }

    print("Starting YouTube OAuth2 setup...")
    print("You will be prompted to visit a URL and enter a code.")
    print()

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(test_url, download=False)
        print()
        print("OAuth2 setup complete! Tokens cached for future use.")
        return True
    except Exception as e:
        print(f"OAuth2 setup failed: {e}")
        return False


def _download_youtube_video(
    url: str,
    cache_dir: str | None = None,
    cookies_file: str | None = None,
    use_oauth2: bool = False,
) -> PreparedVideo:
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

    # Resolve auth method: OAuth2 > cookies file > cookies content env var
    effective_oauth2 = use_oauth2 or os.environ.get("YTDLP_USE_OAUTH2", "").lower() in ("1", "true", "yes")
    effective_cookies: str | None = None

    if not effective_oauth2:
        # Priority: explicit file param > YTDLP_COOKIES_FILE > YTDLP_COOKIES (content)
        if cookies_file:
            cookies_path = Path(cookies_file).expanduser().resolve()
            if not cookies_path.exists():
                raise FileNotFoundError(f"YouTube cookies file not found: {cookies_file}")
            effective_cookies = str(cookies_path)
        elif os.environ.get("YTDLP_COOKIES_FILE"):
            cookies_path = Path(os.environ["YTDLP_COOKIES_FILE"]).expanduser().resolve()
            if not cookies_path.exists():
                raise FileNotFoundError(f"YouTube cookies file not found: {os.environ['YTDLP_COOKIES_FILE']}")
            effective_cookies = str(cookies_path)
        else:
            # Check for cookie content in env vars (base64 or plain text)
            cookies_content = _get_cookies_content_from_env()
            if cookies_content:
                global _cookies_temp_file
                if _cookies_temp_file is None:
                    fd, _cookies_temp_file = tempfile.mkstemp(suffix=".txt", prefix="ytdlp_cookies_")
                    os.write(fd, cookies_content.encode("utf-8"))
                    os.close(fd)
                    atexit.register(lambda: os.unlink(_cookies_temp_file) if os.path.exists(_cookies_temp_file) else None)
                effective_cookies = _cookies_temp_file

    outtmpl = str(cache_root / "%(id)s.%(ext)s")
    ydl_opts: dict = {
        "format": "best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    # OAuth2 takes precedence over cookies
    if effective_oauth2:
        ydl_opts["username"] = "oauth2"
        ydl_opts["password"] = ""
    elif effective_cookies:
        ydl_opts["cookiefile"] = effective_cookies

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


def prepare_video_source(
    source: str,
    cache_dir: str | None = None,
    cookies_file: str | None = None,
    use_oauth2: bool = False,
) -> PreparedVideo:
    trimmed = source.strip()
    if is_url(trimmed):
        if not is_youtube_url(trimmed):
            raise ValueError(
                "Only YouTube URLs are currently supported for remote sources."
            )
        return _download_youtube_video(
            trimmed,
            cache_dir=cache_dir,
            cookies_file=cookies_file,
            use_oauth2=use_oauth2,
        )

    path = Path(trimmed).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    return PreparedVideo(source=trimmed, local_path=str(path), downloaded=False)

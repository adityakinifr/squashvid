"""Squash video analyzer package."""

import base64
import os
from pathlib import Path

__all__ = ["__version__"]
__version__ = "0.1.0"


def _setup_youtube_cookies() -> None:
    """Decode base64 YouTube cookies from env var if present."""
    cookies_b64 = os.environ.get("YOUTUBE_COOKIES_B64")
    if not cookies_b64:
        print("DEBUG: YOUTUBE_COOKIES_B64 not set")
        return

    # Already have a cookies file set? Skip.
    if os.environ.get("YTDLP_COOKIES_FILE"):
        print(f"DEBUG: YTDLP_COOKIES_FILE already set: {os.environ.get('YTDLP_COOKIES_FILE')}")
        return

    try:
        cookies_data = base64.b64decode(cookies_b64)
        cookies_path = Path("/tmp/youtube_cookies.txt")
        cookies_path.write_bytes(cookies_data)
        os.environ["YTDLP_COOKIES_FILE"] = str(cookies_path)
        print(f"DEBUG: Wrote {len(cookies_data)} bytes to {cookies_path}, set YTDLP_COOKIES_FILE")
    except Exception as e:
        print(f"Warning: Failed to decode YOUTUBE_COOKIES_B64: {e}")


_setup_youtube_cookies()

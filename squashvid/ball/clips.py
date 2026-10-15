"""Where the pilot's clips, work products and third-party weights live.

Each match exists twice: the iPhone original (1920x1080 HEVC 10-bit HLG, 8.5 Mb/s,
variable frame rate) and the WhatsApp copy (1024x576, 0.66 Mb/s, resampled to a
constant 29.97 fps). Labels and evaluation use the originals; the copies are the
baseline that shows what compression costs. The two share audio and t = 0, but
copy frame k is not original frame k, so frames are matched by time (see align.py).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
LABELS = WORK / "labels"
THIRD_PARTY = ROOT / "third_party"
WEIGHTS = THIRD_PARTY / "weights"
VENUE = WORK / "venue_court5.json"  # intrinsics for the 1024x576 copies
VENUE_1080 = WORK / "venue_court5_1080p.json"  # intrinsics for the 1920x1080 originals

# Original stem -> WhatsApp copy stem: one junior match, court 5, the same phone behind the back glass.
COPIES = {
    "IMG_4091": "wa_3A16D3EC0BAFDCE1C46D",  # clip 1, 291 s
    "IMG_4089": "wa_3A644526AAA5409BE3EF",  # clip 2, 602 s (the copy stops at 533 s)
    "IMG_4090": "wa_3AE908EFCE489E9F342F",  # clip 3, 321 s
}
ORIGINAL_OF = {copy: orig for orig, copy in COPIES.items()}
CLIPS = {
    **{orig: f"~/Downloads/deepti-originals/{orig}.MOV" for orig in COPIES},
    **{copy: f"~/Downloads/{copy}.mp4" for copy in COPIES.values()},
}
COPY_SCALE = 1920 / 1024  # original pixels per copy pixel (both 16:9)
REGISTRY = "clips.json"  # work/clips.json: stem -> absolute video path, written by `python -m squashvid.process`


def registered(work: Path | None = None) -> dict[str, str]:
    """Clips added without editing code; empty when work/clips.json does not exist."""
    path = (work or WORK) / REGISTRY
    return json.loads(path.read_text()) if path.exists() else {}


def register(stem: str, path: str | Path, work: Path | None = None) -> Path:
    """Record stem -> absolute video path in work/clips.json (a later call for the same stem replaces it)."""
    work = work or WORK
    work.mkdir(parents=True, exist_ok=True)
    reg = {**registered(work), stem: str(Path(path).expanduser().resolve())}
    out = work / REGISTRY
    out.write_text(json.dumps(reg, indent=1, sort_keys=True))
    return out


def clip_path(stem: str) -> Path:
    if stem in CLIPS:
        return Path(os.path.expanduser(CLIPS[stem]))
    reg = registered()
    if stem not in reg:
        raise KeyError(f"unknown clip {stem!r}: not one of {sorted(CLIPS)} and not in {WORK / REGISTRY} (register it with `python -I -m squashvid.process <video> --stem {stem}`)")
    return Path(reg[stem])


def venue_for_size(width: int, height: int, work: Path | None = None) -> Path:
    """The court-5 venue file whose intrinsics belong to this frame size."""
    from squashvid.calib.venue import Venue

    for path in sorted((work or WORK).glob("venue_court5*.json")):
        if tuple(Venue.load(path).size) == (width, height):
            return path
    raise FileNotFoundError(f"no work/venue_court5*.json for {width}x{height}: this command handles court 5 only; a new court needs a venue calibration")


@lru_cache(maxsize=None)
def frame_size(stem: str) -> tuple[int, int]:
    from squashvid.media.video import probe

    info = probe(clip_path(stem))
    return info.width, info.height


def is_original(stem: str) -> bool:
    """A 1920x1080 clip (an iPhone original, or a registered one of that size), as opposed to a 1024x576 copy."""
    return stem in COPIES or (stem not in CLIPS and frame_size(stem)[0] == 1920)


def venue_path(stem: str) -> Path:
    if stem in CLIPS:
        return VENUE_1080 if is_original(stem) else VENUE
    return venue_for_size(*frame_size(stem))


def work_dir(stem: str) -> Path:
    return WORK / stem


def ball_dir(stem: str) -> Path:
    """Per-clip prediction outputs: one parquet per model variant."""
    return WORK / stem / "ball"


@lru_cache(maxsize=None)
def video(stem: str, sdr: bool = True):
    """The clip's Video; HLG originals come out as SDR BT.709 (media.video.hlg_to_sdr) unless sdr=False."""
    from squashvid.media.video import Video

    return Video(clip_path(stem), sdr=sdr)


def frame_range(timestamps: np.ndarray, start: float, end: float) -> tuple[int, int]:
    """Inclusive frame-index range whose PTS lie in [start, end]."""
    i0 = int(np.searchsorted(timestamps, start - 1e-6))
    i1 = int(np.searchsorted(timestamps, end + 1e-6)) - 1
    return i0, max(i0, i1)


def copy_to_original_px(xy: np.ndarray) -> np.ndarray:
    """Map copy pixel coordinates (pixel centres at integers) into the original's pixel grid."""
    return (np.asarray(xy, np.float64) + 0.5) * COPY_SCALE - 0.5

"""Run the player stream on one clip.

    python -m squashvid.players ~/Downloads/<stem>.mp4               # extract (if missing) + offline stages
    python -m squashvid.players ~/Downloads/<stem>.mp4 --extract     # force the heavy pass
    python -m squashvid.players ~/Downloads/<stem>.mp4 --start 20 --end 40 --work work/scratch
    python -m squashvid.players ~/Downloads/deepti-originals/IMG_4091.MOV   # 1080p original

Cameras come from work/<stem>/camera.parquet. The venue file defaults to the
work/venue_court5*.json whose image size matches the video.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.path.expanduser("~/code/squashvid"))


def matching_venue(width: int, height: int):
    """The court-5 venue whose intrinsics belong to this frame size."""
    from squashvid.calib.venue import Venue

    for path in sorted((ROOT / "work").glob("venue_court5*.json")):
        venue = Venue.load(path)
        if tuple(venue.size) == (width, height):
            return venue
    raise FileNotFoundError(f"no work/venue_court5*.json for {width}x{height}; pass --venue")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m squashvid.players")
    p.add_argument("video")
    p.add_argument("--venue", default=None, help="venue json (default: the court-5 venue matching the video size)")
    p.add_argument("--imgsz", type=int, default=1024, help="detector input width")
    p.add_argument("--work", default=None, help="output directory (default work/<stem>)")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--extract", action="store_true", help="rerun the heavy pass even if detections exist")
    p.add_argument("--no-render", action="store_true")
    a = p.parse_args(argv)

    from squashvid.calib.venue import Venue
    from squashvid.media.video import Video
    from squashvid.players.extract import extract
    from squashvid.players.geometry import CameraTrack

    video_path = Path(os.path.expanduser(a.video))
    stem = video_path.stem
    work = Path(a.work) if a.work else ROOT / "work" / stem
    video = Video(video_path)
    venue = Venue.load(a.venue) if a.venue else matching_venue(video.info.width, video.info.height)
    cams = CameraTrack.load(venue, ROOT / "work" / stem / "camera.parquet")
    if a.extract or not (work / "detections.parquet").exists():
        t0 = time.perf_counter()

        def progress(n: int, t: float) -> None:
            if n % 800 == 0:
                print(f"  {stem} t={t:7.1f}s frames={n} {n / (time.perf_counter() - t0):.1f} fps", flush=True)

        extract(video, venue, cams, work, a.start, a.end, progress, a.imgsz)
    from squashvid.players.run import offline

    offline(video, venue, cams, work, render=not a.no_render)
    return 0


if __name__ == "__main__":
    sys.exit(main())

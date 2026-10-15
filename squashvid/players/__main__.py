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
    from squashvid.ball.clips import venue_for_size
    from squashvid.calib.venue import Venue

    return Venue.load(venue_for_size(width, height, ROOT / "work"))


def run(video_path: Path, stem: str | None = None, venue_json: str | None = None, imgsz: int = 1024, work: Path | None = None,
        start: float = 0.0, end: float | None = None, force: bool = False, render: bool = True) -> Path:
    """Extract (when detections are missing or force) and run the offline stages; returns the output directory.

    The camera table is work/<stem>/camera.parquet, so register the clip first."""
    from squashvid.calib.venue import Venue
    from squashvid.media.video import Video
    from squashvid.players.extract import extract
    from squashvid.players.geometry import CameraTrack

    stem = stem or video_path.stem
    work = work or ROOT / "work" / stem
    video = Video(video_path)
    venue = Venue.load(venue_json) if venue_json else matching_venue(video.info.width, video.info.height)
    cams = CameraTrack.load(venue, ROOT / "work" / stem / "camera.parquet")
    if force or not (work / "detections.parquet").exists():
        t0 = time.perf_counter()

        def progress(n: int, t: float) -> None:
            if n % 800 == 0:
                print(f"  {stem} t={t:7.1f}s frames={n} {n / (time.perf_counter() - t0):.1f} fps", flush=True)

        extract(video, venue, cams, work, start, end, progress, imgsz)
    from squashvid.players.run import offline

    offline(video, venue, cams, work, render=render)
    return work


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

    run(Path(os.path.expanduser(a.video)), venue_json=a.venue, imgsz=a.imgsz, work=Path(a.work) if a.work else None,
        start=a.start, end=a.end, force=a.extract, render=not a.no_render)
    return 0


if __name__ == "__main__":
    sys.exit(main())

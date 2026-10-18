"""Calibrate a new court (or a new phone mount) from one frame's landmark clicks.

    python -I -m squashvid.calib.new_venue <video> <clicks.json> --name NAME --out work/venue_NAME_1080p.json

clicks.json holds {"t": seconds, "points": {landmark: [u, v]}} with names from geom.court.landmarks
(at least six, some on the floor so the focal length is observable). The clicks seed a pose and focal
length; the phone is then followed through the clip (it may pan and tilt) and the venue fitted jointly
on frames every 10 s. Overlays of every fitted frame go next to the output for a look.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from squashvid.calib.register import draw_court
from squashvid.calib.venue import collect_frames_chained, fit_venue
from squashvid.geom import court
from squashvid.geom.camera import solve_pnp_focal
from squashvid.media.video import Video


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -I -m squashvid.calib.new_venue", description=__doc__.split("\n\n")[0])
    p.add_argument("video")
    p.add_argument("clicks")
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=float, default=10.0, help="seconds between venue frames")
    a = p.parse_args(argv)

    video = Video(Path(a.video).expanduser())
    clicks = json.loads(Path(a.clicks).read_text())
    lm = court.landmarks(court.WSF)
    names = list(clicks["points"])
    unknown = sorted(set(names) - set(lm))
    if unknown or len(names) < 6:
        p.error(f"need six or more landmarks from {sorted(lm)}; unknown: {unknown}")
    w, h = video.info.width, video.info.height
    world = np.array([lm[n] for n in names])
    image = np.array([clicks["points"][n] for n in names], np.float64)
    seed, rms = solve_pnp_focal(world, image, (w - 1) / 2, (h - 1) / 2, (0.25 * w, 1.2 * w), 4.0)
    print(f"seed: f {seed.f:.0f} px, click rms {rms:.1f} px, camera at {np.round(seed.centre, 2)} m", flush=True)

    times = [0.2, *np.arange(a.every, video.info.duration - 1, a.every)]
    frames = collect_frames_chained(video, Path(a.video).stem, seed, float(clicks["t"]), times, court.WSF)
    print(f"{len(frames)} of {len(times)} venue frames matched", flush=True)
    venue, frames = fit_venue(a.name, frames)
    out = venue.save(a.out)
    print(f"{out}: width {venue.spec.width:.3f} length {venue.spec.length:.3f} f {venue.f:.1f} rms {venue.rms:.2f} px")
    review = out.with_suffix("")
    review.mkdir(exist_ok=True)
    for fr in frames:
        _, _, img = video.frame_at(fr.t)
        cv2.imwrite(str(review / f"t{fr.t:06.1f}.jpg"), cv2.resize(draw_court(img, fr.camera, venue.spec), (w // 2, h // 2)))
    print(f"overlays in {review}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

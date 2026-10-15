"""Which court surface a ball pixel lies against, so detection quality can be reported per region.

A detector that finds the ball against the plain front wall can still miss it
against the door glass or the floor lines, and those failures matter for
different events, so recall is broken down by the surface behind the ball.

Definition (per pixel, using that frame's tracked camera pose):
  1. door_strip: the pixel's ray crosses the back-wall plane (y = 0, between
     the camera and the court) inside one of DOOR_BANDS, the vertical glass
     edges, hinge plates and handle that the camera films through. These
     cover the ball wherever it is in depth, so they win over everything else.
  2. Otherwise the first court surface the ray meets inside the court box:
     front wall below the out line -> front_wall; side walls below their
     out line -> side_walls; floor behind the short line -> floor_back, in
     front of it -> floor_front.
  3. Rays that meet a wall above its out line, or leave the court box without
     meeting a surface, are air (the ball can only be in flight there).
The ball is not on that surface, only seen against it: a lob at the top of
the frame is "front_wall" if the front wall is behind it.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from squashvid.ball.clips import COPIES, is_original, venue_path, video, work_dir
from squashvid.geom import court
from squashvid.geom.camera import Camera

REGIONS = ("front_wall", "side_walls", "floor_back", "floor_front", "air", "door_strip")

# World-x bands on the back-wall plane covered by the glass edges and hardware of court 5's
# back wall: the fixed panel's two edges and hinges, the door's edge and handle block, and
# the door's hinge plates with the next glass edge. Measured on dark/edge pixel columns of
# frame 629 of clip 1 (handle 415-515 px, hinges 650-728 px, edge 742-750 px) with a few px
# of margin, and checked by projection through each clip's own poses.
DOOR_BANDS = ((0.80, 0.95), (1.12, 1.38), (2.40, 2.86), (3.38, 3.86))
DOOR_TOP_Z = 3.0  # above the glass the view is unobstructed; the frame never reaches it


def classify_rays(centre: np.ndarray, rays: np.ndarray, spec: court.CourtSpec, door=DOOR_BANDS) -> np.ndarray:
    """Region name for each ray from `centre` (camera behind the back wall, y < 0)."""
    c = np.asarray(centre, np.float64).reshape(3)
    d = np.asarray(rays, np.float64).reshape(-1, 3)
    W, L = spec.width, spec.length
    out = np.full(len(d), "air", dtype=object)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Back-wall crossing for the door strip.
        tb = (0.0 - c[1]) / d[:, 1]
        pb = c + d * tb[:, None]
        on_back = (tb > 0) & (pb[:, 2] >= 0) & (pb[:, 2] <= DOOR_TOP_Z)
        in_door = np.zeros(len(d), bool)
        for x0, x1 in door:
            in_door |= on_back & (pb[:, 0] >= x0) & (pb[:, 0] <= x1)

        # Distance along the ray to each surface; keep hits that land inside that surface's rectangle.
        cands = []
        t = (L - c[1]) / d[:, 1]
        p = c + d * t[:, None]
        ok = (t > 0) & (p[:, 0] >= 0) & (p[:, 0] <= W) & (p[:, 2] >= 0)
        cands.append((np.where(ok, t, np.inf), np.where(p[:, 2] <= spec.front_out_z, "front_wall", "air")))
        for x_wall in (0.0, W):
            t = (x_wall - c[0]) / d[:, 0]
            p = c + d * t[:, None]
            ok = (t > 0) & (p[:, 1] >= 0) & (p[:, 1] <= L) & (p[:, 2] >= 0)
            cands.append((np.where(ok, t, np.inf), np.where(p[:, 2] <= spec.side_out_z(p[:, 1]), "side_walls", "air")))
        t = -c[2] / d[:, 2]
        p = c + d * t[:, None]
        ok = (t > 0) & (p[:, 0] >= 0) & (p[:, 0] <= W) & (p[:, 1] >= 0) & (p[:, 1] <= L)
        cands.append((np.where(ok, t, np.inf), np.where(p[:, 1] < spec.short_y, "floor_back", "floor_front")))

    ts = np.stack([np.nan_to_num(t_, nan=np.inf) for t_, _ in cands], axis=1)
    names = np.stack([n for _, n in cands], axis=1)
    first = np.argmin(ts, axis=1)
    hit = np.isfinite(ts[np.arange(len(d)), first])
    out[hit] = names[np.arange(len(d)), first][hit]
    out[in_door] = "door_strip"
    return out


def classify(cam: Camera, uv: np.ndarray, spec: court.CourtSpec, door=DOOR_BANDS) -> np.ndarray:
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    if not len(uv):
        return np.array([], dtype=object)
    return classify_rays(cam.centre, cam.rays(uv), spec, door)


@lru_cache(maxsize=None)
def venue(stem: str):
    from squashvid.calib.venue import Venue

    return Venue.load(venue_path(stem))


@lru_cache(maxsize=None)
def poses(stem: str) -> pd.DataFrame:
    """Smoothed per-frame camera pose from calibration (work/<stem>/camera.parquet), indexed by frame."""
    df = pd.read_parquet(work_dir(stem) / "camera.parquet")
    return df.set_index("frame")[["t", "sr0", "sr1", "sr2", "st0", "st1", "st2"]]


def camera_at(stem: str, frame: int) -> Camera:
    """The tracked camera for one frame.

    An original without its own registration yet borrows the copy's pose at the same
    time with the copy's intrinsics rescaled to 1080p (the pose does not depend on the
    resolution); the 1080p venue is used as soon as work/<stem>/camera.parquet exists.
    """
    if is_original(stem) and not (work_dir(stem) / "camera.parquet").exists():
        t = float(video(stem).timestamps[frame])
        copy = COPIES[stem]
        p = poses(copy)
        row = p.iloc[int(np.clip(np.searchsorted(p.t.to_numpy(), t), 0, len(p) - 1))].to_numpy()[1:]
        v = venue(copy)
        return v.resized(1920, 1080).camera(row[:3], row[3:])
    p = poses(stem)
    row = p.loc[int(np.clip(frame, p.index.min(), p.index.max()))].to_numpy()[1:]
    return venue(stem).camera(row[:3], row[3:])


def label_regions(stem: str, frames: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Region per labelled point; NaN positions (ball out of frame) get None."""
    frames = np.asarray(frames).astype(int)
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    out = np.full(len(frames), None, dtype=object)
    spec = venue(stem).spec
    for f in np.unique(frames):
        sel = (frames == f) & np.isfinite(xy).all(axis=1)
        if sel.any():
            out[sel] = classify(camera_at(stem, f), xy[sel], spec)
    return out


def region_map(cam: Camera, spec: court.CourtSpec, size: tuple[int, int], step: int = 4) -> np.ndarray:
    """Region index (into REGIONS) on a coarse pixel grid, for drawing overlays."""
    w, h = size
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    names = classify(cam, np.column_stack([xs.ravel(), ys.ravel()]), spec)
    lut = {n: i for i, n in enumerate(REGIONS)}
    return np.array([lut[n] for n in names]).reshape(xs.shape)

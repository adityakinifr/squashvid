"""Per-frame cameras and the pixel-to-court maths the player stream needs.

Poses come from `work/<stem>/camera.parquet`, written by calib.register. The
smoothed pose is preferred; a frame missing from the table borrows the nearest
registered frame, because the phone barely moves between neighbouring frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from squashvid.calib.venue import Venue
from squashvid.geom.camera import Camera
from squashvid.geom.court import CourtSpec

ANKLE_Z = 0.08  # ankle keypoint height above the floor
REF_WIDTH = 1024  # pixel thresholds in this package were tuned on the 1024-px WhatsApp copies


def px_scale(width: int) -> float:
    """Factor that carries a pixel threshold tuned at REF_WIDTH to a frame of this width (1.875 at 1080p)."""
    return width / REF_WIDTH


class CameraTrack:
    """Camera for any frame index of one clip."""

    def __init__(self, venue: Venue, table: pd.DataFrame):
        t = table.sort_values("frame")
        self.venue = venue
        self.frames = t["frame"].to_numpy(np.int64)
        r = t[["sr0", "sr1", "sr2"]].to_numpy(np.float64)
        v = t[["st0", "st1", "st2"]].to_numpy(np.float64)
        raw_r = t[["r0", "r1", "r2"]].to_numpy(np.float64)
        raw_v = t[["t0", "t1", "t2"]].to_numpy(np.float64)
        bad = ~np.isfinite(r).all(1) | ~np.isfinite(v).all(1)
        r[bad], v[bad] = raw_r[bad], raw_v[bad]
        self.rvecs, self.tvecs = r, v
        self._cache: dict[int, Camera] = {}

    @classmethod
    def load(cls, venue: Venue, path: str | Path) -> "CameraTrack":
        return cls(venue, pd.read_parquet(path))

    def row(self, frame: int) -> int:
        i = int(np.searchsorted(self.frames, frame))
        if i >= len(self.frames) or (i > 0 and frame - self.frames[i - 1] < self.frames[i] - frame):
            i -= 1
        return max(i, 0)

    def __call__(self, frame: int) -> Camera:
        cam = self._cache.get(frame)
        if cam is None:
            i = self.row(frame)
            cam = self.venue.camera(self.rvecs[i], self.tvecs[i])
            if len(self._cache) > 4096:
                self._cache.clear()
            self._cache[frame] = cam
        return cam


def floor_xy(cam: Camera, uv: np.ndarray, spec: CourtSpec, z: float = 0.0) -> np.ndarray:
    """Court (x, y) where each pixel's ray meets the horizontal plane at height z."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    if not len(uv):
        return np.zeros((0, 2))
    return cam.to_plane(uv, "floor", offset=z, spec=spec)[:, :2]


def inside_court(xy: np.ndarray, spec: CourtSpec, margin: float = 0.3) -> np.ndarray:
    """Points on the floor polygon, widened by a margin for foot-point noise and players touching a wall."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    ok = np.isfinite(xy).all(1)
    x, y = xy[:, 0], xy[:, 1]
    with np.errstate(invalid="ignore"):
        return ok & (x > -margin) & (x < spec.width + margin) & (y > -margin) & (y < spec.length + margin)


def stature_px(cam: Camera, xy: np.ndarray, stature: float) -> np.ndarray:
    """Image height of an upright person of the given stature standing at each court point."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    foot = np.column_stack([xy, np.zeros(len(xy))])
    head = np.column_stack([xy, np.full(len(xy), stature)])
    return np.linalg.norm(cam.project(head) - cam.project(foot), axis=1)


def ground_jacobian(cam: Camera, uv: np.ndarray, spec: CourtSpec, z: float = 0.0, eps: float = 0.5) -> np.ndarray:
    """d(x, y)/d(u, v) at each pixel for the plane at height z: (N, 2, 2), metres per pixel.

    Turns pixel noise into court-space covariance, which is what makes the
    smoother trust along-court positions less toward the front wall.
    """
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    du = floor_xy(cam, uv + [eps, 0], spec, z) - floor_xy(cam, uv - [eps, 0], spec, z)
    dv = floor_xy(cam, uv + [0, eps], spec, z) - floor_xy(cam, uv - [0, eps], spec, z)
    return np.stack([du, dv], axis=2) / (2 * eps)


def height_jacobian(cam: Camera, uv: np.ndarray, spec: CourtSpec, z: float, eps: float = 0.02) -> np.ndarray:
    """d(x, y)/dz: how far the court point slides when the assumed plane height is wrong (N, 2)."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    return (floor_xy(cam, uv, spec, z + eps) - floor_xy(cam, uv, spec, z - eps)) / (2 * eps)


def point_height(cam: Camera, uv: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Height of the point on each pixel ray closest to the vertical line through court point xy.

    Used to measure a player's hip height from frames where the feet are seen.
    """
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    c = cam.centre
    d = cam.rays(uv)
    # Minimise horizontal distance between c + s d and (x, y): least squares in s.
    dh = d[:, :2]
    s = np.einsum("ij,ij->i", xy - c[:2], dh) / np.maximum(np.einsum("ij,ij->i", dh, dh), 1e-12)
    return c[2] + s * d[:, 2]

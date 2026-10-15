"""Choose each player's floor-contact point and its court-space uncertainty.

Ladder, in order: the ankle midpoint on the 8 cm ankle plane; the box
bottom-centre on the floor unless the box is truncated by the frame edge or
the door hardware (a truncated box puts the player too far forward); the hip
midpoint on the player's hip plane; otherwise nothing, and the smoother
interpolates. Every point carries a 2x2 court covariance built from pixel noise
through the local floor Jacobian, so along-court noise grows toward the front
wall exactly as the camera geometry says it should.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from squashvid.geom.camera import Camera
from squashvid.geom.court import CourtSpec
from squashvid.players.geometry import ANKLE_Z, floor_xy, ground_jacobian, height_jacobian
from squashvid.players.pose import L_ANK, L_HIP, R_ANK, R_HIP

ANKLE_MIN = 0.45
HIP_MIN = 0.35
HIP_Z_DEFAULT = 0.53 * 1.45  # 0.53 x a junior's stature
SOURCES = ("ankle", "box", "hip", "none")

# Pixel noise per source (u, v), plus the error of the assumed plane height (m).
SIGMA = {"ankle": (3.0, 3.0, 0.03), "box": (None, 4.0, 0.0), "hip": (4.0, 4.0, 0.07)}


@dataclass
class FootInputs:
    """What the ladder needs for one detection, already reduced to pixels and flags."""

    box: np.ndarray  # x1, y1, x2, y2
    kpts: np.ndarray | None  # (17, 2)
    scores: np.ndarray | None  # (17,)
    ankle_on_hardware: bool = False
    bottom_on_hardware: float = 0.0  # share of the strip just below the box covered by hardware
    frame_hw: tuple[int, int] = (576, 1024)  # (height, width)


def ankle_uv(f: FootInputs) -> np.ndarray | None:
    """Midpoint of two confident ankles that sit in the lower body and inside the box."""
    if f.kpts is None or f.scores is None or min(f.scores[L_ANK], f.scores[R_ANK]) < ANKLE_MIN:
        return None
    mid = (f.kpts[L_ANK] + f.kpts[R_ANK]) / 2
    x1, y1, x2, y2 = f.box
    h, w = y2 - y1, x2 - x1
    if not (x1 - 0.1 * w <= mid[0] <= x2 + 0.1 * w and mid[1] >= y1 + 0.65 * h and mid[1] <= y2 + 0.05 * h):
        return None
    return mid.astype(np.float64)


def box_truncated(f: FootInputs) -> bool:
    """The box bottom is not the feet: it hits the frame edge or stops on the door hardware."""
    H, W = f.frame_hw
    x1, _, x2, y2 = f.box
    return bool(y2 >= H - 3 or x1 <= 1 or x2 >= W - 2 or f.bottom_on_hardware > 0.2)


def hip_uv(f: FootInputs) -> np.ndarray | None:
    if f.kpts is None or f.scores is None:
        return None
    sl, sr = f.scores[L_HIP], f.scores[R_HIP]
    if min(sl, sr) >= HIP_MIN:
        return ((f.kpts[L_HIP] + f.kpts[R_HIP]) / 2).astype(np.float64)
    if max(sl, sr) >= 0.5:
        return f.kpts[L_HIP if sl > sr else R_HIP].astype(np.float64)
    return None


def choose(f: FootInputs, hip_z: float = HIP_Z_DEFAULT) -> tuple[str, np.ndarray | None, float, bool]:
    """(source, pixel, plane height, box truncated) for one detection, following the ladder."""
    truncated = box_truncated(f)
    if not f.ankle_on_hardware:
        a = ankle_uv(f)
        if a is not None:
            return "ankle", a, ANKLE_Z, truncated
    if not truncated:
        x1, _, x2, y2 = f.box
        return "box", np.array([(x1 + x2) / 2, y2], np.float64), 0.0, truncated
    hp = hip_uv(f)
    if hp is not None:
        return "hip", hp, hip_z, truncated
    return "none", None, float("nan"), truncated


def court_point(cam: Camera, spec: CourtSpec, source: str, uv: np.ndarray, z: float, box_w: float, px: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Court (x, y) and its 2x2 covariance for one chosen pixel.

    px scales the pixel noise from the 1024-px frames it was set on: keypoint and
    box-edge errors are a share of the person's size, which grows with resolution.
    """
    xy = floor_xy(cam, uv, spec, z)[0]
    su, sv, sz = SIGMA[source]
    if su is None:  # the box centre wanders with arms and racket
        su = max(3.0, 0.12 * box_w / px)
    su, sv = su * px, sv * px
    J = ground_jacobian(cam, uv, spec, z)[0]
    R = J @ np.diag([su**2, sv**2]) @ J.T
    if sz:
        g = height_jacobian(cam, uv, spec, z)[0]
        R = R + sz**2 * np.outer(g, g)
    return xy, R

"""Coarse camera alignment for when line refinement has no nearby start (a cut, or the phone being moved).

The painted lines are thresholded from the black-hat response and turned into
a truncated distance map; a pose is scored by how far its projected paint
samples land from the nearest detected line. Powell's method from a handful of
starts around the previous pose finds the basin, and line refinement finishes.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize

from squashvid.calib.lines import Maps, ModelSamples
from squashvid.geom import court
from squashvid.geom.camera import Camera

PAINT_THRESHOLD = 14.0
TRUNCATE_PX = 25.0  # reference px (see lines.REFERENCE_WIDTH)


def distance_map(maps: Maps, threshold: float = PAINT_THRESHOLD) -> np.ndarray:
    lines = (maps.paint >= threshold).astype(np.uint8)
    return cv2.distanceTransform(1 - lines, cv2.DIST_L2, 3).astype(np.float32)


@dataclass
class CoarseResult:
    camera: Camera
    score: float  # mean truncated distance, frame px


def chamfer(cam: Camera, dist: np.ndarray, pts: np.ndarray, trunc: float = TRUNCATE_PX) -> float:
    h, w = dist.shape
    uv = cam.project(pts)
    inside = (cam.depth(pts) > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    d = np.full(len(pts), trunc, np.float32)
    if inside.any():
        vals = cv2.remap(dist, uv[inside, 0].astype(np.float32).reshape(-1, 1), uv[inside, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR)
        d[inside] = np.minimum(vals.reshape(-1), trunc)
    return float(d.mean())


def coarse_align(
    cam: Camera,
    maps: Maps,
    samples: ModelSamples,
    spec: court.CourtSpec | None = None,
    shifts: tuple[float, ...] = (-0.8, -0.4, 0.0, 0.4, 0.8),
    trunc: float = TRUNCATE_PX,
) -> CoarseResult:
    """Search pose around `cam` (lateral camera shifts as extra starts) to minimise chamfer distance.

    trunc is reference px.
    """
    trunc = trunc * maps.scale
    paint = ~samples.is_edge
    pts = samples.points(spec)[paint]
    dist = distance_map(maps)
    c0 = cam.centre

    def make(x):
        rvec = x[:3]
        centre = x[3:]
        R = cv2.Rodrigues(rvec)[0]
        return cam.copy(rvec=rvec.copy(), tvec=-R @ centre)

    best: CoarseResult | None = None
    for dx in shifts:
        start = np.concatenate([np.asarray(cam.rvec, np.float64), c0 + np.array([dx, 0.0, 0.0])])
        # A lateral move usually comes with the phone turning back toward the court: seed yaw too.
        best_seed = start
        best_seed_score = chamfer(make(start), dist, pts, trunc)
        for dyaw in np.linspace(-0.15, 0.15, 7):
            cand = start.copy()
            R = cv2.Rodrigues(cand[:3])[0]
            Ry = cv2.Rodrigues(np.array([0.0, dyaw, 0.0]))[0]  # yaw about the camera's own vertical axis
            cand[:3] = cv2.Rodrigues(Ry @ R)[0].reshape(3)
            s = chamfer(make(cand), dist, pts, trunc)
            if s < best_seed_score:
                best_seed, best_seed_score = cand, s
        sol = minimize(lambda x: chamfer(make(x), dist, pts, trunc), best_seed, method="Powell", options={"xtol": 1e-4, "ftol": 1e-4, "maxfev": 4000})
        res = CoarseResult(make(sol.x), float(sol.fun))
        if best is None or res.score < best.score:
            best = res
    assert best is not None
    return best

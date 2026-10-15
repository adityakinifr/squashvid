"""Piecewise 3D ballistic fit (gravity + quadratic drag) to each linked segment.

A segment is a run of track frames between impacts (the track's segment_id). Each is fitted by
reprojection error. When an event on a known plane sits at either end of the segment, the fit is
pulled through that world point (anchored); otherwise depth along the ray is a guess among a few
starts and the result is an estimate only (anchored = False).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from squashvid.balltrack import pipeline
from squashvid.ball.regions import venue

G = 9.81
COLUMNS = ["frame", "t", "x", "y", "z", "segment_id", "anchored", "residual_px"]


@dataclass(frozen=True)
class FitParams:
    drag: float = 0.016  # 1/m: 0.5 * rho * Cd * A / m for a squash ball
    min_points: int = 6
    max_frames: int = 90
    anchor_sigma: float = 0.15  # m
    anchor_reach: int = 2  # frames between an event and the segment end that still count
    substeps: int = 4
    depth_starts: tuple[float, ...] = (2.0, 5.0, 8.0)
    pixel_sigma: float = 2.0
    margin: float = 0.5  # m outside the court box that still counts as plausible
    max_height: float = 7.0
    max_rms_px: float = 8.0  # fits worse than this are dropped


def simulate(p0: np.ndarray, v0: np.ndarray, times: np.ndarray, drag: float, substeps: int = 4) -> np.ndarray:
    """Positions (len(times), 3) at times (s, from 0, ascending) under gravity and quadratic drag."""
    out = np.empty((len(times), 3))
    p, v, t = p0.astype(float).copy(), v0.astype(float).copy(), 0.0
    for i, target in enumerate(times):
        n = max(1, int(np.ceil((target - t) / 0.0083 / 1.0))) if target > t else 0
        dt = (target - t) / max(n, 1)
        for _ in range(n):
            a = -drag * np.linalg.norm(v) * v
            a[2] -= G
            v = v + a * dt
            p = p + v * dt
        t = target
        out[i] = p
    return out


def _project(cam_of, frames: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.vstack([cam_of(int(f)).project(pts[i : i + 1]) for i, f in enumerate(frames)])


def _fit_segment(cam_of, frames, T, uv, anchors, fp: FitParams):
    """frames (n,), uv (n,2) detected points; anchors list of (frame, xyz). Returns (pos at all frames, rms) or None."""
    t = T[frames] - T[frames[0]]
    cam0 = cam_of(int(frames[0]))
    ray0, ray1 = cam0.rays(uv[:1])[0], cam0.rays(uv[-1:])[0]
    c0 = cam0.centre

    def resid(theta):
        pos = simulate(theta[:3], theta[3:], t, fp.drag, fp.substeps)
        r = (_project(cam_of, frames, pos) - uv) / fp.pixel_sigma
        extra = []
        for af, axyz in anchors:
            k = int(np.argmin(np.abs(frames - af)))
            extra.append((pos[k] - axyz) / fp.anchor_sigma)
        return np.concatenate([r.ravel()] + extra) if extra else r.ravel()

    best = None
    starts = []
    if anchors:
        k_a, p_a = anchors[0][0], anchors[0][1]
        k = int(np.argmin(np.abs(frames - k_a)))
        starts = [(k, p_a)]
    else:
        for d in fp.depth_starts:
            # point on the first ray at depth d (metres along y from the camera) then a velocity toward the last ray's point
            p_start = c0 + ray0 * ((d - c0[1]) / ray0[1] if abs(ray0[1]) > 1e-6 else 1.0)
            starts.append((0, p_start))
    for k, p_k in starts:
        # initial guess: the end point on the last ray at similar depth
        d_last = p_k[1] if not anchors else p_k[1]
        p_end = c0 + ray1 * ((d_last - c0[1]) / ray1[1] if abs(ray1[1]) > 1e-6 else 1.0)
        p_first = c0 + ray0 * ((d_last - c0[1]) / ray0[1] if abs(ray0[1]) > 1e-6 else 1.0)
        v = (p_end - p_first) / max(t[-1], 1e-3)
        v[2] += 0.5 * G * t[-1]
        if anchors:
            p_first = p_k - v * t[k]
        try:
            sol = least_squares(resid, np.r_[p_first, v], loss="soft_l1", f_scale=2.0, max_nfev=60)
        except (ValueError, FloatingPointError):
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return None
    return best.x, t


def fit(stem: str, track, events: pd.DataFrame, T: np.ndarray, fp: FitParams = FitParams()) -> pd.DataFrame:
    cam_of = pipeline.camera_track(stem)
    spec = venue(stem).spec
    fr = track.frames
    fr = fr[(fr.state != "none") & (fr.segment_id >= 0)]
    rows = []
    ev = events[np.isfinite(events[["x", "y", "z"]]).all(axis=1) & events.kind.isin(["front_wall", "side_wall_left", "side_wall_right", "floor", "back_wall"])]
    for sid, seg in fr.groupby("segment_id"):
        seg = seg.sort_values("frame")
        det = seg[seg.state != "interpolated"]
        if len(det) < fp.min_points or len(seg) > fp.max_frames:
            continue
        frames_all = seg.frame.to_numpy().astype(int)
        lo, hi = frames_all[0], frames_all[-1]
        anchors = [(float(e.frame), np.array([e.x, e.y, e.z])) for e in ev.itertuples()
                   if lo - fp.anchor_reach <= e.frame <= lo + fp.anchor_reach or hi - fp.anchor_reach <= e.frame <= hi + fp.anchor_reach]
        anchors = anchors[:2]
        frames = det.frame.to_numpy().astype(int)
        uv = det[["u", "v"]].to_numpy(float)
        res = _fit_segment(cam_of, frames, T, uv, anchors, fp)
        if res is None:
            continue
        theta, _ = res
        t_all = T[frames_all] - T[frames[0]]
        pos = simulate(theta[:3], theta[3:], t_all, fp.drag, fp.substeps)
        if t_all[0] < 0:
            continue
        m = fp.margin
        if (pos[:, 0].min() < -m or pos[:, 0].max() > spec.width + m or pos[:, 1].min() < -m or pos[:, 1].max() > spec.length + m
                or pos[:, 2].min() < -0.1 or pos[:, 2].max() > fp.max_height):
            continue
        proj = _project(cam_of, frames_all, pos)
        obs = seg[["u", "v"]].to_numpy(float)
        err = np.hypot(*(proj - obs).T)
        use = (seg.state != "interpolated").to_numpy()
        if np.sqrt(np.mean(err[use] ** 2)) > fp.max_rms_px:
            continue
        for f, p, e_px in zip(frames_all, pos, err):
            rows.append((f, T[f], p[0], p[1], p[2], int(sid), bool(anchors), float(e_px)))
    return pd.DataFrame(rows, columns=COLUMNS)

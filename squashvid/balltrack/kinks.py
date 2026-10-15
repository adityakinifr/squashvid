"""Kinks in a 2D ball track: the instants where the image velocity jumps.

An impact (wall, floor, racket) changes the ball's velocity in an instant while
gravity changes it smoothly. Around each trial time tau the samples are fitted two
ways, per image axis:

  smooth  p(s) = p0 + v s + a s^2 / 2                       (one flight)
  kink    p(s) = p0 + v- s (s < 0), p0 + v+ s (s >= 0), plus a s^2 / 2 (same pull, new velocity)

A real kink makes the smooth model fail where the kink model fits, and the jump
v+ - v- is its size. Times are frame indices (floats); velocities are px per frame.
Trial times step by `step` frames, so a kink is placed between frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KinkParams:
    windows: tuple[float, ...] = (4.0, 6.0)  # frames either side of tau that enter the fit; the better-fitting width wins
    min_side: int = 3  # samples needed on each side
    step: float = 0.25  # trial spacing in frames
    sigma: float = 2.0  # px, the noise the gain is measured against
    min_jump: float = 2.5  # px per frame: |v+ - v-|
    min_gain: float = 6.0  # (RSS smooth - RSS kink) / sigma^2, summed over both axes
    min_sep: float = 4.0  # frames between kinks; the stronger one wins
    max_rms: float = 4.0  # px: the kink model must fit its own window, unless it explains most of what the smooth one cannot
    min_rel_jump: float = 0.12  # the jump must be this fraction of the speed on either side
    explained: float = 0.3  # kink RSS / smooth RSS at or below this passes the rms test


@dataclass(frozen=True)
class Kink:
    frame: float  # sub-frame index
    u: float
    v: float
    v_before: tuple[float, float]  # (du, dv) per frame
    v_after: tuple[float, float]
    jump: float  # |v_after - v_before|, px per frame
    gain: float
    rms: float  # px, residual of the kink fit
    n_left: int
    n_right: int
    # How far the nearest sample is on each side: a kink hidden in a gap is less certain.
    gap_left: float
    gap_right: float


def _fit(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    r = y - design @ coef
    return coef, float((r * r).sum())


def _at(f: np.ndarray, pts: np.ndarray, tau: float, window: float, p: KinkParams) -> Kink | None:
    lo, hi = np.searchsorted(f, [tau - window, tau + window + 1e-9])
    s = f[lo:hi] - tau
    left = s < 0
    n_l, n_r = int(left.sum()), int((~left).sum())
    if n_l < p.min_side or n_r < p.min_side:
        return None
    y = pts[lo:hi]
    dk = np.column_stack([np.ones_like(s), np.where(left, s, 0.0), np.where(left, 0.0, s), 0.5 * s * s])
    ds = np.column_stack([np.ones_like(s), s, 0.5 * s * s])
    ck, rk = _fit(dk, y)
    _, rs = _fit(ds, y)
    gain = (rs - rk) / p.sigma**2
    jump_vec = ck[2] - ck[1]
    jump = float(np.hypot(*jump_vec))
    rms = float(np.sqrt(rk / (2 * len(s))))
    speed = 0.5 * (np.hypot(*ck[1]) + np.hypot(*ck[2]))
    if gain < p.min_gain or jump < max(p.min_jump, p.min_rel_jump * speed):
        return None
    if rms > p.max_rms and rk > p.explained * rs:
        return None
    return Kink(tau, float(ck[0, 0]), float(ck[0, 1]), (float(ck[1, 0]), float(ck[1, 1])), (float(ck[2, 0]), float(ck[2, 1])),
                jump, float(gain), rms, n_l, n_r, float(-s[left].max()), float(s[~left].min()))


def detect(frames: np.ndarray, u: np.ndarray, v: np.ndarray, p: KinkParams = KinkParams()) -> list[Kink]:
    """Kinks of one continuous track given by samples (frame, u, v); samples may have gaps."""
    f = np.asarray(frames, np.float64)
    pts = np.column_stack([u, v]).astype(np.float64)
    ok = np.isfinite(f) & np.isfinite(pts).all(axis=1)
    f, pts = f[ok], pts[ok]
    if len(f) < 2 * p.min_side:
        return []
    order = np.argsort(f)
    f, pts = f[order], pts[order]

    taus = np.arange(f[0], f[-1] + 1e-9, p.step)
    found: list[Kink] = []
    for tau in taus:
        best: Kink | None = None
        for window in p.windows:
            k = _at(f, pts, float(tau), window, p)
            if k is not None and (best is None or k.rms < best.rms):
                best = k
        if best is not None:
            found.append(best)
    # Non-maximum suppression: strongest first, drop neighbours closer than min_sep.
    found.sort(key=lambda k: -k.gain)
    kept: list[Kink] = []
    for k in found:
        if all(abs(k.frame - q.frame) >= p.min_sep for q in kept):
            kept.append(k)
    return sorted(kept, key=lambda k: k.frame)

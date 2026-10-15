"""Link per-frame ball candidates into a track by global association.

Candidates are the heatmap blobs of a detector (frame, x, y, peak, mass). One ball
exists at a time, so the best explanation of a whole window is the set of paths
through the candidates that maximises

    sum of node rewards - sum of motion costs - a start cost per path.

A motion cost compares a candidate with where the previous one was heading
(constant velocity from the path's own history). It is capped, so an impact, where
the ball legitimately breaks its prediction, costs a fixed amount instead of a
veto. The best path is found by dynamic programming over (candidate, best
history) states and extracted repeatedly; frames already claimed by a stronger
path are dropped from weaker ones. Static blobs (shoes, fixtures) never enter.

Short gaps between linked candidates are filled by `fill_gaps`, which never
smooths across an impact: if the two sides do not meet smoothly, the gap is
bridged through the intersection of their extrapolations (a kink) or left open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from squashvid.balltrack import kinks as kink_mod

STATES = ("detected", "linked", "interpolated", "none")


@dataclass(frozen=True)
class LinkParams:
    max_gap: int = 8  # frames a path may skip between two candidates
    max_speed: float = 120.0  # px per frame; faster is a different object
    sigma: float = 7.0  # px of prediction error at one frame
    sigma_speed: float = 0.12  # sigma also grows with the path's speed (blur length and curvature grow with it)
    sigma_growth: float = 0.6  # sigma grows by this fraction per skipped frame
    first_sigma: float = 35.0  # px per frame of free movement before the path has a velocity
    cap: float = 5.0  # motion cost ceiling, in units of half squared sigmas
    gap_cost: float = 0.4  # per skipped frame
    start_cost: float = 2.0
    reward_base: float = 0.6
    reward_peak: float = 2.0
    min_score: float = 2.0  # a path's total must exceed this
    min_nodes: int = 3
    blend: float = 0.6  # weight of the newest velocity in the path's running velocity
    merge_px: float = 4.0  # blobs of one frame closer than this are one blob
    # static rejection
    static_px: float = 2.5
    static_frames: int = 25  # distinct frames with a blob this close ...
    static_window: int = 150  # ... within this many frames either side
    rescue_peak: float = 0.8  # an unlinked blob this strong stays, as a one-sample track
    detected_peak: float = 0.5  # a chosen blob at or above this peak is "detected", below it "linked"


def merge_blobs(c: pd.DataFrame, px: float) -> pd.DataFrame:
    """One blob per spot per frame: nearby blobs of a frame are combined, keeping the strongest peak and summed mass."""
    rows = []
    for frame, g in c.groupby("frame", sort=True):
        g = g.sort_values("peak", ascending=False)
        kept: list[dict] = []
        for r in g.itertuples():
            for k in kept:
                if np.hypot(k["x"] - r.x, k["y"] - r.y) < px:
                    k["mass"] += r.mass
                    break
            else:
                kept.append({"frame": int(frame), "x": float(r.x), "y": float(r.y), "peak": float(r.peak), "mass": float(r.mass)})
        rows.extend(kept)
    return pd.DataFrame(rows, columns=["frame", "x", "y", "peak", "mass"])


def static_mask(c: pd.DataFrame, p: LinkParams) -> np.ndarray:
    """True for blobs that sit in one spot for many frames: fixtures, shoes standing still, a ball at rest."""
    if c.empty:
        return np.zeros(0, bool)
    fr = c.frame.to_numpy()
    xy = c[["x", "y"]].to_numpy()
    out = np.zeros(len(c), bool)
    lo = np.searchsorted(fr, fr - p.static_window)
    hi = np.searchsorted(fr, fr + p.static_window, side="right")
    for i in range(len(c)):
        seg = slice(lo[i], hi[i])
        near = np.hypot(xy[seg, 0] - xy[i, 0], xy[seg, 1] - xy[i, 1]) <= p.static_px
        if len(np.unique(fr[seg][near])) >= p.static_frames:
            out[i] = True
    return out


def _extract(c: pd.DataFrame, t: np.ndarray, p: LinkParams) -> list[list[int]]:
    """Dynamic programme over candidates (rows of c, sorted by frame); returns paths as lists of row indices."""
    n = len(c)
    fr = c.frame.to_numpy()
    pos = c[["x", "y"]].to_numpy()
    reward = p.reward_base + p.reward_peak * c.peak.to_numpy()
    score = np.empty(n)
    pred = np.full(n, -1)
    vel = np.zeros((n, 2))
    has_v = np.zeros(n, bool)
    first = np.searchsorted(fr, fr - p.max_gap)  # first row of any candidate within max_gap frames
    for j in range(n):
        best, arg, bv, bhv = -p.start_cost, -1, np.zeros(2), False
        for i in range(first[j], j):
            g = int(fr[j] - fr[i])
            if g < 1:
                continue
            dt = float(t[fr[j]] - t[fr[i]]) if len(t) > fr[j] else float(g)
            dt = dt / (float(np.median(np.diff(t))) if len(t) > 1 else 1.0)  # in nominal frames
            d = pos[j] - pos[i]
            if np.hypot(*d) > p.max_speed * dt:
                continue
            if has_v[i]:
                e = np.hypot(*(d - vel[i] * dt))
                sig = (p.sigma + p.sigma_speed * float(np.hypot(*vel[i]))) * (1.0 + p.sigma_growth * (g - 1))
            else:
                e = np.hypot(*d)
                sig = p.first_sigma * g
            cost = min(0.5 * (e / sig) ** 2, p.cap) + p.gap_cost * (g - 1)
            s = score[i] - cost
            if s > best:
                v_obs = d / dt
                saturated = has_v[i] and 0.5 * (e / sig) ** 2 >= p.cap
                if has_v[i] and not saturated:
                    v_new = p.blend * v_obs + (1 - p.blend) * vel[i]
                else:
                    v_new = v_obs
                best, arg, bv, bhv = s, i, v_new, True
        score[j] = reward[j] + best
        pred[j], vel[j], has_v[j] = arg, bv, bhv

    used = np.zeros(n, bool)
    paths: list[list[int]] = []
    for j in np.argsort(-score):
        if used[j] or score[j] < p.min_score:
            continue
        path, k = [], int(j)
        while k >= 0 and not used[k]:
            path.append(k)
            used[k] = True
            k = int(pred[k])
        path.reverse()
        if len(path) >= p.min_nodes:
            paths.append(path)
    return paths


def link(cands: pd.DataFrame, t: np.ndarray | None = None, p: LinkParams = LinkParams()) -> pd.DataFrame:
    """Chosen candidates, one row per frame at most: frame, x, y, peak, state, track_id.

    `t` are the clip's frame timestamps (to scale motion by real frame spacing); frames index into it.
    """
    cols = ["frame", "x", "y", "peak", "state", "track_id"]
    if cands is None or len(cands) == 0:
        return pd.DataFrame(columns=cols)
    c = merge_blobs(cands, p.merge_px).sort_values(["frame", "peak"], ascending=[True, False]).reset_index(drop=True)
    c = c[~static_mask(c, p)].reset_index(drop=True)
    if c.empty:
        return pd.DataFrame(columns=cols)
    tt = np.arange(int(c.frame.max()) + 1, dtype=float) if t is None else np.asarray(t, float)
    paths = _extract(c, tt, p)
    claimed: set[int] = set()
    rows = []
    for tid, path in enumerate(paths):
        # A path split by frames a stronger path claimed is cut there: a run shorter than min_nodes is not a track.
        run: list[int] = []
        runs = []
        for i in path:
            if int(c.frame[i]) in claimed:
                if run:
                    runs.append(run)
                run = []
            else:
                run.append(i)
        if run:
            runs.append(run)
        for r in runs:
            if len(r) < p.min_nodes:
                continue
            for i in r:
                claimed.add(int(c.frame[i]))
                rows.append((int(c.frame[i]), float(c.x[i]), float(c.y[i]), float(c.peak[i]),
                             "detected" if c.peak[i] >= p.detected_peak else "linked", tid))
    # Strong blobs no path took, on frames no path covers: shoes and fixtures are already gone, so these are
    # mostly the ball in a stretch too short or too erratic to chain.
    if p.rescue_peak < 1.0:
        tid = len(paths)
        for i in c.index[c.peak >= p.rescue_peak]:
            f = int(c.frame[i])
            if f not in claimed:
                claimed.add(f)
                rows.append((f, float(c.x[i]), float(c.y[i]), float(c.peak[i]), "detected", tid))
                tid += 1
    return pd.DataFrame(rows, columns=cols).sort_values("frame").reset_index(drop=True)


@dataclass(frozen=True)
class FillParams:
    max_gap: int = 8  # frames; longer gaps stay open
    smooth_tol: float = 8.0  # px + smooth_tol_per_frame per frame: both extrapolations must reach the far end this closely
    smooth_tol_per_frame: float = 2.0
    kink_tol: float = 10.0  # px: the two extrapolated lines must pass this close to each other, plus ...
    kink_tol_quad: float = 1.5  # ... this times gap^2 (gravity bends a flight away from its tangent lines)
    fit_n: int = 3  # samples per side that give the end velocities
    fit_span: int = 6  # frames


def _end_velocity(f: np.ndarray, pts: np.ndarray, side: str, n: int, span: int) -> np.ndarray | None:
    """Velocity (px/frame) at the end of a run of samples, from a line through the last n samples within span."""
    sel = slice(-n, None) if side == "left" else slice(0, n)
    ff, pp = f[sel], pts[sel]
    ok = np.abs(ff - (f[-1] if side == "left" else f[0])) <= span
    ff, pp = ff[ok], pp[ok]
    if len(ff) < 2:
        return None
    return np.polyfit(ff - ff.mean(), pp, 1)[0]


KINK_COLS = ["frame", "u", "v", "vb_u", "vb_v", "va_u", "va_v", "jump", "gain", "rms", "gap_left", "gap_right", "source", "track_id"]


class Track(NamedTuple):
    frames: pd.DataFrame  # one row per frame: frame, u, v, state, score, track_id, segment_id
    kinks: pd.DataFrame  # velocity jumps: seen in the samples ("track") or found by bridging a gap ("bridge")


def fill_gaps(track: pd.DataFrame, n_frames: int, p: FillParams = FillParams(), kp: kink_mod.KinkParams = kink_mod.KinkParams()) -> Track:
    """Per-frame table (frame, u, v, state, score, track_id, segment_id) and the kinks, from linked candidates.

    Gaps up to `max_gap` frames inside one track are interpolated only when the track either side
    reaches across smoothly (cubic Hermite from the end velocities); when it does not, the gap holds
    an impact and is bridged by the two sides' lines meeting at the kink, with the segment id
    changing there; when even that fails the gap stays open. A kink seen in the detected samples
    also starts a new segment.
    """
    if not track.empty:
        n_frames = max(n_frames, int(track.frame.max()) + 1)
    out = pd.DataFrame({"frame": np.arange(n_frames), "u": np.nan, "v": np.nan, "state": "none", "score": 0.0, "track_id": -1, "segment_id": -1})
    kink_rows: list[dict] = []
    if track.empty:
        return Track(out, pd.DataFrame(columns=KINK_COLS))
    seg_counter = 0
    for tid, g in track.groupby("track_id", sort=True):
        g = g.sort_values("frame")
        f = g.frame.to_numpy().astype(float)
        pts = g[["x", "y"]].to_numpy()
        idx = g.frame.to_numpy().astype(int)
        out.loc[idx, ["u", "v"]] = pts
        out.loc[idx, "state"] = g.state.to_numpy()
        out.loc[idx, "score"] = g.peak.to_numpy()
        out.loc[idx, "track_id"] = tid
        # Segment breaks: kinks of the detected samples, and gaps that could not be filled.
        seen = kink_mod.detect(f, pts[:, 0], pts[:, 1], kp)
        breaks: list[float] = [k.frame for k in seen]
        for k in seen:
            kink_rows.append(dict(frame=k.frame, u=k.u, v=k.v, vb_u=k.v_before[0], vb_v=k.v_before[1], va_u=k.v_after[0], va_v=k.v_after[1],
                                  jump=k.jump, gain=k.gain, rms=k.rms, gap_left=k.gap_left, gap_right=k.gap_right, source="track", track_id=tid))
        fills: list[tuple[int, np.ndarray]] = []
        for a in range(len(f) - 1):
            gap = int(f[a + 1] - f[a])
            if gap < 2:
                continue
            if gap - 1 > p.max_gap:
                breaks.append(f[a] + 0.5)
                continue
            left_f, left_p = f[: a + 1], pts[: a + 1]
            right_f, right_p = f[a + 1 :], pts[a + 1 :]
            va = _end_velocity(left_f, left_p, "left", p.fit_n, p.fit_span)
            vb = _end_velocity(right_f, right_p, "right", p.fit_n, p.fit_span)
            pa, pb = pts[a], pts[a + 1]
            fr = np.arange(f[a] + 1, f[a + 1])
            if va is None or vb is None:
                va = vb = (pb - pa) / gap if va is None and vb is None else (va if va is not None else vb)
            tol = p.smooth_tol + p.smooth_tol_per_frame * gap
            if np.hypot(*(pa + va * gap - pb)) <= tol and np.hypot(*(pb - vb * gap - pa)) <= tol:
                s = (fr - f[a]) / gap
                h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
                fills.append((a, (h00[:, None] * pa + h10[:, None] * va * gap + h01[:, None] * pb + h11[:, None] * vb * gap)))
                continue
            # Lines pa + va*s and pb - vb*(gap - s): the s minimising their distance.
            w = va + vb
            denom = float(w @ w)
            s_star = float(((pb - pa - vb * gap) @ w) / denom) if denom > 1e-9 else gap / 2
            meet_a, meet_b = pa + va * s_star, pb - vb * (gap - s_star)
            if 0.0 < s_star < gap and np.hypot(*(meet_a - meet_b)) <= p.kink_tol + p.kink_tol_quad * gap**2:
                meet = 0.5 * (meet_a + meet_b)
                s_f = fr - f[a]
                fill = np.where((s_f <= s_star)[:, None], pa + (meet - pa) * (s_f / s_star)[:, None],
                                meet + (pb - meet) * ((s_f - s_star) / (gap - s_star))[:, None])
                fills.append((a, fill))
                breaks.append(f[a] + s_star)
                kink_rows.append(dict(frame=f[a] + s_star, u=float(meet[0]), v=float(meet[1]), vb_u=float(va[0]), vb_v=float(va[1]), va_u=float(vb[0]), va_v=float(vb[1]),
                                      jump=float(np.hypot(*(vb - va))), gain=np.nan, rms=np.nan, gap_left=s_star, gap_right=gap - s_star, source="bridge", track_id=tid))
            else:
                breaks.append(f[a] + 0.5)
        for a, fill in fills:
            fr = np.arange(int(f[a]) + 1, int(f[a + 1])).astype(int)
            out.loc[fr, ["u", "v"]] = fill
            out.loc[fr, "state"] = "interpolated"
            out.loc[fr, "track_id"] = tid
        frames = np.arange(int(f[0]), int(f[-1]) + 1)
        seg = np.searchsorted(np.sort(np.asarray(breaks)), frames, side="right")
        member = (out.loc[frames, "track_id"] == tid).to_numpy()
        # Frames inside an unfilled gap carry no track.
        out.loc[frames[member], "segment_id"] = seg_counter + seg[member]
        seg_counter += int(seg.max()) + 1
    return Track(out, pd.DataFrame(kink_rows, columns=KINK_COLS).sort_values("frame").reset_index(drop=True))

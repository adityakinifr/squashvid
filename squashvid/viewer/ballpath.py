"""A ball path on every frame of a rally, for drawing: the tracked ball where it was seen, smooth curves through the
gaps, and a 3D position on every frame.

Image path (pixels). Knots are the tracked ball (detected or tracker-interpolated) and located impacts (ball events
with a pixel). Between two knots of one rally at most MAX_GAP_S apart, the gap is a cubic Hermite curve whose end
tangents are each side's own image velocity (from the nearest knot on that side), so a ball that leaves view and
comes back stays on its arc. An impact (an event, or a change of the tracker's segment_id) is a corner: no tangent
is taken across it, and a gap starting or ending there uses the straight line toward the other end, so a bounce
keeps its V. Tracked points are smoothed with a 5-frame Savitzky-Golay within each flight (never across an impact)
to take out pixel jitter.

3D path (court metres). Knots are the fitted 3D track (ball_3d.parquet) and impacts with a located 3D position. Each
coordinate is interpolated in time with PCHIP, which never overshoots its knots, across gaps up to MAX_GAP_3D_S.
Frames between knots that have an image position are then moved onto their camera ray (the nearest point on it,
kept inside the court), so the 3D ball always lines up with the picture; only the depth along the ray is the
interpolation's guess.

kind per frame: 0 none, 1 measured (tracked pixel / fitted 3D point), 2 filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

MAX_GAP_S = 0.8  # longest gap in the image track bridged with a curve
MAX_GAP_3D_S = 1.2  # longest gap between 3D knots bridged
TANGENT_S = 0.12  # a side's velocity comes from a knot at most this far back (forward)
SMOOTH_WINDOW = 5
PAD_S = 0.3  # the path runs this far either side of a rally
CEILING_M = 6.0
NONE, MEASURED, FILLED = 0, 1, 2


def rally_spans(rallies: list[dict], ts: np.ndarray) -> list[tuple[int, int]]:
    """[a, b] frame spans of the rallies (padded); the whole clip when there are none."""
    n = len(ts)
    if not rallies:
        return [(0, n - 1)] if n else []
    out = []
    for r in rallies:
        s, e = r.get("start_t"), r.get("end_t")
        if s is None or e is None:
            continue
        a, b = int(np.searchsorted(ts, s - PAD_S)), int(np.searchsorted(ts, e + PAD_S, side="right")) - 1
        if b > a:
            out.append((max(a, 0), min(b, n - 1)))
    return out


def _event_frames(events: pd.DataFrame | None, ts: np.ndarray) -> pd.DataFrame:
    if events is None or not len(events):
        return pd.DataFrame(columns=["frame", "u", "v", "x", "y", "z"])
    e = events.copy()
    e["frame"] = np.clip(np.searchsorted(ts, e.t.to_numpy(float)), 0, len(ts) - 1)
    for c in ("u", "v", "x", "y", "z"):
        if c not in e:
            e[c] = np.nan
    return e


def _hermite(p0, p1, v0, v1, T, s):
    s = s[:, None]
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1


def image_path(uv: np.ndarray, segment: np.ndarray, ts: np.ndarray, spans: list[tuple[int, int]], events: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(path (n,2) px, kind (n,) u8). uv is the tracked ball (NaN where unseen); segment the tracker's segment_id
    (-1 unknown)."""
    n = len(ts)
    path, kind = np.full((n, 2), np.nan), np.zeros(n, np.uint8)
    ev = _event_frames(events, ts)
    impact = np.zeros(n, bool)
    impact[ev.frame.to_numpy(int)] = True
    knot_uv = uv.copy()
    located = ev[np.isfinite(ev.u.to_numpy(float)) & np.isfinite(ev.v.to_numpy(float))]
    for r in located.itertuples():
        if not np.isfinite(knot_uv[r.frame]).all():
            knot_uv[r.frame] = (r.u, r.v)
    for a, b in spans:
        k = a + np.flatnonzero(np.isfinite(knot_uv[a : b + 1]).all(axis=1))
        if not len(k):
            continue
        # A corner sits between knots i and i+1 when the tracker started a new flight, or at an impact knot.
        seg = segment[k]
        new_flight = np.r_[False, (seg[1:] != seg[:-1]) & (seg[1:] >= 0) & (seg[:-1] >= 0)]
        corner = impact[k]
        # Smooth tracked points within each flight.
        pts = knot_uv[k].astype(float)
        cut = np.flatnonzero(new_flight | corner | np.r_[False, np.diff(k) > 1])
        for lo, hi in zip(np.r_[0, cut], np.r_[cut, len(k)]):
            if hi - lo >= SMOOTH_WINDOW:
                pts[lo:hi] = savgol_filter(pts[lo:hi], SMOOTH_WINDOW, 2, axis=0)
        path[k] = pts
        kind[k] = MEASURED
        for i in range(len(k) - 1):
            f0, f1 = k[i], k[i + 1]
            T = ts[f1] - ts[f0]
            if f1 - f0 < 2 or T > MAX_GAP_S:
                continue
            p0, p1 = pts[i], pts[i + 1]
            chord = (p1 - p0) / T
            turns = new_flight[i + 1]
            v0 = v1 = chord
            if not corner[i] and not turns and i > 0 and not new_flight[i] and ts[f0] - ts[k[i - 1]] <= TANGENT_S:
                v0 = (p0 - pts[i - 1]) / (ts[f0] - ts[k[i - 1]])
            if not corner[i + 1] and not turns and i + 2 < len(k) and not new_flight[i + 2] and ts[k[i + 2]] - ts[f1] <= TANGENT_S:
                v1 = (pts[i + 2] - p1) / (ts[k[i + 2]] - ts[f1])
            # Keep a wild tangent from looping the curve: no end speed beyond three times the chord's (plus 200 px/s).
            cap = 3 * np.linalg.norm(chord) + 200
            v0, v1 = (v * min(1.0, cap / max(np.linalg.norm(v), 1e-9)) for v in (v0, v1))
            mid = np.arange(f0 + 1, f1)
            path[mid] = _hermite(p0, p1, v0, v1, T, (ts[mid] - ts[f0]) / T)
            kind[mid] = FILLED
    return path, kind


def _snap(c: np.ndarray, d: np.ndarray, p: np.ndarray, box: tuple[float, float, float]) -> np.ndarray:
    """The point on ray c + s d nearest p, with s kept to the stretch of the ray inside the court box."""
    s = float(np.dot(p - c, d))
    lo, hi = 0.0, np.inf
    for ax, top in enumerate(box):
        if abs(d[ax]) < 1e-12:
            if not 0 <= c[ax] <= top:
                return p
            continue
        s0, s1 = sorted(((0 - c[ax]) / d[ax], (top - c[ax]) / d[ax]))
        lo, hi = max(lo, s0), min(hi, s1)
    if lo > hi:
        return p
    return c + min(max(s, lo), hi) * d


def path_3d(xyz: np.ndarray, ts: np.ndarray, spans: list[tuple[int, int]], events: pd.DataFrame | None = None, uv: np.ndarray | None = None,
            camera_of=None, box: tuple[float, float, float] = (6.4, 9.75, CEILING_M)) -> tuple[np.ndarray, np.ndarray]:
    """(path (n,3) m, kind (n,) u8). xyz is the fitted 3D track (NaN elsewhere); uv the image path to snap onto,
    camera_of(frame) -> (centre, unit ray of uv[frame]) or None."""
    n = len(ts)
    path, kind = np.full((n, 3), np.nan), np.zeros(n, np.uint8)
    knot = xyz.copy()
    ev = _event_frames(events, ts)
    ev = ev[np.isfinite(ev[["x", "y", "z"]].to_numpy(float)).all(axis=1)]
    knot[ev.frame.to_numpy(int)] = ev[["x", "y", "z"]].to_numpy(float)  # an impact is exact where it was located
    measured = np.isfinite(knot).all(axis=1)
    for a, b in spans:
        k = a + np.flatnonzero(measured[a : b + 1])
        path[k] = knot[k]
        kind[k] = MEASURED
        if len(k) < 2:
            continue
        f = PchipInterpolator(ts[k], knot[k], axis=0)
        for i in range(len(k) - 1):
            f0, f1 = k[i], k[i + 1]
            if f1 - f0 < 2 or ts[f1] - ts[f0] > MAX_GAP_3D_S:
                continue
            mid = np.arange(f0 + 1, f1)
            est = f(ts[mid])
            for j, fr in enumerate(mid):
                ray = camera_of(fr) if camera_of is not None and uv is not None and np.isfinite(uv[fr]).all() else None
                if ray is not None:
                    est[j] = _snap(ray[0], ray[1], est[j], box)
            path[mid] = est
            kind[mid] = FILLED
    return path, kind


def build(work, ts: np.ndarray, rallies: list[dict], uv: np.ndarray, cam: np.ndarray | None, venue) -> dict[str, np.ndarray]:
    """The arrays the viewer draws: ball_path_uv, ball_path_kind, and with a 3D track ball_path_xyz, ball_path_kind3."""
    from squashvid.geom.camera import Camera

    def read(name):
        f = work / name
        return pd.read_parquet(f) if f.exists() else None

    n = len(ts)
    track, events, b3 = read("ball_track.parquet"), read("ball_events.parquet"), read("ball_3d.parquet")
    segment = np.full(n, -1)
    if track is not None and "segment_id" in track:
        d = track[(track.frame >= 0) & (track.frame < n)].drop_duplicates("frame", keep="last")
        segment[d.frame.to_numpy(int)] = d.segment_id.fillna(-1).to_numpy(int)
    spans = rally_spans(rallies, ts)
    puv, kuv = image_path(uv, segment, ts, spans, events)
    out = {"ball_path_uv": puv, "ball_path_kind": kuv}
    if b3 is None or not len(b3):
        return out
    xyz = np.full((n, 3), np.nan)
    d = b3[(b3.frame >= 0) & (b3.frame < n)].drop_duplicates("frame", keep="last")
    xyz[d.frame.to_numpy(int)] = d[["x", "y", "z"]].to_numpy(float)
    dist = venue.camera(np.zeros(3), np.zeros(3)).dist if venue is not None else np.zeros(5)

    def camera_of(fr):
        if cam is None or not np.isfinite(cam[fr]).all():
            return None
        c = Camera(float(cam[fr, 0]), float(cam[fr, 1]), float(cam[fr, 2]), cam[fr, 3:6].astype(np.float64), cam[fr, 6:9].astype(np.float64), dist)
        return c.centre, c.rays(puv[fr : fr + 1])[0]

    spec = venue.spec if venue is not None else None
    box = (spec.width, spec.length, CEILING_M) if spec is not None else (6.4, 9.75, CEILING_M)
    pxyz, kxyz = path_3d(xyz, ts, spans, events, puv, camera_of, box)
    out.update(ball_path_xyz=pxyz, ball_path_kind3=kxyz)
    return out


__all__ = ["build", "image_path", "path_3d", "rally_spans", "MEASURED", "FILLED"]

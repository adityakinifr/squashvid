"""Measurements on the per-frame player table: coverage, losses, foot-point jitter, movement and crossings.

These are pilot numbers on one camera and one match, unvalidated against
ground truth except where clicked foot positions exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from squashvid.geom.court import CourtSpec
from squashvid.players.tracklets import box_gap

T_RADIUS = 1.5
STILL_WINDOW_S = 1.0
STILL_RADIUS = 0.2  # m: smoothed track stays within this of its window mean
CONTACT_PX = 10.0
EVENT_MERGE_S = 0.5


def wide(table: pd.DataFrame) -> pd.DataFrame:
    """One row per frame with A_ and B_ prefixed columns."""
    a = table[table.player == "A"].set_index("frame").add_prefix("A_")
    b = table[table.player == "B"].set_index("frame").add_prefix("B_")
    w = a.join(b, how="outer")
    w["t"] = w.A_t.fillna(w.B_t)
    return w


def coverage(table: pd.DataFrame) -> dict:
    """Share of frames with both players detected, one, or none."""
    w = wide(table)
    n = w.A_detected.fillna(False).astype(int) + w.B_detected.fillna(False).astype(int)
    out = {k: round(float((n == v).mean()), 4) for k, v in (("both", 2), ("one", 1), ("none", 0))}
    for p in ("A", "B"):
        out[f"{p}_detected"] = round(float(w[f"{p}_detected"].mean()), 4)
        out[f"{p}_placed"] = round(float(w[f"{p}_used"].mean()), 4)
        src = table[table.player == p].src.value_counts(normalize=True)
        out[f"{p}_source"] = {k: round(float(v), 4) for k, v in src.items()}
    return out


def court_zone(x: np.ndarray, y: np.ndarray, spec: CourtSpec) -> np.ndarray:
    """back / mid / front by distance from the back wall, crossed with left / centre / right thirds."""
    depth = np.where(y < spec.box_back_y, "back", np.where(y < spec.short_y + 1.0, "mid", "front"))
    side = np.where(x < spec.width / 3, "left", np.where(x < 2 * spec.width / 3, "centre", "right"))
    return np.char.add(np.char.add(depth.astype(str), "-"), side.astype(str))


def losses(table: pd.DataFrame, spec: CourtSpec, image_u: dict[str, np.ndarray], bands: dict[str, tuple[float, float]], overlap: dict[str, np.ndarray]) -> dict:
    """Where undetected players are: court zone of the smoothed position and image region of its projection.

    image_u: projected u of each player's smoothed foot point per row of that player's table.
    overlap: whether the player's predicted box overlaps the other player's box.
    """
    out = {}
    for p in ("A", "B"):
        tp = table[table.player == p].reset_index(drop=True)
        lost = (~tp.detected.to_numpy()) & np.isfinite(tp.sx.to_numpy())
        if not lost.any():
            out[p] = {"n_lost": 0}
            continue
        zones = pd.Series(court_zone(tp.sx.to_numpy()[lost], tp.sy.to_numpy()[lost], spec)).value_counts(normalize=True)
        u = image_u[p][lost]
        region = np.full(lost.sum(), "open", dtype=object)
        for name, (lo, hi) in bands.items():
            region[(u >= lo) & (u <= hi)] = name
        region[overlap[p][lost]] = "behind_other_player"
        out[p] = {
            "n_lost": int(lost.sum()),
            "share_lost": round(float(lost.mean()), 4),
            "court_zone": {k: round(float(v), 3) for k, v in zones.items()},
            "image_region": {k: round(float(v), 3) for k, v in pd.Series(region).value_counts(normalize=True).items()},
        }
    return out


def still_windows(t: np.ndarray, xy: np.ndarray, window_s: float = STILL_WINDOW_S, radius: float = STILL_RADIUS) -> list[tuple[int, int]]:
    """Non-overlapping index windows of at least window_s where the smoothed track barely moves."""
    out, i, n = [], 0, len(t)
    while i < n:
        j = int(np.searchsorted(t, t[i] + window_s))
        if j >= n:
            break
        seg = xy[i:j]
        if np.isfinite(seg).all() and np.linalg.norm(seg - seg.mean(0), axis=1).max() < radius:
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def jitter(table: pd.DataFrame, spec: CourtSpec) -> dict:
    """Per-window standard deviation of raw ankle-based and box-based foot positions while the player stands still.

    Real weight shifts inflate these, so they bound per-frame noise from above.
    """
    rows = []
    for p in ("A", "B"):
        tp = table[table.player == p].reset_index(drop=True)
        t, xy = tp.t.to_numpy(), tp[["sx", "sy"]].to_numpy()
        for i, j in still_windows(t, xy):
            seg = tp.iloc[i:j]
            rec = {"player": p, "y": float(np.nanmean(seg.sy)), "n": j - i}
            for name, cx, cy in (("ankle", "ankle_x", "ankle_y"), ("box", "box_x", "box_y"), ("smoothed", "sx", "sy")):
                v = seg[[cx, cy]].to_numpy()
                if name == "box":
                    v = v[~seg.truncated.to_numpy()]
                v = v[np.isfinite(v).all(1)]
                rec[f"{name}_sx"] = float(v[:, 0].std()) if len(v) >= 10 else np.nan
                rec[f"{name}_sy"] = float(v[:, 1].std()) if len(v) >= 10 else np.nan
            a = seg[["ankle_x", "ankle_y"]].to_numpy() - seg[["box_x", "box_y"]].to_numpy()
            a = a[np.isfinite(a).all(1)]
            rec["ankle_minus_box_y"] = float(np.median(a[:, 1])) if len(a) else np.nan
            rec["ankle_minus_box_x"] = float(np.median(a[:, 0])) if len(a) else np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_windows": 0}
    df["zone"] = np.where(df.y < spec.box_back_y, "back", np.where(df.y < spec.short_y + 1.0, "mid", "front"))
    cols = [c for c in df.columns if c.endswith("_sx") or c.endswith("_sy") or c.startswith("ankle_minus")]
    by_zone = df.groupby("zone")[cols].median().round(3)
    return {
        "n_windows": int(len(df)),
        "median_sd_m": df[cols].median().round(3).to_dict(),
        "by_zone": {z: r.dropna().to_dict() for z, r in by_zone.iterrows()},
        "windows_per_zone": df.zone.value_counts().to_dict(),
    }


def movement(table: pd.DataFrame, spec: CourtSpec, t0: float | None = None, t1: float | None = None) -> dict:
    """Distance covered, mean distance from the T and share of time within T_RADIUS of it, per player."""
    out = {}
    T = spec.t_point[:2]
    for p in ("A", "B"):
        tp = table[table.player == p]
        if t0 is not None:
            tp = tp[(tp.t >= t0) & (tp.t <= t1)]
        xy = tp[["sx", "sy"]].to_numpy()
        ok = np.isfinite(xy).all(1)
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        step_ok = ok[1:] & ok[:-1]
        dt = np.diff(tp.t.to_numpy())
        dT = np.linalg.norm(xy[ok] - T, axis=1)
        w = np.r_[dt, np.median(dt) if len(dt) else 0][ok]
        out[p] = {
            "distance_m": round(float(step[step_ok].sum()), 1),
            "mean_dist_from_T_m": round(float(np.average(dT, weights=w)), 2) if ok.any() else None,
            "share_within_1.5m_of_T": round(float(np.average(dT < T_RADIUS, weights=w)), 3) if ok.any() else None,
            "seconds": round(float(w.sum()), 1),
            "share_long_gap": round(float(tp.long_gap.mean()), 3),
        }
    return out


def rallies(table: pd.DataFrame, min_len_s: float = 6.0, speed: float = 1.0, merge_s: float = 1.5) -> list[tuple[float, float]]:
    """Heuristic rally windows: both players moving, smoothed over 2 s. Only a stand-in until hit detection exists."""
    w = wide(table).sort_values("t")
    t = w.t.to_numpy()
    v = np.hypot(w.A_svx, w.A_svy).fillna(0).to_numpy() + np.hypot(w.B_svx, w.B_svy).fillna(0).to_numpy()
    k = max(int(round(2.0 * 30)), 1)
    act = np.convolve(v, np.ones(k) / k, mode="same") > 2 * speed
    segs, start = [], None
    for i, a in enumerate(act):
        if a and start is None:
            start = i
        if (not a or i == len(act) - 1) and start is not None:
            segs.append([t[start], t[i]])
            start = None
    merged: list[list[float]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < merge_s:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(round(a, 2), round(b, 2)) for a, b in merged if b - a >= min_len_s]


@dataclass
class Event:
    start: int  # frame
    end: int
    t0: float
    t1: float
    kind: str  # overlap: both boxes seen and touching; hidden: one player predicted behind the other


def crossing_events(w: pd.DataFrame, pred_boxes: dict[str, np.ndarray], gap_px: float = CONTACT_PX, merge_s: float = EVENT_MERGE_S) -> list[Event]:
    """Contiguous spells where the two players' boxes overlap or nearly touch, or one is hidden behind the other.

    pred_boxes: per player (frames, 4) box from the detection when seen, otherwise
    from the smoothed court position projected at the player's stature.
    """
    frames, t = w.index.to_numpy(), w.t.to_numpy()
    hit, kind = np.zeros(len(w), bool), np.zeros(len(w), dtype=object)
    det_a, det_b = w.A_detected.fillna(False).to_numpy(bool), w.B_detected.fillna(False).to_numpy(bool)
    for i in range(len(w)):
        a, b = pred_boxes["A"][i], pred_boxes["B"][i]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        if box_gap(a, b) < gap_px:
            hit[i] = True
            kind[i] = "overlap" if det_a[i] and det_b[i] else "hidden"
    events: list[Event] = []
    idx = np.flatnonzero(hit)
    for i in idx:
        if events and t[i] - events[-1].t1 <= merge_s:
            e = events[-1]
            e.end, e.t1 = int(frames[i]), float(t[i])
            if kind[i] == "hidden":
                e.kind = "hidden"
        else:
            events.append(Event(int(frames[i]), int(frames[i]), float(t[i]), float(t[i]), str(kind[i])))
    return events

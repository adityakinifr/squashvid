"""Rally segments: chains of shots with audio, movement and ball evidence behind them.

Audio alone merges and misses rallies (pilot: overlap 0.51), and both players
move even between rallies (walking, fetching the ball), so none of the three
signals separates rallies by itself. The segmentation is built on the detected
shot chain: consecutive shots no more than CHAIN_GAP_S apart belong to one
rally. The other signals check each chain and set its confidence:

  * audio: impacts (front wall, racket, floor, side wall) heard during the chain;
  * movement: both players moving, not walking or standing;
  * ball: raw detections per second, when the ball detector has run.

Chains with too few shots, or with players standing still, are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from squashvid.analysis.io import PLAYERS, Match

CHAIN_GAP_S = 2.8
MIN_SHOTS = 2
LEAD_S = 0.6  # the serve toss and swing precede the first contact
TAIL_S = 0.9  # the ball is still live after the last contact
MOVING_MPS = 0.6
IMPACT_KINDS = ("front_wall", "racket", "floor", "side_wall")
COLUMNS = ["rally", "start_t", "end_t", "start_frame", "end_frame", "n_shots", "confidence", "sources"]


def evidence(m: Match, bin_s: float = 1.0) -> pd.DataFrame:
    """Per-second evidence used for confidence and for the timeline plot."""
    edges = np.arange(0, m.duration + bin_s, bin_s)
    mid = edges[:-1] + bin_s / 2
    imp = m.onsets[m.onsets.kind.isin(IMPACT_KINDS)].t.to_numpy()
    audio = np.histogram(imp, edges)[0] / bin_s
    sp = np.minimum(m.tracks["A"].speed, m.tracks["B"].speed)
    both = pd.Series(sp).groupby(np.minimum((m.t // bin_s).astype(int), len(mid) - 1)).apply(lambda s: float((s > MOVING_MPS).mean())).reindex(range(len(mid)), fill_value=0.0).to_numpy()
    ball = np.full(len(mid), np.nan)
    d = m.ball_detections
    if d is not None and len(d):
        seen = d[d.visible].t.to_numpy()
        covered = d.t.max()
        ball = np.histogram(seen, edges)[0] / (bin_s * m.fps)
        ball[mid > covered] = np.nan
    return pd.DataFrame({"t": mid, "audio_rate": audio, "both_moving": both, "ball_presence": ball})


def _mean_in(ev: pd.DataFrame, col: str, a: float, b: float) -> float:
    s = ev[(ev.t >= a) & (ev.t <= b)][col]
    return float(s.mean()) if s.notna().any() else float("nan")


# Play score: a logistic on per-second features, fitted against the by-eye rally census of clips 1 and 3
# (24 and 15 rallies, 1 s resolution, itself rough). Cross-clip AUC 0.83 and 0.90.
# Features: slower player's speed (m/s), faster player's speed, swings per second of the less active
# player (7 s mean), impacts per second (7 s mean), distance between the players (m).
PLAY_WEIGHTS = np.array([0.79, 1.31, 1.08, 0.31, -0.76])
PLAY_BIAS = -0.77
BALL_WEIGHT = 1.0  # added per unit of ball presence above its median when the ball detector has run (unvalidated)
ENTER_P, LEAVE_P = 0.55, 0.40


def _roll(a: np.ndarray, k: int) -> np.ndarray:
    return pd.Series(a).rolling(k, center=True, min_periods=1).mean().to_numpy()


def play_features(m: Match, swings: pd.DataFrame) -> np.ndarray:
    """(seconds, 5) feature matrix for the play score."""
    n = int(m.duration)
    sec = np.minimum((m.t // 1).astype(int), n - 1)

    def per_sec(a):
        return pd.Series(a).groupby(sec).mean().reindex(range(n)).ffill().fillna(0).to_numpy()

    sp = [per_sec(m.tracks[k].speed) for k in PLAYERS]
    cnt = [np.histogram(swings[swings.player == k].t, np.arange(0, n + 1))[0].astype(float) for k in PLAYERS]
    imp = m.onsets[m.onsets.kind.isin(IMPACT_KINDS)].t
    tA, tB = m.tracks["A"], m.tracks["B"]
    dp = per_sec(np.hypot(tA.x - tB.x, tA.y - tB.y))
    return np.c_[
        _roll(np.minimum(*sp), 5),
        _roll(np.maximum(*sp), 5),
        np.minimum(_roll(cnt[0], 7), _roll(cnt[1], 7)),
        _roll(np.histogram(imp, np.arange(0, n + 1))[0].astype(float), 7),
        _roll(dp, 5),
    ]


def play_probability(m: Match, swings: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-second probability that a rally is on, and the evidence table behind it."""
    f = play_features(m, swings)
    z = f @ PLAY_WEIGHTS + PLAY_BIAS
    ev = evidence(m)[: len(z)].copy()
    if ev.ball_presence.notna().any():
        base = np.nanmedian(ev.ball_presence)
        bp = _roll(ev.ball_presence.fillna(base).to_numpy(), 5)
        z = z + BALL_WEIGHT * (bp - base) / max(base, 0.05) * ev.ball_presence.notna().to_numpy()
    p = 1 / (1 + np.exp(-z))
    ev["play_p"] = p
    return p, ev


def _intervals(p: np.ndarray, enter: float = ENTER_P, leave: float = LEAVE_P) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(p):
        if start is None and v >= enter:
            start = i
        elif start is not None and v < leave:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(p)))
    return out


def segment(m: Match, hits: pd.DataFrame, swings: pd.DataFrame, chain_gap: float = CHAIN_GAP_S, min_shots: int = MIN_SHOTS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rallies, the hits with rally id and shot index (shots outside any rally drop out), and the evidence table.

    A rally is a stretch where the play score stays up and at least `min_shots`
    shots by both players chain together (gaps no longer than `chain_gap`).
    """
    p, ev = play_probability(m, swings)
    h = hits.sort_values("t").reset_index(drop=True)
    empty = pd.DataFrame(columns=COLUMNS), h.iloc[:0].assign(rally=pd.Series(dtype=int), shot_index=pd.Series(dtype=int)), ev
    if len(h) == 0:
        return empty
    ht = h.t.to_numpy()
    rows, groups = [], []
    for a, b in _intervals(p):
        sel = np.flatnonzero((ht >= a - 1.0) & (ht <= b + 1.0))
        if len(sel) < min_shots:
            continue
        # Split the stretch wherever the shot chain breaks.
        br = np.flatnonzero(np.diff(ht[sel]) > chain_gap) + 1
        for chunk in np.split(sel, br):
            g = h.iloc[chunk]
            if len(g) < min_shots:
                continue
            hit_minor = min((g.hitter == k).mean() for k in PLAYERS)
            if hit_minor < 0.2:  # one player alone is practice, not a rally
                continue
            ta, tb = float(g.t.iloc[0] - LEAD_S), float(g.t.iloc[-1] + TAIL_S)
            ball = _mean_in(ev, "ball_presence", ta, tb)
            parts = {
                "hits": float(g.confidence.mean()),
                "audio": min(1.0, float(g.source.str.contains("audio|ball").mean()) + 0.15 * min(_mean_in(ev, "audio_rate", ta, tb), 2.0)),
                "movement": float(np.clip(_mean_in(ev, "play_p", ta, tb) / 0.7, 0, 1)),
                "alternation": float((g.hitter.to_numpy()[1:] != g.hitter.to_numpy()[:-1]).mean()),
            }
            if np.isfinite(ball):
                base = np.nanmedian(ev.ball_presence)
                parts["ball"] = float(np.clip(ball / max(base * 1.5, 0.1), 0, 1))
            conf = float(np.mean(list(parts.values()))) * min(1.0, 0.55 + 0.15 * len(g))
            rows.append((ta, tb, len(g), conf, "+".join(k for k, v in parts.items() if v >= 0.5) or "weak"))
            groups.append(g.index.to_numpy())
    r = pd.DataFrame(rows, columns=["start_t", "end_t", "n_shots", "confidence", "sources"])
    if len(r) == 0:
        return empty
    r = r.sort_values("start_t").reset_index(drop=True)
    r.insert(0, "rally", np.arange(1, len(r) + 1))
    r["start_t"] = r.start_t.clip(lower=0).round(3)
    r["end_t"] = r.end_t.clip(upper=m.duration).round(3)
    r["start_frame"] = [m.frame_at(x) for x in r.start_t]
    r["end_frame"] = [m.frame_at(x) for x in r.end_t]
    r["confidence"] = r.confidence.round(3)
    parts_ = []
    order = np.argsort([g[0] for g in groups])
    for rid, gi in zip(r.rally, order):
        g = h.loc[groups[gi]].copy()
        g["rally"] = rid
        g["shot_index"] = np.arange(1, len(g) + 1)
        parts_.append(g)
    return r[COLUMNS], pd.concat(parts_, ignore_index=True), ev


__all__ = ["segment", "evidence", "play_probability"]

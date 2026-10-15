"""Closed-world identity: label every tracklet player A (light-blue shirt), player B (dark shirt) or neither.

A two-component colour model is fitted per clip to torso descriptors, seeded
by the share of light-blue pixels, and gives every detection a log-likelihood
ratio. Labels are then chosen jointly by a Viterbi pass over tracklets in start
order whose state is the last tracklet given to each player. That state is
enough to price the jump from where a player was last seen to where the next
tracklet starts, so a crossing is resolved by colour and by court-space
plausibility together, using future frames freely. Two tracklets that overlap
in time can never share a player.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from squashvid.players.appearance import FEATURES

# How to tell the players apart on screen: the colour model is seeded so that A is the light-blue shirt.
LOOKS = {"A": "light-blue shirt", "B": "dark shirt"}
COLOUR_COLS = [FEATURES.index(c) for c in ("frac_blue", "b", "L", "frac_dark")]
COLOUR_SCALE = np.array([1.0, 0.1, 0.02, 1.0])  # bring b* and L* to unit-ish ranges
LLR_CLIP = 4.0
VAR_FLOOR = 0.004  # scaled units (sd ~0.06): one torso sample is never near-certain
X_PENALTY = 1.2  # per frame: dearer than an ambiguous colour (ln 2), cheaper than a wrong one
V_MAX = 8.0  # m/s, Buote-style outlier gate for junior players
V_TYP = 3.0  # m/s, spread of plausible motion over a gap
GATE_PENALTY = 40.0
BEAM = 400


@dataclass
class ColourModel:
    """Diagonal Gaussians for player A and player B in a 4-D torso colour space."""

    mean: np.ndarray  # (2, D): row 0 is A
    var: np.ndarray  # (2, D)
    weight: np.ndarray  # (2,)

    @staticmethod
    def vectors(feat: np.ndarray) -> np.ndarray:
        return np.asarray(feat, np.float64)[:, COLOUR_COLS] * COLOUR_SCALE

    @classmethod
    def fit(cls, feat: np.ndarray, iters: int = 40) -> "ColourModel":
        """EM on reliable torso descriptors, seeded by light-blue share."""
        X = cls.vectors(feat)
        resp = np.column_stack([X[:, 0] > 0.35, X[:, 0] <= 0.35]).astype(np.float64)
        for _ in range(iters):
            nk = resp.sum(0) + 1e-9
            mean = (resp.T @ X) / nk[:, None]
            var = np.maximum((resp.T @ X**2) / nk[:, None] - mean**2, VAR_FLOOR)
            weight = nk / nk.sum()
            ll = np.stack([_log_gauss(X, mean[k], var[k]) + np.log(weight[k]) for k in range(2)], 1)
            ll -= ll.max(1, keepdims=True)
            resp = np.exp(ll)
            resp /= resp.sum(1, keepdims=True)
        if mean[0, 0] < mean[1, 0]:  # keep A as the bluer component
            mean, var, weight = mean[::-1], var[::-1], weight[::-1]
        return cls(mean, var, weight)

    def llr(self, feat: np.ndarray) -> np.ndarray:
        """log p(f | A) - log p(f | B), clipped: positive means light-blue shirt."""
        X = self.vectors(feat)
        out = _log_gauss(X, self.mean[0], self.var[0]) - _log_gauss(X, self.mean[1], self.var[1])
        return np.clip(np.nan_to_num(out), -LLR_CLIP, LLR_CLIP)


def _log_gauss(X: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (((X - mean) ** 2) / var + np.log(2 * np.pi * var)).sum(1)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


@dataclass
class Tracklet:
    tid: int
    rows: np.ndarray  # candidate row indices, in frame order
    f0: int
    f1: int
    t0: float
    t1: float
    p0: np.ndarray  # court xy at start (NaN if never placed)
    p1: np.ndarray
    s0: float  # positional variance at start, m^2
    s1: float
    cost_a: float
    cost_b: float
    cost_x: float
    llr: float = 0.0  # weighted mean colour evidence
    extra: dict = field(default_factory=dict)


def build_tracklets(tids: np.ndarray, frames: np.ndarray, times: np.ndarray, xy: np.ndarray, var: np.ndarray, llr: np.ndarray, w: np.ndarray) -> list[Tracklet]:
    """Summaries the labeller needs; endpoints use the median of the first and last three placed points."""
    out = []
    for tid in np.unique(tids):
        rows = np.flatnonzero(tids == tid)
        rows = rows[np.argsort(frames[rows])]
        placed = rows[np.isfinite(xy[rows]).all(1)]
        head, tail = placed[:3], placed[-3:]
        p0 = np.median(xy[head], 0) if len(head) else np.full(2, np.nan)
        p1 = np.median(xy[tail], 0) if len(tail) else np.full(2, np.nan)
        s0 = float(np.median(var[head])) if len(head) else 1.0
        s1 = float(np.median(var[tail])) if len(tail) else 1.0
        ww, ll = w[rows], llr[rows]
        out.append(
            Tracklet(
                int(tid), rows, int(frames[rows[0]]), int(frames[rows[-1]]), float(times[rows[0]]), float(times[rows[-1]]),
                p0, p1, s0, s1,
                cost_a=float((ww * softplus(-ll)).sum()),
                cost_b=float((ww * softplus(ll)).sum()),
                cost_x=X_PENALTY * len(rows),
                llr=float((ww * ll).sum() / max(ww.sum(), 1e-9)),
            )
        )
    return out


def gap_cost(a: Tracklet, b: Tracklet) -> float:
    """Price of the same player leaving tracklet a and reappearing at the start of tracklet b."""
    if not (np.isfinite(a.p1).all() and np.isfinite(b.p0).all()):
        return 2.0
    dt = max(b.t0 - a.t1, 1 / 60)
    d = float(np.linalg.norm(b.p0 - a.p1))
    s2 = a.s1 + b.s0
    cost = d**2 / (2 * (s2 + (V_TYP * dt) ** 2))
    excess = d - (V_MAX * dt + 3 * np.sqrt(s2))
    if excess > 0:
        cost += GATE_PENALTY + excess**2 / s2
    return cost


def assign(tracklets: list[Tracklet], beam: int = BEAM) -> dict[int, str]:
    """Label each tracklet 'A', 'B' or 'X' by Viterbi over (last A tracklet, last B tracklet)."""
    order = sorted(range(len(tracklets)), key=lambda i: (tracklets[i].f0, tracklets[i].f1))
    # state -> (cost, backpointer); a backpointer is (tracklet index, label, parent)
    states: dict[tuple[int, int], tuple[float, tuple | None]] = {(-1, -1): (0.0, None)}
    gaps: dict[tuple[int, int], float] = {}

    def gap(i: int, k: int) -> float:
        if i < 0:
            return 0.0
        key = (i, k)
        if key not in gaps:
            gaps[key] = gap_cost(tracklets[i], tracklets[k])
        return gaps[key]

    for k in order:
        tk = tracklets[k]
        nxt: dict[tuple[int, int], tuple[float, tuple | None]] = {}

        def relax(state: tuple[int, int], cost: float, bp: tuple) -> None:
            if state not in nxt or cost < nxt[state][0]:
                nxt[state] = (cost, bp)

        for (a, b), (c, bp) in states.items():
            relax((a, b), c + tk.cost_x, (k, "X", bp))
            if a < 0 or tracklets[a].f1 < tk.f0:
                relax((k, b), c + tk.cost_a + gap(a, k), (k, "A", bp))
            if b < 0 or tracklets[b].f1 < tk.f0:
                relax((a, k), c + tk.cost_b + gap(b, k), (k, "B", bp))
        if len(nxt) > beam:
            nxt = dict(sorted(nxt.items(), key=lambda kv: kv[1][0])[:beam])
        states = nxt

    _, bp = min(states.values(), key=lambda v: v[0])
    labels: dict[int, str] = {}
    while bp is not None:
        k, lab, bp = bp
        labels[tracklets[k].tid] = lab
    return labels

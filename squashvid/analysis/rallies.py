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

Two things keep short or patchily detected rallies. Front-wall and racket
impacts (ball events or audio) between two detected shots are shots the hit
detector missed, so they bridge a gap in the chain while play goes on, and two
front-wall impacts between one player's hits show the opponent hit unseen. And
a chain that opens with a confirmed serve (struck from a service box with the
opponent at the back of the other half, and reaching the front wall) is a rally
even when the play score never rises, which it does not over a two- or
three-shot rally; the walking-pace trim never cuts such a chain below the serve
and the return. Confirmed serves also split rallies the chain ran together.
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
# A break in the shot chain only ends a rally when play also looks paused during the gap: in continuous play a
# couple of missed shots leave a gap of 3-5 s while the play score stays high (IMG_4090 at 295 s, checked by eye).
SPLIT_P = 0.7
ALWAYS_SPLIT_S = 6.0
# After a rally ends the players walk, pick up the ball and bounce it before the serve, which pass as shots and
# carry the rally on. Trailing shots where both players move at walking pace (mean speed within TRIM_WINDOW_S)
# are dropped. Fitted on the census clips: shots outside a census rally have the faster player at a median
# 1.2 m/s (max 1.5 on clip 1) and the slower at 0.6, against 2.0 and 1.0 inside. Only the tail is trimmed:
# both players stand still for the serve.
TRIM_FAST_MPS = 1.5
TRIM_SLOW_MPS = 0.8
TRIM_WINDOW_S = 1.0
# Serve shape, read off every rally start of the three clips (IMG_4089 checked by eye): just before the serve the
# server stands in a service box, 3.4-4.8 m from the back wall and within 1.6 m of a side wall, and the receiver
# waits 0.9-2.3 m from the back wall on the other half. The margins allow for foot-position error.
SERVE_WALL_MARGIN_M = 0.6  # beyond the box width
SERVE_DEPTH_MARGIN_M = (0.3, 0.7)  # behind the box and in front of the short line
RECEIVER_BEHIND_M = 1.2  # the receiver stands at least this much further back than the server
SERVE_STILL_MPS = 1.0  # the server's median speed while waiting
SERVE_LOOK_S = (-0.8, -0.1)  # the window before the hit in which positions are read
SERVE_GAP_S = 2.0  # a serve this long after the previous shot starts a new rally
# A serve must reach the front wall. While waiting in the box the server bounces the ball on the floor with the
# racket, and those touches have the serve's shape too, so a serve is confirmed only by a front-wall impact soon
# after it (the serve is then the server's last touch before that impact), or by the ball reaching a side wall at
# the back about a second later when the front-wall impact itself went unseen. The serve is struck from the box, so
# that last touch must be serve-shaped too (the server may be stepping into it). Every false serve-started rally in
# IMG_4089 (pre-serve bounces, the players swapping ends) lacked both; two real serves lacked both too and stay missed.
SERVE_TO_FRONT_S = 1.6
SERVE_SEARCH_S = 4.0  # how far after a serve-shaped touch to look for the front-wall impact
SERVE_TO_SIDE_S = (0.6, 2.0)
LINK_KINDS = ("front_wall", "racket")
# Two front-wall impacts at least this far apart between one player's consecutive hits mean the opponent hit in
# between unseen; a stretch whose detected hits are all one player's is then still a rally, not solo practice.
REPLY_APART_S = 0.8
LINK_MIN_P = 0.5  # links bridge a gap only while the play score stays above this; an impact after a rally ends is no shot


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


def _chain_breaks(p: np.ndarray, t0: float, t1: float, chain_gap: float = CHAIN_GAP_S, split_p: float = SPLIT_P, always: float = ALWAYS_SPLIT_S, links: np.ndarray | None = None) -> bool:
    """Whether the gap between two consecutive shots at t0 and t1 (s) ends a rally, given the per-second play score p.

    links: impact times (s) that mark missed shots; a gap they bridge into steps no longer than chain_gap stays whole
    while play goes on (the score stays above LINK_MIN_P)."""
    gap = t1 - t0
    if gap <= chain_gap:
        return False
    if gap > always:
        return True
    inside = p[int(np.ceil(t0)) : int(np.floor(t1)) + 1]
    if len(inside) == 0:
        return True
    if links is not None and inside.min() >= LINK_MIN_P:
        steps = np.diff(np.r_[t0, links[(links > t0) & (links < t1)], t1])
        if steps.max() <= chain_gap:
            return False
    return bool(inside.min() < split_p)


def walking_pace(m: Match, t: float, window: float = TRIM_WINDOW_S) -> bool:
    """Whether both players move at walking pace around time t (s)."""
    a, b = m.frame_at(t - window), m.frame_at(t + window)
    sp = np.vstack([m.tracks[k].speed[a : b + 1] for k in PLAYERS])
    if not np.isfinite(sp).any(axis=1).all():
        return False
    fast, slow = np.nanmean(np.nanmax(sp, axis=0)), np.nanmean(np.nanmin(sp, axis=0))
    return bool(fast < TRIM_FAST_MPS and slow < TRIM_SLOW_MPS)


def trim_tail(m: Match, chunk: np.ndarray, ht: np.ndarray, keep: int = 0) -> np.ndarray:
    """Drop the chain's trailing shots that happen at walking pace, never cutting below the first `keep`."""
    end = len(chunk)
    while end > keep and walking_pace(m, float(ht[chunk[end - 1]])):
        end -= 1
    return chunk[:end]


def _median_in(m: Match, player: str, a: float, b: float) -> tuple[float, float, float]:
    """Median x, y (m) and speed (m/s) of a player between times a and b."""
    i, j = m.frame_at(a), m.frame_at(b) + 1
    tr = m.tracks[player]
    sl = slice(i, max(j, i + 1))
    with np.errstate(all="ignore"):
        return float(np.nanmedian(tr.x[sl])), float(np.nanmedian(tr.y[sl])), float(np.nanmedian(tr.speed[sl]))


def serve_shaped(m: Match, t: float, hitter: str, still: bool = True) -> bool:
    """Whether a hit at t (s) looks like a serve: the hitter in a service box (and still, unless still=False: the server
    steps into the serve itself), the opponent behind on the other half."""
    s = m.spec
    a, b = t + SERVE_LOOK_S[0], t + SERVE_LOOK_S[1]
    hx, hy, hv = _median_in(m, hitter, a, b)
    ox, oy, _ = _median_in(m, "B" if hitter == "A" else "A", a, b)
    if not np.all(np.isfinite([hx, hy, hv, ox, oy])):
        return False
    wall = s.box_width + SERVE_WALL_MARGIN_M
    in_box = (hx < wall or hx > s.width - wall) and s.short_line_near_edge - s.box_depth - SERVE_DEPTH_MARGIN_M[0] < hy < s.short_line_near_edge + SERVE_DEPTH_MARGIN_M[1]
    other_half = (hx < s.width / 2) != (ox < s.width / 2)
    behind = oy < s.short_line_near_edge - s.box_depth + 0.1 and oy < hy - RECEIVER_BEHIND_M
    return bool(in_box and other_half and behind and (hv < SERVE_STILL_MPS or not still))


def impact_times(m: Match, kinds: tuple[str, ...]) -> np.ndarray:
    """Impact times (s) of the given kinds from the ball events and the audio onsets ("side_wall" matches either wall)."""
    def pick(k: pd.Series) -> np.ndarray:
        return k.isin(kinds).to_numpy() | (k.astype(str).str.startswith("side_wall").to_numpy() & ("side_wall" in kinds))

    parts = [m.onsets.t[pick(m.onsets.kind)].to_numpy(float)]
    if m.ball_events is not None and len(m.ball_events):
        parts.append(m.ball_events.t[pick(m.ball_events.kind)].to_numpy(float))
    return np.sort(np.concatenate(parts))


def link_times(m: Match) -> np.ndarray:
    """Front-wall and racket impact times (s): shots in play, detected or not."""
    return impact_times(m, LINK_KINDS)


def unseen_reply(front: np.ndarray, t: np.ndarray, hitter: np.ndarray) -> bool:
    """Whether two front-wall impacts (sorted times) fall between some pair of one player's consecutive hits."""
    for i in range(len(t) - 1):
        if hitter[i] == hitter[i + 1]:
            f = front[(front > t[i]) & (front < t[i + 1])]
            if len(f) > 1 and f[-1] - f[0] >= REPLY_APART_S:
                return True
    return False


def serves(m: Match, ht: np.ndarray, hitter: np.ndarray) -> np.ndarray:
    """Which hits (sorted times ht, hitters) are confirmed serves; see SERVE_TO_FRONT_S."""
    front, side = impact_times(m, ("front_wall",)), impact_times(m, ("side_wall",))
    out = np.zeros(len(ht), bool)
    for i in np.flatnonzero([serve_shaped(m, float(t), k) for t, k in zip(ht, hitter)]):
        f = front[(front > ht[i]) & (front <= ht[i] + SERVE_SEARCH_S)]
        if len(f):
            k = int(np.searchsorted(ht, f[0])) - 1  # the last touch before the impact, struck from the box
            if hitter[k] == hitter[i] and f[0] - ht[k] <= SERVE_TO_FRONT_S and serve_shaped(m, float(ht[k]), hitter[k], still=False):
                out[k] = True
                continue
        nxt = i + 1 < len(ht) and hitter[i + 1] != hitter[i]
        if nxt and ((side > ht[i] + SERVE_TO_SIDE_S[0]) & (side < ht[i] + SERVE_TO_SIDE_S[1])).any():
            out[i] = True
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
    links, front = link_times(m), impact_times(m, ("front_wall",))
    serve = serves(m, ht, h.hitter.to_numpy())
    h["serve"] = serve

    def breaks(i: int, j: int) -> bool:
        """Whether a rally ends between hits i and j (consecutive)."""
        return _chain_breaks(p, ht[i], ht[j], chain_gap, links=links) or bool(serve[j] and ht[j] - ht[i] > SERVE_GAP_S)

    chunks = []
    for a, b in _intervals(p):
        sel = np.flatnonzero((ht >= a - 1.0) & (ht <= b + 1.0))
        if len(sel) < min_shots:
            continue
        # Split the stretch wherever the shot chain breaks and play pauses in the gap, or a new serve starts.
        br = [i + 1 for i in range(len(sel) - 1) if breaks(sel[i], sel[i + 1])]
        chunks += [(c, "") for c in np.split(sel, br)]
    # A serve starts a rally whatever the play score: follow the chain from each serve no other chunk holds. A chain
    # that runs into a chunk the play score found is that rally's start, so it joins it.
    owner = np.full(len(h), -1)
    for n, (c, _) in enumerate(chunks):
        owner[c] = n
    for i in np.flatnonzero(serve & (owner < 0)):
        j = i
        while j + 1 < len(h) and owner[j + 1] < 0 and not breaks(j, j + 1):
            j += 1
        nxt = owner[j + 1] if j + 1 < len(h) and not breaks(j, j + 1) else -1
        if nxt >= 0 and chunks[nxt][0][0] == j + 1:
            chunks[nxt] = (np.r_[np.arange(i, j + 1), chunks[nxt][0]], chunks[nxt][1] or "serve")
            owner[i : j + 1] = nxt
        else:
            owner[i : j + 1] = len(chunks)
            chunks.append((np.arange(i, j + 1), "serve"))
    rows, groups = [], []
    for chunk, why in chunks:
        if not len(chunk):
            continue
        chunk = trim_tail(m, chunk, ht, keep=2 if serve[chunk[0]] else 0)
        g = h.iloc[chunk]
        if len(g) < min_shots:
            continue
        hit_minor = min((g.hitter == k).mean() for k in PLAYERS)
        if hit_minor < 0.2 and not unseen_reply(front, g.t.to_numpy(), g.hitter.to_numpy()):  # one player alone is practice
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
        rows.append((ta, tb, len(g), conf, "+".join([k for k, v in parts.items() if v >= 0.5] + ([why] if why else [])) or "weak"))
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

"""Rallies as chains of impacts.

In play the ball is struck every 0.5-2 s and hits the front wall once per
shot; between rallies the players walk, retrieve the ball and bounce it before
serving. A rally is therefore a chain of impacts (classify.rally_impacts: in
practice mostly front-wall impacts, which are heard far more reliably than the
racket contacts) with no gap longer than MAX_GAP, kept only if it has enough
impacts and lasts long enough. Pre-serve bouncing also forms chains, but a
regular metronome: chains whose intervals are short and nearly constant are
dropped. n_hits counts the chained impacts, which approximates the number of
shots; the racket contacts themselves are often too quiet to count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_GAP = 2.6  # s; the longest junior shot interval seen (lob, retrieve) is ~2.2 s
MIN_HITS = 3
MIN_DURATION = 2.0  # s, serve to last hit
BOUNCE_MAX_INTERVAL = 0.9  # s; ball bouncing before a serve repeats faster than this
BOUNCE_MAX_CV = 0.15  # and with nearly constant intervals
PAD = (0.0, 1.0)  # s added before the first and after the last hit: the ball dies after the last hit


def segment(times: np.ndarray, max_gap: float = MAX_GAP, min_hits: int = MIN_HITS, min_duration: float = MIN_DURATION) -> pd.DataFrame:
    """Rallies from impact times: start_t, end_t, first/last impact, n_hits (impacts chained), median interval."""
    t = np.sort(np.asarray(times, float))
    rows = []
    if len(t):
        cut = np.flatnonzero(np.diff(t) > max_gap) + 1
        for ch in np.split(t, cut):
            iv = np.diff(ch)
            if len(ch) < min_hits or ch[-1] - ch[0] < min_duration:
                continue
            if len(iv) >= 3 and np.median(iv) < BOUNCE_MAX_INTERVAL and np.std(iv) / np.mean(iv) < BOUNCE_MAX_CV:
                continue
            rows.append(dict(start_t=ch[0] - PAD[0], end_t=ch[-1] + PAD[1], first_hit=ch[0], last_hit=ch[-1], n_hits=len(ch),
                             median_interval=float(np.median(iv))))
    df = pd.DataFrame(rows, columns=["start_t", "end_t", "first_hit", "last_hit", "n_hits", "median_interval"])
    df.insert(0, "rally", np.arange(len(df)))
    return df


def assign(onsets: pd.DataFrame, rallies: pd.DataFrame) -> np.ndarray:
    """Rally index of each onset (-1 outside every rally)."""
    out = np.full(len(onsets), -1)
    t = onsets["t"].to_numpy()
    for r in rallies.itertuples():
        out[(t >= r.start_t) & (t <= r.end_t)] = r.rally
    return out

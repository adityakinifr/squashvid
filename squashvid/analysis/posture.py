"""How low each player got at the strike, from the 3D figures (squashvid.bodies).

hip_drop is how far the pelvis was below the player's standing pelvis height at the hit, as a share of that height:
the lowest pelvis within STRIKE_WINDOW_S of the hit (hit times are good to a few frames), against the 95th
percentile of the pelvis height over the clip's seen frames. Height is the best-observed axis of the lift (the
camera sees it directly; depth is the lifter's guess), so it is the one pose number the coaching uses. A hit gets
no value when the hitter's pose was seen in under MIN_SEEN of the window.

Checked by eye on crops at hit moments across the three clips: under UPRIGHT (0.08) is a player hitting with
nearly straight legs, about 0.15 to 0.3 a bent-knee shot or lunge, and over 0.45 a scramble on the floor. The
most upright "hits" were mostly serves or hits given to the wrong player, which is why the coaching leaves out
each rally's first shot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from squashvid.analysis.io import PLAYERS

STRIKE_WINDOW_S = 0.2
MIN_SEEN = 0.6
STANDING_Q = 95
UPRIGHT = 0.08  # hip_drop below this: hitting with nearly straight legs
PELVIS_Z = 2  # pelvis is joint 0 of the H36M layout; z is its third coordinate
STATE_SEEN = 1


def hip_drop(bodies: pd.DataFrame | None, hits: pd.DataFrame) -> np.ndarray:
    """Per hit (in the order of `hits`), the hitter's hip drop at the strike; NaN without enough pose."""
    out = np.full(len(hits), np.nan)
    if bodies is None or not len(bodies) or not len(hits):
        return out
    for p in PLAYERS:
        b = bodies[bodies.player == p].sort_values("t")
        if not len(b):
            continue
        t = b.t.to_numpy()
        z = np.array([j[PELVIS_Z] for j in b.joints])
        seen = b.state.to_numpy() == STATE_SEEN
        if not seen.any():
            continue
        stand = float(np.percentile(z[seen], STANDING_Q))
        need = MIN_SEEN * 2 * STRIKE_WINDOW_S / float(np.median(np.diff(t))) if len(t) > 1 else np.inf
        for i in np.flatnonzero(hits.hitter.to_numpy() == p):
            ht = float(hits.t.iloc[i])
            a, e = np.searchsorted(t, ht - STRIKE_WINDOW_S), np.searchsorted(t, ht + STRIKE_WINDOW_S, side="right")
            if seen[a:e].sum() >= need:
                out[i] = 1.0 - float(z[a:e].min()) / stand
    return out


__all__ = ["hip_drop", "UPRIGHT"]

"""How low each player got at the strike (from the 3D figures), and the coaching built on it."""

import numpy as np
import pandas as pd

from squashvid.analysis import coach as C
from squashvid.analysis import insights as I
from squashvid.analysis import posture as P

FPS = 30.0
STAND = 0.70


def bodies(player="A", seconds=10.0, dips=(), state=1, gaps=()):
    """One player's figure standing with the pelvis at STAND, dipping to the given (t, pelvis_z) for 0.1 s each."""
    t = np.arange(0, seconds, 1 / FPS)
    z = np.full(len(t), STAND)
    z[::7] = STAND - 0.03  # ordinary bob while moving, under the standing percentile
    for when, low in dips:
        z[np.abs(t - when) <= 0.05] = low
    keep = np.ones(len(t), bool)
    for a, b in gaps:
        keep &= ~((t >= a) & (t <= b))
    joints = [np.r_[[1.0, 2.0, zz], np.zeros(48)] for zz in z[keep]]
    return pd.DataFrame({"t": t[keep], "player": player, "state": np.uint8(state), "joints": joints})


def test_the_drop_is_the_lowest_pelvis_near_the_hit_as_a_share_of_standing_height():
    hits = pd.DataFrame({"t": [3.0, 6.1], "hitter": ["A", "A"]})
    out = P.hip_drop(bodies(dips=[(3.0, 0.56), (6.0, 0.63)]), hits)
    assert np.allclose(out, [0.2, 0.1])  # the second dip is 0.1 s before its hit, inside the window


def test_no_drop_without_enough_of_the_hitter_seen_or_without_figures():
    hits = pd.DataFrame({"t": [3.0, 5.0, 7.0], "hitter": ["A", "A", "B"]})
    out = P.hip_drop(bodies(gaps=[(4.85, 5.15)]), hits)
    assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isnan(out[2])  # a gap at the hit; nobody tracked as B
    assert np.isnan(P.hip_drop(bodies(state=2), hits)).all()  # bridged frames alone are not enough
    assert np.isnan(P.hip_drop(None, hits)).all()


def back_hits(upright_share: dict, n_rallies=12, per=4, cell="back-left"):
    """Rallies of alternating hits from `cell`; each player hits upright on the given share of their shots."""
    rows, rng = [], np.random.default_rng(0)
    for r in range(1, n_rallies + 1):
        for k in range(per):
            p = "AB"[k % 2]
            up = rng.random() < upright_share[p]
            rows.append({"clip": "X", "rally": r, "shot_index": k + 2, "t": r * 20.0 + k, "hitter": p, "cell_hit_from": cell, "confidence": 0.9, "hip_drop": 0.03 if up else 0.25})
    return pd.DataFrame(rows)


def test_the_player_who_hits_upright_more_from_the_back_gets_something_to_try():
    cands = {c["player"]: c for c in I._posture_candidates(back_hits({"A": 0.1, "B": 0.6}))}
    assert not cands["B"]["favourable"] and cands["A"]["favourable"]
    assert cands["B"]["suppressed"] is None and cands["B"]["display"]["upright_pct"] == "8"
    assert all("hips 3% below" in e["label"] for e in cands["B"]["evidence"])  # the upright moments, for a thing to try


def test_mid_court_shots_and_each_rally_first_shot_are_left_out():
    h = pd.concat([back_hits({"A": 0.1, "B": 0.1}), back_hits({"A": 0.0, "B": 1.0}, cell="mid-centre").assign(t=lambda d: d.t + 0.5)])
    first = back_hits({"A": 0.0, "B": 1.0}, per=1).assign(shot_index=1, t=lambda d: d.t + 0.7)
    cands = I._posture_candidates(pd.concat([h, first]))
    assert all(c["n"] == 24 for c in cands)  # 12 rallies x 2 back-court hits each, nothing else


def test_a_posture_item_carries_the_pose_caveat_and_passes_the_verifier():
    found = I.finalise(I._posture_candidates(back_hits({"A": 0.1, "B": 0.6})))
    out, rejected = C.write(found)
    assert rejected == []
    it = out["B"]["try"][0]
    assert it["theme"] == "posture" and C.POSE_CAVEAT in it["detail"] and C.WINNER_CAVEAT not in it["detail"]
    assert "almost upright, hips less than 8% below standing height" in it["why"][0]

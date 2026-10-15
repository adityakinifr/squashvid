"""Offline two-player identity on synthetic tracklets: colour, court-space gating and exclusivity."""

from __future__ import annotations

import numpy as np

from squashvid.players.appearance import FEATURES
from squashvid.players.identity import ColourModel, Tracklet, assign, build_tracklets, gap_cost

DT = 1 / 30


class Scene:
    """Accumulates candidate rows (tracklet id, frame, court xy, colour evidence) for build_tracklets."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, int, float, float, float]] = []

    def add(self, tid: int, frames: range, path, llr: float) -> None:
        for f in frames:
            x, y = path(f)
            self.rows.append((tid, f, x, y, llr))

    def tracklets(self) -> list[Tracklet]:
        r = np.array(self.rows, np.float64)
        tids, frames = r[:, 0].astype(int), r[:, 1].astype(int)
        n = len(r)
        return build_tracklets(tids, frames, frames * DT, r[:, 2:4], np.full(n, 0.01), r[:, 4], np.ones(n))


def lerp(f0: int, f1: int, p0, p1):
    """Straight-line run from p0 at frame f0 to p1 at frame f1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return lambda f: tuple(p0 + (p1 - p0) * (f - f0) / (f1 - f0))


def test_crossing_with_a_merged_box_keeps_both_identities():
    """A runs left to right, B right to left; for 0.3 s the detector sees one merged box.

    After the crossing the tracker has started fresh ids. The merged box carries
    mixed colour, the pieces after the crossing carry each shirt's evidence.
    """
    a_path = lerp(0, 90, (1.0, 4.0), (5.0, 4.6))
    b_path = lerp(0, 90, (5.0, 4.4), (1.0, 4.0))
    s = Scene()
    s.add(0, range(0, 36), a_path, +1.5)
    s.add(1, range(0, 36), b_path, -1.5)
    s.add(2, range(36, 54), lambda f: tuple((np.array(a_path(f)) + b_path(f)) / 2), 0.0)
    s.add(3, range(54, 90), a_path, +1.0)  # same side as A's run, fresh tracker id
    s.add(4, range(54, 90), b_path, -1.0)
    labels = assign(s.tracklets())
    assert labels[0] == "A" and labels[3] == "A"
    assert labels[1] == "B" and labels[4] == "B"
    assert labels[2] in {"A", "B", "X"}


def test_court_space_gate_overrides_weak_wrong_colour():
    """A tracklet whose shirt looks slightly dark appears 5 m from B a tenth of a second
    after B was last seen, but right where A was: only A can have got there."""
    s = Scene()
    s.add(0, range(0, 30), lambda f: (1.0, 2.0), +2.0)  # A, back left
    s.add(1, range(0, 30), lambda f: (5.0, 7.0), -2.0)  # B, front right
    s.add(2, range(33, 60), lambda f: (1.1, 2.1), -0.4)  # back left again, colour weakly dark
    s.add(3, range(33, 60), lambda f: (5.0, 7.0), -2.0)
    labels = assign(s.tracklets())
    assert labels[2] == "A"
    assert labels[3] == "B"


def test_colour_decides_when_motion_is_ambiguous():
    """Two tracklets start side by side after an occlusion; positions cannot tell them apart, shirts can."""
    s = Scene()
    s.add(0, range(0, 30), lambda f: (3.0, 4.0), +2.0)
    s.add(1, range(0, 30), lambda f: (3.2, 4.1), -2.0)
    s.add(2, range(40, 70), lambda f: (3.15, 4.05), -1.5)
    s.add(3, range(40, 70), lambda f: (3.05, 4.05), +1.5)
    labels = assign(s.tracklets())
    assert (labels[2], labels[3]) == ("B", "A")


def test_simultaneous_tracklets_never_share_a_player():
    """Three boxes at once (two players and a stray): each player label is used at most once per frame."""
    s = Scene()
    s.add(0, range(0, 40), lambda f: (2.0, 3.0), +2.0)
    s.add(1, range(0, 40), lambda f: (2.4, 3.2), +0.3)  # stray box beside A, faintly blue
    s.add(2, range(0, 40), lambda f: (4.5, 6.0), -2.0)
    labels = assign(s.tracklets())
    assert labels[0] == "A" and labels[2] == "B"
    assert labels[1] == "X"


def test_gap_cost_gates_impossible_runs():
    s = Scene()
    s.add(0, range(0, 10), lambda f: (1.0, 2.0), 0.0)
    s.add(1, range(13, 20), lambda f: (1.3, 2.0), 0.0)  # 0.3 m in 0.1 s: plausible
    s.add(2, range(13, 20), lambda f: (5.0, 8.0), 0.0)  # 7 m in 0.1 s: not a player
    t = {tr.tid: tr for tr in s.tracklets()}
    assert gap_cost(t[0], t[1]) < 1.0
    assert gap_cost(t[0], t[2]) > 40.0


def test_colour_model_separates_shirts_and_keeps_a_as_the_blue_one():
    rng = np.random.default_rng(0)
    n = 200
    feat = np.zeros((2 * n, len(FEATURES)))
    blue, b, L, dark = (FEATURES.index(c) for c in ("frac_blue", "b", "L", "frac_dark"))
    feat[:n, blue], feat[n:, blue] = rng.normal(0.6, 0.1, n), rng.normal(0.05, 0.03, n)
    feat[:n, b], feat[n:, b] = rng.normal(-12, 3, n), rng.normal(0, 3, n)
    feat[:n, L], feat[n:, L] = rng.normal(65, 6, n), rng.normal(30, 6, n)
    feat[:n, dark], feat[n:, dark] = rng.normal(0.05, 0.03, n), rng.normal(0.7, 0.1, n)
    model = ColourModel.fit(feat[::-1])  # order must not decide which component is A
    llr = model.llr(feat)
    assert (llr[:n] > 0).mean() > 0.97
    assert (llr[n:] < 0).mean() > 0.97


def test_torso_behind_door_hardware_carries_little_colour_weight():
    """A hinge plate in front of the shirt makes the light-blue player read dark; such samples barely count."""
    import pandas as pd

    from squashvid.players.occluders import OccluderTrack
    from squashvid.players.run import colour_weights, torso_on_hardware

    mask = np.zeros((288, 512), bool)
    mask[100:160, 300:340] = True  # a plate at full-resolution x 600-680, y 200-320
    occ = OccluderTrack(np.array([0.0]), mask[None], scale=2.0)
    boxes = [(590.0, 180.0, 690.0, 460.0), (100.0, 180.0, 200.0, 460.0)]  # behind the plate, in the open
    cand = pd.DataFrame(
        {
            "t": [0.0, 0.0],
            "x1": [b[0] for b in boxes], "y1": [b[1] for b in boxes], "x2": [b[2] for b in boxes], "y2": [b[3] for b in boxes],
            "kpts": [np.zeros(34), np.zeros(34)],
            "kscore": [np.zeros(17), np.zeros(17)],  # no confident keypoints: the torso is the fixed box fraction
            "torso_kpts": [False, False],
        }
    )
    hw = torso_on_hardware(cand, occ)
    assert hw[0] > 0.5 and hw[1] == 0.0
    feat = np.zeros((2, len(FEATURES)))
    feat[:, FEATURES.index("n_px")] = 500
    w = colour_weights(cand, feat, np.zeros(2, bool), torso_hw=hw)
    assert w[0] < 0.2 * w[1]

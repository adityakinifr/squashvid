"""Pure parts of hit detection: the alternation DP and the per-hit arithmetic."""

import numpy as np
import pandas as pd

from squashvid.analysis import hits as H
from squashvid.analysis.io import Match, Track
from squashvid.geom.court import CourtSpec

FPS = 30.0


def match_with(a_xy, b_xy, seconds=10.0) -> Match:
    """A match whose players follow the given functions of time (seconds -> (x, y))."""
    n = int(seconds * FPS)
    t = np.arange(n) / FPS
    tracks = {}
    for k, f in (("A", a_xy), ("B", b_xy)):
        xy = np.array([f(x) for x in t])
        step = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)] * FPS
        tracks[k] = Track(x=xy[:, 0], y=xy[:, 1], sd=np.zeros(n), detected=np.ones(n, bool), speed=step, wrist=np.zeros(n))
    return Match(stem="synthetic", spec=CourtSpec(), t=t, fps=FPS, tracks=tracks, onsets=pd.DataFrame({"t": [], "kind": []}), audio_rallies=pd.DataFrame())


def hit_table(times, hitters) -> pd.DataFrame:
    return pd.DataFrame({"rally": 1, "shot_index": np.arange(1, len(times) + 1), "t": times, "frame": (np.array(times) * FPS).astype(int), "hitter": hitters, "confidence": 0.8, "source": "pose"})


def test_dp_keeps_alternating_chain_and_drops_a_spurious_double():
    # True shots every 1.3 s by A, B, A, B; a weak same-player candidate 0.2 s after one of them.
    t = np.array([1.0, 2.3, 2.5, 3.6, 4.9])
    player = np.array(["A", "B", "B", "A", "B"])
    reward = np.array([1.0, 1.0, 0.3, 1.0, 1.0])
    keep = H.select_hits(t, player, reward)
    assert list(t[keep]) == [1.0, 2.3, 3.6, 4.9]


def test_dp_respects_minimum_gap_whoever_hit():
    t = np.array([1.0, 1.2])
    keep = H.select_hits(t, np.array(["A", "B"]), np.array([2.0, 2.0]))
    assert len(keep) == 1


def test_dp_prefers_alternation_to_a_louder_wrong_hitter():
    # Candidates for both players at the same instants; the B candidate at 2.3 s is a bit weaker than A's
    # but alternation with A at 1.0 s and 3.6 s should make B the hitter.
    t = np.array([1.0, 2.3, 2.3, 3.6])
    player = np.array(["A", "A", "B", "A"])
    reward = np.array([1.0, 0.9, 0.8, 1.0])
    keep = H.select_hits(t, player, reward)
    assert [player[i] for i in keep] == ["A", "B", "A"]


def test_dp_returns_nothing_when_nothing_is_worth_keeping():
    assert H.select_hits(np.array([1.0, 3.0]), np.array(["A", "B"]), np.array([-1.0, -1.0])) == []


def test_dp_output_is_time_ordered_even_for_unsorted_input():
    t = np.array([3.6, 1.0, 2.3])
    keep = H.select_hits(t, np.array(["A", "A", "B"]), np.array([1.0, 1.0, 1.0]))
    assert list(t[keep]) == sorted(t[keep])


def test_available_time_and_distances_follow_from_positions():
    spec = CourtSpec()
    T = spec.t_point[:2]
    # A stands 2 m behind the T. B starts at the T and walks 3 m along the court in 3 s (1 m/s).
    m = match_with(lambda s: (T[0], T[1] - 2.0), lambda s: (T[0], T[1] - 1.0 * min(s, 3.0)), seconds=6.0)
    h = H.add_context(hit_table([1.0, 3.0, 5.0], ["A", "B", "A"]), m)
    # Shot 2 is B's reply to A, 2.0 s later; B walked 2 m between the two hits at 1 m/s.
    assert np.isclose(h.available_time[1], 2.0)
    assert np.isclose(h.returner_moved[1], 2.0, atol=0.15)
    assert np.isclose(h.returner_peak_speed[1], 1.0, atol=0.15)
    # At B's hit (t=3) B is 3 m behind the T and A is 2 m behind it.
    assert np.isclose(h.hitter_T_dist[1], 3.0, atol=0.05)
    assert np.isclose(h.opp_T_dist[1], 2.0, atol=0.05)
    assert np.isnan(h.available_time[0])  # the serve has no shot before it


def test_available_time_is_missing_when_the_same_player_hits_twice():
    m = match_with(lambda s: (3.0, 3.0), lambda s: (3.0, 5.0))
    h = H.add_context(hit_table([1.0, 2.5], ["A", "A"]), m)
    assert np.isnan(h.available_time[1])


def test_returned_from_cell_is_the_next_hitters_cell():
    # A hits from the back-left, B then hits from the front-right.
    m = match_with(lambda s: (0.5, 1.0), lambda s: (5.5, 7.0))
    h = H.add_context(hit_table([1.0, 2.5], ["A", "B"]), m)
    assert h.cell_hit_from[0] == "back-left"
    assert h.cell_returned_from[0] == h.cell_hit_from[1]
    assert pd.isna(h.cell_returned_from[1])


def test_recovery_is_censored_at_the_next_hit_when_the_player_never_returns():
    m = match_with(lambda s: (3.0, 0.5), lambda s: (3.0, 4.285))
    h = H.add_context(hit_table([1.0, 2.5], ["A", "B"]), m)
    assert not h.recovered[0]
    assert np.isclose(h.recovery_time[0], 1.5)


def test_more_time_never_makes_a_shot_a_defence():
    for moved in (0.5, 2.0, 4.0):
        for dT in (0.5, 2.0, 3.5):
            assert H.classify_situation(2.2, moved, dT) != "defence"
            assert H.classify_situation(np.nan, moved, dT) is None

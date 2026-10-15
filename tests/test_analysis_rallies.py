"""Where a break in the shot chain ends a rally."""

import numpy as np

from squashvid.analysis import rallies as R


def test_short_gaps_never_split():
    p = np.zeros(20)
    assert not R._chain_breaks(p, 5.0, 5.0 + R.CHAIN_GAP_S - 0.1)


def test_a_chain_gap_in_continuous_play_does_not_split():
    p = np.full(20, 0.95)  # play score high throughout: shots were missed, the rally went on
    assert not R._chain_breaks(p, 5.0, 9.0)


def test_a_chain_gap_with_a_pause_in_play_splits():
    p = np.full(20, 0.95)
    p[7] = R.SPLIT_P - 0.1
    assert R._chain_breaks(p, 5.0, 9.0)


def test_a_long_gap_always_splits():
    p = np.full(30, 0.95)
    assert R._chain_breaks(p, 5.0, 5.0 + R.ALWAYS_SPLIT_S + 0.5)


def _match_with_speeds(fast, slow, n=300, fps=30.0):
    import pandas as pd

    from squashvid.analysis.io import Match, Track
    from squashvid.geom.court import CourtSpec

    z = np.zeros(n)
    tracks = {k: Track(z + 3, z + 4, z, np.ones(n, bool), np.asarray(sp, float), z) for k, sp in (("A", fast), ("B", slow))}
    return Match("stub", CourtSpec(), np.arange(n) / fps, fps, tracks, pd.DataFrame(columns=["t", "kind"]), pd.DataFrame())


def test_trailing_shots_at_walking_pace_are_trimmed_but_earlier_ones_stay():
    n = 300  # 10 s; play for the first 6 s, then both players walk
    fast = np.where(np.arange(n) < 180, 2.5, 1.0)
    slow = np.where(np.arange(n) < 180, 1.2, 0.4)
    m = _match_with_speeds(fast, slow, n)
    ht = np.array([1.0, 2.5, 4.0, 5.0, 8.0, 9.0])
    kept = R.trim_tail(m, np.arange(len(ht)), ht)
    assert list(kept) == [0, 1, 2, 3]


def test_shots_in_play_are_never_trimmed():
    n = 300
    m = _match_with_speeds(np.full(n, 2.0), np.full(n, 1.0), n)
    ht = np.array([1.0, 3.0, 5.0, 7.0])
    assert list(R.trim_tail(m, np.arange(4), ht)) == [0, 1, 2, 3]


def test_unknown_speed_is_not_taken_for_walking():
    n = 300
    m = _match_with_speeds(np.full(n, np.nan), np.full(n, np.nan), n)
    assert not R.walking_pace(m, 5.0)

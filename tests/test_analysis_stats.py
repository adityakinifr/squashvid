"""The bootstrap interval over rallies and the suppression rule."""

import numpy as np
import pandas as pd

from squashvid.analysis import insights as I
from squashvid.analysis import stats as S


def test_bootstrap_interval_contains_the_value_and_is_reproducible():
    rng = np.random.default_rng(1)
    groups = [rng.normal(2.0, 0.5, 6) for _ in range(20)]
    a = S.bootstrap_ci(groups, seed=3)
    b = S.bootstrap_ci(groups, seed=3)
    assert a == b
    assert a["ci"][0] < a["value"] < a["ci"][1]
    assert a["n"] == 120 and a["n_rallies"] == 20


def test_bootstrap_interval_narrows_with_more_rallies():
    rng = np.random.default_rng(2)
    few = [rng.normal(0, 1, 5) for _ in range(8)]
    many = few + [rng.normal(0, 1, 5) for _ in range(72)]
    w = lambda g: np.subtract(*S.bootstrap_ci(g, seed=0)["ci"][::-1])  # noqa: E731
    assert w(many) < w(few)


def test_bootstrap_resamples_rallies_not_shots():
    # Two rallies, each internally constant. Resampling shots would give intermediate values; resampling rallies cannot.
    g = [np.full(50, 1.0), np.full(50, 3.0)]
    res = S.bootstrap_ci(g, n_boot=500, seed=0)
    assert res["ci"][0] >= 1.0 and res["ci"][1] <= 3.0
    draws = {round(float(np.mean(np.concatenate([g[i] for i in p]))), 6) for p in [(0, 0), (0, 1), (1, 1)]}
    assert draws == {1.0, 2.0, 3.0}


def test_bootstrap_with_a_single_rally_has_no_interval():
    assert S.bootstrap_ci([np.array([1.0, 2.0])])["ci"] is None


def test_paired_difference_is_zero_for_identical_players():
    groups = [np.array([1.0, 2.0, 3.0])] * 10
    d = S.bootstrap_diff(groups, groups, seed=0)
    assert d["value"] == 0 and d["ci"] == [0.0, 0.0]


def test_suppression_rule():
    ok = {"n_rallies": 12, "ci": [0.2, 0.9]}
    assert I.suppress(ok) is None
    assert I.suppress({**ok, "ci": [-0.1, 0.9]}) is not None  # crosses zero
    assert I.suppress({**ok, "ci": [-0.9, -0.2]}) is None  # negative but clear
    assert I.suppress({**ok, "n_rallies": 7}) is not None  # too few rallies
    assert I.suppress({**ok, "ci": None}) is not None
    assert I.suppress({**ok, "n_rallies": 7}, min_rallies=5) is None


def _stub_match(n=300, fps=30.0, seed=0):
    from squashvid.analysis.io import Match, Track
    from squashvid.geom.court import CourtSpec

    rng = np.random.default_rng(seed)
    spec = CourtSpec()
    tracks = {}
    for k in "AB":
        x, y = rng.uniform(0.5, spec.width - 0.5, n), rng.uniform(1.0, 8.0, n)
        x[rng.random(n) < 0.2] = np.nan  # frames where the player was not seen
        z = np.zeros(n)
        tracks[k] = Track(x, y, z, np.isfinite(x), z, z)
    return Match("stub", spec, np.arange(n) / fps, fps, tracks, pd.DataFrame(columns=["t", "kind"]), pd.DataFrame())


def test_findings_and_table_read_the_same_distance_from_T_samples():
    m = _stub_match()
    rallies = pd.DataFrame({"rally": [1, 2], "start_t": [0.5, 5.0], "end_t": [4.0, 9.5]})
    hits = pd.DataFrame({"rally": pd.Series(dtype=int), "t": pd.Series(dtype=float)})
    facts = I.Facts.from_match(m, hits, rallies)
    for k in "AB":
        table = S.dist_from_T_by_rally(m, k, rallies)
        assert all(np.isfinite(d).all() for d in table)  # an unseen frame is unknown, not counted as away from the T
        for rid, d in zip(rallies.rally, table):
            np.testing.assert_array_equal(facts.series[rid][k], d)


def test_impacts_below_the_tin_are_neither_near_the_tin_nor_between_tin_and_service_line():
    from squashvid.geom.court import CourtSpec

    s = CourtSpec()
    z = np.array([s.tin_top - 0.3, s.tin_top + 0.1, s.tin_top + S.NEAR_TIN_M + 0.05, s.service_z + 0.1, 3.0])
    assert list(S.near_tin(z, s)) == [False, True, False, False, False]
    assert list(S.front_wall_band(z, s)) == ["below_tin", "tin_to_service_line", "tin_to_service_line", "service_line_to_2.5m", "above_2.5m"]

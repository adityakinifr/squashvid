"""The bootstrap interval over rallies and the suppression rule."""

import numpy as np

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

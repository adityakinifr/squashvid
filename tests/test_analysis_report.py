"""The Markdown report's tables."""

from squashvid.analysis import report as R


def _share(v):
    return {"value": v, "n": 10, "n_rallies": 6, "ci": [v - 0.1, v + 0.1]}


def _stats(with_ball=True):
    player = {
        "shots": 40,
        "mean_dist_from_T_m": {"value": 1.8, "n": 5000, "n_rallies": 6, "ci": [1.6, 2.0]},
        "mean_dist_from_T_at_opponent_strike_m": {"value": 1.6, "n": 40, "n_rallies": 6, "ci": [1.4, 1.8]},
        "share_within_T_radius": _share(0.25),
        "share_in_front_of_short_line": _share(0.3),
        "available_time_s": {"median": 1.4, "n": 30, "share_under_1.1s": _share(0.2)},
        "recovery_to_T": {"share_back_on_T_before_next_hit": _share(0.5)},
        "distance_covered_in_play_m": {"value": 200.0, "n_rallies": 6},
    }
    ball = {
        "n_shots_with_front_wall": 10,
        "band": {"tin_to_service_line": _share(0.5), "service_line_to_2.5m": _share(0.2), "above_2.5m": _share(0.3)},
        "third": {"left": _share(0.3), "centre": _share(0.4), "right": _share(0.3)},
        "near_tin_share": _share(0.1),
    }
    return {"players": {"A": player, "B": player}, "params": {"t_radius_m": 1.0}, "ball": {"A": ball, "B": ball} if with_ball else None}


def test_per_frame_measures_cite_rallies_not_frames():
    rows = "\n".join(R.stats_table(_stats()))
    assert "n=5000" not in rows and "6 rallies" in rows


def test_ball_table_says_how_many_shots_the_shares_rest_on():
    rows = R.ball_table(_stats())
    assert "10 of 40" in rows[2]
    assert len(rows) == 2 + 8

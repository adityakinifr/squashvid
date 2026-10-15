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
        "band": {"below_tin": _share(0.05), "tin_to_service_line": _share(0.5), "service_line_to_2.5m": _share(0.2), "above_2.5m": _share(0.3)},
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
    assert len(rows) == 2 + 9


def test_how_rallies_ended_is_shown_only_when_winners_were_inferred():
    from squashvid.analysis.report import outcomes_section

    assert outcomes_section(None) == [] and outcomes_section({"n_rallies": 5, "n_decided": 0}) == []
    p = {"won": 3, "serving": [2, 4], "receiving": [1, 3], "short": [1, 3], "long": [2, 4], "ended_on_own_shot": [2, 5]}
    lines = outcomes_section({"n_rallies": 9, "n_decided": 7, "long_from_shots": 6, "players": {"A": p, "B": p}})
    text = "\n".join(lines)
    assert "## How rallies ended (inferred)" in text and "2 of 9 rallies have no winner" in text
    assert "| Won when serving | 2 of 4 | 2 of 4 |" in text and "(6 shots or more)" in text


def test_where_the_shots_went_is_shown_only_with_known_directions_and_says_how_many():
    from squashvid.analysis.report import placement_section

    zone = {"n": 3, "direction": {"straight": 2, "cross": 1}, "length": {"deep": 1, "mid": 0, "short": 1}}
    empty = {"n": 0, "direction": {"straight": 0, "cross": 0}, "length": {"deep": 0, "mid": 0, "short": 0}}
    p = {"n": 3, "zones": {"back": zone, "mid": empty, "front": empty}, "volley": {"volley": 0, "bounce": 3}, "tight": {"yes": 1, "no": 1}, "shot_type": {"drive": 3, "boast": 0, "drop": 0, "lob": 0, "kill": 0}}
    assert placement_section(None) == [] and placement_section({"n_shots": 4, "direction_known": 0, "players": {}}) == []
    text = "\n".join(placement_section({"n_shots": 6, "direction_known": 4, "players": {"A": p, "B": p}}))
    assert "known for 4 of 6 shots" in text
    assert "| From the back (behind the service boxes): straight / cross-court | 2 / 1 | 2 / 1 |" in text
    assert "| From mid-court: straight / cross-court | n/a | n/a |" in text

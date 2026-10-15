"""Who served each rally, and who won it, from the rule that the winner of a rally serves the next one."""

import numpy as np
import pandas as pd

from squashvid.analysis import serves as S
from tests.test_analysis_rallies import BACK_LEFT, IN_RIGHT_BOX, _serve_match


def rallies(*servers, gaps=None):
    """Rallies 10 s long, 5 s apart (or the given gaps), served as given: ("A", "R"), ...; None for unknown."""
    rows, t = [], 0.0
    for i, s in enumerate(servers):
        rows.append({"rally": i + 1, "start_t": t, "end_t": t + 10.0, "n_shots": 4, "server": s[0] if s else None, "server_side": s[1] if s else None})
        t += 10.0 + (gaps[i] if gaps else 5.0)
    return pd.DataFrame(rows)


def test_the_winner_is_whoever_serves_next():
    r = S.winners(rallies(("A", "R"), ("B", "L"), ("B", "R")))
    assert list(r.winner) == ["B", "B", None]
    assert list(r.winner_basis) == ["served_next", "served_next", "last_in_clip"]


def test_the_same_server_on_the_same_side_is_a_let_or_a_missed_rally():
    r = S.winners(rallies(("A", "R"), ("A", "R"), ("B", "L")))
    assert r.winner.iloc[0] is None and r.winner_basis.iloc[0] == "same_server_same_side"


def test_after_a_game_break_the_server_may_start_from_either_side():
    r = S.winners(rallies(("A", "R"), ("A", "R"), gaps=[S.GAME_BREAK_S + 5, 5.0]))
    assert r.winner.iloc[0] == "A" and r.winner_basis.iloc[0] == "served_again_after_break"


def test_no_winner_when_the_next_server_is_unknown_but_one_when_only_this_server_is():
    r = S.winners(rallies(("A", "R"), None, ("B", "L")))
    assert list(r.winner)[:2] == [None, "B"]
    assert r.winner_basis.iloc[0] == "next_server_unknown"


def test_a_confirmed_serve_names_the_server_and_the_side():
    m = _serve_match(IN_RIGHT_BOX, BACK_LEFT)
    g = pd.DataFrame({"t": [5.0, 6.0], "hitter": ["A", "B"], "serve": [True, False]})
    assert S.server_of(m, g, 4.4) == ("A", "R", "serve")


def test_a_missed_serve_is_found_from_the_setup_before_the_first_hit():
    a = np.where((np.arange(300) / 30.0 < 5.5)[:, None], IN_RIGHT_BOX, (3.2, 4.5))  # A leaves the box after serving
    m = _serve_match(a, BACK_LEFT)
    g = pd.DataFrame({"t": [6.2, 7.4], "hitter": ["B", "A"], "serve": [False, False]})  # the first detected hit is the return
    assert S.server_of(m, g, 5.6) == ("A", "R", "setup")


def test_no_setup_no_server():
    m = _serve_match((3.2, 4.5), BACK_LEFT)
    g = pd.DataFrame({"t": [6.2, 7.4], "hitter": ["B", "A"], "serve": [False, False]})
    assert S.server_of(m, g, 5.6) == (None, None, "unknown")


def test_outcomes_count_serving_receiving_and_how_rallies_that_ended_on_a_shot_went():
    r = S.winners(rallies(("A", "R"), ("B", "L"), ("B", "R"), ("A", "L")))  # winners B, B, A, none
    hits = pd.DataFrame({"rally": [1, 1, 2, 2, 3, 4], "t": [1, 2, 16, 17, 31, 46], "hitter": ["A", "B", "B", "A", "A", "B"]})
    out = S.outcomes(r, hits)
    assert out["n_decided"] == 3 and out["players"]["B"]["won"] == 2
    assert out["players"]["A"]["serving"] == [0, 1] and out["players"]["B"]["serving"] == [1, 2]
    assert out["players"]["A"]["receiving"] == [1, 2]
    # Rally 1 ended on B's shot (won by B), rally 2 on A's shot (won by B), rally 3 on A's shot (won by A).
    assert out["players"]["A"]["ended_on_own_shot"] == [1, 2] and out["players"]["B"]["ended_on_own_shot"] == [1, 1]

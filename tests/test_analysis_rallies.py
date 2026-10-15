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


def test_impacts_of_missed_shots_bridge_a_gap_while_play_goes_on():
    p = np.full(20, 0.6)  # below SPLIT_P: without the impacts this gap would end the rally
    links = np.array([6.5, 8.0])
    assert R._chain_breaks(p, 5.0, 9.0)
    assert not R._chain_breaks(p, 5.0, 9.0, links=links)


def test_an_impact_after_play_stops_bridges_nothing():
    p = np.full(20, R.LINK_MIN_P - 0.1)
    assert R._chain_breaks(p, 5.0, 9.0, links=np.array([5.3, 7.0]))


def test_a_trim_never_cuts_below_the_floor():
    n = 300
    m = _match_with_speeds(np.full(n, 1.0), np.full(n, 0.4), n)  # walking pace throughout
    ht = np.array([1.0, 2.0, 3.0, 4.0])
    assert list(R.trim_tail(m, np.arange(4), ht)) == []
    assert list(R.trim_tail(m, np.arange(4), ht, keep=2)) == [0, 1]


def _serve_match(a_xy, b_xy, front=(), side=(), a_speed=0.3, n=300, fps=30.0):
    """A stub match on a WSF court: per-frame positions (or one fixed position) and audio impacts."""
    import pandas as pd

    from squashvid.analysis.io import Match, Track
    from squashvid.geom.court import CourtSpec

    def track(xy, speed):
        xy = np.broadcast_to(np.asarray(xy, float), (n, 2))
        return Track(xy[:, 0].copy(), xy[:, 1].copy(), np.zeros(n), np.ones(n, bool), np.broadcast_to(np.asarray(speed, float), n).copy(), np.zeros(n))

    onsets = pd.DataFrame({"t": [*front, *side], "kind": ["front_wall"] * len(front) + ["side_wall_left"] * len(side)})
    return Match("stub", CourtSpec(), np.arange(n) / fps, fps, {"A": track(a_xy, a_speed), "B": track(b_xy, 0.2)}, onsets, pd.DataFrame())


IN_RIGHT_BOX, BACK_LEFT, BACK_RIGHT = (5.4, 3.9), (1.5, 1.3), (5.0, 1.3)


def test_a_serve_is_struck_from_a_box_with_the_receiver_behind_on_the_other_half():
    assert R.serve_shaped(_serve_match(IN_RIGHT_BOX, BACK_LEFT), 5.0, "A")
    assert not R.serve_shaped(_serve_match(IN_RIGHT_BOX, BACK_RIGHT), 5.0, "A")  # receiver on the same half
    assert not R.serve_shaped(_serve_match(BACK_RIGHT, BACK_LEFT), 5.0, "A")  # hitter at the back


def test_the_server_may_step_into_the_serve_but_waits_still_before_it():
    m = _serve_match(IN_RIGHT_BOX, BACK_LEFT, a_speed=1.5)
    assert not R.serve_shaped(m, 5.0, "A")
    assert R.serve_shaped(m, 5.0, "A", still=False)


def test_the_serve_is_the_last_touch_before_the_front_wall_not_the_bounces_before_it():
    m = _serve_match(IN_RIGHT_BOX, BACK_LEFT, front=[2.3])
    assert list(R.serves(m, np.array([1.0, 2.0, 3.0]), np.array(["A", "A", "B"]))) == [False, True, False]


def test_a_touch_that_never_reaches_the_front_wall_is_no_serve():
    m = _serve_match(IN_RIGHT_BOX, BACK_LEFT)
    assert not R.serves(m, np.array([1.0, 2.0, 3.0]), np.array(["A", "A", "B"])).any()


def test_a_ball_reaching_a_side_wall_at_the_back_confirms_an_unseen_serve():
    m = _serve_match(IN_RIGHT_BOX, BACK_LEFT, side=[3.0])
    assert list(R.serves(m, np.array([2.0, 3.5]), np.array(["A", "B"]))) == [True, False]


def test_a_shot_after_the_return_is_not_taken_for_the_serve():
    a = np.where((np.arange(300) / 30.0 < 2.5)[:, None], IN_RIGHT_BOX, BACK_LEFT)  # A leaves the box after serving
    m = _serve_match(a, BACK_LEFT, front=[3.6])
    assert not R.serves(m, np.array([1.0, 2.2, 3.4]), np.array(["A", "B", "A"])).any()


def test_two_front_wall_impacts_between_one_players_hits_mean_the_opponent_hit_unseen():
    t, who = np.array([1.0, 4.0]), np.array(["B", "B"])
    assert R.unseen_reply(np.array([1.3, 3.2]), t, who)
    assert not R.unseen_reply(np.array([1.3]), t, who)  # solo practice: one impact per shot
    assert not R.unseen_reply(np.array([1.3, 1.5]), t, who)  # one impact heard twice

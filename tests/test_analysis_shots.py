"""Per-shot facts and rally endings from synthetic ball events. Nothing here touches work/."""

import numpy as np
import pandas as pd
import pytest

from squashvid.analysis import endings as E
from squashvid.analysis import shots as S
from squashvid.geom.court import CourtSpec

SPEC = CourtSpec()
L = SPEC.length


def ev(*rows):
    """rows: (t, kind, x, y, z[, evidence[, player]])"""
    out = []
    for r in rows:
        t, kind, x, y, z = r[:5]
        out.append({"t": t, "kind": kind, "x": x, "y": y, "z": z, "evidence": r[5] if len(r) > 5 else "track", "confidence": 0.5, "player": r[6] if len(r) > 6 else None})
    return pd.DataFrame(out, columns=["t", "kind", "x", "y", "z", "evidence", "confidence", "player"])


def fw(t, x, z, evidence="track"):
    return (t, "front_wall", x, L, z, evidence)


def fl(t, x, y):
    return (t, "floor", x, y, 0.02)


def hits(*rows):
    """rows: (t, hitter, hit_x, hit_y)"""
    return pd.DataFrame(
        [{"rally": 1, "shot_index": i + 1, "t": t, "hitter": h, "hitter_x": x, "hitter_y": y, "confidence": 0.8, "source": "pose", "cell_hit_from": "mid-centre"} for i, (t, h, x, y) in enumerate(rows)]
    )


def rallies(end_t):
    return pd.DataFrame([{"rally": 1, "start_t": 0.0, "end_t": end_t, "n_shots": 2}])


def trace(events, t_hit=10.0, t_end=12.0, hit_y=3.0):
    return S.trace_shot(SPEC, events, t_hit, t_end, hit_y)


def shot(events, hit_x=1.5, hit_y=3.0):
    r = trace(events, hit_y=hit_y)
    return S.classify(SPEC, r), r


# ---------- trace ----------


def test_audio_only_front_wall_is_seen_but_not_placed():
    r = trace(ev((10.4, "front_wall", np.nan, L, np.nan, "audio")))
    assert r["fw_seen"] and not r["fw_placed"] and np.isnan(r["fw_z"]) and r["fw_band"] == S.UNKNOWN


def test_placed_front_wall_takes_position_and_band():
    r = trace(ev(fw(10.4, 2.0, 1.2)))
    assert r["fw_placed"] and r["fw_x"] == 2.0 and r["fw_band"] == "tin_to_service_line" or r["fw_band"] == "service_line_to_2.5m"


def test_cluster_prefers_the_placed_event_over_the_audio_one():
    r = trace(ev((10.40, "front_wall", np.nan, L, np.nan, "audio"), fw(10.55, 3.0, 1.0)))
    assert r["fw_placed"] and r["fw_x"] == 3.0


def test_side_wall_before_front_wall_is_reported_and_none_seen_otherwise():
    r = trace(ev((10.2, "side_wall_left", 0.0, 4.5, 1.5), fw(10.5, 3.0, 1.0)))
    assert r["side_first"] == "left"
    assert trace(ev(fw(10.5, 3.0, 1.0)))["side_first"] == "none_seen"
    assert trace(ev())["side_first"] == S.UNKNOWN  # no front wall either: nothing to say


def test_side_wall_behind_the_hitter_does_not_count():
    r = trace(ev((10.2, "side_wall_left", 0.0, 1.0, 1.5), fw(10.5, 3.0, 1.0)), hit_y=3.0)
    assert r["side_first"] == "none_seen"


def test_front_wall_event_after_a_floor_bounce_is_discarded():
    r = trace(ev(fl(10.6, 3.0, 5.0), fw(11.5, 3.0, 0.1)))
    assert r["fw_after_floor"] and not r["fw_seen"] and r["bounce_y"] == 5.0


def test_bounce_and_second_bounce_and_back_wall():
    r = trace(ev(fw(10.4, 3.0, 1.2), fl(10.9, 3.0, 2.0), fl(11.4, 3.0, 1.0), (11.6, "back_wall", 3.0, 0.0, 1.0)))
    assert r["bounce_known"] and r["bounce_y"] == 2.0 and r["second_bounce"] == "yes" and r["back_wall"] == "yes"
    r = trace(ev(fw(10.4, 3.0, 1.2), fl(10.9, 3.0, 2.0)))
    assert r["second_bounce"] == "none_seen"


def test_two_floor_events_within_the_dedupe_gap_are_one_bounce():
    r = trace(ev(fw(10.4, 3.0, 1.2), fl(10.9, 3.0, 2.0), fl(10.95, 3.0, 2.1)))
    assert r["second_bounce"] == "none_seen"


def test_nothing_known_without_events():
    r = trace(ev())
    assert not r["fw_seen"] and not r["bounce_known"] and r["second_bounce"] == S.UNKNOWN and np.isnan(r["flight_speed"])


def test_implausibly_short_or_fast_flight_gives_no_speed():
    assert np.isnan(trace(ev(fw(10.1, 3.0, 1.0)), hit_y=3.0)["flight_speed"])  # 0.1 s
    assert np.isnan(trace(ev(fw(10.22, 3.0, 1.0)), hit_y=0.5)["flight_speed"])  # >40 m/s
    assert trace(ev(fw(10.6, 3.0, 1.0)), hit_y=3.0)["flight_speed"] == pytest.approx((L - 3.0) / 0.6)


# ---------- shot type rules ----------


def test_boast_when_side_wall_comes_first():
    (st, rule), _ = shot(ev((10.2, "side_wall_right", SPEC.width, 4.5, 1.5), fw(10.5, 3.0, 1.0)))
    assert (st, rule) == ("boast", "side_wall_then_front")


def test_boast_without_front_wall_needs_a_close_side_wall_soon_after_the_hit():
    (st, rule), _ = shot(ev((10.3, "side_wall_right", SPEC.width, 5.0, 1.5)))
    assert (st, rule) == ("boast", "side_wall_soon_after_hit")
    (st, _), _ = shot(ev((10.9, "side_wall_right", SPEC.width, 5.0, 1.5)))  # too late: a rebound
    assert st == S.UNKNOWN
    (st, _), _ = shot(ev((10.3, "side_wall_right", SPEC.width, 9.0, 1.5)))  # too far ahead: a front-corner rebound
    assert st == S.UNKNOWN


def test_lob_by_front_wall_height():
    (st, rule), _ = shot(ev(fw(10.8, 3.0, S.LOB_MIN_Z + 0.2)))
    assert (st, rule) == ("lob", "front_wall_high")


def test_kill_is_low_and_fast():
    (st, _), r = shot(ev(fw(10.4, 3.0, 0.7), fl(10.8, 3.0, 8.0)))
    assert r["flight_speed"] >= S.KILL_MIN_SPEED and st == "kill"


def test_drop_is_soft_and_lands_short():
    t_fw = 10.0 + (L - 3.0) / 8.0  # 8 m/s
    (st, _), r = shot(ev(fw(t_fw, 3.0, 0.8), fl(t_fw + 0.3, 3.0, SPEC.short_y + 1.5)))
    assert st == "drop" and r["flight_speed"] < S.DROP_MAX_SPEED


def test_soft_short_without_wall_height_is_still_a_drop():
    t_fw = 10.0 + (L - 3.0) / 8.0
    (st, rule), _ = shot(ev((t_fw, "front_wall", np.nan, L, np.nan, "audio"), fl(t_fw + 0.3, 3.0, SPEC.short_y + 1.5)))
    assert (st, rule) == ("drop", "soft_short")


def test_drive_is_mid_height_even_when_it_lands_short():
    (st, _), _ = shot(ev(fw(10.5, 3.0, 1.5), fl(10.9, 3.0, SPEC.short_y + 1.0)))
    assert st == "drive"
    (st, _), _ = shot(ev(fw(10.5, 3.0, 1.5), fl(10.9, 3.0, 1.0)))
    assert st == "drive"


def test_below_tin_is_never_a_kill():
    (st, rule), _ = shot(ev(fw(10.5, 3.0, SPEC.tin_top - 0.2)))
    assert (st, rule) == (S.UNKNOWN, "front_wall_below_tin_top")


def test_low_wall_with_ambiguous_pace_is_unknown():
    t_fw = 10.0 + (L - 3.0) / 12.0  # between the drop and kill thresholds
    (st, rule), _ = shot(ev(fw(t_fw, 3.0, 0.7)))
    assert (st, rule) == (S.UNKNOWN, "low_front_wall_ambiguous_pace")


def test_no_height_deep_and_fast_is_a_drive_and_deep_and_slow_a_lob():
    t_fast = 10.0 + (L - 3.0) / 25.0
    (st, rule), _ = shot(ev((t_fast, "front_wall", np.nan, L, np.nan, "audio"), fl(t_fast + 0.6, 3.0, 1.0)))
    assert (st, rule) == ("drive", "deep_fast_no_height")
    t_slow = 10.0 + (L - 3.0) / 6.0
    (st, rule), _ = shot(ev((t_slow, "front_wall", np.nan, L, np.nan, "audio"), fl(t_slow + 0.6, 3.0, 1.0)))
    assert (st, rule) == ("lob", "deep_slow_no_height")


def test_unknown_without_any_ball_evidence():
    (st, rule), _ = shot(ev())
    assert (st, rule) == (S.UNKNOWN, "no_front_wall_height")


# ---------- direction, length, tight ----------


def test_direction_straight_cross_and_near_centre():
    h = SPEC.half_x
    assert S.direction_of(SPEC, 1.0, 0.8, np.nan) == ("straight", "bounce")
    assert S.direction_of(SPEC, 1.0, SPEC.width - 1.0, np.nan) == ("cross", "bounce")
    assert S.direction_of(SPEC, 1.0, np.nan, SPEC.width - 1.0) == ("cross", "front_wall")
    assert S.direction_of(SPEC, h + 0.1, 1.0, np.nan)[0] == S.UNKNOWN  # hit from the middle line
    assert S.direction_of(SPEC, 1.0, h + 0.1, np.nan)[0] == S.UNKNOWN  # lands on the middle line
    assert S.direction_of(SPEC, 1.0, np.nan, np.nan) == (S.UNKNOWN, "no_target")


def test_length_uses_the_spec_lines():
    assert S.length_of(SPEC, SPEC.box_back_y - 0.1) == "deep"
    assert S.length_of(SPEC, (SPEC.box_back_y + SPEC.short_y) / 2) == "mid"
    assert S.length_of(SPEC, SPEC.short_y + 0.1) == "short"
    assert S.length_of(SPEC, np.nan) == S.UNKNOWN


def test_length_follows_a_changed_spec():
    narrow = SPEC.with_(short_line_near_edge=SPEC.short_line_near_edge - 1.0)
    y = SPEC.short_y - 0.5
    assert S.length_of(SPEC, y) == "mid" and S.length_of(narrow, y) == "short"


def test_tight_from_bounce_or_side_wall_after_the_front():
    assert S.tight_of(SPEC, 0.4, False) == ("yes", "bounce")
    assert S.tight_of(SPEC, SPEC.width - 0.4, False) == ("yes", "bounce")
    assert S.tight_of(SPEC, 2.0, False) == ("no", "bounce")
    assert S.tight_of(SPEC, np.nan, True) == ("yes", "side_wall_event")
    assert S.tight_of(SPEC, np.nan, False)[0] == S.UNKNOWN


# ---------- volley ----------


def b3(z, t=10.0):
    return pd.DataFrame({"t": [t - 0.05, t, t + 0.02], "x": 3.0, "y": 4.0, "z": z})


def test_volley_fact_floor_event_in_gap_means_bounce():
    events = ev(fl(9.3, 3.0, 5.0))
    assert S.volley_fact(SPEC, events, b3(1.4), 8.5, 10.0)[:2] == ("bounce", "floor_event")


def test_volley_fact_uses_height_when_no_floor_event():
    assert S.volley_fact(SPEC, ev(), b3(1.4), 8.5, 10.0)[:2] == ("volley", "height")
    assert S.volley_fact(SPEC, ev(), b3(0.2), 8.5, 10.0)[:2] == ("bounce", "height")
    assert S.volley_fact(SPEC, ev(), b3(0.7), 8.5, 10.0)[0] == S.UNKNOWN
    assert S.volley_fact(SPEC, ev(), None, 8.5, 10.0)[:2] == (S.UNKNOWN, "no_evidence")


def test_first_shot_of_a_rally_is_a_serve():
    assert S.volley_fact(SPEC, ev(), None, None, 10.0)[0] == "serve"


# ---------- tables ----------


def two_shot_rally(last_events):
    h = hits((10.0, "A", 1.5, 3.0), (11.5, "B", 3.5, 4.0))
    events = ev(fw(10.4, 2.0, 1.2), fl(10.9, 2.0, 2.0), fl(11.2, 2.0, 3.0), *last_events)
    return h, events


def test_shot_table_has_one_row_per_hit_and_known_columns():
    h, events = two_shot_rally([fw(11.9, 3.0, 1.5)])
    t = S.shot_table("X", SPEC, h, rallies(12.4), events)
    assert list(t.columns) == S.COLUMNS and len(t) == 2
    assert t.loc[0, "rally_first"] and t.loc[0, "volley"] == "serve"
    assert t.loc[0, "shot_type"] == "drive" and t.loc[0, "length"] == "deep"
    assert t.loc[1, "volley"] == "bounce"  # a floor event sits between the first front wall and this hit


def test_shot_table_without_ball_events_is_all_unknown_but_complete():
    h = hits((10.0, "A", 1.5, 3.0), (11.5, "B", 3.5, 4.0))
    t = S.shot_table("X", SPEC, h, rallies(12.4), None)
    assert len(t) == 2 and (t.shot_type == S.UNKNOWN).all() and not t.fw_seen.any()
    assert (t.n_known == 0).all()


def test_coverage_counts_known_shares_and_type_mix():
    h, events = two_shot_rally([fw(11.9, 3.0, 1.5)])
    c = S.coverage(S.shot_table("X", SPEC, h, rallies(12.4), events))
    assert c["coverage"]["n_shots"] == 2 and 0 < c["coverage"]["shot_type"] <= 1
    assert sum(c["shot_type_mix"]["A"].values()) == 1


def test_run_writes_both_tables_under_the_given_directory(tmp_path, monkeypatch):
    h, events = two_shot_rally([fw(11.9, 3.0, 1.5)])
    monkeypatch.setattr(S, "load_inputs", lambda stem: (SPEC, h, rallies(12.4), events, None))
    monkeypatch.setattr(S, "out_dir", lambda stem: tmp_path)
    shots, ends = S.run("X")
    assert (tmp_path / "shots.parquet").exists() and (tmp_path / "endings.parquet").exists()
    assert list(pd.read_parquet(tmp_path / "shots.parquet").columns) == S.COLUMNS
    assert list(pd.read_parquet(tmp_path / "endings.parquet").columns) == E.COLUMNS
    assert len(shots) == 2 and len(ends) == 1


# ---------- endings ----------


def ending(last_events, last_hitter="B", **kw):
    """A two-shot rally whose last shot (by `last_hitter`) is followed by `last_events`."""
    first = "A" if last_hitter == "B" else "B"
    h = hits((10.0, first, 1.5, 3.0), (11.5, last_hitter, 3.5, 4.0))
    events = ev(fw(10.4, 2.0, 1.2), fl(10.9, 2.0, 2.0), *last_events)
    shots = S.shot_table("X", SPEC, h, rallies(12.4), events, kw.get("b3"))
    return E.ending_table("X", SPEC, shots, rallies(12.4), events, kw.get("b3")).iloc[0]


def test_tin_when_the_last_front_wall_impact_is_below_the_tin_top():
    r = ending([fw(11.9, 3.0, SPEC.tin_top - 0.2)])
    assert r.ending == "tin" and r.inferred_winner == "A" and r.last_hitter == "B" and r.confidence > 0.5


def test_tin_with_a_ball_that_came_back_is_unclear():
    r = ending([fw(11.9, 3.0, SPEC.tin_top - 0.2), fl(12.4, 3.0, 4.0)])
    assert r.ending == "unclear" and r.inferred_winner == "none" and r.confidence == 0


def test_tin_with_the_ball_flying_back_in_ball_3d_is_unclear():
    flown = pd.DataFrame({"t": [12.1, 12.2, 12.3, 12.4], "x": 3.0, "y": 6.0, "z": 1.0})
    r = ending([fw(11.9, 3.0, SPEC.tin_top - 0.2)], b3=flown)
    assert r.ending == "unclear"


@pytest.mark.parametrize(
    "event,kind",
    [
        (fw(11.9, 3.0, SPEC.front_out_z + 0.4), "front"),
        ((11.9, "side_wall_left", 0.0, 2.0, SPEC.side_out_z(2.0) + 0.4), "side"),
        ((11.9, "back_wall", 3.0, 0.0, SPEC.back_out_z + 0.4), "back"),
    ],
)
def test_out_over_each_out_line(event, kind):
    r = ending([event])
    assert r.ending == "out" and r.out_kind == kind and r.inferred_winner == "A"


def test_side_wall_below_its_out_line_is_not_out():
    r = ending([(11.9, "side_wall_left", 0.0, 2.0, SPEC.side_out_z(2.0) - 0.3), fw(12.0, 3.0, 1.5)])
    assert r.ending != "out"


def test_not_returned_needs_a_double_bounce_and_credits_the_last_hitter():
    r = ending([fw(11.9, 3.0, 1.5), fl(12.3, 3.0, 4.0), fl(12.8, 3.0, 5.0)])
    assert r.ending == "not_returned" and r.inferred_winner == "B" and r.double_bounce and r.confidence >= 0.7


def test_one_bounce_is_not_enough():
    r = ending([fw(11.9, 3.0, 1.5), fl(12.3, 3.0, 4.0)])
    assert r.ending == "unclear" and "one_bounce_only" in r.evidence
    r = ending([fw(11.9, 3.0, 1.5)])
    assert r.ending == "unclear" and "no_floor_event_seen" in r.evidence


def test_audio_only_front_wall_is_not_returned_only_with_a_double_bounce():
    audio = (11.9, "front_wall", np.nan, L, np.nan, "audio")
    assert ending([audio]).ending == "unclear"
    r = ending([audio, fl(12.3, 3.0, 4.0), fl(12.8, 3.0, 5.0)])
    assert r.ending == "not_returned" and r.confidence < 0.5


def test_no_front_wall_after_the_last_hit_is_unclear():
    r = ending([])
    assert r.ending == "unclear" and r.inferred_winner == "none" and r.evidence == "no_front_wall_event_after_last_hit"


def test_an_opponent_racket_event_after_the_last_hit_is_unclear():
    r = ending([fw(11.9, 3.0, 1.5), fl(12.3, 3.0, 4.0), fl(12.8, 3.0, 5.0), (12.1, "racket", 3.0, 4.0, 0.5, "track", "A")])
    assert r.ending == "unclear" and r.racket_after


def test_a_front_wall_event_after_a_floor_bounce_does_not_make_a_tin():
    r = ending([fl(11.8, 3.0, 5.0), fw(12.2, 3.0, 0.1)])
    assert r.ending == "unclear"


def test_shot_before_last_is_carried_into_the_ending_row():
    r = ending([fw(11.9, 3.0, 1.5)])
    assert r.shot_before_last_index == 1 and r.shot_before_last_hitter == "A" and r.shot_before_last_type == "drive"


def test_no_ball_events_means_every_rally_is_unclear():
    h = hits((10.0, "A", 1.5, 3.0), (11.5, "B", 3.5, 4.0))
    shots = S.shot_table("X", SPEC, h, rallies(12.4), None)
    e = E.ending_table("X", SPEC, shots, rallies(12.4), None)
    assert e.ending.tolist() == ["unclear"] and e.inferred_winner.tolist() == ["none"]


def test_placement_counts_known_facts_by_where_the_shot_was_hit_from_and_leaves_out_serves():
    rows = [
        {"rally_first": True, "hitter": "A", "cell_hit_from": "mid-left", "direction": "straight", "length": "deep", "volley": "serve", "tight": "no", "shot_type": "drive"},
        {"rally_first": False, "hitter": "B", "cell_hit_from": "back-left", "direction": "cross", "length": "short", "volley": "bounce", "tight": "yes", "shot_type": "drive"},
        {"rally_first": False, "hitter": "B", "cell_hit_from": "back-right", "direction": "unknown", "length": "unknown", "volley": "unknown", "tight": "unknown", "shot_type": "unknown"},
        {"rally_first": False, "hitter": "A", "cell_hit_from": None, "direction": "straight", "length": "mid", "volley": "volley", "tight": "no", "shot_type": "boast"},
    ]
    pl = S.placement(pd.DataFrame(rows))
    assert pl["n_shots"] == 3 and pl["direction_known"] == 2
    b = pl["players"]["B"]
    assert b["n"] == 2 and b["zones"]["back"]["n"] == 2
    assert b["zones"]["back"]["direction"] == {"straight": 0, "cross": 1} and b["zones"]["back"]["length"] == {"deep": 0, "mid": 0, "short": 1}
    a = pl["players"]["A"]
    assert sum(z["n"] for z in a["zones"].values()) == 0 and a["shot_type"]["boast"] == 1  # no cell: counted overall, in no zone

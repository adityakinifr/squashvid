"""Coaching: choosing what to say, verifying it, and filling in the players' names."""

from squashvid.analysis import coach as C
from squashvid.analysis.insights import verify_sentence


def cand(i: int, metric: str, favourable: bool, player: str = "A", generator: str = "vs_opponent", diff: float = 0.12, ci=(0.06, 0.18), **display) -> dict:
    """A candidate whose display numbers agree with each other (percent metrics are scaled by 100)."""
    d = {"value": "17", "comparison": "29", "diff": "12", "diff_lo": "6", "diff_hi": "18", "n": "158", "n_rallies": "17", **display}
    return {
        "id": f"c{i}",
        "player": player,
        "generator": generator,
        "metric": metric,
        "diff": diff if favourable else -diff,
        "ci": list(ci) if favourable else [-ci[1], -ci[0]],
        "n_label": "shots",
        "favourable": favourable,
        "display": d,
        "evidence": [{"clip": "IMG_4091", "t0": 20.0 + i, "t1": 26.0 + i, "label": f"rally {i}: {{{player}}} was away from the T"}],
    }


def test_every_reason_and_detail_passes_the_verifier():
    cands = [cand(1, "t_share", False), cand(2, "deep_bounce", True), cand(3, "t_share", True, player="B")]
    out, rejected = C.write(cands)
    by_id = {c["id"]: c for c in cands}
    assert rejected == []
    items = [it for p in out for kind in ("try", "keep") for it in out[p][kind]]
    assert items
    for it in items:
        for text in it["why"] + it["detail"]:
            assert verify_sentence(text, it["cites"], by_id)[0], text


def test_a_sentence_with_a_number_no_candidate_holds_is_dropped():
    bad = cand(1, "t_share", False, value="9.9")
    by_id = {"c1": bad}
    it = C.item(bad, "try", [bad])
    it = {**it, "detail": it["detail"] + ["{A} was 4.2 m from the T."]}
    rejected: list[dict] = []
    kept = C._verified(it, by_id, rejected, "A")
    assert kept is not None and "{A} was 4.2 m from the T." not in kept["detail"]
    assert [r["text"] for r in rejected] == ["{A} was 4.2 m from the T."]


def test_at_most_three_to_try_and_two_to_keep_one_per_theme():
    cands = [
        cand(1, "t_share", False), cand(2, "t_overall", False), cand(3, "time_given", False, diff=0.14, ci=(0.02, 0.2)),
        cand(4, "deep_bounce", False), cand(5, "t_at_strike", False, generator="early_vs_late"),
        cand(6, "deep_bounce", True, player="A"),
    ]
    picked = C.choose(cands, "A")
    assert len(picked["try"]) == C.MAX_TRY and len(picked["keep"]) <= C.MAX_KEEP
    themes = [C.theme(c) for c in picked["try"]]
    assert len(themes) == len(set(themes))


def test_never_praise_and_criticise_the_same_theme():
    picked = C.choose([cand(1, "t_share", False, diff=0.2), cand(2, "t_overall", True)], "A")
    assert [C.theme(c) for k in ("try", "keep") for c in picked[k]] == ["position"]


def test_a_narrow_thing_to_try_may_sit_beside_a_broad_one_to_keep():
    picked = C.choose([cand(1, "t_share", True, diff=0.2), cand(2, "recover_from_back", False)], "A")
    assert [c["metric"] for c in picked["keep"]] == ["t_share"]
    assert [c["metric"] for c in picked["try"]] == ["recover_from_back"]


def test_a_narrow_thing_to_keep_does_not_excuse_a_broad_one_to_try():
    picked = C.choose([cand(1, "recover_from_back", True, diff=0.2), cand(2, "t_share", False)], "A")
    assert len(picked["keep"]) + len(picked["try"]) == 1


def test_the_why_line_of_a_position_item_says_what_being_off_the_T_cost():
    main = cand(1, "t_share", False)
    cost = {**cand(2, "off_T_cost", False, value="83", comparison="50", far_m="2"), "role": "why"}
    it = C.item(main, "try", [main, cost])
    assert it["cites"][:2] == ["c1", "c2"] and len(it["why"]) == 2
    assert "rushed 83% of the time" in it["why"][1]
    assert C.choose([main, cost], "A")["try"] == [main]  # the cost line never stands as an item of its own


def test_fill_swaps_tokens_for_names_and_leaves_other_text():
    assert C.fill("{A} hit; {B} replied. A and B stay.", {"A": "Deepti", "B": "Maya"}) == "Deepti hit; Maya replied. A and B stay."


def test_clear_only_when_the_whole_interval_clears_the_smallest_gap_worth_coaching():
    assert C.confidence(cand(1, "t_share", True, ci=(0.1, 0.2))) == "clear"  # min_effect 0.08
    assert C.confidence(cand(1, "t_share", True, ci=(0.05, 0.2))) == "likely"

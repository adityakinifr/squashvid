"""Verifier, banned phrases, number display and candidate suppression."""

import pytest

from squashvid.analysis import insights as I

CAND = {
    "id": "c1",
    "player": "A",
    "generator": "vs_opponent",
    "metric": "t_at_strike",
    "title": "how far from the T you are when your opponent hits",
    "comparison": "vs opponent",
    "diff": -0.4,
    "ci": [-0.7, -0.1],
    "n": 46,
    "n_label": "shots",
    "n_rallies": 12,
    "favourable": True,
    "display": {"value": "1.6", "comparison": "2.0", "diff": "0.4", "diff_lo": "0.1", "diff_hi": "0.7", "n": "46", "n_rallies": "12"},
    "evidence": [{"clip": "IMG_4091", "t0": 112.0, "t1": 115.0, "label": "rally 7: A was 1.2 m from the T when B hit"}],
    "suppressed": None,
}
BY_ID = {"c1": CAND}


def test_verifier_accepts_numbers_from_the_cited_candidate():
    ok, why = I.verify_sentence("A was 1.6 m from the T, against 2.0 m, over 12 rallies. Watch 1:52 to 1:55.", ["c1"], BY_ID)
    assert ok, why


def test_verifier_rejects_a_number_that_is_not_in_the_candidate():
    ok, why = I.verify_sentence("A was 1.8 m from the T.", ["c1"], BY_ID)
    assert not ok and "1.8" in why


def test_verifier_rejects_a_timestamp_that_is_not_in_the_evidence():
    ok, _ = I.verify_sentence("Watch 3:10 to 3:13.", ["c1"], BY_ID)
    assert not ok


def test_verifier_rejects_a_sentence_with_no_citation_or_an_unknown_one():
    assert not I.verify_sentence("A was close to the T.", [], BY_ID)[0]
    assert not I.verify_sentence("A was close to the T.", ["c9"], BY_ID)[0]


def test_a_number_cited_from_a_different_candidate_does_not_count():
    other = {**CAND, "id": "c2", "display": {**CAND["display"], "value": "9.9"}, "evidence": []}
    ok, _ = I.verify_sentence("A was 9.9 m away.", ["c1"], {"c1": CAND, "c2": other})
    assert not ok
    ok, _ = I.verify_sentence("A was 9.9 m away.", ["c1", "c2"], {"c1": CAND, "c2": other})
    assert ok


@pytest.mark.parametrize(
    "text",
    [
        "Change your grip on the backhand.",
        "Your racket face is open at contact.",
        "Keep your head up.",
        "Watch the ball onto the racket.",
        "Your head position was wrong.",
        "You won more rallies than you lost.",
        "That is better than elite players manage.",
        "Pro level recovery.",
    ],
)
def test_banned_phrases_are_caught(text):
    assert I.banned_hits(text)
    assert not I.verify_sentence(text, ["c1"], BY_ID)[0]


@pytest.mark.parametrize("text", ["Get back to the T before they hit.", "Take a first step towards the middle after each shot.", "An attempted winner needs time."])
def test_ordinary_coaching_language_passes_the_filter(text):
    assert not I.banned_hits(text)


def test_clock_formats_minutes_and_seconds():
    assert I.clock(112.0) == "1:52"
    assert I.clock(59.6) == "1:00"
    assert I.clock(0) == "0:00"


def test_the_quoted_difference_matches_the_quoted_numbers():
    d = I._display("time_given", 1.4689, 1.3349, [0.01, 0.24], 45, 17)
    assert (d["value"], d["comparison"], d["diff"]) == ("1.47", "1.33", "0.14")
    d = I._display("t_share", 0.2654, 0.1684, [0.04, 0.17], 185, 17)
    assert (d["value"], d["comparison"], d["diff"]) == ("27", "17", "10")


def test_a_gap_too_small_to_matter_is_suppressed_even_when_certain():
    small = {**CAND, "id": None, "diff": -0.2, "ci": [-0.25, -0.15]}  # t_at_strike: min_effect 0.3 m
    out = I.finalise([dict(CAND), small], with_suppressed=True)
    assert [c["suppressed"] for c in out] == [None, "too small to matter"]
    assert [c["id"] for c in I.finalise([dict(CAND), dict(small)])] == ["c1"]


def test_a_candidate_with_no_interval_is_suppressed():
    out = I.finalise([{**CAND, "display": {}}], with_suppressed=True)
    assert out[0]["suppressed"] == "no interval"


def test_won_and_lost_pass_only_in_sentences_citing_a_candidate_built_on_the_inferred_winners():
    plain = {**CAND, "id": "c2"}
    winners = {**CAND, "id": "c3", "allows": ["won_lost"]}
    by_id = {"c2": plain, "c3": winners}
    assert not I.verify_sentence("A won 12 rallies.", ["c2"], by_id)[0]
    ok, why = I.verify_sentence("A won 12 rallies.", ["c3"], by_id)
    assert ok, why
    assert not I.verify_sentence("Keep your head still; A won 12 rallies.", ["c3"], by_id)[0]  # only the named ban is lifted

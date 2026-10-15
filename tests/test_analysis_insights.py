"""Verifier, banned phrases and the editors."""

import json

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


def test_template_editor_output_passes_its_own_verifier():
    items = I.template_editor([CAND])
    kept, rejected = I.verify_items(items, [CAND])
    assert rejected == []
    assert len(kept["A"]["strengths"]) == 1 and kept["A"]["improvements"] == []
    assert kept["A"]["strengths"][0]["evidence"] == [{k: e[k] for k in ("t0", "t1", "label", "clip")} for e in CAND["evidence"]]


def test_at_most_three_items_per_kind():
    cands = [{**CAND, "id": f"c{i}", "metric": m} for i, m in enumerate(["t_at_strike", "t_overall", "t_share", "recovered", "time_given"], start=1)]
    items = I.template_editor(cands)
    assert len(items["A"]["strengths"]) == I.MAX_PER_KIND


def test_llm_editor_is_off_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert I.llm_editor([CAND]) is None
    _, _, editor = I.edit([CAND])
    assert editor == "template"


def test_llm_sentences_with_invented_numbers_are_dropped_and_the_template_fills_in():
    reply = {"A": {"strengths": [{"candidate": "c1", "sentences": [{"text": "A was 1.6 m from the T over 12 rallies.", "cites": ["c1"]}, {"text": "A was 0.9 m from the T.", "cites": ["c1"]}, {"text": "No citation here.", "cites": []}]}], "improvements": []}, "B": {"strengths": [], "improvements": []}}
    items, rejected, editor = I.edit([CAND], llm_call=lambda prompt: json.dumps(reply))
    kept = [s["text"] for s in items["A"]["strengths"][0]["sentences"]]
    assert editor == "llm+template"
    assert "A was 1.6 m from the T over 12 rallies." in kept
    assert all("0.9" not in s for s in kept)
    reasons = {r["text"]: r["reason"] for r in rejected}
    assert "A was 0.9 m from the T." in reasons and "No citation here." in reasons


def test_llm_garbage_falls_back_to_the_template():
    items, _, editor = I.edit([CAND], llm_call=lambda prompt: "not json")
    assert editor == "template" and items["A"]["strengths"]


def test_clock_formats_minutes_and_seconds():
    assert I.clock(112.0) == "1:52"
    assert I.clock(59.6) == "1:00"
    assert I.clock(0) == "0:00"


def test_template_editor_never_praises_and_criticises_the_same_behaviour():
    late = {**CAND, "id": "c2", "generator": "early_vs_late", "diff": -0.2, "ci": [-0.3, -0.1]}
    worse = {**CAND, "id": "c3", "diff": 0.3, "ci": [0.05, 0.55], "favourable": False,
             "display": {**CAND["display"], "value": "2.0", "comparison": "1.7", "diff": "0.3", "diff_lo": "0.05", "diff_hi": "0.55"}}
    items = I.template_editor([CAND, late, worse])["A"]
    used = [it["metric"] for kind in ("strengths", "improvements") for it in items[kind]]
    assert used == ["t_at_strike"]
    assert items["strengths"][0]["candidate"] == "c2"  # the largest gap relative to its interval wins


def test_the_quoted_difference_matches_the_quoted_numbers():
    d = I._display("time_given", 1.4689, 1.3349, [0.01, 0.24], 45, 17)
    assert (d["value"], d["comparison"], d["diff"]) == ("1.47", "1.33", "0.14")
    d = I._display("t_share", 0.2654, 0.1684, [0.04, 0.17], 185, 17)
    assert (d["value"], d["comparison"], d["diff"]) == ("27", "17", "10")

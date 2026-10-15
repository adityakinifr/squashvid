"""Run the analysis stage for one clip and write its outputs under work/<stem>/analysis/."""

from __future__ import annotations

import json
import logging

import pandas as pd

from squashvid.analysis import hits as H
from squashvid.analysis import rallies as R
from squashvid.analysis import stats as S
from squashvid.analysis.io import PLAYERS, Match, load, out_dir
from squashvid.analysis.serves import add_servers

log = logging.getLogger("squashvid.analysis")

HIT_COLUMNS = ["rally", "shot_index", "t", "frame", "hitter", "confidence", "source"]


def detect(m: Match) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shots, rallies and the evidence table, using ball events if the match has them."""
    swings = H.swing_candidates(m)
    raw = H.detect_hits(m)
    rallies, hits, ev = R.segment(m, raw, swings)
    hits = H.add_context(hits, m)
    return add_servers(m, rallies, hits), hits, ev


def write_tables(m: Match, rallies: pd.DataFrame, hits: pd.DataFrame) -> None:
    d = out_dir(m.stem)
    rallies.to_parquet(d / "rallies.parquet", index=False)
    hits.to_parquet(d / "hits.parquet", index=False)


def run(stem: str, use_ball: bool = True, review: bool = True, insights: bool = True, tag: str = "") -> dict:
    m = load(stem, use_ball=use_ball)
    rallies, hits, ev = detect(m)
    write_tables(m, rallies, hits)
    stats = S.write(m, hits, rallies)
    summary = {"clip": stem, "ball_events": m.ball_events is not None, "rallies": len(rallies), "hits": len(hits)}
    if review:
        from squashvid.analysis import review as V

        V.timeline(m, rallies, hits, ev, tag=tag)
    if insights:
        from squashvid.analysis import insights as I
        from squashvid.analysis import report as P

        ins = I.build(m, hits, rallies, stats)
        (out_dir(stem) / "insights.json").write_text(json.dumps(ins, indent=1))
        (out_dir(stem) / "report.md").write_text(P.render(stem, stats, ins))
    log.info("%s", summary)
    return summary


def run_pooled(stems: list[str], use_ball: bool = True) -> dict:
    """Insights and a report over several clips together (more rallies, so more findings survive suppression).

    Assumes A and B are the same two people in every clip. Output goes to work/analysis/.
    """
    from squashvid.analysis import coach
    from squashvid.analysis import insights as I
    from squashvid.analysis import report as P
    from squashvid.analysis.io import WORK
    from squashvid.analysis.serves import outcomes

    parts, m0 = [], None
    for i, stem in enumerate(stems):
        m = load(stem, use_ball=use_ball)
        r = pd.read_parquet(out_dir(stem) / "rallies.parquet")
        h = pd.read_parquet(out_dir(stem) / "hits.parquet")
        parts.append(I.Facts.from_match(m, h, r, offset=I.RALLY_BLOCK * (i + 1), ball=I.ball_table(m, h)))
        m0 = m0 or m
    facts = I.Facts.concat(parts)
    all_c = I.generate(facts, m0.T, with_suppressed=True)
    players, rejected = coach.write([c for c in all_c if c["suppressed"] is None])
    ins = {
        "clip": "+".join(stems),
        "editor": "coach",
        "n_rallies": int(facts.rallies.rally.nunique()),
        "outcomes": outcomes(facts.rallies, facts.hits),
        "players": players,
        "candidates": [I._pub_c(c) for c in all_c],
        "rejected_sentences": rejected,
    }
    d = WORK / "analysis"
    d.mkdir(exist_ok=True)
    (d / "insights.json").write_text(json.dumps(ins, indent=1))
    (d / "report.md").write_text(P.render_pooled(ins))
    return {"clip": ins["clip"], "rallies": ins["n_rallies"], "items": {p: {k: len(v) for k, v in ins["players"][p].items()} for p in PLAYERS}}

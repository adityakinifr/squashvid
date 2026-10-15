"""Pick the rally windows to label: two per clip, consecutive frames, ~15 s each.

Rallies come from the audio stream (work/<stem>/audio_rallies.parquet, racket hits
chained in time; audio is shared by an original and its copy). A rally window is
labelled frame by frame because a ball track needs continuity, and two windows per
clip with different hit tempos give a spread of drives, drops and lobs without
labelling idle time. Knock-ups and pre-serve bouncing show up as very regular
fast chains, so rallies with a median hit interval under MIN_INTERVAL are skipped.

    python -m squashvid.ball.windows        # writes work/labels/windows.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from squashvid.ball.clips import COPIES, LABELS, frame_range, video, work_dir

WINDOW_S = 15.5  # ~465 frames: six windows make the ~2,700-frame labelling target
LEAD_S, TAIL_S = 1.0, 1.0  # serve wind-up before the first hit; the ball dies after the last
MIN_INTERVAL, MIN_HITS = 0.45, 8
MIN_APART_S = 30.0
# Spans to keep out of windows (original timeline): clip 2's copy ends at 533 s and its camera jumps near 299 s.
EXCLUDE = {"IMG_4089": [(295.0, 305.0), (530.0, 1e9)]}


def load_rallies(stem: str) -> pd.DataFrame | None:
    for s in (stem, COPIES.get(stem)):
        p = work_dir(s) / "audio_rallies.parquet" if s else None
        if p is not None and p.exists():
            return pd.read_parquet(p)
    return None


def choose(rallies: pd.DataFrame, n: int = 2, exclude: list[tuple[float, float]] = ()) -> list[dict]:
    """The rally with the most hits, then the busiest one far from it whose tempo differs most."""
    r = rallies[(rallies.median_interval >= MIN_INTERVAL) & (rallies.n_hits >= MIN_HITS)].copy()
    r["start"] = r.first_hit - LEAD_S
    r["end"] = np.minimum(r.last_hit + TAIL_S, r.start + WINDOW_S)
    for a, b in exclude:
        r = r[(r.end < a) | (r.start > b)]
    r["hits_in"] = [int(((rallies.first_hit <= e) & (rallies.last_hit >= s)).sum()) for s, e in zip(r.start, r.end)]
    picked: list[pd.Series] = []
    for _ in range(n):
        cand = r[[all(abs(row.start - p.start) > MIN_APART_S for p in picked) for row in r.itertuples()]]
        if not len(cand):
            break
        if not picked:
            best = cand.loc[cand.n_hits.idxmax()]
        else:
            tempo = np.abs(np.log(cand.median_interval / picked[0].median_interval))
            best = cand.loc[(cand.n_hits.rank(pct=True) + tempo.rank(pct=True)).idxmax()]
        picked.append(best)
    return [dict(rally=int(p.rally), start=float(p.start), end=float(p.end), n_hits=int(p.n_hits), median_interval=float(p.median_interval)) for p in picked]


def build(stems=tuple(COPIES)) -> list[dict]:
    out = []
    for stem in stems:
        rl = load_rallies(stem)
        if rl is None:
            print(f"{stem}: no audio_rallies.parquet yet; skipped", file=sys.stderr)
            continue
        ts = video(stem).timestamps
        for w in choose(rl, exclude=EXCLUDE.get(stem, [])):
            i0, i1 = frame_range(ts, w["start"], w["end"])
            note = f"rally {w['rally']}: {w['n_hits']} hits, {w['median_interval']:.2f} s apart"
            out.append({"id": f"{stem}-r{w['rally']}", "stem": stem, "start": float(ts[i0]), "end": float(ts[i1]), "i0": i0, "i1": i1, "note": note})
    return out


def main() -> int:
    wins = build()
    LABELS.mkdir(parents=True, exist_ok=True)
    (LABELS / "windows.json").write_text(json.dumps(wins, indent=1))
    for w in wins:
        print(f"{w['id']:16s} {w['start']:7.2f}-{w['end']:7.2f} s  frames {w['i0']}-{w['i1']} ({w['i1'] - w['i0'] + 1})  {w['note']}")
    print("total frames", sum(w["i1"] - w["i0"] + 1 for w in wins))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

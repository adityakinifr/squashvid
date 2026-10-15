"""How each rally ended, inferred from the ball after the last detected hit. Nobody labels winners.

One row per rally. The ending is one of:

  tin          the last shot's placed front-wall impact is below the top of the tin (spec.tin_top).
  out          the placed front-wall impact is above the front out line, or a side-wall impact is above
               the (sloping) side-wall out line, or a back-wall impact is above the back out line.
  not_returned a valid placed front-wall impact (or an audio-only front-wall impact) followed by a double
               bounce (two floor events at least 0.25 s apart) with no later hit: the opponent did not play it.
               One bounce, or none, is not enough: it is also what a missed opponent hit looks like.
  unclear      no front-wall evidence after the last hit, or the evidence contradicts itself, or a front
               wall impact was heard but is not placed and there is no double bounce to back it.

`inferred_winner` is the last hitter for not_returned, the opponent for tin and out, and "none" for unclear.

What this cannot know, and what would make it wrong:

  * Lets and strokes. A let or stroke is awarded by the referee, not by the ball. A rally that stops for a
    let looks like an ending here, and a stroke can be given when the ball was in fact returnable.
  * A missed final hit. Hit recall is about 0.87. If the last real hit was missed, the last detected hitter
    is the player before, and the inferred winner is the wrong player (a tin or out by the real last hitter
    is credited to the other side). `last_hit_confidence` and the `racket_after` evidence flag point at
    the likely cases but cannot remove the risk.
  * Hidden or unplaced impacts. Most front-wall events are audio-only, with no height, so most rallies
    cannot be called tin or out. Heights have an error of about 0.1 to 0.3 m, which is the size of the tin.
  * A ball hit into the tin can still be heard as a front-wall impact; an out ball is usually stopped by
    the players, so the ball is seen no further.

`confidence` is an ordinal 0 to 1 score built from the evidence listed in `evidence`; it is not a
calibrated probability. Unclear rows carry 0.0 and no winner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from squashvid.analysis.io import other
from squashvid.analysis.shots import UNKNOWN
from squashvid.geom.court import CourtSpec

ENDINGS = ("tin", "out", "not_returned", "unclear")
TIN_SIGMA_M = 0.25  # front-wall height error near the tin; certainty is full only this far below the line
OUT_SIGMA_M = 0.4  # likewise above the out line, which the ball rarely clears by much
RETURN_3D_SAMPLES = 3  # ball_3d samples back in the court after a tin-height impact that mean the ball came back
TIN_BACK_M = 1.5  # a floor bounce this far in front of the front wall after a "tin" means the ball came back
RACKET_AFTER_S = 0.3  # an opponent racket event this long after the last hit is a possible missed return
GUARD_S = 0.04
COLUMNS = [
    "stem", "rally", "start_t", "end_t", "n_shots", "last_shot_index", "last_t", "last_hitter", "last_hit_confidence", "last_hit_source",
    "ending", "inferred_winner", "confidence", "evidence",
    "fw_seen", "fw_placed", "fw_z", "fw_x", "out_kind", "n_floor_after", "double_bounce", "racket_after",
    "last_shot_type", "last_shot_volley",
    "shot_before_last_index", "shot_before_last_hitter", "shot_before_last_type", "shot_before_last_length", "shot_before_last_tight",
]  # fmt: skip


def _out_kind(spec: CourtSpec, w: pd.DataFrame, fw_z: float) -> tuple[str, float]:
    """Which out line was cleared and by how much (metres); ("none", 0) if none."""
    best = ("none", 0.0)
    if np.isfinite(fw_z) and fw_z - spec.front_out_z > 0:
        best = ("front", float(fw_z - spec.front_out_z))
    sides = w[w.kind.isin(["side_wall_left", "side_wall_right"]) & w.z.notna()]
    for _, s in sides.iterrows():
        over = float(s.z - spec.side_out_z(s.y))
        if over > best[1]:
            best = ("side", over)
    back = w[(w.kind == "back_wall") & w.z.notna()]
    for _, s in back.iterrows():
        over = float(s.z - spec.back_out_z)
        if over > best[1]:
            best = ("back", over)
    return best


def judge(spec: CourtSpec, last: pd.Series, ev: pd.DataFrame, b3: pd.DataFrame | None = None) -> dict:
    """The ending of one rally from its last shot row and the ball events after it. See the module docstring."""
    w = ev[(ev.t > last.t + GUARD_S) & (ev.t < last.t_end)].sort_values("t")
    fw_z = float(last.fw_z) if last.fw_placed else np.nan
    opp = other(last.hitter)
    floors = w[w.kind == "floor"]
    fl_after = floors[floors.t > last.fw_t] if last.fw_seen else floors
    n_fl = int(len(fl_after))
    dbl = last.second_bounce == "yes"
    racket = w[(w.kind == "racket") & (w.player == opp) & (w.t > last.t + RACKET_AFTER_S)] if "player" in w else w.iloc[:0]
    notes: list[str] = []
    out_kind, over = _out_kind(spec, w, fw_z)
    res = dict(ending="unclear", confidence=0.0, out_kind=out_kind, n_floor_after=n_fl, double_bounce=bool(dbl), racket_after=bool(len(racket)))

    def done(ending: str, conf: float) -> dict:
        res.update(ending=ending, confidence=round(float(np.clip(conf, 0.0, 0.95)), 3) if ending != "unclear" else 0.0, evidence="; ".join(notes))
        return res

    if out_kind != "none" and over > 0:
        notes.append(f"{out_kind}_wall_impact_{over:.2f}m_over_out_line")
        return done("out", 0.45 + 0.45 * min(1.0, over / OUT_SIGMA_M))
    if not last.fw_seen:
        notes.append("no_front_wall_event_after_last_hit")
        return done("unclear", 0)
    if len(racket):
        notes.append("opponent_racket_event_after_last_hit_possible_missed_return")
        return done("unclear", 0)
    if last.fw_placed:
        if fw_z < spec.tin_top:
            under = spec.tin_top - fw_z
            back = fl_after[fl_after.y < spec.length - TIN_BACK_M]
            flown = pd.DataFrame()
            if b3 is not None and len(b3):
                flown = b3[(b3.t > last.fw_t + 0.15) & (b3.t < last.t_end) & (b3.y < spec.length - TIN_BACK_M) & (b3.z > 0.3)]
            if len(flown) >= RETURN_3D_SAMPLES:
                notes.append(f"front_wall_z_{fw_z:.2f}_under_tin_but_ball_flew_back_{len(flown)}_samples")
                return done("unclear", 0)
            if len(back):
                notes.append(f"front_wall_z_{fw_z:.2f}_under_tin_but_ball_bounced_{spec.length - float(back.y.iloc[0]):.1f}m_out_from_wall")
                return done("unclear", 0)
            notes.append(f"front_wall_z_{fw_z:.2f}_under_tin_top_{spec.tin_top:.2f}")
            return done("tin", 0.45 + 0.4 * min(1.0, under / TIN_SIGMA_M) + (0.05 if last.fw_evidence != "audio" else 0.0))
        notes.append(f"valid_front_wall_z_{fw_z:.2f}")
        if not dbl:
            # One bounce, or none, does not show the ball was not played: the opponent's hit may have been missed.
            notes.append("one_bounce_only" if n_fl else "no_floor_event_seen")
            return done("unclear", 0)
        notes.append("double_bounce")
        return done("not_returned", 0.70 + 0.10 * float(last.hit_confidence >= 0.5))
    # front wall heard, not placed
    if dbl:
        notes += ["front_wall_heard_not_placed", "double_bounce"]
        return done("not_returned", 0.30 + 0.10 * float(last.hit_confidence >= 0.5))
    notes.append("front_wall_heard_not_placed_no_double_bounce")
    return done("unclear", 0)


def ending_table(stem: str, spec: CourtSpec, shots: pd.DataFrame, rallies: pd.DataFrame, events: pd.DataFrame | None, b3: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per rally. `shots` is the table from shots.shot_table; `events` may be None (everything unclear)."""
    ev = events if events is not None else pd.DataFrame({c: [] for c in ("t", "kind", "x", "y", "z", "player")})
    rows = []
    for _, r in rallies.iterrows():
        g = shots[shots.rally == r.rally].sort_values("shot_index")
        if len(g) == 0:
            continue
        last = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else None
        j = judge(spec, last, ev, b3) if events is not None else dict(ending="unclear", confidence=0.0, out_kind="none", n_floor_after=0, double_bounce=False, racket_after=False, evidence="no_ball_events")
        win = last.hitter if j["ending"] == "not_returned" else other(last.hitter) if j["ending"] in ("tin", "out") else "none"
        rows.append(
            {
                "stem": stem, "rally": int(r.rally), "start_t": float(r.start_t), "end_t": float(r.end_t), "n_shots": int(len(g)),
                "last_shot_index": int(last.shot_index), "last_t": float(last.t), "last_hitter": last.hitter,
                "last_hit_confidence": float(last.hit_confidence), "last_hit_source": last.hit_source,
                **{k: j[k] for k in ("ending", "confidence", "evidence")}, "inferred_winner": win,
                "fw_seen": bool(last.fw_seen), "fw_placed": bool(last.fw_placed), "fw_z": float(last.fw_z), "fw_x": float(last.fw_x),
                **{k: j[k] for k in ("out_kind", "n_floor_after", "double_bounce", "racket_after")},
                "last_shot_type": last.shot_type, "last_shot_volley": last.volley,
                "shot_before_last_index": int(prev.shot_index) if prev is not None else -1,
                "shot_before_last_hitter": prev.hitter if prev is not None else UNKNOWN,
                "shot_before_last_type": prev.shot_type if prev is not None else UNKNOWN,
                "shot_before_last_length": prev.length if prev is not None else UNKNOWN,
                "shot_before_last_tight": prev.tight if prev is not None else UNKNOWN,
            }
        )  # fmt: skip
    return pd.DataFrame(rows, columns=COLUMNS)

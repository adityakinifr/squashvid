"""Per-shot facts: where each shot went, and what kind of shot it was.

One row per detected hit, keyed by (rally, shot_index). Everything here is read
from the ball events (`ball_events.parquet`) in the interval between this hit
and the next hit of the same rally (or the rally end for the last shot). It
writes no coaching text.

Nothing is guessed. Every fact comes with the evidence it rests on, and a fact
whose evidence is missing is the string "unknown" (or NaN for a number). Only
about 42% of shots get a *placed* front-wall impact: most front-wall events are
audio-only (the time is known, x and height are not) and the dark door frame
hides some impacts. Floor and racket events have lower recall again, and an
absent side-wall event is weak evidence of "no side wall" because side-wall
recall is only about 0.5 to 0.7. So the columns say "none_seen", not "none".

Shot types: drive, boast, drop, lob, kill, unknown. The rules are ordered and
the rule that fired is stored in `shot_type_rule`. They are heuristics on noisy
inputs, checked by eye on three clips (docs/main/shots_validation.md), not
verified against labels. The thresholds below were fitted by looking at the
same clips they are reported on, so any agreement is in-sample.

Court frame: x across from the left wall, y from the back wall (0) to the front
wall (L), z up. All lines come from the venue's CourtSpec, never WSF numbers.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from squashvid.analysis.io import CLIPS, PLAYERS, out_dir, work_dir
from squashvid.analysis.stats import front_wall_band
from squashvid.geom.court import CourtSpec

log = logging.getLogger("squashvid.analysis.shots")

UNKNOWN = "unknown"

# Event bookkeeping.
GUARD_S = 0.04  # events this close after the hit time are the racket contact itself
RALLY_TAIL_S = 1.5  # how long after the last hit of a rally the ball is still followed
MAX_FLIGHT_S = 3.5  # a shot's interval never extends past this, however long the next hit takes
FW_CLUSTER_S = 0.25  # front-wall events closer than this are one impact seen by audio and track
FLOOR_DEDUPE_S = 0.10  # floor events closer than this are one bounce
FLOOR_BEFORE_FRONT_M = 1.5  # a front-wall event after a floor bounce this far from the front wall is not the shot's first impact
SECOND_BOUNCE_S = 0.25  # a floor event this long after the first is a second bounce, not the same one

# Shot-type rules (metres, seconds, m/s).
LOB_MIN_Z = 3.2  # front-wall height from which a ball is a lob; drives rarely climb this high
LOW_FW_Z = 0.9  # below this on the front wall a ball is a kill or drop candidate (tin top is 0.48)
MIN_FLIGHT_S = 0.2  # a hit-to-front-wall time shorter than this means the hit time or the impact time is wrong
MAX_PLAUSIBLE_SPEED = 40.0  # mean foot-to-wall speed above this (144 km/h) is a timing error, not a shot
DROP_MAX_Z = 1.5  # a drop that lands short can still hit the wall this high given height error of ~0.3 m
DROP_MAX_SPEED = 11.0  # mean hitter-to-front-wall speed under which a ball is soft (a drop)
KILL_MIN_SPEED = 13.0  # and above this it is hit hard (a kill); between the two is ambiguous
DRIVE_MIN_SPEED = 13.0  # a deep ball with unknown wall height is a drive only when fast
LOB_MAX_SPEED = 8.0  # a deep ball with unknown wall height is a lob only when this slow
BOAST_MAX_S = 0.5  # without a front-wall event, a side wall hit this soon after the hit counts as "side wall first" (a ball that went to the front wall first needs longer to come back)
BOAST_MAX_AHEAD_M = 4.0  # ...and only if it is this close in front of the hitter (a side-wall hit near the front wall is a rebound)
SIDE_WALL_AHEAD_M = 0.5  # side-wall impact must be this far in front of (or level with) the hitter to be a boast

# Direction, length, tightness.
CENTRE_MARGIN_M = 0.4  # within this of the middle line a ball is neither left nor right
TIGHT_M = 0.6  # a bounce (or path) this close to a side wall is "tight"

# Volley from height when no floor event is in the gap.
VOLLEY_Z = 1.0  # ball centre this high at the hit: struck out of the air
BOUNCE_Z = 0.4  # ball centre this low at the hit: struck after it bounced
HEIGHT_WINDOW_S = (-0.10, 0.05)  # ball_3d samples around the hit used for its height

SHOT_TYPES = ("drive", "boast", "drop", "lob", "kill", UNKNOWN)

COLUMNS = [
    "stem", "rally", "shot_index", "t", "hitter", "hit_confidence", "hit_source", "rally_first", "hit_x", "hit_y", "cell_hit_from", "t_end",
    "fw_seen", "fw_t", "fw_x", "fw_z", "fw_placed", "fw_evidence", "fw_band", "fw_after_floor",
    "side_first", "side_first_y", "flight_s", "flight_speed",
    "bounce_t", "bounce_x", "bounce_y", "bounce_known", "floor_before_front", "second_bounce", "back_wall",
    "ball_z_at_hit", "volley", "volley_source",
    "shot_type", "shot_type_rule", "direction", "direction_source", "length", "length_source", "tight", "tight_source",
    "n_known",
]  # fmt: skip


def _dedupe(ev: pd.DataFrame, gap: float) -> pd.DataFrame:
    """Keep the first of events that follow each other within `gap` seconds."""
    if len(ev) < 2:
        return ev
    keep, last = [], -1e9
    for i, t in enumerate(ev.t.to_numpy()):
        if t - last >= gap:
            keep.append(i)
        last = t
    return ev.iloc[keep]


def _first_front_wall(fw: pd.DataFrame) -> pd.Series | None:
    """The first front-wall impact in `fw`: the best-placed event among those within FW_CLUSTER_S of the first.

    Audio-only events have no x or z. If the cluster has a placed event, that one carries the
    position and the cluster's earliest time is not used (the track event time is the better clock).
    """
    if len(fw) == 0:
        return None
    fw = fw.sort_values("t")
    cl = fw[fw.t <= fw.t.iloc[0] + FW_CLUSTER_S]
    placed = cl[cl.x.notna() & cl.z.notna()]
    return (placed if len(placed) else cl).sort_values("confidence", ascending=False).iloc[0]


def ball_height_at(b3: pd.DataFrame | None, t: float) -> float:
    """Median ball_3d height around a hit; NaN without samples. Unanchored fits have unconstrained depth, so this is rough."""
    if b3 is None or len(b3) == 0:
        return np.nan
    q = b3[(b3.t >= t + HEIGHT_WINDOW_S[0]) & (b3.t <= t + HEIGHT_WINDOW_S[1])]
    return float(q.z.median()) if len(q) else np.nan


def trace_shot(spec: CourtSpec, ev: pd.DataFrame, t_hit: float, t_end: float, hit_y: float) -> dict:
    """What the ball did between a hit and the next hit: front wall, side wall first, floor bounces, back wall."""
    w = ev[(ev.t > t_hit + GUARD_S) & (ev.t < t_end)].sort_values("t")
    fw_all = w[w.kind == "front_wall"]
    f0 = _first_front_wall(fw_all)
    out: dict = {"fw_seen": f0 is not None, "fw_t": np.nan, "fw_x": np.nan, "fw_z": np.nan, "fw_placed": False, "fw_evidence": UNKNOWN, "fw_band": UNKNOWN}
    if f0 is not None:
        placed = bool(np.isfinite(f0.x) and np.isfinite(f0.z))
        out.update(fw_t=float(f0.t), fw_placed=placed, fw_evidence=str(f0.evidence))
        if placed:
            out.update(fw_x=float(f0.x), fw_z=float(f0.z), fw_band=str(front_wall_band(np.array([f0.z]), spec)[0]))
    # A front-wall event that follows a floor bounce well out in the court cannot be this shot's first
    # impact (the real one was missed, and this is a later ricochet or a stray event): discard it.
    out["fw_after_floor"] = False
    if f0 is not None:
        fl0 = w[(w.kind == "floor") & (w.t < f0.t) & (w.y < spec.length - FLOOR_BEFORE_FRONT_M)]
        if len(fl0):
            out["fw_after_floor"] = True
            f0 = None
            out.update(fw_seen=False, fw_t=np.nan, fw_x=np.nan, fw_z=np.nan, fw_placed=False, fw_evidence=UNKNOWN, fw_band=UNKNOWN)
    t_fw = out["fw_t"] if f0 is not None else None

    # Side wall before the front wall (boast-like path), only walls in front of the hitter.
    sides = w[w.kind.isin(["side_wall_left", "side_wall_right"]) & (w.y >= hit_y - SIDE_WALL_AHEAD_M)]
    out.update(side_first=UNKNOWN, side_first_y=np.nan)
    if f0 is not None:
        pre = sides[sides.t < t_fw]
        if len(pre):
            s = pre.iloc[0]
            out.update(side_first="left" if s.kind == "side_wall_left" else "right", side_first_y=float(s.y))
        else:
            out["side_first"] = "none_seen"
    elif len(sides) and sides.t.iloc[0] - t_hit <= BOAST_MAX_S and sides.y.iloc[0] - hit_y <= BOAST_MAX_AHEAD_M:
        s = sides.iloc[0]
        out.update(side_first="left" if s.kind == "side_wall_left" else "right", side_first_y=float(s.y))

    # Mean speed from the hitter's feet to the front wall: a rough measure of pace, not ball speed.
    out["flight_s"] = out["flight_speed"] = np.nan
    if f0 is not None and t_fw > t_hit + MIN_FLIGHT_S:
        out["flight_s"] = float(t_fw - t_hit)
        v = float(max(spec.length - hit_y, 0.0) / (t_fw - t_hit))
        out["flight_speed"] = v if v <= MAX_PLAUSIBLE_SPEED else np.nan

    # Floor: the first bounce after the front wall; without any front-wall event, the first after the hit (flagged).
    fl = _dedupe(w[w.kind == "floor"], FLOOR_DEDUPE_S)
    out["floor_before_front"] = bool(f0 is not None and len(fl[fl.t < t_fw]) > 0)
    after = fl[fl.t > t_fw] if f0 is not None else fl
    b1 = after.iloc[0] if len(after) else None
    out.update(bounce_t=np.nan, bounce_x=np.nan, bounce_y=np.nan, bounce_known=False)
    if b1 is not None:
        out.update(bounce_t=float(b1.t), bounce_x=float(b1.x), bounce_y=float(b1.y), bounce_known=True)
        later = after[after.t >= b1.t + SECOND_BOUNCE_S]
        out["second_bounce"] = "yes" if len(later) else "none_seen"
        bw = w[(w.kind == "back_wall") & (w.t > t_fw if f0 is not None else True)]
        out["back_wall"] = "yes" if len(bw) else "none_seen"
    else:
        out["second_bounce"] = out["back_wall"] = UNKNOWN
    return out


def volley_fact(spec: CourtSpec, ev: pd.DataFrame, b3: pd.DataFrame | None, prev_ref_t: float | None, t: float) -> tuple[str, str, float]:
    """Volley, bounce or unknown for the hit at `t`, given the previous shot's front-wall time (or hit time).

    A floor event in the gap is a bounce (strong). Without one, the ball's height near the hit decides if it
    is clearly high (volley) or low (bounce); the middle is unknown. A missing floor event alone is not
    evidence: floor recall is only about 0.57.
    """
    z = ball_height_at(b3, t)
    if prev_ref_t is None:
        return "serve", "first_shot", z
    fl = ev[(ev.kind == "floor") & (ev.t > prev_ref_t + GUARD_S) & (ev.t < t - GUARD_S)]
    if len(fl):
        return "bounce", "floor_event", z
    if np.isfinite(z):
        if z >= VOLLEY_Z:
            return "volley", "height", z
        if z <= BOUNCE_Z:
            return "bounce", "height", z
    return UNKNOWN, "no_evidence", z


def length_of(spec: CourtSpec, bounce_y: float) -> str:
    """deep: bounced behind the back of the service box; short: in front of the short line; mid between."""
    if not np.isfinite(bounce_y):
        return UNKNOWN
    if bounce_y < spec.box_back_y:
        return "deep"
    return "short" if bounce_y > spec.short_y else "mid"


def direction_of(spec: CourtSpec, hit_x: float, bounce_x: float, fw_x: float) -> tuple[str, str]:
    """straight if the ball goes to the side of the middle line the hitter is on, cross if the other side.

    The bounce is the target when known; the front-wall impact stands in otherwise. Near the middle line
    (CENTRE_MARGIN_M) either end makes the call ambiguous, so it is unknown.
    """
    half = spec.half_x
    if np.isfinite(bounce_x):
        tgt, src = bounce_x, "bounce"
    elif np.isfinite(fw_x):
        tgt, src = fw_x, "front_wall"
    else:
        return UNKNOWN, "no_target"
    if not np.isfinite(hit_x) or abs(hit_x - half) < CENTRE_MARGIN_M or abs(tgt - half) < CENTRE_MARGIN_M:
        return UNKNOWN, "near_centre"
    return ("straight" if (hit_x - half) * (tgt - half) > 0 else "cross"), src


def tight_of(spec: CourtSpec, bounce_x: float, sides_after_front: bool) -> tuple[str, str]:
    """tight if the ball bounced within TIGHT_M of a side wall or struck a side wall after the front wall."""
    if np.isfinite(bounce_x):
        if min(bounce_x, spec.width - bounce_x) <= TIGHT_M:
            return "yes", "bounce"
        return ("yes", "side_wall_event") if sides_after_front else ("no", "bounce")
    return ("yes", "side_wall_event") if sides_after_front else (UNKNOWN, "no_bounce")


def classify(spec: CourtSpec, r: dict) -> tuple[str, str]:
    """Shot type and the rule that fired, from a traced shot `r` (see trace_shot).

    Ordered rules, first match wins:
      boast   a side wall was hit before the front wall, in front of the hitter.
      lob     placed front-wall height >= LOB_MIN_Z.
      unknown a placed impact below the tin top: an error shot or a mis-measured impact, never called a kill.
      kill    placed height < LOW_FW_Z and the ball was fast (>= KILL_MIN_SPEED).
      drop    soft (< DROP_MAX_SPEED) and the first bounce in front of the short line, with the wall height
              unknown or < DROP_MAX_Z.
      drive   placed height in [LOW_FW_Z, LOB_MIN_Z) at any pace the drop rule did not take; where it bounced is
              reported in `length`, not used here. Without wall height: a deep bounce and fast ball (>= DRIVE_MIN_SPEED).
      lob     (no height) deep bounce and a slow ball (< LOB_MAX_SPEED).
      unknown anything else, including low-and-ambiguous pace and every case without a front-wall time.
    Speed is the mean speed from the hitter's feet to the front wall; it is rough (a few tenths of a second
    of timing error moves it by a few m/s).
    """
    z, spd, by = r["fw_z"], r["flight_speed"], r["bounce_y"]
    short = np.isfinite(by) and by > spec.short_y
    deep = np.isfinite(by) and by < spec.box_back_y
    if r["side_first"] in ("left", "right"):
        return "boast", "side_wall_then_front" if r["fw_seen"] else "side_wall_soon_after_hit"
    if np.isfinite(z) and z >= LOB_MIN_Z:
        return "lob", "front_wall_high"
    if np.isfinite(z) and z < spec.tin_top:
        return UNKNOWN, "front_wall_below_tin_top"  # a tin (error) or a mis-measured impact; not a kill
    if np.isfinite(z) and z < LOW_FW_Z:
        if np.isfinite(spd) and spd >= KILL_MIN_SPEED:
            return "kill", "low_front_wall_fast"
        if short and np.isfinite(spd) and spd < DROP_MAX_SPEED:
            return "drop", "low_front_wall_soft_short"
        return UNKNOWN, "low_front_wall_ambiguous_pace"
    if short and np.isfinite(spd) and spd < DROP_MAX_SPEED and (not np.isfinite(z) or z < DROP_MAX_Z):
        return "drop", "soft_short" if not np.isfinite(z) else "soft_short_low_wall"
    if np.isfinite(z):
        return "drive", "front_wall_mid"  # a fast ball that still lands short is a short-length drive (see `length`)
    if np.isfinite(spd) and deep:
        if spd >= DRIVE_MIN_SPEED:
            return "drive", "deep_fast_no_height"
        if spd < LOB_MAX_SPEED:
            return "lob", "deep_slow_no_height"
    return UNKNOWN, "no_front_wall_height"


def shot_table(stem: str, spec: CourtSpec, hits: pd.DataFrame, rallies: pd.DataFrame, events: pd.DataFrame | None, b3: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per hit. With no ball events every ball fact is unknown (the hit columns are still filled)."""
    ev = events if events is not None else pd.DataFrame({c: [] for c in ("t", "kind", "x", "y", "z", "evidence", "confidence")})
    ev = ev.sort_values("t").reset_index(drop=True)
    end_of = dict(zip(rallies.rally, rallies.end_t)) if len(rallies) else {}
    rows = []
    for rid, g in hits.sort_values(["rally", "shot_index"]).groupby("rally"):
        g = g.reset_index(drop=True)
        prev_ref: float | None = None
        for i, h in g.iterrows():
            last = i == len(g) - 1
            t_end = float(g.t[i + 1]) if not last else max(float(end_of.get(rid, h.t)), float(h.t)) + RALLY_TAIL_S
            t_end = min(t_end, float(h.t) + MAX_FLIGHT_S)
            tr = trace_shot(spec, ev, float(h.t), t_end, float(h.hitter_y))
            vol, vsrc, z_hit = volley_fact(spec, ev, b3, prev_ref, float(h.t))
            st, rule = classify(spec, tr)
            dirn, dsrc = direction_of(spec, float(h.hitter_x), tr["bounce_x"], tr["fw_x"])
            ln = length_of(spec, tr["bounce_y"])
            side_after = False
            if tr["fw_seen"]:
                lim = tr["bounce_t"] if tr["bounce_known"] else t_end
                sw = ev[ev.kind.isin(["side_wall_left", "side_wall_right"]) & (ev.t > tr["fw_t"]) & (ev.t < lim)]
                side_after = len(sw) > 0
            tg, tsrc = tight_of(spec, tr["bounce_x"], side_after)
            row = {"stem": stem, "rally": int(rid), "shot_index": int(h.shot_index), "t": float(h.t), "hitter": h.hitter, "hit_confidence": float(h.confidence), "hit_source": str(h.source), "rally_first": i == 0, "hit_x": float(h.hitter_x), "hit_y": float(h.hitter_y), "cell_hit_from": h.get("cell_hit_from"), "t_end": t_end}
            row.update(tr)
            row.update(ball_z_at_hit=z_hit, volley=vol, volley_source=vsrc, shot_type=st, shot_type_rule=rule, direction=dirn, direction_source=dsrc, length=ln, length_source="floor_event" if ln != UNKNOWN else "no_bounce", tight=tg, tight_source=tsrc)
            row["n_known"] = int(sum(v != UNKNOWN for v in (st, dirn, ln, tg)))
            rows.append(row)
            prev_ref = tr["fw_t"] if tr["fw_seen"] else float(h.t)
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df


def load_inputs(stem: str):
    from squashvid.ball.regions import venue

    wd = work_dir(stem)
    ad = out_dir(stem)
    ev = pd.read_parquet(wd / "ball_events.parquet") if (wd / "ball_events.parquet").exists() else None
    b3 = pd.read_parquet(wd / "ball_3d.parquet") if (wd / "ball_3d.parquet").exists() else None
    return venue(stem).spec, pd.read_parquet(ad / "hits.parquet"), pd.read_parquet(ad / "rallies.parquet"), ev, b3


def coverage(shots: pd.DataFrame) -> dict:
    """Share of shots with each field known, and the shot-type mix per player."""
    n = len(shots)
    known = lambda s: float((s != UNKNOWN).mean()) if n else float("nan")  # noqa: E731
    cov = {
        "n_shots": n,
        "front_wall_seen": float(shots.fw_seen.mean()) if n else float("nan"),
        "front_wall_placed": float(shots.fw_placed.mean()) if n else float("nan"),
        "bounce_known": float(shots.bounce_known.mean()) if n else float("nan"),
        "side_first_known": known(shots.side_first),
        "volley_known": known(shots.volley),
        "shot_type": known(shots.shot_type),
        "direction": known(shots.direction),
        "length": known(shots.length),
        "tight": known(shots.tight),
    }
    mix = {p: g.shot_type.value_counts().to_dict() for p, g in shots.groupby("hitter")}
    return {"coverage": cov, "shot_type_mix": mix}


ZONES = ("back", "mid", "front")


def _counts(col: pd.Series, keys: tuple[str, ...]) -> dict[str, int]:
    return {k: int((col == k).sum()) for k in keys}


def placement(shots: pd.DataFrame) -> dict:
    """Where each player's shots went, by the part of the court they were hit from (the cell's back/mid/front):
    counts of known facts only, so a reader sees "of how many". Each rally's first shot (the serve) is left out."""
    s = shots[~shots.rally_first.astype(bool)] if len(shots) else shots
    out: dict = {"n_shots": int(len(s)), "direction_known": int((s.direction != UNKNOWN).sum()) if len(s) else 0, "players": {}}
    for p in PLAYERS:
        mine = s[s.hitter == p] if len(s) else s
        zone = mine.cell_hit_from.fillna("").str.split("-").str[0] if len(mine) else pd.Series(dtype=str)
        out["players"][p] = {
            "n": int(len(mine)),
            "zones": {z: {"n": int((zone == z).sum()), "direction": _counts(mine.direction[zone == z], ("straight", "cross")), "length": _counts(mine.length[zone == z], ("deep", "mid", "short"))} for z in ZONES},
            "volley": _counts(mine.volley, ("volley", "bounce")),
            "tight": _counts(mine.tight, ("yes", "no")),
            "shot_type": _counts(mine.shot_type, ("drive", "boast", "drop", "lob", "kill")),
        }
    return out


def run(stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from squashvid.analysis import endings as E

    spec, hits, rallies, ev, b3 = load_inputs(stem)
    shots = shot_table(stem, spec, hits, rallies, ev, b3)
    ends = E.ending_table(stem, spec, shots, rallies, ev, b3)
    d = out_dir(stem)
    shots.to_parquet(d / "shots.parquet", index=False)
    ends.to_parquet(d / "endings.parquet", index=False)
    return shots, ends


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -I -m squashvid.analysis.shots", description=__doc__.splitlines()[0])
    ap.add_argument("stems", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    stems = list(CLIPS) if a.all else a.stems
    if not stems:
        ap.error("give stems or --all")
    for stem in stems:
        shots, ends = run(stem)
        c = coverage(shots)
        print(f"\n== {stem}: {c['coverage']['n_shots']} shots, {len(ends)} rallies")
        for k, v in c["coverage"].items():
            if k != "n_shots":
                print(f"  {k:20s} {v:6.1%}")
        for p, mix in c["shot_type_mix"].items():
            print(f"  shot types {p}: {mix}")
        for p, g in ends.groupby("last_hitter", dropna=False):
            print(f"  endings, last hitter {p}: {g.ending.value_counts().to_dict()}")
        print(f"  endings overall: {ends.ending.value_counts().to_dict()}")


if __name__ == "__main__":
    main()

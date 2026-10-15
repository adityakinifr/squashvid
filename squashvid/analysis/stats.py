"""Per-player, per-clip statistics with sample sizes and bootstrap intervals over rallies.

Position and time are what the coaching research rewards (T position at the
opponent's strike, time available per shot), so those lead. Distance covered is
reported only because people ask; it does not separate winners from losers
(Murray et al. 2017, Cohen's d 0.04-0.09). "Play" means inside a detected rally.
Every headline number is {"value", "n", "ci"} where ci is a 90% bootstrap
interval that resamples whole rallies (shots in a rally are not independent).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from squashvid.analysis.hits import RECOVERY_RADIUS_M, SITUATION_RULES, T_RADIUS_M, path_length, zone_cells
from squashvid.analysis.io import PLAYERS, Match, other, out_dir

BOOT = 2000
ALPHA = 0.10
MIN_RALLIES = 8
FRONT_EDGES_NOTE = "rows are 1 m deep at the back, then to the short line, then two coarser rows towards the front wall, where depth resolution is worse"


def bootstrap_ci(groups: list[np.ndarray], stat=np.mean, n_boot: int = BOOT, alpha: float = ALPHA, seed: int = 0) -> dict:
    """Value of `stat` on all data, with a percentile interval from resampling groups (rallies) with replacement."""
    groups = [np.asarray(g, float)[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g)]
    n = int(sum(len(g) for g in groups))
    if n == 0:
        return {"value": None, "n": 0, "n_rallies": 0, "ci": None}
    value = float(stat(np.concatenate(groups)))
    if len(groups) < 2:
        return {"value": value, "n": n, "n_rallies": len(groups), "ci": None}
    rng = np.random.default_rng(seed)
    k = len(groups)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, k)
        out[b] = stat(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": value, "n": n, "n_rallies": k, "ci": [float(lo), float(hi)]}


def bootstrap_diff(a: list[np.ndarray], b: list[np.ndarray], stat=np.mean, n_boot: int = BOOT, alpha: float = ALPHA, seed: int = 0) -> dict:
    """stat(a) - stat(b) where a[i], b[i] come from the same rally i, resampling rallies jointly."""
    pairs = [(np.asarray(x, float)[np.isfinite(x)], np.asarray(y, float)[np.isfinite(y)]) for x, y in zip(a, b)]
    pairs = [(x, y) for x, y in pairs if len(x) and len(y)]
    if not pairs:
        return {"value": None, "n": 0, "n_rallies": 0, "ci": None}
    n = int(sum(len(x) + len(y) for x, y in pairs))
    value = float(stat(np.concatenate([x for x, _ in pairs])) - stat(np.concatenate([y for _, y in pairs])))
    if len(pairs) < 2:
        return {"value": value, "n": n, "n_rallies": len(pairs), "ci": None}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(pairs), len(pairs))
        out[i] = stat(np.concatenate([pairs[j][0] for j in pick])) - stat(np.concatenate([pairs[j][1] for j in pick]))
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": value, "n": n, "n_rallies": len(pairs), "ci": [float(lo), float(hi)]}


def play_mask(m: Match, rallies: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(m.n, bool)
    for r in rallies.itertuples():
        mask[(m.t >= r.start_t) & (m.t <= r.end_t)] = True
    return mask


def grid_edges(m: Match) -> tuple[np.ndarray, np.ndarray]:
    s = m.spec
    ys = np.array([0.0, 1.0, 2.0, 3.0, s.short_y, s.short_y + 2.0, s.length])
    xs = np.array([0.0, s.width / 3, 2 * s.width / 3, s.width])
    return ys, xs


def occupancy(m: Match, player: str, mask: np.ndarray) -> dict:
    ys, xs = grid_edges(m)
    tr = m.tracks[player]
    ok = mask & np.isfinite(tr.x) & np.isfinite(tr.y)
    h, _, _ = np.histogram2d(tr.y[ok], tr.x[ok], bins=[ys, xs])
    total = h.sum()
    return {
        "y_edges_m": [round(float(v), 2) for v in ys],
        "x_edges_m": [round(float(v), 2) for v in xs],
        "share": (h / total).round(4).tolist() if total else None,
        "n_frames": int(total),
        "note": FRONT_EDGES_NOTE,
        "rows_from": "back wall (row 0) to front wall; columns left to right as seen from behind the back wall",
    }


def _num(x, nd=4):  # 4 places, so a percentage shown to 0 places is rounded once, as in the insights
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _round_ci(d: dict, nd: int = 4) -> dict:
    out = dict(d)
    if out.get("value") is not None:
        out["value"] = round(out["value"], nd)
    if out.get("ci"):
        out["ci"] = [round(v, nd) for v in out["ci"]]
    return out


def per_rally_groups(h: pd.DataFrame, rally_ids, col: str, sel: pd.Series) -> list[np.ndarray]:
    return [h[(h.rally == r) & sel][col].to_numpy(float) for r in rally_ids]


def time_in_play_series(m: Match, player: str, rallies: pd.DataFrame, fn) -> list[np.ndarray]:
    out = []
    for r in rallies.itertuples():
        a, b = m.frame_at(r.start_t), m.frame_at(r.end_t)
        out.append(fn(m, player, a, b))
    return out


def player_stats(m: Match, h: pd.DataFrame, rallies: pd.DataFrame, player: str, t_radius: float = T_RADIUS_M) -> dict:
    opp = other(player)
    mask = play_mask(m, rallies)
    tr = m.tracks[player]
    dT = np.hypot(tr.x - m.T[0], tr.y - m.T[1])
    rids = rallies.rally.tolist()
    ok = mask & np.isfinite(dT)
    play_seconds = float(ok.sum() / m.fps)

    def seg(fn):
        return [fn(dT[m.frame_at(r.start_t) : m.frame_at(r.end_t) + 1], m.frame_at(r.start_t), m.frame_at(r.end_t)) for r in rallies.itertuples()]

    mean_dT = bootstrap_ci([x for x in seg(lambda d, a, b: d[np.isfinite(d)])])
    share_T = bootstrap_ci(seg(lambda d, a, b: (d[np.isfinite(d)] <= t_radius).astype(float)))
    front = bootstrap_ci(seg(lambda d, a, b: (tr.y[a : b + 1][np.isfinite(tr.y[a : b + 1])] > m.spec.short_y).astype(float)))
    # T distance at the opponent's strike: this player's distance when the opponent hits.
    by_opp = h[h.hitter == opp]
    at_strike = bootstrap_ci([np.hypot(g[f"{player}_x"].to_numpy() - m.T[0], g[f"{player}_y"].to_numpy() - m.T[1]) for g in (by_opp[by_opp.rally == r] for r in rids)])
    own = h[h.hitter == player]
    avail = own.available_time.dropna()
    avail_stats = {
        "n": int(len(avail)),
        "mean": _num(avail.mean()) if len(avail) else None,
        "median": _num(avail.median()) if len(avail) else None,
        "p10_p90": [_num(avail.quantile(0.1)), _num(avail.quantile(0.9))] if len(avail) else None,
        "share_under_1.1s": bootstrap_ci(per_rally_groups(own.assign(u=(own.available_time < 1.1).astype(float).where(own.available_time.notna())), rids, "u", pd.Series(True, index=own.index))),
        "histogram_counts_0.5s_bins_0_to_3s": np.histogram(avail, bins=np.arange(0, 3.5, 0.5))[0].tolist() if len(avail) else None,
    }
    avail_stats["share_under_1.1s"] = _round_ci(avail_stats["share_under_1.1s"])
    need = own[~(own.hitter_T_dist <= RECOVERY_RADIUS_M)]
    rec = {
        "n": int(len(need)),
        "median_s": _num(need.recovery_time.median()) if len(need) else None,
        "share_back_on_T_before_next_hit": bootstrap_ci(per_rally_groups(need.assign(u=need.recovered.astype(float)), rids, "u", pd.Series(True, index=need.index))),
        "radius_m": RECOVERY_RADIUS_M,
        "note": "hits from within the radius are excluded; time is censored at the next hit",
    }
    rec["share_back_on_T_before_next_hit"] = _round_ci(rec["share_back_on_T_before_next_hit"])
    sit = own.situation.dropna()
    situations = {k: {"share": _num((sit == k).mean()) if len(sit) else None, "n": int((sit == k).sum())} for k in SITUATION_RULES}
    sit_n = int(len(sit))
    # Distance covered and peak speeds (play only).
    dist = [path_length(m, player, r.start_t, r.end_t) for r in rallies.itertuples()]
    speeds = np.concatenate([tr.speed[m.frame_at(r.start_t) : m.frame_at(r.end_t) + 1] for r in rallies.itertuples()]) if len(rallies) else np.array([])
    cells = zone_cells(own.hitter_x, own.hitter_y, m) if len(own) else np.array([])
    cell_share = {c: _num(float((cells == c).mean())) for c in sorted(set(cells))} if len(cells) else {}
    out = {
        "shots": int(len(own)),
        "shot_share": _num(len(own) / max(len(h), 1)),
        "mean_dist_from_T_m": _round_ci(mean_dT, 2),
        "mean_dist_from_T_at_opponent_strike_m": _round_ci(at_strike, 2),
        "share_within_T_radius": _round_ci(share_T),
        "t_radius_m": t_radius,
        "share_in_front_of_short_line": _round_ci(front),
        "share_behind_short_line": _round_ci({**front, "value": None if front["value"] is None else 1 - front["value"], "ci": None if not front["ci"] else [1 - front["ci"][1], 1 - front["ci"][0]]}),
        "play_seconds": _num(play_seconds, 1),
        "occupancy": occupancy(m, player, mask),
        "distance_covered_in_play_m": {"value": _num(sum(dist), 1), "per_rally_median": _num(np.median(dist)) if dist else None, "n_rallies": len(dist)},
        "speed_mps": {"p95": _num(np.percentile(speeds, 95)) if len(speeds) else None, "max": _num(min(speeds.max(), 8.0)) if len(speeds) else None, "n_frames": int(len(speeds)), "note": "distance and speed do not separate winners from losers"},
        "available_time_s": avail_stats,
        "recovery_to_T": rec,
        "situations": {"n": sit_n, "shares": situations, "note": "rule-based approximation of Murray's six clusters"},
        "hit_from_cells": {"n": int(len(cells)), "share": cell_share},
    }
    return out


def match_stats(m: Match, h: pd.DataFrame, rallies: pd.DataFrame) -> dict:
    dur = rallies.end_t - rallies.start_t
    shots = rallies.n_shots
    return {
        "rallies": int(len(rallies)),
        "shots_total": int(len(h)),
        "rally_shots": {"median": _num(shots.median()) if len(shots) else None, "mean": _num(shots.mean()) if len(shots) else None, "max": int(shots.max()) if len(shots) else None, "values": shots.tolist()},
        "rally_seconds": {"median": _num(dur.median(), 1) if len(dur) else None, "mean": _num(dur.mean(), 1) if len(dur) else None, "max": _num(dur.max(), 1) if len(dur) else None},
        "play_share_of_clip": _num(float(dur.sum() / m.duration)) if len(dur) else None,
        "clip_seconds": _num(m.duration, 1),
    }


def per_rally(m: Match, h: pd.DataFrame, rallies: pd.DataFrame) -> list[dict]:
    rows = []
    for r in rallies.itertuples():
        g = h[h.rally == r.rally]
        row = {"rally": int(r.rally), "start_t": float(r.start_t), "end_t": float(r.end_t), "n_shots": int(r.n_shots), "confidence": float(r.confidence)}
        for k in PLAYERS:
            own, opp = g[g.hitter == k], g[g.hitter == other(k)]
            a, b = m.frame_at(r.start_t), m.frame_at(r.end_t)
            tr = m.tracks[k]
            d = np.hypot(tr.x[a : b + 1] - m.T[0], tr.y[a : b + 1] - m.T[1])
            row[k] = {
                "shots": int(len(own)),
                "mean_dist_from_T_m": _num(np.nanmean(d), 2),
                "mean_dist_from_T_at_opponent_strike_m": _num(np.nanmean(np.hypot(opp[f"{k}_x"] - m.T[0], opp[f"{k}_y"] - m.T[1])), 2) if len(opp) else None,
                "mean_available_time_s": _num(own.available_time.mean(), 2) if own.available_time.notna().any() else None,
            }
        rows.append(row)
    return rows


# --- ball-dependent statistics --------------------------------------------------------------

NEAR_TIN_M = 0.25  # front-wall contact within this height above the top of the tin


def ball_stats(m: Match, h: pd.DataFrame) -> dict | None:
    """Front-wall height bands, width thirds, near-tin share and first-bounce length, per hitter; None without ball events."""
    ev = m.ball_events
    if ev is None or len(h) == 0:
        return None
    s = m.spec
    fw = ev[ev.kind == "front_wall"]
    fl = ev[ev.kind == "floor"]
    rows = []
    h = h.sort_values(["rally", "shot_index"])
    nxt = h.groupby("rally").t.shift(-1).fillna(h.t + 3.0)
    for (_, r), t_next in zip(h.iterrows(), nxt):
        f = fw[(fw.t > r.t) & (fw.t < min(t_next, r.t + 1.5))]
        if len(f) == 0:
            continue
        f0 = f.iloc[0]
        b = fl[(fl.t > f0.t) & (fl.t < max(t_next, f0.t + 1.2))]
        by = float(b.iloc[0].y) if len(b) else np.nan
        rows.append({"rally": r.rally, "hitter": r.hitter, "z": float(f0.z), "x": float(f0.x), "bounce_y": by})
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d["band"] = np.where(d.z < s.service_z, "tin_to_service_line", np.where(d.z < 2.5, "service_line_to_2.5m", "above_2.5m"))
    d["third"] = np.where(d.x < s.width / 3, "left", np.where(d.x < 2 * s.width / 3, "centre", "right"))
    d["near_tin"] = (d.z - s.tin_top) < NEAR_TIN_M
    d["length"] = np.where(d.bounce_y.isna(), None, np.where(d.bounce_y < s.box_back_y, "back", np.where(d.bounce_y < s.short_y + 1.0, "mid", "front")))
    out = {"note": "front-wall impacts attributed to the preceding shot; height z in metres above the floor"}
    rids = sorted(d.rally.unique())
    for k in PLAYERS:
        g = d[d.hitter == k]
        entry = {"n_shots_with_front_wall": int(len(g))}
        for col, vals in (("band", ["tin_to_service_line", "service_line_to_2.5m", "above_2.5m"]), ("third", ["left", "centre", "right"])):
            entry[col] = {v: _round_ci(bootstrap_ci([(d[(d.rally == r) & (d.hitter == k)][col] == v).to_numpy(float) for r in rids])) for v in vals}
        entry["near_tin_share"] = _round_ci(bootstrap_ci([d[(d.rally == r) & (d.hitter == k)].near_tin.to_numpy(float) for r in rids]))
        gl = g[g.length.notna()]
        entry["first_bounce_length"] = {"n": int(len(gl)), **{v: _num(float((gl.length == v).mean())) if len(gl) else None for v in ("back", "mid", "front")}}
        out[k] = entry
    return out


def compute(m: Match, h: pd.DataFrame, rallies: pd.DataFrame, t_radius: float = T_RADIUS_M) -> dict:
    out = {
        "clip": m.stem,
        "ball_events_used": m.ball_events is not None,
        "params": {"t_radius_m": t_radius, "recovery_radius_m": RECOVERY_RADIUS_M, "bootstrap": {"resamples": BOOT, "interval": 1 - ALPHA, "unit": "rally"}, "t_point_m": [round(float(v), 3) for v in m.T], "short_line_y_m": round(float(m.spec.short_y), 3)},
        "match": match_stats(m, h, rallies),
        "players": {k: player_stats(m, h, rallies, k, t_radius) for k in PLAYERS},
        "per_rally": per_rally(m, h, rallies),
        "situation_rules": SITUATION_RULES,
        "caveats": [
            "Positions are smoothed foot points; foot position has not been checked against tape marks.",
            "Front-court depth resolution is coarse.",
            "Hitter and contact time come from pose, audio and alternation (ball events when present); see docs/main/analysis.md for accuracy.",
            "Who won each rally is not known, so nothing here compares winners and losers.",
        ],
    }
    out["ball"] = ball_stats(m, h)
    return out


def write(m: Match, h: pd.DataFrame, rallies: pd.DataFrame) -> dict:
    s = compute(m, h, rallies)
    (out_dir(m.stem) / "stats.json").write_text(json.dumps(s, indent=1, default=lambda o: None if o is None else float(o)))
    return s

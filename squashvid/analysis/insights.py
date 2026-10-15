"""Coaching insights: facts -> candidate generators -> editor -> verifier.

1. Facts: the hits table (positions, available time, recovery) per rally.
2. Candidates: rule-based comparisons (player vs opponent, early vs late, recovery
   after short vs long time available, ball metrics when ball events exist). Each has a
   value, a comparison, n, a 90% bootstrap interval over rallies and 2-3 evidence
   clips. A candidate is suppressed when its interval crosses zero or it rests on
   fewer than MIN_RALLIES rallies.
3. Editor: picks at most three strengths and three things to work on per player and
   phrases them. The deterministic templated editor always runs; an LLM editor runs
   only when ANTHROPIC_API_KEY is already set in the environment.
4. Verifier: rejects any sentence whose numbers or timestamps are not in the candidates it
   cites, any sentence that cites nothing, and any banned claim (technique, won/lost,
   comparison to elite norms).

No win/loss claims are made: who won each rally is not known.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from squashvid.analysis.hits import RECOVERY_RADIUS_M, T_RADIUS_M
from squashvid.analysis.io import PLAYERS, Match, other
from squashvid.analysis.stats import dist_from_T_by_rally

MIN_RALLIES = 8
N_BOOT = 600
ALPHA = 0.10
MAX_PER_KIND = 3
SHORT_AVAILABLE_S = 1.1
LLM_MODEL = "claude-sonnet-5-5"

# --- verifier ---------------------------------------------------------------------------------

BANNED = {
    "technique": r"\b(grip|racket face|racquet face|face of the racket|head position|head up|head down|keep your head|watch the ball|watch your ball|eyes? on the ball|eye on the ball|swing path|follow[- ]through|elbow|wrist|backswing|racket preparation|racquet preparation|technique)\b",
    "won_lost": r"\b(won|lost|lose|losing|loses|winning|wins|beat|beaten|defeat\w*)\b",
    "elite_norms": r"\b(elite|professional|professionals|pros?|world[- ]class|top players?|world ranking|psa|wsf|norms?|national standard|international)\b",
}
NUMBER = re.compile(r"\d+(?::\d+)?(?:\.\d+)?")


def banned_hits(text: str) -> list[str]:
    """Names of the banned categories a sentence falls into."""
    return [k for k, rx in BANNED.items() if re.search(rx, text, flags=re.I)]


def numbers_in(text: str) -> list[str]:
    return NUMBER.findall(text)


def allowed_numbers(cands: list[dict]) -> set[str]:
    """Number tokens a sentence may use: those displayed in the candidates it cites."""
    ok: set[str] = set()
    for c in cands:
        ev = c.get("evidence", [])
        for s in list(c.get("display", {}).values()) + [e.get("label", "") for e in ev] + [e.get("clip", "") for e in ev] + [clock(e["t0"]) for e in ev] + [clock(e["t1"]) for e in ev]:
            ok.update(numbers_in(str(s)))
    return ok


def verify_sentence(text: str, cites: list[str], by_id: dict[str, dict]) -> tuple[bool, str]:
    """(ok, reason). Numbers and timestamps must come from the cited candidates."""
    if not cites:
        return False, "no citation"
    missing = [c for c in cites if c not in by_id]
    if missing:
        return False, f"unknown candidate {missing[0]}"
    bad = banned_hits(text)
    if bad:
        return False, f"banned: {bad[0]}"
    ok = allowed_numbers([by_id[c] for c in cites])
    extra = [n for n in numbers_in(text) if n not in ok]
    if extra:
        return False, f"number not in cited candidates: {extra[0]}"
    return True, ""


def clock(t: float) -> str:
    t = max(float(t), 0.0)
    return f"{int(t // 60)}:{int(round(t % 60)) % 60:02d}" if round(t % 60) < 60 else f"{int(t // 60) + 1}:00"


# --- bootstrap over rallies -------------------------------------------------------------------


class _Resampler:
    """Rows of several rallies in one frame, so a resample is one positional take instead of many concats."""

    def __init__(self, groups: dict):
        keys = [k for k, g in groups.items() if len(g)]
        self.keys = keys
        self.frame = pd.concat([groups[k] for k in keys], ignore_index=True) if keys else pd.DataFrame()
        sizes = [len(groups[k]) for k in keys]
        ends = np.cumsum(sizes)
        self.pos = [np.arange(e - n, e) for e, n in zip(ends, sizes)]

    def take(self, pick: np.ndarray) -> pd.DataFrame:
        return self.frame.iloc[np.concatenate([self.pos[i] for i in pick])]


def cluster_bootstrap(groups: dict, stat: Callable[[pd.DataFrame], float], n_boot: int = N_BOOT, alpha: float = ALPHA, seed: int = 0) -> dict:
    """stat(all rows) with a percentile interval from resampling whole rallies with replacement."""
    rs = _Resampler(groups)
    k = len(rs.keys)
    if k == 0:
        return {"value": float("nan"), "ci": None, "n_rallies": 0}
    value = float(stat(rs.frame))
    if k < 2:
        return {"value": value, "ci": None, "n_rallies": k}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        v = stat(rs.take(rng.integers(0, k, k)))
        if np.isfinite(v):
            out.append(v)
    if len(out) < n_boot // 2:
        return {"value": value, "ci": None, "n_rallies": k}
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": value, "ci": [float(lo), float(hi)], "n_rallies": k}


def suppress(c: dict, min_rallies: int = MIN_RALLIES) -> str | None:
    """Reason to drop a candidate, or None: too few rallies, no interval, or an interval that crosses zero."""
    if c.get("n_rallies", 0) < min_rallies:
        return f"fewer than {min_rallies} rallies"
    ci = c.get("ci")
    if not ci or not np.isfinite(ci).all():
        return "no interval"
    if ci[0] <= 0 <= ci[1]:
        return "interval crosses zero"
    return None


# --- facts ------------------------------------------------------------------------------------


@dataclass
class Facts:
    """Everything the generators read. Rally ids are unique across clips."""

    hits: pd.DataFrame  # clip, rally (unique), t, hitter, opp/hitter positions, available_time, recovery...
    rallies: pd.DataFrame  # clip, rally (unique), start_t, end_t
    series: dict = field(default_factory=dict)  # rally -> {"A": stats.dist_from_T_by_rally samples, "B": ...}
    ball: pd.DataFrame | None = None  # per shot with a front-wall impact: clip, rally, hitter, z, x, bounce_y, t
    t_radius: float = T_RADIUS_M
    short_s: float = SHORT_AVAILABLE_S

    @classmethod
    def from_match(cls, m: Match, hits: pd.DataFrame, rallies: pd.DataFrame, offset: int = 0, ball: pd.DataFrame | None = None) -> "Facts":
        h = hits.copy()
        h.insert(0, "clip", m.stem)
        h["rally"] = h.rally + offset
        r = rallies.copy()
        r.insert(0, "clip", m.stem)
        r["rally"] = r.rally + offset
        per_player = {k: dist_from_T_by_rally(m, k, rallies) for k in PLAYERS}
        series = {rid: {k: per_player[k][i] for k in PLAYERS} for i, rid in enumerate(r.rally)}
        if ball is not None:
            ball = ball.copy()
            ball.insert(0, "clip", m.stem)
            ball["rally"] = ball.rally + offset
        return cls(h, r, series, ball)

    @classmethod
    def concat(cls, parts: list["Facts"]) -> "Facts":
        series: dict = {}
        for p in parts:
            series.update(p.series)
        balls = [p.ball for p in parts if p.ball is not None]
        ball = pd.concat(balls, ignore_index=True) if balls else None
        if ball is not None:
            ball.attrs.update(balls[0].attrs)
        return cls(pd.concat([p.hits for p in parts], ignore_index=True), pd.concat([p.rallies for p in parts], ignore_index=True), series, ball)


# --- candidate generators ---------------------------------------------------------------------

METRICS = {
    "t_at_strike": {"unit": "m", "better": "lower", "what": "how far from the T you are when your opponent hits", "dec": 1},
    "t_overall": {"unit": "m", "better": "lower", "what": "how far from the T you are during rallies", "dec": 1},
    "t_share": {"unit": "%", "better": "higher", "what": "how much of each rally you spend within about one metre of the T", "dec": 0, "scale": 100},
    "recovered": {"unit": "%", "better": "higher", "what": "how often you are back near the T before the next shot", "dec": 0, "scale": 100},
    "time_given": {"unit": "s", "better": "lower", "what": "how little time your shots leave your opponent", "dec": 2},
    "deep_bounce": {"unit": "%", "better": "higher", "what": "how often your shots first bounce in the back of the court", "dec": 0, "scale": 100},
}

COACHING = {
    "t_at_strike": ("Be back near the T before they hit.", "Ghosting: move to a corner, play an imaginary shot, then get back to the T and stop before the next call."),
    "t_overall": ("After every shot, take your first step back towards the middle.", "Hit and recover: play a shot, then touch the T with your foot before the next ball."),
    "t_share": ("Hit, then straight back to the T.", "Solo ghosting around the T: eight corners, always finishing on the T."),
    "recovered": ("Hit, then straight back to the T.", "Pressure ghosting: a partner calls corners quickly and you finish each one on the T."),
    "time_given": ("Hit early and deep so they have less time.", "Boast and drive: hit to the back, move up early to volley the reply, and keep the pace."),
    "deep_bounce": ("Aim for the back corners.", "Length drill: hit rails and see how many bounce behind the back of the service box."),
    "early_late": ("Keep your feet moving in the last minutes.", "Finish every practice with a minute of ghosting at full effort."),
    "short_time": ("When you are rushed, still get back to the T.", "Rushed-shot recovery: a partner feeds quick balls and you reset to the T after each one."),
}


def _f(x: float, dec: int, scale: float = 1.0) -> str:
    v = x * scale
    return f"{abs(v):.{dec}f}" if dec else f"{abs(round(v)):d}"


def _display(metric: str, value: float, comp: float | None, ci: list[float], n: int, n_rallies: int, extra: dict | None = None) -> dict:
    """Every number a sentence may quote, as the strings it will show. Differences are shown as sizes (the sign is in the wording)."""
    spec = METRICS.get(metric, {"dec": 2, "scale": 1})
    dec, scale = spec["dec"], spec.get("scale", 1)
    lo, hi = sorted((abs(ci[0]), abs(ci[1])))
    d = {"value": _f(value, dec, scale), "diff_lo": _f(lo, dec, scale), "diff_hi": _f(hi, dec, scale), "n": str(n), "n_rallies": str(n_rallies)}
    if comp is not None:
        d["comparison"] = _f(comp, dec, scale)
        d["diff"] = _f(value - comp, dec, scale)
    d.update({k: str(v) for k, v in (extra or {}).items()})
    return d


def _windows(rows: pd.DataFrame, k: int = 3, pad: float = 1.5, label_fn=None) -> list[dict]:
    """Up to k clip windows around hits in `rows` (in the order given), at least 8 s apart."""
    out: list[dict] = []
    for r in rows.itertuples():
        if any(e["clip"] == r.clip and abs(e["t0"] + pad - r.t) < 8 for e in out):
            continue
        out.append({"clip": r.clip, "t0": round(max(r.t - pad, 0), 2), "t1": round(r.t + pad, 2), "label": label_fn(r) if label_fn else f"rally {r.rally}"})
        if len(out) == k:
            break
    return out


def _by_rally(df: pd.DataFrame) -> dict:
    return {k: g for k, g in df.groupby("rally")}


def _order(rows: pd.DataFrame, favourable: bool, lower_better: bool) -> pd.DataFrame:
    """Best examples first for a strength, worst first for something to improve (by column `pick`)."""
    return rows.sort_values("pick", ascending=(lower_better == favourable))


def _candidate(p: str, generator: str, metric: str, comparison: str, value: float, comp_value: float, res: dict, n: int, n_rallies: int, title: str | None = None, extra: dict | None = None, n_label: str = "shots") -> dict:
    better = METRICS[metric]["better"]
    c = {"player": p, "generator": generator, "metric": metric, "comparison": comparison, "title": title or METRICS[metric]["what"], "value": value, "comparison_value": comp_value, "diff": res["value"], "ci": res["ci"], "n": n, "n_label": n_label, "n_rallies": n_rallies, "better": better}
    c["favourable"] = bool(np.isfinite(res["value"]) and ((res["value"] < 0) == (better == "lower")))
    if res["ci"]:
        c["display"] = _display(metric, value, comp_value, res["ci"], n, n_rallies, extra)
    return c


def generate(facts: Facts, m_T: np.ndarray, with_suppressed: bool = False) -> list[dict]:
    """All candidates that survive suppression (or all, each with a `suppressed` reason, if asked)."""
    cands: list[dict] = []
    h = facts.hits.copy()
    for p in PLAYERS:
        h[f"dT_{p}"] = np.hypot(h[f"{p}_x"] - m_T[0], h[f"{p}_y"] - m_T[1])
    rr = facts.rallies.copy()
    rids = sorted(rr.rally.unique())
    early_ids, late_ids = rids[: len(rids) // 2], rids[len(rids) // 2 :]
    R = RECOVERY_RADIUS_M

    # Metrics on the hits table: fn(d, q) for player q, rows(d, q) giving the rows used and the `pick` column.
    def strike_fn(d, q):
        g = d[d.hitter == other(q)]
        return float(g[f"dT_{q}"].mean()) if len(g) else np.nan

    def strike_rows(d, q):
        g = d[d.hitter == other(q)].copy()
        g["pick"] = g[f"dT_{q}"]
        return g

    def recovered_fn(d, q):
        g = d[(d.hitter == q) & (d.hitter_T_dist > R)]
        return float(g.recovered.mean()) if len(g) else np.nan

    def recovered_rows(d, q):
        g = d[(d.hitter == q) & (d.hitter_T_dist > R)].copy()
        g["pick"] = g.recovered.astype(float) * 10 - g.recovery_time
        return g

    hit_metrics = {
        "t_at_strike": (strike_fn, strike_rows, lambda r, q: f"rally {r.rally}: {q} was {r.pick:.1f} m from the T when {other(q)} hit"),
        "recovered": (recovered_fn, recovered_rows, lambda r, q: f"rally {r.rally}: {q} {'got back to the T' if r.pick > 5 else 'did not get back to the T'} before the next shot"),
    }
    for metric, (fn, rows_fn, label) in hit_metrics.items():
        better_lower = METRICS[metric]["better"] == "lower"
        for p in PLAYERS:
            o = other(p)
            res = cluster_bootstrap(_by_rally(h), lambda d, p=p, o=o, fn=fn: fn(d, p) - fn(d, o))
            rows = rows_fn(h, p)
            c = _candidate(p, "vs_opponent", metric, "vs opponent", fn(h, p), fn(h, o), res, int(len(rows)), res["n_rallies"])
            c["evidence"] = _windows(_order(rows, c["favourable"], better_lower), label_fn=lambda r, p=p, label=label: label(r, p))
            c["suppressed"] = suppress(c)
            cands.append(c)

    # Position metrics from the same distance-from-T samples as stats.py, so the report's table and its findings agree.
    def series_fn(metric):
        def fn(d, q):
            arrs = [facts.series[r][q] for r in d.rally.unique() if r in facts.series]
            if not arrs:
                return np.nan
            v = np.concatenate(arrs)
            if not len(v):
                return np.nan
            return float(np.mean(v)) if metric == "t_overall" else float(np.mean(v <= facts.t_radius))

        return fn

    for metric in ("t_overall", "t_share"):
        fn = series_fn(metric)
        lower = METRICS[metric]["better"] == "lower"
        for p in PLAYERS:
            o = other(p)
            res = cluster_bootstrap(_by_rally(rr), lambda d, p=p, o=o, fn=fn: fn(d, p) - fn(d, o))
            seconds = int(round(float((rr.end_t - rr.start_t).sum())))
            c = _candidate(p, "vs_opponent", metric, "vs opponent", fn(rr, p), fn(rr, o), res, seconds, res["n_rallies"], n_label="seconds of play")
            per = rr.copy()
            per["pick"] = [fn(rr[rr.rally == r], p) for r in rr.rally]
            ev = []
            for r in _order(per, c["favourable"], lower).itertuples():
                if len(ev) == 3:
                    break
                if any(e["clip"] == r.clip and abs(e["t0"] - r.start_t) < 8 for e in ev):
                    continue
                shown = f"{r.pick * 100:.0f}% of the time within 1 m of the T" if metric == "t_share" else f"{r.pick:.1f} m from the T on average"
                ev.append({"clip": r.clip, "t0": round(r.start_t, 2), "t1": round(min(r.end_t, r.start_t + 8), 2), "label": f"rally {r.rally}: {p} {shown}"})
            c["evidence"] = ev
            c["suppressed"] = suppress(c)
            cands.append(c)

    # Time each player's shots left the opponent, against the time they received (pressing).
    for p in PLAYERS:
        o = other(p)

        def given_minus_received(d, q=p, oo=o):
            a, b = d[d.hitter == oo].available_time, d[d.hitter == q].available_time
            return float(a.mean() - b.mean()) if a.notna().any() and b.notna().any() else np.nan

        res = cluster_bootstrap(_by_rally(h), given_minus_received)
        given, recv = float(h[h.hitter == o].available_time.mean()), float(h[h.hitter == p].available_time.mean())
        rows = h[(h.hitter == o) & h.available_time.notna()].copy()
        c = _candidate(p, "vs_opponent", "time_given", "time given to the opponent vs time received", given, recv, res, int(len(rows)), res["n_rallies"])
        rows["pick"] = rows.available_time
        c["evidence"] = _windows(_order(rows, c["favourable"], True), label_fn=lambda r, p=p: f"rally {r.rally}: {r.pick:.1f} s between {p}'s shot and the reply")
        c["suppressed"] = suppress(c)
        cands.append(c)

    # Early against late: the first half of the rallies against the second half, late minus early.
    for p in PLAYERS:
        for metric, (fn, rows_fn, label) in hit_metrics.items():
            ge, gl = _by_rally(h[h.rally.isin(early_ids)]), _by_rally(h[h.rally.isin(late_ids)])
            res = _two_sample(ge, gl, lambda d, fn=fn, p=p: fn(d, p))
            late_h = h[h.rally.isin(late_ids)]
            c = _candidate(p, "early_vs_late", metric, "late half vs early half", fn(late_h, p), fn(h[h.rally.isin(early_ids)], p), res, int(len(rows_fn(late_h, p))), min(len(ge), len(gl)))
            pool = h[h.rally.isin(late_ids)]  # the later rallies show the change: best examples if it improved, worst if not
            c["evidence"] = _windows(_order(rows_fn(pool, p), c["favourable"], METRICS[metric]["better"] == "lower"), label_fn=lambda r, p=p, label=label: label(r, p))
            c["suppressed"] = suppress(c)
            cands.append(c)

    # Recovery after a short against a long time available (split at the player's own median).
    for p in PLAYERS:
        g = h[(h.hitter == p) & (h.hitter_T_dist > R) & h.available_time.notna()]
        if len(g) < MIN_RALLIES:
            continue
        med = float(g.available_time.median())

        def short_minus_long(d, med=med):
            s_, l_ = d[d.available_time < med], d[d.available_time >= med]
            return float(s_.recovered.mean() - l_.recovered.mean()) if len(s_) and len(l_) else np.nan

        res = cluster_bootstrap(_by_rally(g), short_minus_long)
        short, long_ = g[g.available_time < med], g[g.available_time >= med]
        c = _candidate(p, "short_vs_long", "recovered", "after short vs long time available", float(short.recovered.mean()), float(long_.recovered.mean()), res, int(len(g)), res["n_rallies"], title="getting back to the T when you are rushed", extra={"split": f"{med:.1f}"})
        c["favourable"] = bool(np.isfinite(res["value"]) and res["value"] >= 0)
        miss = short[~short.recovered].copy()
        miss["pick"] = miss.recovery_time
        c["evidence"] = _windows(miss.sort_values("pick", ascending=False), label_fn=lambda r: f"rally {r.rally}: {r.pick:.1f} s without getting back to the T")
        c["suppressed"] = suppress(c)
        cands.append(c)

    if facts.ball is not None and len(facts.ball):
        cands.extend(_ball_candidates(facts))

    out = []
    for i, c in enumerate(cands):
        c["id"] = f"c{i + 1}"
        c.setdefault("display", {})
        c.setdefault("evidence", [])
        if c.get("suppressed") is None and not c["display"]:
            c["suppressed"] = "no interval"
        if with_suppressed or c["suppressed"] is None:
            out.append(c)
    return out


def _two_sample(ga: dict, gb: dict, stat, n_boot: int = N_BOOT, alpha: float = ALPHA, seed: int = 0) -> dict:
    """stat(b) - stat(a) with the two sets of rallies resampled independently."""
    ra, rb = _Resampler(ga), _Resampler(gb)
    ka, kb = len(ra.keys), len(rb.keys)
    if not ka or not kb:
        return {"value": float("nan"), "ci": None}
    value = float(stat(rb.frame) - stat(ra.frame))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        v = stat(rb.take(rng.integers(0, kb, kb))) - stat(ra.take(rng.integers(0, ka, ka)))
        if np.isfinite(v):
            out.append(v)
    if len(out) < n_boot // 2:
        return {"value": value, "ci": None}
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": value, "ci": [float(lo), float(hi)]}


def _ball_candidates(facts: Facts) -> list[dict]:
    b = facts.ball
    # Only length is judged. How close to the tin a shot goes is a choice (drops and kills aim there), so it is
    # reported in the stats table and never ranked as a strength or a weakness.
    b = b.assign(deep=(b.bounce_y.notna() & (b.bounce_y < b.attrs.get("box_back_y", 2.6))).astype(float))
    out = []
    for p in PLAYERS:
        o = other(p)
        for metric, col in (("deep_bounce", "deep"),):

            def fn(d, q, col=col):
                g = d[d.hitter == q][col]
                return float(g.mean()) if len(g) else np.nan

            res = cluster_bootstrap(_by_rally(b), lambda d, p=p, o=o, fn=fn: fn(d, p) - fn(d, o))
            c = _candidate(p, "vs_opponent", metric, "vs opponent", fn(b, p), fn(b, o), res, int((b.hitter == p).sum()), res["n_rallies"])
            rows = b[b.hitter == p].copy()
            rows["pick"] = rows[col]
            rows = rows[rows.pick > 0] if (METRICS[metric]["better"] == "lower") != c["favourable"] else rows[rows.pick == 0]
            c["evidence"] = _windows(rows, label_fn=lambda r: f"rally {r.rally}: front wall at {r.z:.1f} m high")
            c["suppressed"] = suppress(c)
            out.append(c)
    return out


# --- editor -----------------------------------------------------------------------------------


STEMS = {
    "t_at_strike": "{p} was {v} m from the T when {o} hit",
    "t_overall": "{p} was on average {v} m from the T during rallies",
    "t_share": "{p} spent {v}% of rally time within about 1 m of the T",
    "recovered": "{p} was back near the T before the next shot {v}% of the time",
    "time_given": "{p}'s shots left {o} {v} s before the reply",
    "deep_bounce": "{p}'s shots first bounced in the back of the court {v}% of the time",
}


def _phrase(c: dict) -> list[str]:
    """Templated sentences for one candidate; every number comes from c['display']."""
    d, p, o = c["display"], c["player"], other(c["player"])
    metric, gen = c["metric"], c["generator"]
    unit = METRICS[metric]["unit"]
    u = {"%": "%", "m": " m", "s": " s"}[unit]
    pu = " percentage points" if unit == "%" else u
    size = f"The difference is about {d['diff']}{pu}, probably between {d['diff_lo']} and {d['diff_hi']}{pu}"
    base = f"({d['n_rallies']} rallies, {d['n']} {c.get('n_label', 'shots')}). {size}."
    if gen == "short_vs_long":
        return [f"After shots hit with less than {d['split']} s since the opponent's shot, {p} was back near the T before the next ball {d['value']}% of the time, against {d['comparison']}% after shots with more time {base}"]
    stem = STEMS[metric].format(p=p, o=o, v=d["value"])
    if gen == "early_vs_late":
        return [f"In the later rallies of the clip, {stem}, against {d['comparison']}{u.strip() if unit == '%' else u} in the earlier ones {base}"]
    if metric == "time_given":
        return [f"{stem}, and {o}'s shots left {p} {d['comparison']} s {base}"]
    return [f"{stem}, against {d['comparison']}{u.strip() if unit == '%' else u} for {o} {base}"]


def template_editor(cands: list[dict]) -> dict:
    """At most three strengths and three improvements per player, templated and cited."""
    by_player: dict = {}
    for p in PLAYERS:
        mine = [c for c in cands if c["player"] == p]
        # One item per metric per player, across strengths and improvements alike, so a behaviour is never praised and
        # criticised in the same report: the clearest evidence (largest standardised gap) wins.
        def key(c): return abs(c["diff"]) / max((c["ci"][1] - c["ci"][0]) / 2, 1e-9)
        items = {"strengths": [], "improvements": []}
        seen: set[str] = set()
        for c in sorted(mine, key=key, reverse=True):
            kind = "strengths" if c["favourable"] else "improvements"
            if c["metric"] in seen or len(items[kind]) == MAX_PER_KIND:
                continue
            seen.add(c["metric"])
            items[kind].append(_item(c, kind))
        by_player[p] = items
    return by_player


def _item(c: dict, kind: str) -> dict:
    cue_key = "early_late" if c["generator"] == "early_vs_late" else "short_time" if c["generator"] == "short_vs_long" else c["metric"]
    cue, drill = COACHING[cue_key]
    sents = [{"text": s, "cites": [c["id"]]} for s in _phrase(c)]  # the clips travel in `evidence`, not as a sentence
    if kind == "improvements":
        sents.append({"text": f"Cue: {cue}", "cites": [c["id"]]})
        sents.append({"text": f"Drill: {drill}", "cites": [c["id"]]})
    else:
        sents.append({"text": f"Keep it going. Cue: {cue}", "cites": [c["id"]]})
    return {"behaviour": c["title"], "metric": c["metric"], "candidate": c["id"], "sentences": sents, "evidence": [{"t0": e["t0"], "t1": e["t1"], "label": e["label"], "clip": e["clip"]} for e in c["evidence"]], "cue": cue, "drill": drill, "number": {"value": c["display"]["value"], "unit": METRICS[c["metric"]]["unit"], "n": c["n"], "n_label": c.get("n_label", "shots"), "n_rallies": c["n_rallies"], "difference_size_interval": [c["display"]["diff_lo"], c["display"]["diff_hi"]]}}


# --- LLM editor (optional) --------------------------------------------------------------------

LLM_SYSTEM = (
    "You edit coaching notes for a junior squash player's parent and coach. You are given candidate findings as JSON, "
    "each with an id, display numbers and evidence clips. Choose at most three strengths and three things to work on per player. "
    "Write plain, kind sentences. Every sentence must cite the candidate ids it relies on, use only numbers shown in those candidates, "
    "and make no technique claims, no won/lost claims and no comparison to elite players. Reply with JSON only: "
    '{"A": {"strengths": [{"candidate": id, "sentences": [{"text": str, "cites": [id]}]}], "improvements": [...]}, "B": {...}}.'
)


def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def default_llm_call(prompt: str) -> str:  # pragma: no cover - needs a key and network
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(model=LLM_MODEL, max_tokens=3000, system=LLM_SYSTEM, messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def llm_editor(cands: list[dict], call: Callable[[str], str] | None = None) -> dict | None:
    """Parse the model's JSON into items; None when no key, or nothing usable. Sentences are verified later."""
    if call is None:
        if not llm_available():
            return None
        call = default_llm_call
    slim = [{k: c[k] for k in ("id", "player", "metric", "title", "comparison", "favourable", "display", "evidence")} for c in cands]
    try:
        raw = call(json.dumps(slim))
        data = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
    except Exception:  # noqa: BLE001 - any failure falls back to the template
        return None
    by_id = {c["id"]: c for c in cands}
    out: dict = {}
    for p in PLAYERS:
        out[p] = {"strengths": [], "improvements": []}
        for kind in ("strengths", "improvements"):
            for it in (data.get(p, {}).get(kind, []) or [])[:MAX_PER_KIND]:
                c = by_id.get(it.get("candidate"))
                if not c or c["player"] != p or c["favourable"] != (kind == "strengths"):
                    continue
                item = _item(c, kind)
                item["sentences"] = [{"text": str(s.get("text", "")), "cites": [x for x in s.get("cites", []) if isinstance(x, str)]} for s in it.get("sentences", [])]
                item["editor"] = "llm"
                out[p][kind].append(item)
    return out


def verify_items(items: dict, cands: list[dict]) -> tuple[dict, list[dict]]:
    """Drop sentences that fail verification; drop items left without a sentence. Returns (kept, rejections)."""
    by_id = {c["id"]: c for c in cands}
    rejected: list[dict] = []
    kept: dict = {}
    for p, kinds in items.items():
        kept[p] = {}
        for kind, lst in kinds.items():
            kept[p][kind] = []
            for it in lst:
                good = []
                for s in it["sentences"]:
                    ok, why = verify_sentence(s["text"], s["cites"], by_id)
                    if ok:
                        good.append(s)
                    else:
                        rejected.append({"player": p, "text": s["text"], "reason": why})
                if good:
                    kept[p][kind].append({**it, "sentences": good})
    return kept, rejected


def edit(cands: list[dict], llm_call: Callable[[str], str] | None = None) -> tuple[dict, list[dict], str]:
    """Editor + verifier. Uses the LLM editor when available, falls back to the template per player/kind."""
    template, rejected_t = verify_items(template_editor(cands), cands)
    editor = "template"
    llm = llm_editor(cands, llm_call)
    if llm:
        kept, rejected = verify_items(llm, cands)
        for p in PLAYERS:
            for kind in ("strengths", "improvements"):
                if not kept[p][kind]:
                    kept[p][kind] = template[p][kind]
        template, rejected_t, editor = kept, rejected_t + rejected, "llm+template"
    return template, rejected_t, editor


# --- entry point ------------------------------------------------------------------------------


def build(m: Match, hits: pd.DataFrame, rallies: pd.DataFrame, stats: dict | None = None, facts: Facts | None = None) -> dict:
    """Insights for one clip (or pass `facts` built over several clips)."""
    if facts is None:
        ball = ball_table(m, hits)
        facts = Facts.from_match(m, hits, rallies, ball=ball)
    all_cands = generate(facts, m.T, with_suppressed=True)
    cands = [c for c in all_cands if c["suppressed"] is None]
    items, rejected, editor = edit(cands)
    n_r = int(facts.rallies.rally.nunique())
    return {
        "clip": m.stem,
        "editor": editor,
        "ball_events_used": m.ball_events is not None,
        "n_rallies": n_r,
        "note": "No finding is reported unless its 90% interval over rallies excludes zero and it rests on at least eight rallies; who won each rally is not known, so nothing compares winners and losers.",
        "players": {p: {"strengths": [_pub(i) for i in items[p]["strengths"]], "improvements": [_pub(i) for i in items[p]["improvements"]]} for p in PLAYERS},
        "candidates": [_pub_c(c) for c in all_cands],
        "rejected_sentences": rejected,
    }


def _pub(i: dict) -> dict:
    return {"behaviour": i["behaviour"], "metric": i["metric"], "candidate": i["candidate"], "number": i["number"], "text": " ".join(s["text"] for s in i["sentences"]), "sentences": i["sentences"], "evidence": i["evidence"], "cue": i["cue"], "drill": i["drill"]}


def _pub_c(c: dict) -> dict:
    keep = ("id", "player", "generator", "metric", "title", "comparison", "value", "comparison_value", "diff", "ci", "n", "n_label", "n_rallies", "favourable", "suppressed", "evidence", "display")
    return {k: (None if isinstance(c.get(k), float) and not np.isfinite(c[k]) else c.get(k)) for k in keep}


def ball_table(m: Match, hits: pd.DataFrame) -> pd.DataFrame | None:
    """Per shot with a front-wall impact: height, width and first bounce, from ball events. None without ball events."""
    ev = m.ball_events
    if ev is None or len(hits) == 0:
        return None
    fw, fl = ev[ev.kind == "front_wall"], ev[ev.kind == "floor"]
    rows = []
    h = hits.sort_values(["rally", "shot_index"])
    nxt = h.groupby("rally").t.shift(-1).fillna(h.t + 3.0)
    for (_, r), t_next in zip(h.iterrows(), nxt):
        f = fw[(fw.t > r.t) & (fw.t < min(t_next, r.t + 1.5))]
        if len(f) == 0:
            continue
        f0 = f.iloc[0]
        b = fl[(fl.t > f0.t) & (fl.t < max(t_next, f0.t + 1.2))]
        rows.append({"rally": r.rally, "hitter": r.hitter, "t": r.t, "z": float(f0.z), "x": float(f0.x), "bounce_y": float(b.iloc[0].y) if len(b) else np.nan})
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d.attrs["box_back_y"] = float(m.spec.box_back_y)
    return d

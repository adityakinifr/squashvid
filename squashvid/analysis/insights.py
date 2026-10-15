"""Coaching insights: facts -> candidate generators -> editor -> verifier.

1. Facts: the hits table (positions, available time, recovery) per rally.
2. Candidates: rule-based comparisons (player vs opponent, early vs late, recovery
   after short vs long time available, ball metrics when ball events exist). Each has a
   value, a comparison, n, a 90% bootstrap interval over rallies and 2-3 evidence
   clips. A candidate is suppressed when its interval crosses zero or it rests on
   fewer than MIN_RALLIES rallies.
   Situational generators compare a player with themself (how often the next shot was
   rushed when they were far from the T, against near it) so the coaching can say what a
   habit cost. A candidate smaller than its metric's `min_effect` is suppressed as too
   small to matter, however sure the interval.
3. Editor: squashvid.analysis.coach groups candidates into themes and writes the coaching.
4. Verifier: rejects any sentence whose numbers or timestamps are not in the candidates it
   cites, any sentence that cites nothing, and any banned claim (technique, won/lost,
   comparison to elite norms). A candidate may lift a ban for the sentences that cite it
   (`allows`): those built on the inferred rally winners may say won and lost.

Players appear in text as {A} and {B} (see `tok`); coach.fill puts in the saved names.

Who won a rally is inferred from who served the next one (squashvid.analysis.serves), never
from the ball alone. Rallies without an inferred winner are left out of those candidates.

How low a player got at the strike comes from the 3D figures (squashvid.analysis.posture); hits
without a figure are left out of that candidate.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from squashvid.analysis.hits import RECOVERY_RADIUS_M, T_RADIUS_M
from squashvid.analysis.io import PLAYERS, Match, other
from squashvid.analysis.posture import UPRIGHT
from squashvid.analysis.stats import dist_from_T_by_rally

MIN_RALLIES = 8
N_BOOT = 600
ALPHA = 0.10
SHORT_AVAILABLE_S = 1.1

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
    lifted = {b for c in cites for b in by_id[c].get("allows", ())}
    bad = [b for b in banned_hits(text) if b not in lifted]
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

# Pooled facts shift each clip's rally ids by a multiple of RALLY_BLOCK to keep them unique; labels show the clip's own number.
RALLY_BLOCK = 1000


def rally_no(rally) -> int:
    return int(rally) % RALLY_BLOCK


def tok(player: str) -> str:
    """How generated text names a player; coach.fill swaps in the saved name."""
    return "{" + player + "}"


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

# min_effect: the smallest difference worth coaching. A 0.2 m gap in average distance from the T can be certain and still
# mean nothing on court, so findings below these sizes are dropped whatever their interval.
METRICS = {
    "t_at_strike": {"unit": "m", "better": "lower", "what": "how far from the T you are when your opponent hits", "dec": 1, "min_effect": 0.3},
    "t_overall": {"unit": "m", "better": "lower", "what": "how far from the T you are during rallies", "dec": 1, "min_effect": 0.3},
    "t_share": {"unit": "%", "better": "higher", "what": "how much of each rally you spend within about one metre of the T", "dec": 0, "scale": 100, "min_effect": 0.08},
    "recovered": {"unit": "%", "better": "higher", "what": "how often you are back near the T before the next shot", "dec": 0, "scale": 100, "min_effect": 0.08},
    "time_given": {"unit": "s", "better": "lower", "what": "how little time your shots leave your opponent", "dec": 2, "min_effect": 0.1},
    "deep_bounce": {"unit": "%", "better": "higher", "what": "how often your shots first bounce in the back of the court", "dec": 0, "scale": 100, "min_effect": 0.1},
    "recover_from_back": {"unit": "%", "better": "higher", "what": "getting back to the T after a shot from the back of the court", "dec": 0, "scale": 100, "min_effect": 0.1},
    "off_T_cost": {"unit": "%", "better": "lower", "what": "how often your next shot is rushed when you are far from the T as your opponent hits", "dec": 0, "scale": 100, "min_effect": 0.1},
    "won_on_T": {"unit": "%", "better": "higher", "what": "how often you win the rallies in which you hold the T more than your opponent, against the others", "dec": 0, "scale": 100, "min_effect": 0.1},
    "last_shot_error": {"unit": "%", "better": "lower", "what": "how often a rally that ends on your shot goes to your opponent", "dec": 0, "scale": 100, "min_effect": 0.1},
    "upright_at_back": {"unit": "%", "better": "lower", "what": "how often you hit from the back of the court standing almost upright", "dec": 0, "scale": 100, "min_effect": 0.1},
}
FAR_FROM_T_M = 2.0  # "far from the T" at the opponent's strike, for off_T_cost
RUSHED = ("defence", "pressure")  # hits.SITUATION_RULES classes with under about 1.1 s, or stretched


def _f(x: float, dec: int, scale: float = 1.0) -> str:
    v = x * scale
    return f"{abs(v):.{dec}f}" if dec else f"{abs(round(v)):d}"


# Fractions a reader takes in at a glance, for the plain summary ("about 1 in 3").
EVERYDAY_FRACTIONS = [(1, 10), (1, 8), (1, 7), (1, 6), (1, 5), (1, 4), (3, 10), (1, 3), (2, 5), (1, 2), (3, 5), (2, 3), (3, 4), (4, 5), (9, 10)]


def share_words(x: float) -> str:
    """A share as an everyday fraction: "about 1 in 3", "almost none", "almost all"."""
    if not np.isfinite(x):
        return "unknown"
    if x < 0.05:
        return "almost none"
    if x > 0.95:
        return "almost all"
    a, b = min(EVERYDAY_FRACTIONS, key=lambda f: abs(f[0] / f[1] - x))
    return f"about {a} in {b}"


def _display(metric: str, value: float, comp: float | None, ci: list[float], n: int, n_rallies: int, extra: dict | None = None) -> dict:
    """Every number a sentence may quote, as the strings it will show. Differences are shown as sizes (the sign is in the wording)."""
    spec = METRICS.get(metric, {"dec": 2, "scale": 1})
    dec, scale = spec["dec"], spec.get("scale", 1)
    lo, hi = sorted((abs(ci[0]), abs(ci[1])))
    d = {"value": _f(value, dec, scale), "diff_lo": _f(lo, dec, scale), "diff_hi": _f(hi, dec, scale), "n": str(n), "n_rallies": str(n_rallies)}
    if comp is not None:
        d["comparison"] = _f(comp, dec, scale)
        # The size is the difference of the two numbers as shown, so a reader's own subtraction agrees with it.
        d["diff"] = _f(round(value * scale, dec) - round(comp * scale, dec), dec)
    if spec.get("unit") == "%":
        d["v_in"] = share_words(value)
        if comp is not None:
            d["c_in"] = share_words(comp)
    d.update({k: str(v) for k, v in (extra or {}).items()})
    return d


def _windows(rows: pd.DataFrame, k: int = 3, pad: float = 1.5, label_fn=None) -> list[dict]:
    """Up to k clip windows around hits in `rows` (in the order given), at least 8 s apart."""
    out: list[dict] = []
    for r in rows.itertuples():
        if any(e["clip"] == r.clip and abs(e["t0"] + pad - r.t) < 8 for e in out):
            continue
        out.append({"clip": r.clip, "t0": round(max(r.t - pad, 0), 2), "t1": round(r.t + pad, 2), "label": label_fn(r) if label_fn else f"rally {rally_no(r.rally)}"})
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
        "t_at_strike": (strike_fn, strike_rows, lambda r, q: f"rally {rally_no(r.rally)}: {tok(q)} was {r.pick:.1f} m from the T when {tok(other(q))} hit"),
        "recovered": (recovered_fn, recovered_rows, lambda r, q: f"rally {rally_no(r.rally)}: {tok(q)} {'got back to the T' if r.pick > 5 else 'did not get back to the T'} before the next shot"),
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
                shown = f"was within 1 m of the T for {r.pick * 100:.0f}% of the rally" if metric == "t_share" else f"was {r.pick:.1f} m from the T on average"
                ev.append({"clip": r.clip, "t0": round(r.start_t, 2), "t1": round(min(r.end_t, r.start_t + 8), 2), "label": f"rally {rally_no(r.rally)}: {tok(p)} {shown}"})
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
        c["evidence"] = _windows(_order(rows, c["favourable"], True), label_fn=lambda r, p=p: f"rally {rally_no(r.rally)}: {r.pick:.1f} s between {tok(p)}'s shot and the reply")
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
        c["evidence"] = _windows(miss.sort_values("pick", ascending=False), label_fn=lambda r: f"rally {rally_no(r.rally)}: {r.pick:.1f} s without getting back to the T")
        c["suppressed"] = suppress(c)
        cands.append(c)

    cands.extend(_situational_candidates(h))
    if "winner" in rr:
        cands.extend(_winner_candidates(facts, h))
    if facts.ball is not None and len(facts.ball):
        cands.extend(_ball_candidates(facts))
    if "hip_drop" in h and h.hip_drop.notna().any():
        cands.extend(_posture_candidates(h))

    return finalise(cands, with_suppressed)


def finalise(cands: list[dict], with_suppressed: bool = False) -> list[dict]:
    """Number the candidates and suppress those with no interval or a gap below their metric's min_effect."""
    out = []
    for i, c in enumerate(cands):
        c["id"] = f"c{i + 1}"
        c.setdefault("display", {})
        c.setdefault("evidence", [])
        if c.get("suppressed") is None and not c["display"]:
            c["suppressed"] = "no interval"
        if c.get("suppressed") is None and abs(c["diff"]) < METRICS[c["metric"]].get("min_effect", 0):
            c["suppressed"] = "too small to matter"
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


def _situational_candidates(h: pd.DataFrame) -> list[dict]:
    """A player against themself or the opponent in one situation: the cost of being far from the T, and recovery from the back."""
    out = []
    h = h.sort_values(["rally", "shot_index"])
    nxt = h.groupby("rally")[["hitter", "situation"]].shift(-1)
    pairs = h.assign(next_hitter=nxt.hitter, next_rushed=nxt.situation.isin(RUSHED))
    pairs = pairs[pairs.next_hitter.notna() & (pairs.next_hitter != pairs.hitter)]
    for p in PLAYERS:
        o = other(p)
        # The opponent hits while p waits (p is the hit's `opp`); was p's next shot rushed?
        g = pairs[pairs.hitter == o].assign(far=lambda d: d.opp_T_dist > FAR_FROM_T_M)

        def rushed(d, far):
            x = d[d.far == far].next_rushed
            return float(x.mean()) if len(x) else np.nan

        res = cluster_bootstrap(_by_rally(g), lambda d: rushed(d, True) - rushed(d, False))
        c = _candidate(p, "situational", "off_T_cost", f"more than {FAR_FROM_T_M:g} m from the T vs nearer", rushed(g, True), rushed(g, False), res, int(g.far.sum()), res["n_rallies"], extra={"far_m": f"{FAR_FROM_T_M:g}"}, n_label="moments far from the T")
        c["role"] = "why"  # a consequence, told as the reason behind a position finding rather than on its own
        rows = g[g.far & g.next_rushed].assign(pick=lambda d: d.opp_T_dist).sort_values("pick", ascending=False)
        c["evidence"] = _windows(rows, label_fn=lambda r, p=p, o=o: f"rally {rally_no(r.rally)}: {tok(p)} was {r.pick:.1f} m from the T as {tok(o)} hit")
        c["suppressed"] = suppress(c)
        out.append(c)

    # Recovery after a shot from the back third (the cell behind the service boxes), player against opponent.
    back = h[h.cell_hit_from.fillna("").str.startswith("back") & (h.hitter_T_dist > RECOVERY_RADIUS_M)]

    def rec(d, q):
        x = d[d.hitter == q].recovered
        return float(x.mean()) if len(x) else np.nan

    for p in PLAYERS:
        o = other(p)
        res = cluster_bootstrap(_by_rally(back), lambda d, p=p, o=o: rec(d, p) - rec(d, o))
        c = _candidate(p, "vs_opponent", "recover_from_back", "vs opponent", rec(back, p), rec(back, o), res, int((back.hitter == p).sum()), res["n_rallies"], n_label="shots from the back")
        rows = back[back.hitter == p].assign(pick=lambda d: d.recovered.astype(float) * 10 - d.recovery_time)
        c["evidence"] = _windows(_order(rows, c["favourable"], False), label_fn=lambda r, p=p, o=o: f"rally {rally_no(r.rally)}: {tok(p)} {'got back to the T' if r.pick > 5 else 'was still away from the T'} when {tok(o)} replied")
        c["suppressed"] = suppress(c)
        out.append(c)
    return out


def _winner_candidates(facts: Facts, h: pd.DataFrame) -> list[dict]:
    """From the inferred rally winners: how the rallies that ended on a player's shot went, and how often a player
    won when holding the T more than the opponent (a reason behind the position findings)."""
    out = []
    d = facts.rallies[facts.rallies.winner.notna()].copy()
    if not len(d):
        return out
    last = h.sort_values("t").groupby("rally").tail(1).set_index("rally")
    d["last"] = d.rally.map(last.hitter)
    d["last_t"] = d.rally.map(last.t)
    d = d[d["last"].notna()]
    groups = {r: g for r, g in d.groupby("rally")}

    def lost_own(x, q):
        g = x[x["last"] == q]
        return float((g.winner != q).mean()) if len(g) else np.nan

    for p in PLAYERS:
        o = other(p)
        mine, theirs = d[d["last"] == p], d[d["last"] == o]
        res = cluster_bootstrap(groups, lambda x, p=p, o=o: lost_own(x, p) - lost_own(x, o))
        counts = {"n_own": len(mine), "lost_own": int((mine.winner != p).sum()), "o_n_own": len(theirs), "o_lost_own": int((theirs.winner != o).sum())}
        c = _candidate(p, "vs_opponent", "last_shot_error", "vs opponent", lost_own(d, p), lost_own(d, o), res, len(mine), res["n_rallies"], extra=counts, n_label="rallies that ended on their shot")
        c["allows"] = ["won_lost"]
        rows = mine[(mine.winner != p) != c["favourable"]].assign(t=lambda x: x.last_t)
        c["evidence"] = _windows(rows, label_fn=lambda r, p=p, o=o: f"rally {rally_no(r.rally)}: {tok(p)} hit the last shot, then {tok(r.winner)} served the next rally")
        c["suppressed"] = suppress(c)
        out.append(c)

    share = {r: {q: float(np.mean(v[q] <= facts.t_radius)) if len(v[q]) else np.nan for q in PLAYERS} for r, v in facts.series.items()}
    for p in PLAYERS:
        o = other(p)
        x = d.assign(more=[share.get(r, {}).get(p, np.nan) > share.get(r, {}).get(o, np.nan) for r in d.rally], win=d.winner == p)
        more, less = x[x.more], x[~x.more]

        def gap(y):
            a, b = y[y.more].win, y[~y.more].win
            return float(a.mean() - b.mean()) if len(a) and len(b) else np.nan

        res = cluster_bootstrap({r: g for r, g in x.groupby("rally")}, gap)
        counts = {"n_more": len(more), "won_more": int(more.win.sum()), "n_less": len(less), "won_less": int(less.win.sum())}
        c = _candidate(p, "situational", "won_on_T", "holding the T more vs less than the opponent", float(more.win.mean()) if len(more) else np.nan, float(less.win.mean()) if len(less) else np.nan, res, len(more), res["n_rallies"], extra=counts, n_label="rallies holding the T more")
        c["role"] = "why"
        c["allows"] = ["won_lost"]
        rows = less[~less.win].assign(t=lambda y: y.start_t + 1.5)
        c["evidence"] = _windows(rows, label_fn=lambda r, p=p, o=o: f"rally {rally_no(r.rally)}: {tok(o)} held the T more, then served the next rally")
        c["suppressed"] = suppress(c)
        out.append(c)
    return out


def _posture_candidates(h: pd.DataFrame) -> list[dict]:
    """How often each player hit from behind the service boxes standing almost upright, against the opponent.
    Behind the boxes nearly every shot is played off a low bounce, so straight legs there are worth a coach's look;
    in mid-court a volley is rightly hit upright. Each rally's first shot is left out: serves are hit standing,
    and the most upright "hits" were mostly serves or hits given to the wrong player."""
    back = h[h.cell_hit_from.fillna("").str.startswith("back") & (h.shot_index > 1) & h.hip_drop.notna()]
    back = back.assign(upright=back.hip_drop < UPRIGHT)

    def share(d, q):
        x = d[d.hitter == q].upright
        return float(x.mean()) if len(x) else np.nan

    out = []
    for p in PLAYERS:
        o = other(p)
        res = cluster_bootstrap(_by_rally(back), lambda d, p=p, o=o: share(d, p) - share(d, o))
        mine = back[back.hitter == p]
        c = _candidate(p, "vs_opponent", "upright_at_back", "vs opponent", share(back, p), share(back, o), res, len(mine), res["n_rallies"], extra={"upright_pct": f"{UPRIGHT * 100:g}"}, n_label="shots from the back")
        # The surest hits first; upright ones for something to try, bent-knee ones (short of a scramble) for a strength.
        rows = mine[mine.upright] if not c["favourable"] else mine[mine.hip_drop.between(0.15, 0.45)]
        c["evidence"] = _windows(rows.sort_values("confidence", ascending=False), label_fn=lambda r, p=p: f"rally {rally_no(r.rally)}: {tok(p)} hit from the back with the hips {r.hip_drop * 100:.0f}% below standing height")
        c["suppressed"] = suppress(c)
        out.append(c)
    return out


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
            c["evidence"] = _windows(rows, label_fn=lambda r: f"rally {rally_no(r.rally)}: front wall at {r.z:.1f} m high")
            c["suppressed"] = suppress(c)
            out.append(c)
    return out


# --- entry point ------------------------------------------------------------------------------


def build(m: Match, hits: pd.DataFrame, rallies: pd.DataFrame, stats: dict | None = None, facts: Facts | None = None) -> dict:
    """Insights for one clip (or pass `facts` built over several clips)."""
    from squashvid.analysis import coach

    if facts is None:
        ball = ball_table(m, hits)
        facts = Facts.from_match(m, hits, rallies, ball=ball)
    from squashvid.analysis.serves import outcomes

    all_cands = generate(facts, m.T, with_suppressed=True)
    players, rejected = coach.write([c for c in all_cands if c["suppressed"] is None])
    return {
        "clip": m.stem,
        "editor": "coach",
        "ball_events_used": m.ball_events is not None,
        "n_rallies": int(facts.rallies.rally.nunique()),
        "outcomes": outcomes(facts.rallies, facts.hits),
        "note": "A finding is used only when its 90% interval over rallies excludes zero, it rests on at least eight rallies and the difference is big enough to matter on court.",
        "players": players,
        "candidates": [_pub_c(c) for c in all_cands],
        "rejected_sentences": rejected,
    }


def _pub_c(c: dict) -> dict:
    keep = ("id", "player", "generator", "metric", "title", "comparison", "value", "comparison_value", "diff", "ci", "n", "n_label", "n_rallies", "favourable", "suppressed", "evidence", "display", "role", "allows")
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

"""Score audio onsets and rallies against visual labels.

Visual and audio clocks differ by a constant container offset plus the time
sound takes to reach the camera: 343 m/s is 2.9 ms per metre, so a hit 10 m
away is heard about a frame late. Labels are therefore moved onto the audio
clock (visual time + travel + offset) before matching, and the offset itself is
estimated from the clearest labels only.

Matching is one-to-one and greedy by distance, so a burst of onsets around one
hit counts one true positive and the rest as false positives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPEED_OF_SOUND = 343.0  # m/s at ~20 C
CAMERA_BEHIND_BACK_WALL = 2.0  # m, phone on the walkway behind the back glass
CAMERA_HEIGHT = 1.2  # m
FRAME = 1 / 30


def travel_s(x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray = 1.0, court_width: float = 6.15) -> np.ndarray:
    """Seconds for sound from court point (x across from the left wall, y from the back wall, z up) to the camera."""
    d = np.sqrt((np.asarray(x) - court_width / 2) ** 2 + (np.asarray(y) + CAMERA_BEHIND_BACK_WALL) ** 2 + (np.asarray(z) - CAMERA_HEIGHT) ** 2)
    return d / SPEED_OF_SOUND


def match(pred: np.ndarray, true: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Greedy one-to-one pairs (i_pred, i_true) with |pred - true| <= tol, closest pairs first."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    if len(pred) == 0 or len(true) == 0:
        return []
    j = np.searchsorted(pred, true)
    pairs = []
    for it, t in enumerate(true):
        for ip in range(max(0, j[it] - 3), min(len(pred), j[it] + 4)):
            d = abs(pred[ip] - t)
            if d <= tol:
                pairs.append((d, ip, it))
    used_p, used_t, out = set(), set(), []
    for _, ip, it in sorted(pairs):
        if ip not in used_p and it not in used_t:
            used_p.add(ip)
            used_t.add(it)
            out.append((ip, it))
    return sorted(out, key=lambda p: p[1])


def precision_recall(pred: np.ndarray, true: np.ndarray, tol: float, span: tuple[float, float] | None = None) -> dict:
    """P/R of predicted times against true times; with span, only predictions inside it count."""
    pred = np.sort(np.asarray(pred, float))
    if span is not None:
        pred = pred[(pred >= span[0]) & (pred <= span[1])]
    m = match(pred, np.sort(np.asarray(true, float)), tol)
    tp = len(m)
    return dict(tol_ms=round(tol * 1e3, 1), tp=tp, fp=len(pred) - tp, fn=len(true) - tp,
                precision=tp / len(pred) if len(pred) else float("nan"), recall=tp / len(true) if len(true) else float("nan"))


def estimate_offset(pred: np.ndarray, true: np.ndarray, window: float = 0.1, lock: float = 0.025) -> dict:
    """Audio minus (travel-corrected) visual time, locked onto the most common lag.

    Every onset within +-window of every label votes with its lag; the densest
    lag (20 ms kernel) wins, because the busy soundtrack puts a confounder near
    almost every label and a nearest-onset median drifts toward them. Each label
    then contributes its onset closest to that lag, if within +-lock of it.
    Returns the median of those, the median absolute deviation, a bootstrap 90%
    interval of the median, and N.
    """
    pred = np.sort(np.asarray(pred, float))
    true = np.asarray(true, float)
    lags = np.concatenate([pred[np.abs(pred - t) <= window] - t for t in true]) if len(true) else np.array([])
    if len(lags) == 0:
        return dict(n=0, median_ms=float("nan"), mad_ms=float("nan"), ci90_ms=(float("nan"), float("nan")), diffs_ms=[])
    grid = np.arange(-window, window + 1e-9, 0.001)
    dens = np.exp(-0.5 * ((grid[:, None] - lags[None, :]) / 0.01) ** 2).sum(1)
    mode = grid[np.argmax(dens)]
    d = []
    for t in true:
        near = pred[np.abs(pred - t - mode) <= lock]
        if len(near):
            d.append(near[np.argmin(np.abs(near - t - mode))] - t)
    d = np.array(d)
    if len(d) == 0:
        return dict(n=0, median_ms=float("nan"), mad_ms=float("nan"), ci90_ms=(float("nan"), float("nan")), diffs_ms=[])
    rng = np.random.default_rng(0)
    boots = np.median(rng.choice(d, (2000, len(d))), axis=1)
    med = float(np.median(d))
    return dict(n=len(d), n_labels=len(true), mode_ms=1e3 * float(mode), median_ms=1e3 * med, mad_ms=1e3 * float(np.median(np.abs(d - med))),
                ci90_ms=tuple(1e3 * np.percentile(boots, [5, 95])), diffs_ms=list(np.round(1e3 * d, 1)))


def interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def score_rallies(pred: pd.DataFrame, true: pd.DataFrame, min_iou: float = 0.3) -> dict:
    """Match predicted to true rallies by IoU (greedy); report counts and boundary errors in seconds."""
    pairs = []
    for i, p in pred.iterrows():
        for j, t in true.iterrows():
            inter = interval_overlap((p.start_t, p.end_t), (t.start_t, t.end_t))
            union = max(p.end_t, t.end_t) - min(p.start_t, t.start_t)
            if union > 0 and inter / union >= min_iou:
                pairs.append((inter / union, i, j))
    used_p, used_t, m = set(), set(), []
    for iou, i, j in sorted(pairs, reverse=True):
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            m.append((i, j, iou))
    ds = np.array([pred.loc[i, "start_t"] - true.loc[j, "start_t"] for i, j, _ in m])
    de = np.array([pred.loc[i, "end_t"] - true.loc[j, "end_t"] for i, j, _ in m])
    return dict(n_true=len(true), n_pred=len(pred), matched=len(m), missed=len(true) - len(m), spurious=len(pred) - len(m),
                mean_iou=float(np.mean([x[2] for x in m])) if m else float("nan"),
                start_err_median_s=float(np.median(ds)) if len(ds) else float("nan"), start_abs_err_median_s=float(np.median(np.abs(ds))) if len(ds) else float("nan"),
                end_err_median_s=float(np.median(de)) if len(de) else float("nan"), end_abs_err_median_s=float(np.median(np.abs(de))) if len(de) else float("nan"),
                pairs=[(int(i), int(j), round(iou, 2)) for i, j, iou in m])


DEFAULT_TRAVEL_M = {"racket": 6.0, "front_wall": 11.94, "side_wall": 8.0, "back_glass": 2.0, "floor": 5.0}
IMPACT_EVENTS = ("racket", "front_wall", "side_wall", "back_glass", "floor")


def label_spans(labels: pd.DataFrame, pad: float = 0.4) -> list[tuple[float, float]]:
    """The stretches a labeller covered: each rally from its first to its last labelled event (audio clock)."""
    g = labels.groupby("rally")["t_audio"]
    return sorted((lo - pad, hi + pad) for lo, hi in zip(g.min(), g.max()))


def evaluate_clip(onsets: pd.DataFrame, rallies: pd.DataFrame, labels: pd.DataFrame, offset_ms: float | None = None,
                  offset_conf: tuple[str, ...] = ("high", "med")) -> dict:
    """Score one clip's onsets against its visual labels (hit_labels.csv rows).

    The offset is estimated from racket labels of the given confidences against
    all onsets (any class) unless given. Precision counts predictions inside
    the labelled stretches only, since the rest of the clip was not labelled.
    any_onset_recall_chance is the recall that onsets at the same rate but at
    random times would reach (1 - exp(-2 * rate * tol)): on a busy soundtrack a
    quarter of labels or more find an onset by luck.
    """
    lab = labels.copy()
    if "travel_s" not in lab or lab["travel_s"].isna().any():
        dflt = lab["event"].map(DEFAULT_TRAVEL_M).fillna(6.0) / SPEED_OF_SOUND
        lab["travel_s"] = lab["travel_s"].fillna(dflt) if "travel_s" in lab else dflt
    all_t = np.sort(onsets["t"].to_numpy())
    rk = lab[lab.event == "racket"]
    est = estimate_offset(all_t, (rk[rk.confidence.isin(offset_conf)].t + rk[rk.confidence.isin(offset_conf)].travel_s).to_numpy())
    off = (offset_ms if offset_ms is not None else est["median_ms"]) / 1e3
    off = 0.0 if not np.isfinite(off) else off
    lab["t_audio"] = lab["t"] + lab["travel_s"] + off
    rk = lab[lab.event == "racket"]
    spans = label_spans(lab[lab.event.isin(IMPACT_EVENTS)])
    inside = lambda t: np.array([any(a <= x <= b for a, b in spans) for x in t], bool)  # noqa: E731
    racket_t = np.sort(onsets.loc[onsets.kind == "racket", "t"].to_numpy())
    racket_t = racket_t[inside(racket_t)]
    any_t = all_t[inside(all_t)]
    span_s = sum(b - a for a, b in spans)
    rate = len(any_t) / span_s if span_s > 0 else 0.0  # onsets per second inside the labelled stretches
    out = dict(offset_estimate=est, offset_used_ms=1e3 * off, spans=[(round(a, 2), round(b, 2)) for a, b in spans],
               onsets_per_s_in_spans=round(rate, 2),
               n_labels={f"{e}/{c}": int(n) for (e, c), n in lab.groupby(["event", "confidence"]).size().items()})
    res = {}
    for conf_name, confs in [("high+med", ("high", "med")), ("all", ("high", "med", "low"))]:
        tr = np.sort(rk[rk.confidence.isin(confs)].t_audio.to_numpy())
        imp = np.sort(lab[lab.event.isin(IMPACT_EVENTS) & lab.confidence.isin(confs)].t_audio.to_numpy())
        for tol_name, tol in [("1f", FRAME), ("2f", 2 * FRAME)]:
            res[f"racket_class/{conf_name}/{tol_name}"] = precision_recall(racket_t, tr, tol)
            res[f"any_onset_recall/{conf_name}/{tol_name}"] = precision_recall(any_t, tr, tol)["recall"]
            res[f"any_onset_recall_chance/{tol_name}"] = 1 - float(np.exp(-rate * 2 * tol))
            res[f"any_onset_vs_impacts/{conf_name}/{tol_name}"] = precision_recall(any_t, imp, tol)
    out["scores"] = res
    m = match(racket_t, np.sort(rk.t_audio.to_numpy()), 2 * FRAME)
    used = {i for i, _ in m}
    out["racket_fp_times"] = [round(float(t), 3) for i, t in enumerate(racket_t) if i not in used]
    rows = []
    for r in lab.itertuples():
        k = np.searchsorted(all_t, r.t_audio)
        near = all_t[max(0, k - 2) : k + 2]
        dt = float(near[np.argmin(np.abs(near - r.t_audio))] - r.t_audio) if len(near) else float("nan")
        rows.append(dict(t=r.t, event=r.event, confidence=r.confidence, t_audio=round(r.t_audio, 3), nearest_onset_dt_ms=round(1e3 * dt, 1)))
    out["per_label"] = rows
    return out

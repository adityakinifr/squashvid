"""Held-out evaluation of the track and event stages on the six labelled windows.

    python -I -m squashvid.balltrack.evaluate [--folds AB C] [--json out.json]

Input is each window's cross-validation candidates (the fold model that never saw the window), the
same blobs a full-clip run would save. Three things are scored:

* Track points against the user's labels at 7.5 px: the raw detector (top blob per frame at threshold
  0.5), the linked track (detected + linked frames), and the interpolated frames separately.
* Events by kind against events derived from the labelled trajectory itself: the user's dense labels
  are run through the same kink detector and the same physics and player tests as the predicted track,
  with no audio, so the benchmark isolates what detection and linking cost. Hand-checked samples are in
  work/balltrack_review/.
* Front-wall impact position (metres on the wall) against the label-derived position.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from squashvid.ball import evaluate as ball_eval
from squashvid.ball.regions import venue
from squashvid.balltrack import data, events, link, pipeline

TOL_PX = 7.5
EDGE = 6  # frames at a window's ends where a kink cannot be seen from both sides
LABEL_RUN_GAP = 4  # frames: label runs are split at larger gaps
FOLD_GROUPS = {"AB": ["IMG_4091-r12", "IMG_4091-r16", "IMG_4089-r3", "IMG_4089-r11"], "C": ["IMG_4090-r18", "IMG_4090-r9"]}


def prf(tp: int, fp: int, fn: int) -> dict:
    return ball_eval.prf(int(tp), int(fp), int(fn))


def _pred_frame(pts: pd.DataFrame | None) -> pd.DataFrame:
    if pts is None or len(pts) == 0:
        return pd.DataFrame({"frame": [], "x": [], "y": [], "score": [], "visible": []})
    return pd.DataFrame({"frame": pts.frame, "x": pts.x, "y": pts.y, "score": 1.0, "visible": True})


def point_metrics(wid: str, params: pipeline.Params = pipeline.Params()) -> dict:
    """Per-frame track scores of one window (counts, so windows can be pooled)."""
    w = data.windows()[wid]
    gt = data.window_labels(wid)
    T = data.timestamps(w["stem"])
    raw = data.cv_raw(wid, 0.5)
    raw = raw[raw.visible & raw.x.notna()]
    track = pipeline.build_track(data.cv_candidates(wid), T, w["i1"] + 1, params)
    fr = track.frames
    fr = fr[(fr.frame >= w["i0"]) & (fr.frame <= w["i1"])]
    det = fr[fr.state.isin(["detected", "linked"])].rename(columns={"u": "x", "v": "y"})
    itp = fr[fr.state == "interpolated"].rename(columns={"u": "x", "v": "y"})
    out = {}
    for name, pts in (("raw_0.5", raw), ("linked", det), ("linked+interpolated", pd.concat([det, itp]))):
        m = ball_eval.match(gt, _pred_frame(pts), TOL_PX)
        out[name] = {k: int(m[k].sum()) for k in ("tp", "fp", "fn")}
    # Interpolated frames on their own: where labels can judge them, and how many fall where the ball is labelled out.
    g = gt.set_index("frame")
    vis = itp[itp.frame.isin(g.index[g.visibility == "visible"])]
    d = np.hypot(vis.x.to_numpy() - g.loc[vis.frame, "x"].to_numpy(), vis.y.to_numpy() - g.loc[vis.frame, "y"].to_numpy())
    covered = set(det.frame[np.isfinite(det.x)])
    recovered = int(((d <= TOL_PX) & ~vis.frame.isin(covered).to_numpy()).sum()) if len(vis) else 0
    out["interpolated"] = {"frames": int(len(itp)), "on_visible_label": int(len(vis)), "within_7.5px": int((d <= TOL_PX).sum()) if len(vis) else 0,
                           "on_out_label": int(itp.frame.isin(g.index[g.visibility == "out"]).sum()), "recovered_misses": recovered}
    out["frames"] = {"visible": int((gt.visibility == "visible").sum()), "out": int((gt.visibility == "out").sum())}
    return out


def label_track(wid: str) -> pd.DataFrame:
    """The labelled visible points as a linked track (one track per run, runs split where a gap exceeds what fill_gaps bridges)."""
    gt = data.window_labels(wid)
    gt = gt[gt.visibility == "visible"]
    run = (gt.frame.diff().fillna(1) > link.FillParams().max_gap + 1).cumsum()
    return pd.DataFrame({"frame": gt.frame.to_numpy(), "x": gt.x.to_numpy(), "y": gt.y.to_numpy(), "peak": 1.0, "state": "detected", "track_id": run.to_numpy()})


def label_kinks(wid: str, params: pipeline.Params = pipeline.Params()) -> pd.DataFrame:
    """Kinks of the user's labelled trajectory (seen, or bridged across short unlabelled gaps), in the schema of link.Track.kinks."""
    w = data.windows()[wid]
    return link.fill_gaps(label_track(wid), w["i1"] + 1, params.fill, params.kink).kinks


def label_events(wid: str, params: pipeline.Params = pipeline.Params()) -> pd.DataFrame:
    w = data.windows()[wid]
    stem = w["stem"]
    T = data.timestamps(stem)
    pl = data.players(stem)
    pl = pl[(pl.frame >= w["i0"] - 2) & (pl.frame <= w["i1"] + 2)]
    onsets = data.audio_onsets(stem)  # only to confirm faint front-wall kinks, as in the predicted events
    ev = events.detect_events(label_kinks(wid, params), pipeline.camera_track(stem), venue(stem).spec, T, onsets, data.audio_offset(stem), pl, None, params.events, audio_only=False)
    return ev[(ev.frame >= w["i0"] + EDGE) & (ev.frame <= w["i1"] - EDGE)].reset_index(drop=True)


def predicted_events(wid: str, params: pipeline.Params = pipeline.Params(), audio: bool = True, audio_only: bool = True) -> tuple[pd.DataFrame, link.Track]:
    w = data.windows()[wid]
    T = data.timestamps(w["stem"])
    track = pipeline.build_track(data.cv_candidates(wid), T, w["i1"] + 1, params)
    ev = pipeline.build_events(w["stem"], track, T, params, w["i0"], w["i1"], audio=audio, audio_only=audio_only)
    return ev[(ev.frame >= w["i0"] + EDGE) & (ev.frame <= w["i1"] - EDGE)].reset_index(drop=True), track


def verifiable(wid: str, frames: np.ndarray, radius: int = 4, need: float = 0.6) -> np.ndarray:
    """Whether labels can judge an event at each frame: most of the frames around it are labelled visible."""
    gt = data.window_labels(wid)
    vis = set(gt.frame[gt.visibility == "visible"])
    return np.array([np.mean([(int(round(f)) + d) in vis for d in range(-radius, radius + 1)]) >= need for f in frames])


def match_times(a: np.ndarray, b: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Optimal one-to-one pairing of two sets of times within tol."""
    if len(a) == 0 or len(b) == 0:
        return []
    cost = np.abs(a[:, None] - b[None, :])
    big = 1e6
    c = np.where(cost <= tol, cost, big)
    r, cidx = linear_sum_assignment(c)
    return [(int(i), int(j)) for i, j in zip(r, cidx) if c[i, j] < big]


def event_metrics(wid: str, gt: pd.DataFrame, pred: pd.DataFrame, tol: float = 3.0) -> dict:
    """Counts by kind: tp / fp / fn plus the frame errors and, for front-wall matches, the position errors.

    Predicted events at frames labels cannot judge (ball mostly unlabelled around them) are counted apart,
    not as false positives.
    """
    ok = verifiable(wid, pred.frame.to_numpy()) if len(pred) else np.array([], bool)
    unver = pred[~ok]
    pv = pred[ok].reset_index(drop=True)
    out: dict = {"unverifiable": {k: int((unver.kind == k).sum()) for k in events.KINDS}}
    pos_err = []
    for kind in (*events.KINDS, "any"):
        g = gt if kind == "any" else gt[gt.kind == kind]
        pr = pv if kind == "any" else pv[pv.kind == kind]
        pairs = match_times(g.frame.to_numpy(float), pr.frame.to_numpy(float), tol)
        out[kind] = {"tp": len(pairs), "fp": len(pr) - len(pairs), "fn": len(g) - len(pairs),
                     "dt": [float(pr.frame.iloc[j] - g.frame.iloc[i]) for i, j in pairs]}
        if kind == "front_wall":
            for i, j in pairs:
                a, b = g.iloc[i], pr.iloc[j]
                pos_err.append(dict(dx=float(b.x - a.x), dz=float(b.z - a.z), evidence=b.evidence, conf=float(b.confidence), sa=float(b.sigma_a), sb=float(b.sigma_b)))
    out["front_wall_pos"] = pos_err
    return out


def _sum(dicts: list[dict], keys=("tp", "fp", "fn")) -> dict:
    return {k: sum(d[k] for d in dicts) for k in keys}


def run(groups: dict[str, list[str]] = FOLD_GROUPS, params: pipeline.Params = pipeline.Params()) -> dict:
    res: dict = {"points": {}, "events": {}}
    for name, wids in groups.items():
        pts = [point_metrics(w, params) for w in wids]
        res["points"][name] = {
            m: {**prf(**_sum([p[m] for p in pts])), **_sum([p[m] for p in pts])} for m in ("raw_0.5", "linked", "linked+interpolated")
        }
        res["points"][name]["interpolated"] = {k: sum(p["interpolated"][k] for p in pts) for k in pts[0]["interpolated"]}
        res["points"][name]["frames"] = {k: sum(p["frames"][k] for p in pts) for k in pts[0]["frames"]}
        evs = []
        for w in wids:
            gt = label_events(w, params)
            pred, _ = predicted_events(w, params)
            evs.append(event_metrics(w, gt, pred))
        res["events"][name] = summarise_events(evs)
    return res


def summarise_events(evs: list[dict]) -> dict:
    out: dict = {}
    for kind in (*events.KINDS, "any"):
        tp = sum(e[kind]["tp"] for e in evs)
        fp = sum(e[kind]["fp"] for e in evs)
        fn = sum(e[kind]["fn"] for e in evs)
        dt = np.array([x for e in evs for x in e[kind]["dt"]])
        out[kind] = {**prf(tp, fp, fn), "n_gt": tp + fn, "dt_median": float(np.median(dt)) if len(dt) else np.nan,
                     "dt_abs_median": float(np.median(np.abs(dt))) if len(dt) else np.nan, "unverifiable": sum(e["unverifiable"].get(kind, 0) for e in evs) if kind != "any" else sum(sum(e["unverifiable"].values()) for e in evs)}
    pe = [x for e in evs for x in e["front_wall_pos"]]
    if pe:
        err = np.hypot([x["dx"] for x in pe], [x["dz"] for x in pe])
        out["front_wall_pos"] = {"n": len(pe), "median_m": float(np.median(err)), "mean_m": float(err.mean()), "p90_m": float(np.percentile(err, 90)),
                                 "median_dx": float(np.median(np.abs([x["dx"] for x in pe]))), "median_dz": float(np.median(np.abs([x["dz"] for x in pe]))),
                                 "within_0.25m": float(np.mean(err <= 0.25))}
    return out


def hit_label_recall(stem: str, tol: float = 3.0) -> dict:
    """Recall of the full-clip events against hit_labels.csv, by label kind (precision is not measurable: labels cover few hits)."""
    h = pd.read_csv(data.output_path(stem, "hit_labels.csv"))
    ev = pd.read_parquet(data.output_path(stem, "ball_events.parquet"))
    out = {}
    for name in ("front_wall", "floor", "racket", "side_wall"):
        hk = h[h.event == name]
        ek = ev[ev.kind.str.startswith(name)]
        if hk.empty:
            continue
        cost = np.abs(hk.frame.to_numpy()[:, None] - ek.frame.to_numpy()[None, :]).astype(float)
        cost[cost > tol] = 1e6
        rows, cols = linear_sum_assignment(cost) if len(ek) else ([], [])
        dts = [abs(hk.frame.iloc[i] - ek.frame.iloc[j]) for i, j in zip(rows, cols) if cost[i, j] < 1e6]
        out[name] = {"labels": len(hk), "matched": len(dts), "recall": round(len(dts) / len(hk), 2),
                     "median_abs_dt_frames": float(np.median(dts)) if dts else None}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=None, help="write the full result here")
    a = ap.parse_args(argv)
    res = run()
    res["hit_labels"] = {st: hit_label_recall(st) for st in ("IMG_4091", "IMG_4089") if data.output_path(st, "ball_events.parquet").exists()}
    text = json.dumps(res, indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o))
    if a.json:
        open(a.json, "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

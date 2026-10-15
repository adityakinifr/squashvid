"""Score ball detections against labels, per court region, and measure model agreement without labels.

    python -m squashvid.ball.evaluate [--labeller claude-self] [--variants blurball@576x1024-coreml ...]

Matching is strict and per frame, as in the TrackNet/WASB papers: a visible
prediction within `tol` px of a visible label is a true positive; one further
away counts as a false positive and the label as a miss; a prediction on a
frame labelled out is a false positive; no prediction on a visible label is a
miss. Occluded labels are best guesses, so they are scored separately (recall
only) and never create false positives. Misses and hits are binned by the
labelled point's region, false positives by the predicted point's region.

Labels are in original 1080p pixels. Copy predictions are mapped onto the
original frame that shows the same picture (align.py) and scaled by 1.875.
Tolerances are given in 1080p pixels: 7.5 and 15 px are the 4 and 8 px of the
1024x576 copies the brief was written for, and 4 px is the 1080p-native check.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from squashvid.ball import labels as L
from squashvid.ball.clips import COPIES, LABELS, ORIGINAL_OF, ball_dir, copy_to_original_px

TOLS = (4.0, 7.5, 15.0)
AGREE_PX = 20.0


def match(gt: pd.DataFrame, pred: pd.DataFrame, tol: float, threshold: float | None = None) -> pd.DataFrame:
    """Per labelled frame: tp, fp, fn, dist and the point that decides the region (gt for tp/fn, pred for fp)."""
    p = pred.set_index("frame")
    rows = []
    for g in gt.itertuples():
        has = g.frame in p.index
        px, py = (float(p.at[g.frame, "x"]), float(p.at[g.frame, "y"])) if has else (np.nan, np.nan)
        score = float(p.at[g.frame, "score"]) if has else 0.0
        vis = has and np.isfinite(px) and (bool(p.at[g.frame, "visible"]) if threshold is None else score >= threshold)
        d = float(np.hypot(px - g.x, py - g.y)) if vis and g.visibility != "out" else np.nan
        tp = fp = fn = 0
        if g.visibility == "visible":
            tp = int(vis and d <= tol)
            fn = 1 - tp
            fp = int(vis and d > tol)
        elif g.visibility == "out":
            fp = int(vis)
        rows.append({"frame": g.frame, "visibility": g.visibility, "gt_x": g.x, "gt_y": g.y, "pred_x": px if vis else np.nan,
                     "pred_y": py if vis else np.nan, "score": score, "pred_visible": vis, "dist": d, "tp": tp, "fp": fp, "fn": fn,
                     "occ_hit": int(g.visibility == "occluded" and vis and d <= tol)})
    return pd.DataFrame(rows)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if tp and np.isfinite(p) and np.isfinite(r) else (0.0 if tp + fp + fn else np.nan)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def by_region(m: pd.DataFrame, gt_region: np.ndarray, pred_region: np.ndarray, regions: tuple[str, ...]) -> pd.DataFrame:
    """P/R per region: tp and fn binned by the label's region, fp by the prediction's region."""
    rows = []
    for r in (*regions, "all"):
        g = np.ones(len(m), bool) if r == "all" else gt_region == r
        q = np.ones(len(m), bool) if r == "all" else pred_region == r
        occ = m.visibility == "occluded"
        rows.append({"region": r, **prf(int(m.tp[g].sum()), int(m.fp[q].sum()), int(m.fn[g].sum())),
                     "n_visible": int(((m.visibility == "visible") & g).sum()), "n_occluded": int((occ & g).sum()),
                     "occ_recall": float(m.occ_hit[occ & g].mean()) if (occ & g).any() else np.nan})
    return pd.DataFrame(rows)


def sweep(gt: pd.DataFrame, pred: pd.DataFrame, tol: float, thresholds=np.round(np.arange(0.05, 1.0, 0.05), 2)) -> pd.DataFrame:
    """P/R as the score threshold moves (a model's 'visible' flag is its own default threshold)."""
    return pd.DataFrame([{"threshold": t, **prf(*(int(match(gt, pred, tol, t)[k].sum()) for k in ("tp", "fp", "fn")))} for t in thresholds])


def agreement(a: pd.DataFrame, b: pd.DataFrame, tol: float = AGREE_PX) -> dict:
    """Label-free proxy: how often two models report the ball, and agree on where, on the same frames."""
    j = a.merge(b, on="frame", suffixes=("_a", "_b"))
    both = j.visible_a & j.visible_b
    d = np.hypot(j.x_a - j.x_b, j.y_a - j.y_b)
    agree = both & (d <= tol)
    either = j.visible_a | j.visible_b
    return {"frames": len(j), "both_visible": int(both.sum()), "agree": int(agree.sum()), "either_visible": int(either.sum()),
            "agree_of_either": float(agree.sum() / either.sum()) if either.any() else np.nan,
            "agree_of_both": float(agree.sum() / both.sum()) if both.any() else np.nan}


def candidate_recall(gt: pd.DataFrame, cands: pd.DataFrame, tol: float) -> dict:
    """How often any heatmap blob lies within tol of a visible label: the ceiling a better selector could reach."""
    v = gt[gt.visibility == "visible"]
    c = cands.groupby("frame")
    hit = 0
    for g in v.itertuples():
        if g.frame in c.groups:
            cc = c.get_group(g.frame)
            hit += bool((np.hypot(cc.x - g.x, cc.y - g.y) <= tol).any())
    n_c = cands.groupby("frame").size().reindex(v.frame).fillna(0)
    return {"n_visible": int(len(v)), "any_within": int(hit), "ceiling": hit / len(v) if len(v) else np.nan,
            "median_cands_per_frame": float(n_c.median()) if len(n_c) else np.nan}


def load_prediction(variant_path: Path) -> tuple[str, pd.DataFrame]:
    """A prediction parquet as (original stem, frames and pixels in the original's terms)."""
    stem = variant_path.parent.parent.name
    df = pd.read_parquet(variant_path)
    if stem in ORIGINAL_OF:
        al = pd.read_parquet(ball_dir(stem) / "orig_frames.parquet").set_index("frame")
        df = df[df.frame.isin(al.index)].copy()
        df["frame"] = df.frame.map(al.orig_frame).astype(int)
        xy = copy_to_original_px(df[["x", "y"]].to_numpy())
        df["x"], df["y"] = xy[:, 0], xy[:, 1]
        df = df.drop_duplicates("frame", keep="first")
        return ORIGINAL_OF[stem], df
    return stem, df


def predictions(variants: list[str] | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    """{variant label: {original stem: predictions}}; copy variants are labelled 'copy:<variant>'."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for orig in COPIES:
        for stem, tag in ((orig, ""), (COPIES[orig], "copy:")):
            d = ball_dir(stem)
            for path in sorted(d.glob("*.parquet")) if d.exists() else []:
                if path.stem == "orig_frames" or (variants and path.stem not in variants and tag + path.stem not in variants):
                    continue
                if tag and not (d / "orig_frames.parquet").exists():
                    continue
                o, df = load_prediction(path)
                out.setdefault(tag + path.stem, {})[o] = df
    return out


def evaluate(labeller: str | None, variants: list[str] | None, root: Path = LABELS, tols=TOLS) -> dict:
    from squashvid.ball.regions import REGIONS, label_regions

    gts = {orig: L.load(orig, root, labeller) for orig in COPIES}
    gts = {k: v for k, v in gts.items() if len(v)}
    preds = predictions(variants)
    regions = {}
    for orig, gt in gts.items():
        regions[orig] = label_regions(orig, gt.frame.to_numpy(), gt[["x", "y"]].to_numpy(float))
    report: dict = {"labeller": labeller, "labels": {k: v.visibility.value_counts().to_dict() for k, v in gts.items()}, "models": {}}
    for variant, per_clip in preds.items():
        res = {}
        for tol in tols:
            ms, gr, pr, used = [], [], [], []
            for orig, gt in gts.items():
                if orig not in per_clip:
                    continue
                pred = per_clip[orig]
                g = gt[gt.frame.isin(pred.frame)]  # score only frames the variant ran on
                if not len(g):
                    continue
                m = match(g, pred, tol)
                ms.append(m)
                used.append(orig)
                rg = pd.Series(regions[orig], index=gt.frame.to_numpy()).loc[g.frame.to_numpy()].to_numpy()
                gr.append(rg)
                pr.append(np.where(m.pred_visible, label_regions(orig, m.frame.to_numpy(), m[["pred_x", "pred_y"]].to_numpy(float)), None))
            if not ms:
                continue
            m = pd.concat(ms, ignore_index=True)
            tab = by_region(m, np.concatenate(gr), np.concatenate(pr), REGIONS)
            res[f"tol{tol:g}"] = tab.replace({np.nan: None}).to_dict(orient="records")
            res["n_frames"] = int(len(m))
            if tol == 7.5:
                med = m.dist[m.tp == 1]
                res["median_err_px_tp"] = float(med.median()) if len(med) else None
                res["per_clip"] = {o: {**prf(*(int(mm[k].sum()) for k in ("tp", "fp", "fn"))), "n": int(len(mm))} for o, mm in zip(used, ms)}
        if res:
            report["models"][variant] = res
    # Label-free agreement between every pair of variants on the frames both ran on.
    agree = {}
    for a, b in combinations(sorted(preds), 2):
        for orig in set(preds[a]) & set(preds[b]):
            agree.setdefault(f"{a} | {b}", {})[orig] = agreement(preds[a][orig], preds[b][orig])
    report["agreement"] = agree
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m squashvid.ball.evaluate")
    p.add_argument("--labeller", default=None)
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    rep = evaluate(a.labeller, a.variants)
    out = a.out or LABELS / f"eval_{a.labeller or 'all'}.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print("labels:", rep["labels"])
    rows = []
    for v, res in rep["models"].items():
        for key in ("tol4", "tol7.5", "tol15"):
            for r in res.get(key, []):
                rows.append({"variant": v, "tol": key, **r})
    if rows:
        df = pd.DataFrame(rows)
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            print(df[df.region == "all"][["variant", "tol", "tp", "fp", "fn", "precision", "recall", "f1"]].round(3).to_string(index=False))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

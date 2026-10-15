"""Compare two player runs over the same footage (e.g. the WhatsApp copy and the iPhone original).

    python -m squashvid.players.compare work/wa_3A16D3EC0BAFDCE1C46D work/IMG_4091 [--out compare.json]

Frames are matched by presentation time, nearest neighbour within one frame
period of the first run; the two encodes do not share frame indices. Reports,
per run and on matched frames: detection coverage, keypoint scores, position
agreement, label agreement, foot-point jitter on the same still windows, and
extraction throughput.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from squashvid.players import metrics as M
from squashvid.players.pose import L_ANK, R_ANK
from squashvid.players.tracklets import iou_matrix

PLAYERS = ("A", "B")


def match_times(ta: np.ndarray, tb: np.ndarray, tol: float) -> np.ndarray:
    """Index into ta of the nearest time to each tb, or -1 when it is further than tol."""
    j = np.clip(np.searchsorted(ta, tb), 1, len(ta) - 1)
    left, right = ta[j - 1], ta[j]
    k = np.where(np.abs(tb - left) <= np.abs(right - tb), j - 1, j)
    return np.where(np.abs(ta[k] - tb) <= tol, k, -1)


def keypoint_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Mean keypoint score and the weaker ankle's score for each detected row."""
    d = table[table.detected & table.kscore.notna()]
    ks = np.stack([np.asarray(v, np.float64) for v in d.kscore]) if len(d) else np.zeros((0, 17))
    return pd.DataFrame({"player": d.player.to_numpy(), "kp_mean": ks.mean(1), "ankle_min": np.minimum(ks[:, L_ANK], ks[:, R_ANK]), "conf": d.conf.to_numpy()}, index=d.index)


def run_summary(table: pd.DataFrame) -> dict:
    ks = keypoint_scores(table)
    out = {"coverage": M.coverage(table)}
    for p in PLAYERS:
        k = ks[ks.player == p]
        out[p] = {
            "det_conf_median": round(float(k.conf.median()), 3),
            "kp_mean_median": round(float(k.kp_mean.median()), 3),
            "ankle_min_median": round(float(k.ankle_min.median()), 3),
            "ankle_ge_045": round(float((k.ankle_min >= 0.45).mean()), 3),
        }
    return out


def second_difference_noise(table: pd.DataFrame, col_x: str, col_y: str) -> dict:
    """Per-frame noise of raw foot points from the second difference over consecutive detected frames.

    For white noise of SD s, x[k+1] - 2x[k] + x[k-1] has SD s*sqrt(6). A median
    absolute deviation keeps real accelerations and the odd wrong point out.
    At 30 fps a 10 m/s^2 lunge adds about 1 cm, so this bounds noise from above.
    """
    out = {}
    for p in PLAYERS:
        tp = table[table.player == p].sort_values("frame")
        v = tp[[col_x, col_y]].to_numpy(np.float64)
        f = tp.frame.to_numpy()
        ok = np.isfinite(v).all(1)
        d2 = []
        for i in range(1, len(tp) - 1):
            if ok[i - 1] and ok[i] and ok[i + 1] and f[i + 1] - f[i - 1] == 2:
                d2.append(v[i + 1] - 2 * v[i] + v[i - 1])
        d2 = np.asarray(d2)
        if len(d2) < 50:
            out[p] = None
            continue
        mad = 1.4826 * np.median(np.abs(d2 - np.median(d2, 0)), 0) / np.sqrt(6)
        out[p] = {"sd_x_cm": round(100 * float(mad[0]), 2), "sd_y_cm": round(100 * float(mad[1]), 2), "n": int(len(d2))}
    return out


def same_window_jitter(ta: pd.DataFrame, tb: pd.DataFrame, k_of_b: dict[str, np.ndarray], spec) -> dict:
    """Ankle and box foot-point SD over still windows found in run a, evaluated in both runs on the same times."""
    rows = []
    for p in PLAYERS:
        a = ta[ta.player == p].reset_index(drop=True)
        b = tb[tb.player == p].reset_index(drop=True)
        kb = k_of_b[p]  # row of a matched to each row of b (-1 when none)
        for i, j in M.still_windows(a.t.to_numpy(), a[["sx", "sy"]].to_numpy()):
            rb = b[(kb >= i) & (kb < j)]
            rec = {"player": p, "y": float(a.sy.iloc[i:j].mean())}
            for run, seg in (("a", a.iloc[i:j]), ("b", rb)):
                for name, cx, cy in (("ankle", "ankle_x", "ankle_y"), ("box", "box_x", "box_y")):
                    v = seg[[cx, cy]].to_numpy(np.float64)
                    if name == "box":
                        v = v[~seg.truncated.to_numpy(bool)]
                    v = v[np.isfinite(v).all(1)]
                    ok = len(v) >= 10
                    rec[f"{run}_{name}_sx"] = float(v[:, 0].std()) if ok else np.nan
                    rec[f"{run}_{name}_sy"] = float(v[:, 1].std()) if ok else np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_windows": 0}
    df["zone"] = np.where(df.y < spec.box_back_y, "back", np.where(df.y < spec.short_y + 1.0, "mid", "front"))
    cols = [c for c in df.columns if c.endswith(("_sx", "_sy"))]
    both = df.dropna(subset=["a_ankle_sx", "b_ankle_sx"])
    return {
        "n_windows": int(len(df)),
        "n_windows_ankle_both": int(len(both)),
        "median_sd_cm": (100 * df[cols].median()).round(2).to_dict(),
        "median_sd_cm_ankle_both": (100 * both[cols].median()).round(2).to_dict(),
        "by_zone_cm": {z: (100 * g[cols].median()).round(2).dropna().to_dict() for z, g in df.groupby("zone")},
        "windows_per_zone": df.zone.value_counts().to_dict(),
    }


def frame_size(work: Path) -> list[int]:
    """Frame size the run was extracted at; runs older than that field were all on the 1024x576 copies."""
    return json.loads((work / "extract_timing.json").read_text()).get("frame_size", [1024, 576])


def compare(work_a: Path, work_b: Path, spec) -> dict:
    ta = pd.read_parquet(work_a / "players.parquet")
    tb = pd.read_parquet(work_b / "players.parquet")
    size_a, size_b = frame_size(work_a), frame_size(work_b)
    scale = size_b[0] / size_a[0]
    times_a = np.sort(ta.t.unique())
    tol = float(np.median(np.diff(times_a)))
    tb = tb[tb.t <= times_a[-1] + tol]  # the original may run past the copy's end

    out: dict = {"a": str(work_a), "b": str(work_b), "tolerance_s": round(tol, 4), "frame_size": {"a": size_a, "b": size_b}}
    out["run_a"], out["run_b"] = run_summary(ta), run_summary(tb)

    # Matched frames, per player.
    k_of_b, matched = {}, {}
    offsets = {}
    for p in PLAYERS:
        a = ta[ta.player == p].sort_values("t").reset_index(drop=True)
        b = tb[tb.player == p].sort_values("t").reset_index(drop=True)
        k = match_times(a.t.to_numpy(), b.t.to_numpy(), tol)
        k_of_b[p] = k
        ok = k >= 0
        am, bm = a.iloc[k[ok]].reset_index(drop=True), b[ok].reset_index(drop=True)
        da, db = am.detected.to_numpy(bool), bm.detected.to_numpy(bool)
        both = da & db
        d_s = np.linalg.norm(am[["sx", "sy"]].to_numpy() - bm[["sx", "sy"]].to_numpy(), axis=1)
        d_a = np.linalg.norm(am[["ankle_x", "ankle_y"]].to_numpy() - bm[["ankle_x", "ankle_y"]].to_numpy(), axis=1)
        boxa = am[["x1", "y1", "x2", "y2"]].to_numpy(np.float64) * scale
        boxb = bm[["x1", "y1", "x2", "y2"]].to_numpy(np.float64)
        iou = np.array([iou_matrix(x[None], y[None])[0, 0] if m else np.nan for x, y, m in zip(boxa, boxb, both)])
        matched[p] = {
            "n_matched": int(ok.sum()),
            "detected_both": round(float(both.mean()), 4),
            "detected_only_a": round(float((da & ~db).mean()), 4),
            "detected_only_b": round(float((~da & db).mean()), 4),
            "same_box_when_both": round(float((iou[both] > 0.3).mean()), 4) if both.any() else None,  # same person under this label in both runs
            "n_different_box": int((iou[both] <= 0.3).sum()),
            "smoothed_diff_m": {"median": round(float(np.nanmedian(d_s)), 3), "p90": round(float(np.nanpercentile(d_s, 90)), 3)},
            "ankle_diff_m": {"median": round(float(np.nanmedian(d_a)), 3), "p90": round(float(np.nanpercentile(d_a, 90)), 3)},
        }
        # Timeline check: the frame shift that best aligns the two smoothed tracks (0 when the times agree).
        sa = a[["sx", "sy"]].to_numpy()
        res = {}
        for s in range(-3, 4):
            kk = k[ok] + s
            good = (kk >= 0) & (kk < len(a))
            res[s] = float(np.nanmedian(np.linalg.norm(sa[kk[good]] - b[ok][["sx", "sy"]].to_numpy()[good], axis=1)))
        offsets[p] = {"best": min(res, key=res.get), "median_diff_m": {s: round(v, 4) for s, v in res.items()}}
    out["matched"] = matched
    out["best_frame_offset"] = offsets
    out["jitter_same_windows"] = same_window_jitter(ta, tb, k_of_b, spec)
    out["second_difference_noise"] = {
        run: {name: second_difference_noise(t, f"{name}_x", f"{name}_y") for name in ("ankle", "box")} for run, t in (("a", ta), ("b", tb))
    }
    for run, w in (("a", work_a), ("b", work_b)):
        tj = json.loads((w / "extract_timing.json").read_text())
        out[f"throughput_{run}"] = {"imgsz": tj.get("imgsz"), "n_frames": tj["n_frames"], "wall_s": tj["wall_s"], "ms_per_frame": tj["ms_per_frame"]}
    return out


def main(argv: list[str] | None = None) -> int:
    from squashvid.calib.venue import Venue

    p = argparse.ArgumentParser(prog="python -m squashvid.players.compare")
    p.add_argument("work_a")
    p.add_argument("work_b")
    p.add_argument("--venue", default="work/venue_court5.json")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    res = compare(Path(a.work_a), Path(a.work_b), Venue.load(a.venue).spec)
    text = json.dumps(res, indent=1, default=float)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

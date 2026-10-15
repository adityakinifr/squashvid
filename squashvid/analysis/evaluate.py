"""Scores for hits and rallies against the thin ground truth.

Hit labels (`hit_labels.csv`) were made by Claude agents over short spans of
each clip, 10 to 40 racket contacts per clip. Precision is only meaningful
inside those spans (outside them nothing was labelled), so detections are
scored there. Rally boundaries are scored against rally windows that the
same labellers wrote in the label file where available, and by eye otherwise.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from squashvid.analysis.io import PLAYERS, work_dir

COLOUR = {"blue": "A", "dark": "B", "a": "A", "b": "B"}


def load_labels(stem: str, events=("racket",)) -> pd.DataFrame:
    lab = pd.read_csv(work_dir(stem) / "hit_labels.csv")
    lab = lab[lab.event.isin(events)].copy()
    lab["player"] = lab.player.astype(str).str.strip().str.lower().map(COLOUR)
    return lab.reset_index(drop=True)


def label_spans(stem: str, pad: float = 0.5) -> list[tuple[float, float]]:
    """Time spans the labellers covered exhaustively (from the audio evaluation)."""
    f = work_dir(stem) / "audio_eval.json"
    if f.exists():
        spans = json.load(open(f)).get("spans") or []
        if spans:
            return [(a - pad, b + pad) for a, b in spans]
    lab = load_labels(stem, ("racket", "front_wall", "floor", "side_wall", "handling", "air_swing"))
    if lab.empty:
        return []
    # Fall back to clusters of labels no more than 4 s apart.
    t = np.sort(lab.t.to_numpy())
    spans, a = [], t[0]
    for x, y in zip(t[:-1], t[1:]):
        if y - x > 4:
            spans.append((a - pad, x + pad))
            a = y
    spans.append((a - pad, t[-1] + pad))
    return spans


def in_spans(t, spans) -> np.ndarray:
    t = np.asarray(t, float)
    ok = np.zeros(len(t), bool)
    for a, b in spans:
        ok |= (t >= a) & (t <= b)
    return ok


def match_times(det: np.ndarray, truth: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """One-to-one nearest matching within tol (greedy by smallest error)."""
    pairs = sorted(((abs(d - g), i, j) for i, d in enumerate(det) for j, g in enumerate(truth) if abs(d - g) <= tol))
    used_i, used_j, out = set(), set(), []
    for _, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j))
    return out


def score_hits(det: pd.DataFrame, truth: pd.DataFrame, spans, tol: float = 0.2) -> dict:
    """Precision, recall, timing error and hitter accuracy of detections against labelled contacts."""
    truth = truth[in_spans(truth.t, spans)].reset_index(drop=True)
    det = det[in_spans(det.t, spans)].reset_index(drop=True)
    pairs = match_times(det.t.to_numpy(), truth.t.to_numpy(), tol)
    err = np.array([det.t[i] - truth.t[j] for i, j in pairs])
    hit_ok = [det.hitter[i] == truth.player[j] for i, j in pairs if truth.player[j] in PLAYERS]
    tp = len(pairs)
    return {
        "tol_s": tol,
        "n_truth": len(truth),
        "n_detected": len(det),
        "tp": tp,
        "precision": round(tp / len(det), 3) if len(det) else None,
        "recall": round(tp / len(truth), 3) if len(truth) else None,
        "timing_median_s": round(float(np.median(err)), 3) if tp else None,
        "timing_mad_s": round(float(np.median(abs(err - np.median(err)))), 3) if tp else None,
        "hitter_accuracy": round(float(np.mean(hit_ok)), 3) if hit_ok else None,
        "n_hitter": len(hit_ok),
    }


def pooled(scores: list[dict]) -> dict:
    tp = sum(s["tp"] for s in scores)
    nd = sum(s["n_detected"] for s in scores)
    nt = sum(s["n_truth"] for s in scores)
    nh = sum(s["n_hitter"] for s in scores)
    ok = sum(round((s["hitter_accuracy"] or 0) * s["n_hitter"]) for s in scores)
    return {
        "tp": tp,
        "n_detected": nd,
        "n_truth": nt,
        "precision": round(tp / nd, 3) if nd else None,
        "recall": round(tp / nt, 3) if nt else None,
        "hitter_accuracy": round(ok / nh, 3) if nh else None,
        "n_hitter": nh,
    }


def census(stem: str) -> pd.DataFrame | None:
    """Rally windows written by eye from 1 s contact sheets (clips 1 and 3 only; rough, often 'low' confidence)."""
    from squashvid.ball.clips import WORK

    f = {"IMG_4091": "census_c1", "IMG_4090": "census_c3"}.get(stem)
    path = WORK / "audio_scratch" / str(f) / "rallies.csv"
    return pd.read_csv(path) if f and path.exists() else None


def score_rallies(det: pd.DataFrame, truth: pd.DataFrame, duration: float, min_iou: float = 0.3) -> dict:
    """Per-second agreement and rally-level matching (IoU of the two windows)."""
    n = int(duration)
    a, b = np.zeros(n, bool), np.zeros(n, bool)
    for r in det.itertuples():
        a[int(r.start_t) : int(r.end_t)] = True
    for r in truth.itertuples():
        b[int(r.start_t) : int(r.end_t)] = True
    pairs = []
    for i, d in enumerate(det.itertuples()):
        for j, g in enumerate(truth.itertuples()):
            inter = max(0.0, min(d.end_t, g.end_t) - max(d.start_t, g.start_t))
            iou = inter / (max(d.end_t, g.end_t) - min(d.start_t, g.start_t))
            if iou >= min_iou:
                pairs.append((iou, i, j))
    used_i, used_j, matched = set(), set(), []
    for iou, i, j in sorted(pairs, reverse=True):
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            matched.append((i, j))
    sb = [det.start_t.iloc[i] - truth.start_t.iloc[j] for i, j in matched]
    eb = [det.end_t.iloc[i] - truth.end_t.iloc[j] for i, j in matched]
    return {
        "n_detected": len(det),
        "n_truth": len(truth),
        "matched": len(matched),
        "precision": round(len(matched) / len(det), 3) if len(det) else None,
        "recall": round(len(matched) / len(truth), 3) if len(truth) else None,
        "seconds_iou": round(float((a & b).sum() / max((a | b).sum(), 1)), 3),
        "start_error_median_s": round(float(np.median(sb)), 2) if sb else None,
        "end_error_median_s": round(float(np.median(eb)), 2) if eb else None,
    }


# --- trajectory-derived benchmark from the user's hand-labelled ball frames ---------------------------


def trajectory_contacts(stem: str, turn_deg: float = 50.0, near_px: float = 110.0, min_speed: float = 6.0) -> pd.DataFrame:
    """Racket contacts inferred from the hand-labelled ball: the ball path turns sharply while it is near a player's wrist.

    Independent of the pose-based detector except that it reads the same wrist keypoints. Noisy: the ball is often
    hidden at contact, so contacts are missed, and a ball bounce next to a hand can look like one.
    """
    from squashvid.ball import labels

    lab = labels.load(stem, labeller="aditya")
    lab = lab[lab.visibility == "visible"].reset_index(drop=True)
    if lab.empty:
        return pd.DataFrame(columns=["t", "player"])
    p = pd.read_parquet(work_dir(stem) / "players.parquet")
    wr = {}
    for k in PLAYERS:
        q = p[p.player == k].set_index("frame")
        wr[k] = {int(f): np.asarray(v, float).reshape(17, 2)[9:11] for f, v in q.kpts.items() if v is not None}
    f, xy = lab.frame.to_numpy(), lab[["x", "y"]].to_numpy()
    rows = []
    for i in range(2, len(lab) - 2):
        if f[i + 2] - f[i - 2] > 8:
            continue
        v0, v1 = xy[i] - xy[i - 2], xy[i + 2] - xy[i]
        s0, s1 = np.hypot(*v0) / (f[i] - f[i - 2]), np.hypot(*v1) / (f[i + 2] - f[i])
        if min(s0, s1) < min_speed:
            continue
        ang = np.degrees(np.arccos(np.clip(np.dot(v0, v1) / (np.hypot(*v0) * np.hypot(*v1) + 1e-9), -1, 1)))
        if ang < turn_deg:
            continue
        best, who = np.inf, None
        for k in PLAYERS:
            for dfr in (-2, -1, 0, 1, 2):
                w = wr[k].get(int(f[i]) + dfr)
                if w is not None:
                    d = np.hypot(*(w - xy[i]).T).min()
                    if d < best:
                        best, who = d, k
        if best <= near_px:
            rows.append((float(lab.t[i]), who, best, ang))
    df = pd.DataFrame(rows, columns=["t", "player", "dist_px", "turn_deg"])
    # one contact per 0.4 s: keep the sharpest turn
    keep = []
    for r in df.sort_values("turn_deg", ascending=False).itertuples():
        if all(abs(r.t - df.t[j]) > 0.4 for j in keep):
            keep.append(r.Index)
    return df.loc[sorted(keep)].reset_index(drop=True)


def ball_label_windows(stem: str) -> list[tuple[float, float]]:
    import json

    from squashvid.ball.clips import WORK

    w = json.load(open(WORK / "labels" / "windows.json"))
    return [(x["start"], x["end"]) for x in w if x["stem"] == stem]


# --- the full evaluation --------------------------------------------------------------------------


def run_all(use_ball: bool = True, tag: str | None = None) -> dict:
    """Hit and rally scores for all clips: full detector, pose-only ablation, and the audio rally baseline."""
    import json
    from dataclasses import replace

    from squashvid.analysis import hits as H
    from squashvid.analysis import rallies as R
    from squashvid.analysis.io import CLIPS, REVIEW_DIR, load

    out: dict = {"ball_events_used": {}, "hits": {}, "hits_pose_only": {}, "hits_vs_ball_labels": {}, "rallies": {}, "rallies_audio_baseline": {}}
    sc = {t: [] for t in (0.1, 0.2, 0.3)}
    sc_pose = []
    sc_traj = []
    for stem in CLIPS:
        m = load(stem, use_ball=use_ball)
        out["ball_events_used"][stem] = m.ball_events is not None
        truth, spans = load_labels(stem), label_spans(stem)
        raw = H.detect_hits(m)
        out["hits"][stem] = {str(t): score_hits(raw, truth, spans, t) for t in sc}
        for t in sc:
            sc[t].append(out["hits"][stem][str(t)])
        mp = replace(m, onsets=m.onsets.iloc[:0], ball_events=None)
        pose = H.detect_hits(mp)
        out["hits_pose_only"][stem] = score_hits(pose, truth, spans, 0.2)
        sc_pose.append(out["hits_pose_only"][stem])
        traj = trajectory_contacts(stem)
        wins = ball_label_windows(stem)
        traj = traj[in_spans(traj.t, wins)].reset_index(drop=True)
        s = score_hits(raw, traj.rename(columns={"player": "player"}), wins, 0.2)
        out["hits_vs_ball_labels"][stem] = {k: s[k] for k in ("n_truth", "tp", "recall", "hitter_accuracy", "n_hitter", "timing_median_s", "timing_mad_s")}
        sc_traj.append(s)
        sw = H.swing_candidates(m)
        rallies, _, _ = R.segment(m, raw, sw)
        c = census(stem)
        if c is not None:
            out["rallies"][stem] = score_rallies(rallies, c, m.duration)
            out["rallies_audio_baseline"][stem] = score_rallies(m.audio_rallies[["start_t", "end_t"]], c, m.duration)
        else:
            out["rallies"][stem] = {"n_detected": len(rallies), "note": "no by-eye rally census for this clip"}
    out["hits_pooled"] = {str(t): pooled(v) for t, v in sc.items()}
    out["hits_pose_only_pooled"] = pooled(sc_pose)
    tp = sum(s["tp"] for s in sc_traj)
    nt = sum(s["n_truth"] for s in sc_traj)
    nh = sum(s["n_hitter"] for s in sc_traj)
    ok = sum(round((s["hitter_accuracy"] or 0) * s["n_hitter"]) for s in sc_traj)
    out["hits_vs_ball_labels_pooled"] = {"n_truth": nt, "tp": tp, "recall": round(tp / nt, 3) if nt else None, "hitter_accuracy": round(ok / nh, 3) if nh else None}
    out["truth_sizes"] = {"racket_labels_in_spans": int(sum(s["n_truth"] for s in sc[0.2])), "ball_label_contacts": int(nt)}
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    name = tag if tag is not None else ("ball" if any(out["ball_events_used"].values()) else "noball")
    (REVIEW_DIR / f"evaluation_{name}.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    import json

    print(json.dumps(run_all(), indent=1))


if __name__ == "__main__":
    main()

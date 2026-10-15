"""Per-clip pipeline: players.parquet + camera.parquet -> work/<stem>/bodies.parquet (+ bodies_summary.json)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from squashvid.bodies import lift as L
from squashvid.bodies import place as P
from squashvid.bodies.skeleton import PARENT, coco_to_h36m
from squashvid.players.geometry import ANKLE_Z

log = logging.getLogger(__name__)
PLAYERS = ("A", "B")
MIN_SCORE = 0.3  # a keypoint scoring below this is treated as not seen
GAP_S = 0.5  # gaps up to this long are bridged; anything longer breaks the track
MIN_SEG_S = 0.3  # a pose track shorter than this is dropped
CORE = [0, 1, 2, 3, 4, 5, 6, 8, 11, 14]  # H36M joints a frame needs to be lifted: pelvis chain + shoulders
NEED = [0, 1, 4, 8, 11, 14]  # must be seen: pelvis, both hips, thorax, both shoulders
ALPHAS = (0.0, 0.5, 1.0)
STATE_NONE, STATE_SEEN, STATE_BRIDGED = 0, 1, 2


@dataclass
class Track:
    """One player's lifted track over a clip, all arrays indexed by frame."""

    joints: np.ndarray  # (n,17,3) court metres, NaN where there is no pose
    conf: np.ndarray  # (n,) 0..1; 0 means no pose
    state: np.ndarray  # (n,) uint8
    info: dict = field(default_factory=dict)


def segments(valid: np.ndarray, gap: int, min_len: int) -> list[tuple[int, int]]:
    """[start, end) runs of valid frames; gaps of at most `gap` frames are bridged, longer ones split the run."""
    idx = np.flatnonzero(valid)
    out: list[tuple[int, int]] = []
    if not len(idx):
        return out
    s = p = int(idx[0])
    for i in idx[1:]:
        if i - p - 1 > gap:
            out.append((s, p + 1))
            s = int(i)
        p = int(i)
    out.append((s, p + 1))
    return [(a, b) for a, b in out if int(valid[a:b].sum()) >= min_len]


def fill_2d(x: np.ndarray, conf: np.ndarray, seen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate each joint over the frames where it (or the whole frame) was not seen.

    `seen` (T,17) says which entries are real. Entries with no real neighbour keep the nearest real value and
    confidence 0; the lifter is told through the confidence channel."""
    t = np.arange(len(x))
    x2, c2 = x.copy(), conf.copy()
    for j in range(x.shape[1]):
        ok = seen[:, j]
        if ok.all():
            continue
        if not ok.any():
            x2[:, j] = x2[:, 8]
            c2[:, j] = 0
            continue
        for d in range(2):
            x2[:, j, d] = np.interp(t, t[ok], x[ok, j, d])
        c2[~ok, j] = 0
    return x2, c2


def smooth(a: np.ndarray, window: int = 7, order: int = 2) -> np.ndarray:
    """Savitzky-Golay over axis 0, with the window shrunk for short tracks."""
    w = min(window, len(a) - (1 - len(a) % 2))
    if w < order + 2 or w % 2 == 0:
        return a
    return savgol_filter(a, w, order, axis=0, mode="interp")


def player_arrays(df: pd.DataFrame, n: int) -> dict:
    sub = df.drop_duplicates("frame", keep="last").set_index("frame")
    kp = np.full((n, 17, 2), np.nan)
    ks = np.zeros((n, 17))
    pos = np.full((n, 2), np.nan)
    det = np.zeros(n, bool)
    src = np.array([""] * n, dtype=object)
    idx = sub.index.to_numpy(int)
    ok = idx < n
    for i, (f, k, s) in enumerate(zip(idx, sub["kpts"], sub["kscore"], strict=False)):
        if f < n and k is not None and s is not None and len(k) == 34 and len(s) == 17:
            kp[f] = np.asarray(k, float).reshape(17, 2)
            ks[f] = np.asarray(s, float)
    pos[idx[ok]] = sub[["sx", "sy"]].to_numpy(float)[ok]
    det[idx[ok]] = sub["detected"].to_numpy(bool)[ok]
    src[idx[ok]] = sub["src"].fillna("").to_numpy(object)[ok]
    return {"kp": kp, "ks": ks, "pos": pos, "det": det, "src": src}


def lift_player(arr: dict, cam: np.ndarray, dist: np.ndarray, fps: float, model, device: str = "cpu") -> Track:
    n = len(cam)
    gap = int(round(GAP_S * fps))
    h2d, hconf = coco_to_h36m(arr["kp"], arr["ks"])
    seen = (hconf >= MIN_SCORE) & np.isfinite(h2d).all(axis=2)
    valid = arr["det"] & seen[:, NEED].all(axis=1) & np.isfinite(arr["pos"]).all(axis=1)
    segs = segments(valid, gap, int(round(MIN_SEG_S * fps)))
    R = P.rotations(cam)
    joints = np.full((n, 17, 3), np.nan)
    conf = np.zeros(n)
    state = np.zeros(n, np.uint8)
    if not segs:
        return Track(joints, conf, state, {"segments": 0})

    lifted: dict[tuple[int, int], np.ndarray] = {}
    for a, b in segs:
        x, c = fill_2d(h2d[a:b], hconf[a:b], seen[a:b] & valid[a:b, None])
        lifted[(a, b)] = smooth(L.lift_segment(model, x, c, device))
    allf = np.concatenate(list(lifted.values()))
    props = P.bone_proportions(allf)
    par = np.array(PARENT[1:])
    ln = np.linalg.norm(allf[:, 1:] - allf[:, par], axis=2)
    cv = (ln / ln.sum(axis=1, keepdims=True)).std(axis=0) / (ln / ln.sum(axis=1, keepdims=True)).mean(axis=0)  # raw lifter, before lengths are fixed
    anchor_pelvis = np.array([s == "hip" for s in arr["src"]])
    frames = np.concatenate([np.arange(a, b) for a, b in segs])
    best = None
    for alpha in ALPHAS:
        M = P.model_to_court(R, alpha)
        dirs = np.full((n, 16, 3), np.nan)
        for (a, b), lf in lifted.items():
            dirs[a:b] = np.einsum("tij,tkj->tki", M[a:b], P.directions(lf))
        total, err = P.fit_scale(dirs[frames], props, arr["pos"][frames], anchor_pelvis[frames], cam[frames], dist,
                                 arr["kp"][frames], arr["ks"][frames], R[frames])
        log.info("alpha %.1f: bone-chain %.2f m, median reprojection %.2f px", alpha, total, err)
        if best is None or err < best[0]:
            best = (err, alpha, total, dirs)
    err, alpha, total, dirs = best
    lengths = props * total
    rel = np.full((n, 17, 3), np.nan)
    for a, b in segs:
        rel[a:b] = P.build(dirs[a:b], lengths)
    pos = np.nan_to_num(arr["pos"])
    placed = P.place(np.nan_to_num(rel), pos, anchor_pelvis)
    for a, b in segs:  # smooth the floor offset so the figure does not hop frame to frame, then forbid penetration
        zraw = ANKLE_Z - np.minimum(rel[a:b, 3, 2], rel[a:b, 6, 2])
        zs = smooth(zraw[:, None], 9, 2)[:, 0]
        placed[a:b, :, 2] += (zs - zraw)[:, None]
        low = np.minimum(placed[a:b, 3, 2], placed[a:b, 6, 2])
        placed[a:b, :, 2] += np.maximum(ANKLE_Z - low, 0)[:, None]
        joints[a:b] = placed[a:b]
        got = valid[a:b]
        state[a:b] = np.where(got, STATE_SEEN, STATE_BRIDGED)
        core = hconf[a:b][:, CORE].mean(axis=1)
        tt = np.arange(b - a)
        core = np.where(got, core, np.interp(tt, tt[got], core[got]))  # a bridged frame inherits its neighbours' scores
        conf[a:b] = np.clip(core, 0, 1) * np.where(got, 1.0, 0.5)
    heights = joints[state > 0][:, 10, 2] - ANKLE_Z
    info = {"segments": len(segs), "alpha": alpha, "bone_chain_m": total, "fit_reproj_px": err,
            "raw_bone_cv_median": float(np.median(cv)), "raw_bone_cv_max": float(cv.max()),
            "stature_m_p90": float(np.percentile(heights, 90) + ANKLE_Z), "bone_m": {int(j): float(lengths[j]) for j in range(1, 17)}}
    return Track(joints, conf, state, info)


def run(stem: str, root: Path | None = None, device: str = "cpu", out: Path | None = None) -> dict:
    from squashvid.ball.regions import venue as get_venue
    from squashvid.viewer.export import build_camera

    from squashvid.ball.clips import work_dir

    work = Path(root) / stem if root else work_dir(stem)
    players = pd.read_parquet(work / "players.parquet")
    cam_df = pd.read_parquet(work / "camera.parquet")
    ven = get_venue(stem)
    n = int(max(players.frame.max(), cam_df.frame.max())) + 1
    cam = build_camera(cam_df, ven, n).astype(np.float64)
    dist = ven.camera(np.zeros(3), np.zeros(3)).dist
    ts = np.interp(np.arange(n), *[players.drop_duplicates("frame").sort_values("frame")[c].to_numpy() for c in ("frame", "t")])
    fps = (n - 1) / (ts[-1] - ts[0])
    model = L.load_model(device)
    rows, summary = [], {"stem": stem, "fps": fps, "players": {}}
    for p in PLAYERS:
        arr = player_arrays(players[players.player == p], n)
        tr = lift_player(arr, cam, dist, fps, model, device)
        summary["players"][p] = {**tr.info, "frames_seen": int((tr.state == STATE_SEEN).sum()), "frames_bridged": int((tr.state == STATE_BRIDGED).sum())}
        ok = np.flatnonzero(tr.state > 0)
        rows.append(pd.DataFrame({"frame": ok, "t": ts[ok], "player": p, "state": tr.state[ok], "conf": tr.conf[ok].astype(np.float32),
                                  "joints": list(tr.joints[ok].reshape(len(ok), 51).astype(np.float32))}))
    df = pd.concat(rows, ignore_index=True)
    dest = out or work
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "bodies.parquet")
    (dest / "bodies_summary.json").write_text(json.dumps(summary, indent=1))
    return summary

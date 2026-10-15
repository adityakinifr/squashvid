"""Validation of bodies.parquet against the detector, the camera and the old flat lift. Writes bodies_validation.json."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from squashvid.bodies import place as P
from squashvid.bodies.run import PLAYERS, STATE_SEEN, player_arrays
from squashvid.bodies.skeleton import COCO_OF_H36M, PARENT
from squashvid.players.geometry import ANKLE_Z

COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def flat_lift(cam: np.ndarray, dist: np.ndarray, floor_xy: np.ndarray, kp: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """The viewer's old lift (data.js liftKeypoint): every keypoint ray meets the vertical plane through the floor point
    that faces the camera; height is clamped at 0. (T,17,3), NaN where unseen. Returns also pre-clamp heights in [...,3]."""
    T = len(kp)
    out = np.full((T, 17, 4), np.nan)
    for t in range(T):
        if not np.isfinite(floor_xy[t]).all() or not np.isfinite(cam[t]).all():
            continue
        K = np.array([[cam[t, 0], 0, cam[t, 1]], [0, cam[t, 0], cam[t, 2]], [0, 0, 1.0]])
        R = cv2.Rodrigues(cam[t, 3:6])[0]
        c = -R.T @ cam[t, 6:9]
        n = np.array([c[0] - floor_xy[t, 0], c[1] - floor_xy[t, 1]])
        n /= np.linalg.norm(n) or 1
        for j in range(17):
            if ks[t, j] <= 0.3 or not np.isfinite(kp[t, j]).all():
                continue
            xy = cv2.undistortPoints(kp[t, j].reshape(1, 1, 2).astype(np.float64), K, dist).reshape(2)
            d = R.T @ np.array([xy[0], xy[1], 1.0])
            d /= np.linalg.norm(d)
            den = n[0] * d[0] + n[1] * d[1]
            if abs(den) < 1e-6:
                continue
            tt = (n[0] * (floor_xy[t, 0] - c[0]) + n[1] * (floor_xy[t, 1] - c[1])) / den
            if tt > 0:
                p = c + d * tt
                out[t, j, :3] = p
                out[t, j, 3] = p[2]
                out[t, j, 2] = max(0.0, p[2])
    return out


def ankle_plane_xy(cam: np.ndarray, dist: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """Where each ankle keypoint's ray meets the plane z = ANKLE_Z: (T,2 feet [R,L],2) court metres (NaN if no pose)."""
    T = len(kp)
    out = np.full((T, 2, 2), np.nan)
    for t in range(T):
        if not np.isfinite(cam[t]).all():
            continue
        K = np.array([[cam[t, 0], 0, cam[t, 1]], [0, cam[t, 0], cam[t, 2]], [0, 0, 1.0]])
        R = cv2.Rodrigues(cam[t, 3:6])[0]
        c = -R.T @ cam[t, 6:9]
        for k, j in enumerate((16, 15)):  # COCO R ankle, L ankle
            if not np.isfinite(kp[t, j]).all():
                continue
            xy = cv2.undistortPoints(kp[t, j].reshape(1, 1, 2).astype(np.float64), K, dist).reshape(2)
            d = R.T @ np.array([xy[0], xy[1], 1.0])
            if abs(d[2]) > 1e-6 and (ANKLE_Z - c[2]) / d[2] > 0:
                out[t, k] = (c + d * ((ANKLE_Z - c[2]) / d[2]))[:2]
    return out


def stats(a: np.ndarray) -> dict:
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "median": round(float(np.median(a)), 3), "mean": round(float(a.mean()), 3), "p90": round(float(np.percentile(a, 90)), 3)}


def cv_per_bone(J: np.ndarray, edges) -> dict:
    out = {}
    for a, b in edges:
        ln = np.linalg.norm(J[:, a] - J[:, b], axis=1)
        ln = ln[np.isfinite(ln)]
        if len(ln) > 10:
            out[f"{a}-{b}"] = float(ln.std() / ln.mean())
    return out


def report(stem: str, root: Path | None = None) -> dict:
    from squashvid.ball.clips import work_dir
    from squashvid.ball.regions import venue as get_venue
    from squashvid.viewer.export import build_camera

    work = Path(root) / stem if root else work_dir(stem)
    bodies = pd.read_parquet(work / "bodies.parquet")
    players = pd.read_parquet(work / "players.parquet")
    cam_df = pd.read_parquet(work / "camera.parquet")
    ven = get_venue(stem)
    summary = json.loads((work / "bodies_summary.json").read_text()) if (work / "bodies_summary.json").exists() else {}
    n = int(max(players.frame.max(), cam_df.frame.max())) + 1
    cam = build_camera(cam_df, ven, n).astype(np.float64)
    dist = ven.camera(np.zeros(3), np.zeros(3)).dist
    R = P.rotations(cam)
    hits_p = work / "analysis" / "hits.parquet"
    hits = pd.read_parquet(hits_p) if hits_p.exists() else None
    ts = np.interp(np.arange(n), *[players.drop_duplicates("frame").sort_values("frame")[c].to_numpy() for c in ("frame", "t")])
    out: dict = {"stem": stem, "players": {}}
    hj = list(COCO_OF_H36M)
    cj = [COCO_OF_H36M[j] for j in hj]
    for p in PLAYERS:
        b = bodies[(bodies.player == p) & (bodies.state == STATE_SEEN)]
        arr = player_arrays(players[players.player == p], n)
        fr = b.frame.to_numpy()
        J = np.stack(b.joints.to_numpy()).reshape(-1, 17, 3).astype(np.float64)
        kp, ks = arr["kp"][fr], arr["ks"][fr]
        e = P.reproj_error(P.project(cam[fr], dist, J[:, hj], R[fr]), kp[:, cj], ks[:, cj])
        box_h = (players[(players.player == p)].drop_duplicates("frame").set_index("frame").loc[fr, "y2"] - players[(players.player == p)].drop_duplicates("frame").set_index("frame").loc[fr, "y1"]).to_numpy()
        per_frame = np.nanmedian(e, axis=1)
        flat = flat_lift(cam[fr], dist, arr["pos"][fr], kp, ks)
        flat_edges = COCO_EDGES
        # --- bones
        new_bones = cv_per_bone(J, [(PARENT[j], j) for j in range(1, 17)])
        raw = summary.get("players", {}).get(p, {})
        flat_cv = cv_per_bone(flat[..., :3], flat_edges)
        # --- feet
        lowest = np.minimum(J[:, 3, 2], J[:, 6, 2])
        higher = np.maximum(J[:, 3, 2], J[:, 6, 2])
        flat_ank = np.nanmin(flat[:, [15, 16], 3], axis=1) if np.isfinite(flat[:, [15, 16], 3]).any() else np.array([np.nan])
        # --- depth: lifted ankle y vs the ray / ankle-plane position (independent of the lifter)
        ray = ankle_plane_xy(cam[fr], dist, kp)
        okf = ks[:, [16, 15]] >= 0.5
        lift_xy = J[:, [3, 6], :2]
        flat_xy = flat[:, [16, 15], :2]
        dy_new = np.where(okf, np.abs(lift_xy[..., 1] - ray[..., 1]), np.nan)
        dy_flat = np.where(okf, np.abs(flat_xy[..., 1] - ray[..., 1]), np.nan)
        dx_new = np.where(okf, np.abs(lift_xy[..., 0] - ray[..., 0]), np.nan)
        stance = (J[:, [3, 6], 2] < ANKLE_Z + 0.06) & okf
        # sign agreement of the feet's front/back order when the rays say they differ clearly (a lunge or split step)
        ray_d = ray[:, 0, 1] - ray[:, 1, 1]
        big = np.isfinite(ray_d) & (np.abs(ray_d) > 0.3) & okf.all(axis=1)
        agree_new = float(np.mean(np.sign(lift_xy[big, 0, 1] - lift_xy[big, 1, 1]) == np.sign(ray_d[big]))) if big.any() else None
        fd = flat_xy[big, 0, 1] - flat_xy[big, 1, 1]
        agree_flat = float(np.mean(np.sign(fd) == np.sign(ray_d[big]))) if big.any() else None
        res = {
            "frames_seen": int(len(fr)),
            "reproj_px_all_joints": stats(e.ravel()),
            "reproj_px_per_frame_median": stats(per_frame),
            "reproj_pct_of_box_height": stats(per_frame / np.maximum(box_h, 1) * 100),
            "reproj_px_flat_lift": "0 by construction: every flat-lift point lies on its keypoint ray",
            "bone_cv_new_final": {"median": round(float(np.median(list(new_bones.values()))), 6), "max": round(float(np.max(list(new_bones.values()))), 6)},
            "bone_cv_raw_lifter_before_fixing_lengths": {"median": raw.get("raw_bone_cv_median"), "max": raw.get("raw_bone_cv_max")},
            "bone_cv_flat_lift": {"median": round(float(np.median(list(flat_cv.values()))), 3), "max": round(float(np.max(list(flat_cv.values()))), 3), "per_bone": {k: round(v, 3) for k, v in flat_cv.items()}},
            "foot_lowest_minus_ankle_plane_m": stats(lowest - ANKLE_Z),
            "foot_higher_above_ankle_plane_m": stats(higher - ANKLE_Z),
            "frames_both_feet_over_5cm": round(float(np.mean(lowest - ANKLE_Z > 0.05)), 4),
            "flat_lift_lowest_ankle_height_m": stats(flat_ank),
            "flat_lift_ankle_below_floor_frac": round(float(np.nanmean(flat_ank < ANKLE_Z - 0.08)), 4) if np.isfinite(flat_ank).any() else None,
            "ankle_y_error_vs_ray_m_new": stats(dy_new),
            "ankle_y_error_vs_ray_m_flat": stats(dy_flat),
            "ankle_y_error_vs_ray_m_new_stance": stats(np.where(stance, dy_new, np.nan)),
            "ankle_x_error_vs_ray_m_new": stats(dx_new),
            "foot_order_agreement_new": agree_new,
            "foot_order_agreement_flat": agree_flat,
            "foot_order_frames": int(big.sum()),
        }
        # --- hit moments: hitter's feet / knees vs hip along y
        rows = []
        if hits is not None and "hitter" in hits:
            for _, h in hits[hits.hitter == p].iterrows():
                f = int(np.argmin(np.abs(ts - h["t"])))
                i = np.flatnonzero(fr == f)
                if not len(i) or not np.isfinite(ray[i[0]]).all():
                    continue
                j = J[i[0]]
                ry = ray[i[0], :, 1]
                lead = int(np.argmax(ry))  # 0 = right foot, 1 = left foot (ray says which is further toward the front wall)
                a_i, k_i = (3, 2) if lead == 0 else (6, 5)
                fl = flat[i[0]]
                a_f, k_f = (16, 14) if lead == 0 else (15, 13)
                hip_f = np.nanmean(flat[i[0], [11, 12], 1])
                rows.append({"t": round(float(h["t"]), 2), "ray_foot_gap_m": round(float(abs(ry[0] - ry[1])), 2),
                             "new_lead_foot_minus_hip_y": round(float(j[a_i, 1] - j[0, 1]), 2), "new_lead_knee_minus_hip_y": round(float(j[k_i, 1] - j[0, 1]), 2),
                             "ray_lead_foot_minus_ankles_mid_y": round(float(ry[lead] - ry.mean()), 2),
                             "flat_lead_foot_minus_hip_y": round(float(fl[a_f, 1] - hip_f), 2) if np.isfinite(fl[a_f, 1]) else None})
        rows.sort(key=lambda r: -r["ray_foot_gap_m"])
        res["hit_moments_largest_foot_gap"] = rows[:8]
        if rows:
            big_r = [r for r in rows if r["ray_foot_gap_m"] >= 0.3]
            res["hits_with_foot_gap_ge_0.3m"] = len(big_r)
            res["hits_lead_foot_ahead_of_hip_new"] = int(sum(r["new_lead_foot_minus_hip_y"] > 0 for r in big_r))
            res["hits_lead_foot_ahead_of_hip_flat"] = int(sum((r["flat_lead_foot_minus_hip_y"] or 0) > 0 for r in big_r))
        out["players"][p] = res
    (work / "bodies_validation.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out

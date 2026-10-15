"""Offline stages over stored detections: foot points, tracklets, identity, smoothing, outputs.

Everything here is cheap (seconds per clip), so it reruns freely while the
heavy extraction pass stays cached in work/<stem>/detections.parquet.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from squashvid.calib.venue import Venue
from squashvid.media.video import Video
from squashvid.players import footpoint as fp
from squashvid.players import identity as ident
from squashvid.players.appearance import FEATURES, torso_polygon
from squashvid.players.geometry import ANKLE_Z, CameraTrack, floor_xy, point_height, px_scale
from squashvid.players.occluders import OccluderTrack
from squashvid.players.smooth import rts_smooth
from squashvid.players.tracklets import CONTACT_GAP_PX, candidates, contact_flags, run_tracker, split_tracklets

LONG_GAP_S = 2.0
HIP_Z_RANGE = (0.55, 0.95)
PLAYERS = ("A", "B")
# The phone films through the open door in the back wall. People passing right in front of it
# (spectators, players leaving between games) show as boxes cut by the frame bottom, which the
# box-bottom court filter keeps; their hip-plane point lands ~0.6-0.9 m behind the back wall.
# Players on court read > -0.4 m even when truncated (clips 1-3, WhatsApp copies).
BEHIND_BACK_WALL_M = 0.45
# Torso samples with this share of their rectangle on door hardware are nearly ignored: plate pixels
# brighter than the appearance filter's black cut count as dark shirt. On clip 1 the light-blue
# player read dark in 4.5 % of such samples on the WhatsApp copy and 16 % on the HLG original
# (desaturated shirt), against <= 1.5 % below this share.
TORSO_HW_MAX = 0.3


def _arr(col: pd.Series, shape: tuple[int, ...]) -> np.ndarray:
    return np.stack([np.asarray(v, np.float32).reshape(shape) for v in col]) if len(col) else np.zeros((0, *shape), np.float32)


def foot_points(
    cand: pd.DataFrame, cams: CameraTrack, occ: OccluderTrack, venue: Venue, hip_z: np.ndarray | float = fp.HIP_Z_DEFAULT, frame_hw: tuple[int, int] = (576, 1024)
) -> pd.DataFrame:
    """Ladder choice, court point and covariance per candidate, plus ankle-only and box-only points for comparison."""
    spec = venue.spec
    px = px_scale(frame_hw[1])
    kpts, ksc = _arr(cand.kpts, (17, 2)), _arr(cand.kscore, (17,))
    boxes = cand[["x1", "y1", "x2", "y2"]].to_numpy(np.float64)
    hip_z = np.broadcast_to(np.asarray(hip_z, np.float64), (len(cand),))
    out = {k: [] for k in ("src", "fu", "fv", "fz", "x", "y", "cxx", "cxy", "cyy", "truncated", "ankle_x", "ankle_y", "box_x", "box_y", "hw_bottom")}
    for i, (frame, t) in enumerate(zip(cand.frame.to_numpy(), cand.t.to_numpy())):
        cam = cams(int(frame))
        x1, y1, x2, y2 = boxes[i]
        w = x2 - x1
        ank = kpts[i, [15, 16]]
        hw = occ.under_box(t, boxes[i], px)
        f = fp.FootInputs(boxes[i], kpts[i], ksc[i], bool(occ.contains(t, ank).any()), hw, frame_hw)
        src, uv, z, trunc = fp.choose(f, float(hip_z[i]))
        a_uv = fp.ankle_uv(f)
        ax, ay = floor_xy(cam, a_uv, spec, ANKLE_Z)[0] if a_uv is not None else (np.nan, np.nan)
        bx, by = floor_xy(cam, [(x1 + x2) / 2, y2], spec)[0]
        if uv is not None:
            xy, R = fp.court_point(cam, spec, src, uv, z, w, px)
        else:
            uv, xy, R = np.full(2, np.nan), np.full(2, np.nan), np.full((2, 2), np.nan)
        for k, v in zip(out, (src, uv[0], uv[1], z, xy[0], xy[1], R[0, 0], R[0, 1], R[1, 1], trunc, ax, ay, bx, by, hw)):
            out[k].append(v)
    return pd.DataFrame(out, index=cand.index)


def behind_back_wall(feet: pd.DataFrame, margin: float = BEHIND_BACK_WALL_M) -> np.ndarray:
    """Candidates whose chosen floor point lies clearly behind the back wall, on the camera's side."""
    return (feet.y.to_numpy(np.float64) < -margin) & (feet.src.to_numpy() != "none")


def torso_on_hardware(cand: pd.DataFrame, occ: OccluderTrack) -> np.ndarray:
    """Share of each candidate's torso rectangle (as appearance.torso_features drew it) covered by door hardware."""
    kpts, ksc = _arr(cand.kpts, (17, 2)), _arr(cand.kscore, (17,))
    boxes = cand[["x1", "y1", "x2", "y2"]].to_numpy(np.float64)
    out = np.zeros(len(cand))
    for i, t in enumerate(cand.t.to_numpy()):
        quad, _ = torso_polygon(boxes[i], kpts[i], ksc[i])
        (x0, y0), (x1, y1) = quad.min(0), quad.max(0)
        out[i] = occ.fraction(float(t), x0, y0, x1, y1)
    return out


def colour_weights(cand: pd.DataFrame, feat: np.ndarray, contact: np.ndarray, px: float = 1.0, torso_hw: np.ndarray | None = None) -> np.ndarray:
    """Trust in each torso sample: keypoint torsos with enough pixels, no neighbour overlapping and
    no door hardware in front count fully."""
    n_px = feat[:, FEATURES.index("n_px")]
    w = np.where(cand.torso_kpts.astype("boolean").fillna(False).to_numpy(bool), 1.0, 0.5)
    w = w * np.where(n_px >= 40 * px**2, 1.0, 0.4) * np.where(contact, 0.3, 1.0)
    if torso_hw is not None:
        w = w * np.where(torso_hw > TORSO_HW_MAX, 0.1, 1.0)
    return np.where(n_px > 0, w, 0.0)


def hip_heights(cand: pd.DataFrame, feet: pd.DataFrame, labels: np.ndarray, cams: CameraTrack) -> dict[str, float]:
    """Each player's hip-plane height, measured where both the ankles and hips are seen."""
    kpts, ksc = _arr(cand.kpts, (17, 2)), _arr(cand.kscore, (17,))
    out = {}
    for p in PLAYERS:
        hs = []
        for i in np.flatnonzero((labels == p) & (feet.src.to_numpy() == "ankle")):
            if min(ksc[i, 11], ksc[i, 12]) < 0.5:
                continue
            hip = (kpts[i, 11] + kpts[i, 12]) / 2
            hs.append(point_height(cams(int(cand.frame.iat[i])), hip, [[feet.x.iat[i], feet.y.iat[i]]])[0])
        hs = np.asarray(hs)
        hs = hs[np.isfinite(hs)]
        out[p] = float(np.clip(np.median(hs), *HIP_Z_RANGE)) if len(hs) > 30 else fp.HIP_Z_DEFAULT
        out[f"{p}_n"] = int(len(hs))
    return out


def seconds_to_nearest(times: np.ndarray, marks: np.ndarray) -> np.ndarray:
    """Distance in seconds from each time to the nearest mark (inf when there are none)."""
    if not len(marks):
        return np.full(len(times), np.inf)
    j = np.searchsorted(marks, times)
    left = np.abs(times - marks[np.clip(j - 1, 0, len(marks) - 1)])
    right = np.abs(marks[np.clip(j, 0, len(marks) - 1)] - times)
    return np.minimum(left, right)


def player_tracks(frames: np.ndarray, times: np.ndarray, cand: pd.DataFrame, feet: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """One row per frame per player, smoothed; frames without an accepted measurement are interpolated."""
    tables = []
    pos = {f: i for i, f in enumerate(frames)}
    for p in PLAYERS:
        sel = np.flatnonzero(labels == p)
        T = len(frames)
        z = np.full((T, 2), np.nan)
        R = np.tile(np.eye(2), (T, 1, 1))
        row = np.full(T, -1)
        for i in sel:
            k = pos.get(int(cand.frame.iat[i]))
            if k is None:
                continue
            row[k] = i
            if feet.src.iat[i] != "none" and np.isfinite([feet.x.iat[i], feet.y.iat[i]]).all():
                z[k] = (feet.x.iat[i], feet.y.iat[i])
                R[k] = [[feet.cxx.iat[i], feet.cxy.iat[i]], [feet.cxy.iat[i], feet.cyy.iat[i]]]
        sm = rts_smooth(times, z, R)
        has_det = row >= 0
        tab = pd.DataFrame({"frame": frames, "t": times, "player": p, "cand": row})
        tab["detected"] = has_det
        tab["sx"], tab["sy"] = sm.xy[:, 0], sm.xy[:, 1]
        tab["svx"], tab["svy"] = sm.vel[:, 0], sm.vel[:, 1]
        tab["s_sd_x"], tab["s_sd_y"] = np.sqrt(sm.cov[:, 0, 0]), np.sqrt(sm.cov[:, 1, 1])
        tab["used"], tab["gated"] = sm.used, sm.gated
        acc_t = times[sm.used]
        tab["gap_s"] = seconds_to_nearest(times, acc_t)
        if len(acc_t):  # never extrapolate past the data
            tab.loc[(times < acc_t[0]) | (times > acc_t[-1]), ["sx", "sy", "svx", "svy"]] = np.nan
        tables.append(tab)
    return pd.concat(tables, ignore_index=True)


def assemble(tracks: pd.DataFrame, cand: pd.DataFrame, feet: pd.DataFrame, llr: np.ndarray, contact: np.ndarray, tids: np.ndarray) -> pd.DataFrame:
    """Attach detection-level columns and the flag string to the smoothed per-frame table."""
    c = tracks.cand.to_numpy()
    has = c >= 0
    ci = np.where(has, c, 0)

    def take(values, fill=np.nan):
        v = np.asarray(values)[ci]
        return np.where(has, v, fill)

    out = tracks.copy()
    for col in ("x1", "y1", "x2", "y2", "conf"):
        out[col] = take(cand[col].to_numpy(np.float64))
    for col in ("fu", "fv", "fz", "x", "y", "ankle_x", "ankle_y", "box_x", "box_y"):
        out[col] = take(feet[col].to_numpy(np.float64))
    src = np.where(has, np.asarray(feet.src.to_numpy(), dtype=object)[ci], "interp")
    src = np.where(out.gated.to_numpy(), "gated", src)
    src = np.where(src == "none", "interp", src)
    out["src"] = src
    out["llr"] = take(llr)
    out["tracklet"] = take(tids, -1).astype(np.int64)
    out["contact"] = take(contact, False).astype(bool)
    out["truncated"] = take(feet.truncated.to_numpy(bool), False).astype(bool)
    out["interpolated"] = ~out.used.to_numpy()
    out["long_gap"] = out.gap_s.to_numpy() > LONG_GAP_S
    out["confidence"] = np.where(has, out.conf.to_numpy() * np.where(src == "ankle", 1.0, np.where(src == "box", 0.8, np.where(src == "hip", 0.5, 0.2))), 0.0)
    kp = np.asarray(cand.kpts.to_numpy(), dtype=object)
    ks = np.asarray(cand.kscore.to_numpy(), dtype=object)
    out["kpts"] = [kp[i] if h else None for i, h in zip(ci, has)]
    out["kscore"] = [ks[i] if h else None for i, h in zip(ci, has)]
    flags = []
    for row in out[["interpolated", "long_gap", "contact", "truncated", "gated"]].itertuples(index=False):
        flags.append(",".join(n for n, v in zip(("interpolated", "long_gap", "contact", "truncated", "gated"), row) if v))
    out["flags"] = flags
    return out.drop(columns=["cand"])


def offline(video: Video, venue: Venue, cams: CameraTrack, work: Path, render: bool = True) -> dict:
    """All offline stages for one clip; writes players.parquet and players_summary.json into work."""
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    dets = pd.read_parquet(work / "detections.parquet")
    occ = OccluderTrack.load(work / "occluders.npz")
    frames_all = np.arange(len(video.timestamps))
    times_all = video.timestamps
    lo, hi = int(dets.frame.min()), int(dets.frame.max())
    frames, times = frames_all[lo : hi + 1], times_all[lo : hi + 1]
    frame_hw = (video.info.height, video.info.width)
    px = px_scale(video.info.width)
    cand = candidates(dets, occ, px)
    timing["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    feet = foot_points(cand, cams, occ, venue, frame_hw=frame_hw)
    off = behind_back_wall(feet)
    cand, feet = cand[~off].reset_index(drop=True), feet[~off].reset_index(drop=True)
    timing["footpoint"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    track_id = run_tracker(cand, frames)
    contact = contact_flags(cand, CONTACT_GAP_PX * px)
    tids = split_tracklets(cand, track_id, contact, feet[["x", "y"]].to_numpy())
    timing["tracking"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    feat = _arr(cand.feat, (len(cand.feat.iat[0]),))
    w = colour_weights(cand, feat, contact, px, torso_on_hardware(cand, occ))
    model = ident.ColourModel.fit(feat[(w >= 0.5)])
    llr = np.where(w > 0, model.llr(feat), 0.0)
    var = np.maximum(feet.cxx.to_numpy(), feet.cyy.to_numpy())
    tl = ident.build_tracklets(tids, cand.frame.to_numpy(), cand.t.to_numpy(), feet[["x", "y"]].to_numpy(), np.nan_to_num(var, nan=1.0), llr, w)
    lab_by_tid = ident.assign(tl)
    labels = np.array([lab_by_tid.get(int(t), "X") for t in tids])
    timing["identity"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    hips = hip_heights(cand, feet, labels, cams)
    hip_z = np.where(labels == "A", hips["A"], np.where(labels == "B", hips["B"], fp.HIP_Z_DEFAULT))
    feet = foot_points(cand, cams, occ, venue, hip_z, frame_hw)
    tracks = player_tracks(frames, times, cand, feet, labels)
    table = assemble(tracks, cand, feet, llr, contact, tids)
    timing["smooth"] = time.perf_counter() - t0
    table.to_parquet(work / "players.parquet", index=False)
    cand.assign(track_id=track_id, tracklet=tids, label=labels, llr=llr, contact=contact, **{c: feet[c] for c in feet.columns}).drop(columns=["feat"]).to_parquet(work / "candidates.parquet", index=False)

    summary = {
        "n_candidates": int(len(cand)),
        "n_dropped_behind_back_wall": int(off.sum()),
        "n_tracker_ids": int(len(np.unique(track_id[track_id >= 0]))),
        "n_tracklets": int(len(tl)),
        "labels": {k: int((labels == k).sum()) for k in ("A", "B", "X")},
        "colour_model": {"mean_A": model.mean[0].round(3).tolist(), "mean_B": model.mean[1].round(3).tolist(), "weight": model.weight.round(3).tolist()},
        "hip_height_m": hips,
        "timing_s": {k: round(v, 2) for k, v in timing.items()},
    }
    (work / "players_summary.json").write_text(json.dumps(summary, indent=2))
    if render:
        from squashvid.players.report import report

        report(video, venue, cams, work)
    return summary

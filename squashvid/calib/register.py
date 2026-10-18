"""Per-frame camera registration: track the phone's pose through a clip with the venue held fixed.

Each frame is refined from the previous frame's pose. When the match quality
collapses (the phone was knocked or re-aimed) the frame is re-acquired with
the coarse chamfer search. Poses are then split into stable segments at
motion spikes and smoothed only within a segment, so a real jump is never
blurred into the frames around it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from squashvid.calib.coarse import coarse_align
from squashvid.calib.lines import model_samples, refine_joint, response_maps
from squashvid.calib.venue import Venue
from squashvid.geom.camera import Camera
from squashvid.media.video import Video

REACQUIRE_MATCH_FRAC = 0.45
REACQUIRE_RMS = 1.2  # reference px (see lines.REFERENCE_WIDTH)
JUMP_PX = 3.0  # reference px of frame-to-frame court motion that starts a new segment
SMOOTH_FRAMES = 9
# Where the phone sits behind the back glass of court 5 (IMG_4091 frame 0). Every clip so far starts within a few cm and
# degrees of this, and the first frame's coarse search covers that spread. A venue fitted later carries its own seed.
SEED_POSE = (
    np.array([1.633313124097702, -0.004703997290973849, 0.02261087069225295]),
    np.array([-2.8095257861677725, 1.234736875538943, 2.0130706223337573]),
)


@dataclass
class FramePose:
    idx: int
    t: float
    rvec: np.ndarray
    tvec: np.ndarray
    rms: float
    match_frac: float
    n_matched: int
    reacquired: bool


def _probe_points(venue: Venue) -> np.ndarray:
    """Court points spread over the floor and front wall, used to measure image motion between poses."""
    s = venue.spec
    xs = np.linspace(0.5, s.width - 0.5, 4)
    floor = [(x, y, 0.0) for x in xs for y in (2.0, s.short_y, 7.5)]
    front = [(x, s.length, z) for x in xs for z in (s.tin_z, s.service_z)]
    return np.array(floor + front, dtype=np.float64)


def motion_px(a: Camera, b: Camera, probe: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(a.project(probe) - b.project(probe), axis=1)))


def track(video: Video, venue: Venue, seed: Camera, start: float = 0.0, end: float | None = None, step: float = 0.08, progress=None) -> list[FramePose]:
    samples = model_samples(spec=venue.spec, step=step)
    prev = venue.camera(seed.rvec, seed.tvec)
    out: list[FramePose] = []
    first = True
    for idx, t, img in video.frames(start, end):
        maps = response_maps(img)
        reacq = False
        if first:
            prev = coarse_align(prev, maps, samples, venue.spec).camera
            reacq = True
            first = False
        fit = refine_joint([prev], [maps], samples, radii=(5.0, 3.0, 2.0), spec=venue.spec)
        fr = fit.frames[0]
        if fr.match_frac < REACQUIRE_MATCH_FRAC or fr.rms > REACQUIRE_RMS * maps.scale:
            start_cam = coarse_align(prev, maps, samples, venue.spec).camera
            alt = refine_joint([start_cam], [maps], samples, radii=(8.0, 5.0, 3.0, 2.0), spec=venue.spec)
            if alt.frames[0].n_inliers > fr.n_inliers:
                fit, fr, reacq = alt, alt.frames[0], True
        cam = fit.cameras[0]
        out.append(FramePose(idx, t, cam.rvec.copy(), cam.tvec.copy(), fr.rms, fr.match_frac, fr.n_matched, reacq))
        prev = cam
        if progress is not None:
            progress(idx, t, fr)
    return out


def to_frame(poses: list[FramePose]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [p.idx for p in poses],
            "t": [p.t for p in poses],
            **{f"r{i}": [p.rvec[i] for p in poses] for i in range(3)},
            **{f"t{i}": [p.tvec[i] for p in poses] for i in range(3)},
            "rms": [p.rms for p in poses],
            "match_frac": [p.match_frac for p in poses],
            "n_matched": [p.n_matched for p in poses],
            "reacquired": [p.reacquired for p in poses],
        }
    )


def segment_and_smooth(df: pd.DataFrame, venue: Venue, jump_px: float = JUMP_PX, window: int = SMOOTH_FRAMES) -> pd.DataFrame:
    """Add motion (frame px), segment ids and within-segment smoothed poses (columns sr*, st*).

    jump_px is reference px.
    """
    s = venue.pixel_scale
    probe = _probe_points(venue)
    cams = [venue.camera(row[["r0", "r1", "r2"]].to_numpy(float), row[["t0", "t1", "t2"]].to_numpy(float)) for _, row in df.iterrows()]
    motion = np.zeros(len(cams))
    for i in range(1, len(cams)):
        motion[i] = motion_px(cams[i - 1], cams[i], probe)
    # A jump is a motion spike well above the frame's local jitter.
    local = pd.Series(motion).rolling(15, center=True, min_periods=1).median().to_numpy()
    jump = (motion > jump_px * s) & (motion > 4 * np.maximum(local, 0.25 * s))
    seg = np.cumsum(jump)
    df = df.copy()
    df["motion_px"] = motion
    df["segment"] = seg
    cols = ["r0", "r1", "r2", "t0", "t1", "t2"]
    for c in cols:
        df["s" + c] = df[c]
    for _, idx in df.groupby("segment").indices.items():
        n = len(idx)
        if n < 5:
            continue
        w = min(window, n if n % 2 else n - 1)
        if w < 5:
            continue
        for c in cols:
            df.loc[df.index[idx], "s" + c] = savgol_filter(df[c].to_numpy()[idx], w, 2)
    return df


def draw_court(img: np.ndarray, cam: Camera, spec, step: float = 0.03) -> np.ndarray:
    """Draw every court line in place (paint red, junctions yellow), skipping parts outside the lens field."""
    from squashvid.geom import court

    h, w = img.shape[:2]
    for line in court.court_lines(spec):
        pts = line.sample(step)
        ok = cam.in_field(pts, w, h)
        uv = np.round(cam.project(pts) * 8).astype(np.int32)
        colour = (0, 230, 255) if line.kind == "edge" else (40, 40, 255)
        # Split the polyline wherever it leaves the valid field.
        runs = np.split(np.arange(len(pts)), np.flatnonzero(np.diff(ok.astype(int))) + 1)
        for run in runs:
            if len(run) > 1 and ok[run[0]]:
                cv2.polylines(img, [uv[run]], False, colour, 1, cv2.LINE_AA, 3)
    return img


def save(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def summarise(df: pd.DataFrame) -> dict:
    """Quality numbers of a registered clip: how well the court lines matched, and how often the phone moved."""
    return {
        "frames": int(len(df)),
        "match_frac_median": round(float(df.match_frac.median()), 4),
        "match_frac_p5": round(float(df.match_frac.quantile(0.05)), 4),
        "rms_median": round(float(df.rms.median()), 4),
        "rms_p95": round(float(df.rms.quantile(0.95)), 4),
        "segments": int(df.segment.nunique()),
        "reacquired": int(df.reacquired.sum()),
    }


def register_video(video: Video, venue: Venue, out_dir: str | Path, progress=None) -> dict:
    """Track every frame from the venue's seed (court 5: SEED_POSE), smooth within segments, write camera.parquet and camera_summary.json."""
    out_dir = Path(out_dir)
    poses = track(video, venue, venue.camera(*(venue.seed or SEED_POSE)), progress=progress)
    df = segment_and_smooth(to_frame(poses), venue)
    save(df, out_dir / "camera.parquet")
    summary = summarise(df)
    (out_dir / "camera_summary.json").write_text(json.dumps(summary, indent=1))
    return summary

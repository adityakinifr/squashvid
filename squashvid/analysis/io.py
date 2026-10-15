"""Inputs for the match-analysis stage: the per-frame player table, court, audio onsets and (when present) ball events.

Everything downstream works on a `Match`, so a clip's evidence is read once.
The ball files are optional: `Match.ball_events` is None until the ball stage
has written them, and every consumer must cope with that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from squashvid.ball.clips import WORK
from squashvid.geom.court import CourtSpec

CLIPS = ("IMG_4089", "IMG_4090", "IMG_4091")
PLAYERS = ("A", "B")
L_WRIST, R_WRIST = 9, 10
WRIST_MIN_SCORE = 0.3
OUT_DIR = "analysis"
REVIEW_DIR = WORK / "analysis_review"


def work_dir(stem: str) -> Path:
    return WORK / stem


def out_dir(stem: str) -> Path:
    d = work_dir(stem) / OUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def other(player: str) -> str:
    return "B" if player == "A" else "A"


@dataclass
class Track:
    """One player's per-frame arrays, aligned to the shared frame index."""

    x: np.ndarray  # court metres across, NaN where unknown
    y: np.ndarray  # court metres from the back wall
    sd: np.ndarray  # positional uncertainty, metres
    detected: np.ndarray
    speed: np.ndarray  # m/s, smoothed
    wrist: np.ndarray  # box-heights per second, NaN when no wrist was seen


@dataclass
class Match:
    stem: str
    spec: CourtSpec
    t: np.ndarray
    fps: float
    tracks: dict[str, Track]
    onsets: pd.DataFrame
    audio_rallies: pd.DataFrame
    ball_events: pd.DataFrame | None = None
    ball_detections: pd.DataFrame | None = None
    bodies: pd.DataFrame | None = None  # 3D figures (t, player, state, joints) when squashvid.bodies has run
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    @cached_property
    def T(self) -> np.ndarray:
        return np.asarray(self.spec.t_point[:2], float)

    def frame_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, self.n - 1))

    def pos(self, player: str, t: float) -> np.ndarray:
        i = self.frame_at(t)
        tr = self.tracks[player]
        return np.array([tr.x[i], tr.y[i]])

    def dist_T(self, player: str) -> np.ndarray:
        tr = self.tracks[player]
        return np.hypot(tr.x - self.T[0], tr.y - self.T[1])


def _interp(a: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a)
    if ok.all() or not ok.any():
        return a
    idx = np.arange(len(a))
    return np.interp(idx, idx[ok], a[ok])


def wrist_speed(kpts: np.ndarray, kscore: np.ndarray, height: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Faster of the two wrists, in box heights per second, from consecutive frames."""
    n = len(t)
    best = np.full(n, np.nan)
    dt = np.r_[np.nan, np.diff(t)]
    for w in (L_WRIST, R_WRIST):
        p = kpts[:, w, :].copy()
        p[kscore[:, w] < WRIST_MIN_SCORE] = np.nan
        step = np.r_[np.nan, np.hypot(*np.diff(p, axis=0).T)]
        v = step / np.maximum(height, 50.0) / dt
        best = np.fmax(best, v)
    return best


def _unpack(col: pd.Series, shape: tuple[int, ...]) -> np.ndarray:
    out = np.full((len(col),) + shape, np.nan)
    for i, v in enumerate(col.to_numpy()):
        if v is not None:
            out[i] = np.asarray(v, float).reshape(shape)
    return out


def load_tracks(stem: str) -> tuple[np.ndarray, dict[str, Track]]:
    p = pd.read_parquet(work_dir(stem) / "players.parquet")
    tracks, t_ref = {}, None
    for k in PLAYERS:
        q = p[p.player == k].sort_values("frame").reset_index(drop=True)
        t = q.t.to_numpy()
        if t_ref is None:
            t_ref = t
        kp, ks = _unpack(q.kpts, (17, 2)), _unpack(q.kscore, (17,))
        h = (q.y2 - q.y1).to_numpy()
        ws = wrist_speed(kp, ks, h, t)
        sp = np.hypot(q.svx.to_numpy(), q.svy.to_numpy())
        tracks[k] = Track(
            x=_interp(q.sx.to_numpy()),
            y=_interp(q.sy.to_numpy()),
            sd=np.hypot(q.s_sd_x.to_numpy(), q.s_sd_y.to_numpy()),
            detected=q.detected.to_numpy(bool),
            speed=np.nan_to_num(sp),
            wrist=ws,
        )
    return t_ref, tracks


def load_ball_events(stem: str) -> pd.DataFrame | None:
    f = work_dir(stem) / "ball_events.parquet"
    if not f.exists():
        return None
    ev = pd.read_parquet(f)
    return ev if len(ev) else None


def load_ball_detections(stem: str) -> pd.DataFrame | None:
    f = work_dir(stem) / "ball" / "blurball-ft-final@576x1024-coreml.parquet"
    if not f.exists():
        return None
    try:
        return pd.read_parquet(f)
    except Exception:  # file still being written by the ball run
        return None


def load_bodies(stem: str) -> pd.DataFrame | None:
    f = work_dir(stem) / "bodies.parquet"
    return pd.read_parquet(f, columns=["t", "player", "state", "joints"]) if f.exists() else None


def load(stem: str, use_ball: bool = True) -> Match:
    from squashvid.ball.regions import venue

    t, tracks = load_tracks(stem)
    fps = float(1.0 / np.median(np.diff(t)))
    wd = work_dir(stem)
    return Match(
        stem=stem,
        spec=venue(stem).spec,
        t=t,
        fps=fps,
        tracks=tracks,
        onsets=pd.read_parquet(wd / "audio_onsets.parquet"),
        audio_rallies=pd.read_parquet(wd / "audio_rallies.parquet"),
        ball_events=load_ball_events(stem) if use_ball else None,
        ball_detections=load_ball_detections(stem),
        bodies=load_bodies(stem),
    )

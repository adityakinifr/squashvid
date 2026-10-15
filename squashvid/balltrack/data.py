"""Inputs of the stage: windows, labels, candidates (held-out folds or full-clip runs), audio, camera, players."""

from __future__ import annotations

import glob
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from squashvid.ball import labels as ball_labels
from squashvid.ball.clips import LABELS, WORK, ball_dir, video, work_dir
from squashvid.ball.finetune import FOLD_TESTS

LABELLER = "aditya"
SUFFIX = "@576x1024-coreml.parquet"
FOLD_OF = {w: f for f, ws in FOLD_TESTS.items() for w in ws}
MIN_PEAK = 0.5  # full-clip candidate files hold blobs from this peak up


def windows() -> dict[str, dict]:
    return {w["id"]: w for w in json.loads((LABELS / "windows.json").read_text())}


@lru_cache(maxsize=None)
def timestamps(stem: str) -> np.ndarray:
    return np.asarray(video(stem).timestamps, dtype=np.float64)


def window_labels(wid: str) -> pd.DataFrame:
    w = windows()[wid]
    df = ball_labels.load(w["stem"], labeller=LABELLER)
    return df[(df.frame >= w["i0"]) & (df.frame <= w["i1"])].reset_index(drop=True)


def cv_candidates(wid: str, min_peak: float = MIN_PEAK) -> pd.DataFrame:
    """Held-out candidates of a labelled window, as the full-clip run would give them.

    The cross-validation runs saved every blob with peak >= 0.7, and the top blob of each frame at the
    lower thresholds of the sweep (work/ball_ft/sweep_cache). Their union stands in for the full-clip
    file, which keeps every blob with peak >= 0.5: it lacks the second and third blobs between 0.5 and 0.7.
    """
    w = windows()[wid]
    fold = FOLD_OF[wid]
    stem = w["stem"]
    c = pd.read_parquet(ball_dir(stem) / "cands" / f"blurball-ft-fold{fold}{SUFFIX}")
    c = c[(c.frame >= w["i0"]) & (c.frame <= w["i1"])][["frame", "x", "y", "peak", "mass"]]
    files = sorted(glob.glob(str(WORK / "ball_ft" / "sweep_cache" / f"fold{fold}_*_{wid}_{min_peak:g}.parquet")))
    if files:
        top = pd.read_parquet(files[-1])
        top = top[top.visible & top.x.notna() & (top.score < 0.7)]
        extra = pd.DataFrame({"frame": top.frame, "x": top.x, "y": top.y, "peak": top.score, "mass": 1.0})
        c = pd.concat([c, extra[(extra.frame >= w["i0"]) & (extra.frame <= w["i1"])]], ignore_index=True)
    return c.sort_values(["frame", "peak"], ascending=[True, False]).reset_index(drop=True)


def cv_raw(wid: str, threshold: float = 0.5) -> pd.DataFrame:
    """The detector's own per-frame answer (top blob) at a threshold, from the sweep cache."""
    w = windows()[wid]
    files = sorted(glob.glob(str(WORK / "ball_ft" / "sweep_cache" / f"fold{FOLD_OF[wid]}_*_{wid}_{threshold:g}.parquet")))
    df = pd.read_parquet(files[-1])
    return df[(df.frame >= w["i0"]) & (df.frame <= w["i1"])].reset_index(drop=True)


def clip_candidates(stem: str, variant: str = "blurball-ft-final") -> pd.DataFrame:
    path = ball_dir(stem) / "cands" / f"{variant}{SUFFIX}"
    return pd.read_parquet(path)[["frame", "x", "y", "peak", "mass"]].sort_values(["frame", "peak"], ascending=[True, False]).reset_index(drop=True)


def audio_onsets(stem: str) -> pd.DataFrame:
    return pd.read_parquet(work_dir(stem) / "audio_onsets.parquet")


def audio_offset(stem: str) -> float:
    """Audio minus video, seconds (work/<stem>/audio_eval.json)."""
    return float(json.loads((work_dir(stem) / "audio_eval.json").read_text())["offset_used_ms"]) / 1e3


def players(stem: str) -> pd.DataFrame:
    return pd.read_parquet(work_dir(stem) / "players.parquet")


def output_path(stem: str, name: str) -> Path:
    return work_dir(stem) / name

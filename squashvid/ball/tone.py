"""Superseded: a per-channel histogram-matched curve from raw HLG code values to the copy's SDR look.

media.video now converts HLG originals to SDR BT.709 itself (hlg_to_sdr), so the
pipeline no longer applies this; applying it to Video output would convert twice.
It is kept as the record behind the '-tone' variants in work/<stem>/ball/ (the
first zero-shot evaluation) and works only on raw code values, i.e. frames from
video(stem, sdr=False).

Why it existed: PyAV decoded the iPhone's 10-bit HLG frames to bgr24 without tone
mapping, so the picture was bright and flat (median grey ~183 against ~146 in the
WhatsApp copy), and a faint ball on a flat bright wall had less contrast than the
SDR-trained detectors expect. Histogram-matching the raw original to the copy at
content-matched frames gave a 256-entry lookup table per channel.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from squashvid.ball.clips import WORK


def match_curve(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Monotonic 256-entry map taking the value distribution of src (uint8) onto that of ref."""
    hs = np.bincount(src.ravel(), minlength=256).astype(np.float64)
    hr = np.bincount(ref.ravel(), minlength=256).astype(np.float64)
    cs, cr = np.cumsum(hs) / hs.sum(), np.cumsum(hr) / hr.sum()
    lut = np.interp(cs, cr, np.arange(256))
    return np.clip(np.round(np.maximum.accumulate(lut)), 0, 255).astype(np.uint8)


def fit(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """(256, 3) BGR lookup table from (original, copy) frame pairs; the original is resized to the copy."""
    src, ref = [], []
    for orig, copy in pairs:
        src.append(cv2.resize(orig, (copy.shape[1], copy.shape[0]), interpolation=cv2.INTER_AREA))
        ref.append(copy)
    src, ref = np.stack(src), np.stack(ref)
    return np.stack([match_curve(src[..., c], ref[..., c]) for c in range(3)], axis=1)


class Tone:
    """Callable BGR -> BGR applying a (256, 3) table."""

    def __init__(self, lut: np.ndarray):
        self.lut = np.asarray(lut, np.uint8).reshape(256, 1, 3)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        return cv2.LUT(bgr, self.lut)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.lut.reshape(256, 3).T.tolist()))
        return path

    @classmethod
    def load(cls, path: Path) -> "Tone":
        return cls(np.array(json.loads(path.read_text()), np.uint8).T)


TONE_PATH = WORK / "labels" / "tone_hlg_to_copy.json"


def fit_from_clips(times: dict[str, list[float]]) -> Tone:
    """Fit on content-matched frames: {original stem: [times]}; the copy's frame k shows the original ~1 frame earlier."""
    from squashvid.ball.align import match_window
    from squashvid.ball.clips import COPIES, video

    pairs = []
    for orig, ts in times.items():
        for t in ts:
            m = match_window(orig, t, t + 0.01)
            if not len(m):
                continue
            r = m.iloc[0]
            _, _, copy_img = video(COPIES[orig]).frame_at(float(r.copy_t))
            _, _, orig_img = video(orig, sdr=False).frame_at(float(r.t))
            pairs.append((orig_img, copy_img))
    return Tone(fit(pairs))


def default() -> Tone | None:
    return Tone.load(TONE_PATH) if TONE_PATH.exists() else None

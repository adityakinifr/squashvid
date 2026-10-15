"""TrackNetV3 (qaz812345/TrackNetV3): an 8-frame U-Net with a median background, plus InpaintNet.

The checkpoint was trained on badminton at 288x512 with bg_mode='concat': the
input is the clip's median frame followed by 8 RGB frames (27 channels), and
the network emits 8 sigmoid heatmaps. The official predictor ensembles the 8
predictions each frame receives from overlapping windows with triangular
weights, thresholds at 0.5 and keeps the centre of the largest blob's bounding
box. The U-Net is fully convolutional (three 2x pools), so it also runs at
native 576x1024, which keeps the 1.6 px front-wall ball from halving.

The median is taken over the requested window, not the whole clip as upstream
does, because this camera drifts by tens of pixels over a match and a
clip-wide median would smear the walls the ball is seen against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from squashvid.ball.base import Clip, Detector, blobs, plan_blocks
from squashvid.ball.clips import WEIGHTS
from squashvid.ball.vendor import load_checkpoint, load_module

TRACKNET_CKPT = WEIGHTS / "tracknetv3" / "ckpts" / "TrackNet_best.pt"
INPAINT_CKPT = WEIGHTS / "tracknetv3" / "ckpts" / "InpaintNet_best.pt"
COOR_TH = 50 / np.hypot(288, 512)  # upstream: normalised coords below this on both axes mean "no ball"


def ensemble_weights(n: int) -> np.ndarray:
    """Upstream 'weight' mode: 1, 2, ..., n/2, n/2, ..., 2, 1 (normalised later)."""
    half = np.arange(1, (n + 1) // 2 + 1, dtype=np.float64)
    return np.concatenate([half, half[: n // 2][::-1]])[:n]


class TrackNetV3(Detector):
    name = "tracknetv3"
    frames_in = 8
    merge = "mean"
    threshold = 0.5
    batch_default = {"default": 4, "native": 2}

    def __init__(self, size="default", roi=None, ensemble: bool = True, inpaint: bool = False, median_samples: int = 120, **kw):
        self.ensemble = ensemble
        self.use_inpaint = inpaint
        self.median_samples = median_samples
        self.median: np.ndarray | None = None
        super().__init__(str(TRACKNET_CKPT), size, roi, **kw)
        if inpaint:
            mod = load_module("TrackNetV3/model.py", "tracknetv3_model")
            ck = load_checkpoint(INPAINT_CKPT)
            self.inpaint_len = int(ck["param_dict"]["seq_len"])
            self.inpaintnet = mod.InpaintNet().eval()
            self.inpaintnet.load_state_dict(ck["model"])
        if inpaint:
            self.name = "tracknetv3-inp"
        if not ensemble:
            self.name += "-fast"

    def build(self) -> torch.nn.Module:
        mod = load_module("TrackNetV3/model.py", "tracknetv3_model")
        ck = load_checkpoint(self.weights)
        p = ck["param_dict"]
        assert p["bg_mode"] == "concat" and p["seq_len"] == self.frames_in, p
        net = mod.TrackNet(in_dim=(self.frames_in + 1) * 3, out_dim=self.frames_in)
        net.load_state_dict(ck["model"])
        return net

    def prepare(self, video, lo: int, hi: int) -> None:
        import cv2

        ts = video.timestamps
        step = max(1, (hi - lo + 1) // self.median_samples)
        imgs = [cv2.cvtColor(self.geom.warp(bgr), cv2.COLOR_BGR2RGB) for _, _, bgr in video.frames(float(ts[lo]), float(ts[hi]), step=step)]
        self.median = self.preprocess(np.median(np.stack(imgs), axis=0).astype(np.uint8))

    def tensor(self, frames: list[np.ndarray]) -> np.ndarray:
        return np.concatenate([self.median, *frames], axis=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # sigmoid is inside the network

    def plan(self, i0: int, i1: int, lo: int, hi: int) -> list[Clip]:
        if self.ensemble:
            return plan_blocks(i0, i1, self.frames_in, 1, lo, hi, ensemble_weights(self.frames_in))
        return plan_blocks(i0, i1, self.frames_in, self.frames_in, lo, hi)

    def decode(self, hm: np.ndarray):
        found = blobs(hm, self.threshold)
        if not found:
            return []
        # Upstream keeps the blob with the largest bounding box; the centroid stands in for its box centre.
        x, y, peak, mass, _ = max(found, key=lambda b: b[4])
        return [(x, y, peak, mass)]

    def select(self, cands):
        out = []
        for c in cands:
            out.append((c[0][0], c[0][1], c[0][2], True) if c else (np.nan, np.nan, 0.0, False))
        return out

    def run(self, video, start, end, batch_size=None) -> pd.DataFrame:
        df = super().run(video, start, end, batch_size)
        if self.use_inpaint:
            df = inpaint_track(df, self.inpaintnet, self.frame_hw, self.inpaint_len)
        return df


def inpaint_mask(y: np.ndarray, vis: np.ndarray, th: float) -> np.ndarray:
    """Port of upstream generate_inpaint_mask: gaps bounded by in-frame detections get repaired.

    A gap whose neighbours sit in the top `th` pixels is taken as the ball leaving the
    top of the frame and is left alone.
    """
    n = len(vis)
    mask = np.zeros(n, dtype=np.float32)
    i = j = 0
    while j < n:
        while i < n - 1 and vis[i]:
            i += 1
        j = i
        while j < n - 1 and not vis[j]:
            j += 1
        if j == i:
            break
        if i == 0 and y[j] > th:
            mask[:j] = 1
        elif i > 1 and y[i - 1] > th and y[j] > th:
            mask[i:j] = 1
        i = j
    return mask


def inpaint_track(df: pd.DataFrame, net: torch.nn.Module, frame_hw: tuple[int, int], seq_len: int = 16) -> pd.DataFrame:
    """Fill gaps in a TrackNet track with InpaintNet, ensembling overlapping 16-frame windows."""
    H, W = frame_hw
    vis = df["visible"].to_numpy(bool)
    xy = np.where(vis[:, None], np.c_[df["x"].fillna(0) / W, df["y"].fillna(0) / H], 0.0).astype(np.float32)
    mask = inpaint_mask(df["y"].fillna(0).to_numpy(), vis, th=H * 0.05)
    n = len(df)
    if n == 0 or not mask.any():
        return df.assign(inpainted=False)
    pad = max(0, seq_len - n)
    xy_p = np.concatenate([xy, np.repeat(xy[-1:], pad, 0)])
    m_p = np.concatenate([mask, np.repeat(mask[-1:], pad, 0)])
    starts = range(0, len(xy_p) - seq_len + 1)
    X = torch.from_numpy(np.stack([xy_p[s : s + seq_len] for s in starts]))
    M = torch.from_numpy(np.stack([m_p[s : s + seq_len] for s in starts]))[..., None]
    with torch.inference_mode():
        P = (net(X, M) * M + X * (1 - M)).numpy()
    w = ensemble_weights(seq_len)
    acc = np.zeros_like(xy_p, dtype=np.float64)
    wsum = np.zeros(len(xy_p))
    for k, s in enumerate(starts):
        acc[s : s + seq_len] += w[:, None] * P[k]
        wsum[s : s + seq_len] += w
    res = (acc / wsum[:, None])[:n]
    new_vis = ~((res[:, 0] < COOR_TH) & (res[:, 1] < COOR_TH))
    filled = new_vis & ~vis & (mask > 0)
    out = df.copy()
    out.loc[filled, "x"] = res[filled, 0] * W
    out.loc[filled, "y"] = res[filled, 1] * H
    out.loc[filled, "score"] = np.nan  # no heatmap behind an inpainted point
    out.loc[filled, "visible"] = True
    out["inpainted"] = filled
    return out

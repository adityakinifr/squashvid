"""Temporal 2D -> 3D lifting with MotionBERT (Apache-2.0; H36M-17 in, root-relative 3D out, camera-axis frame).

Weights: https://huggingface.co/walterzhu/MotionBERT (checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin, no login),
stored at work/weights/ or third_party/weights/lifters/motionbert_ft_h36m.bin. Output axes: x right, y down, z away from
the camera, in units where the window's 2D bounding box spans 2 (so the scale is not metric; `place` sets it).
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
WEIGHTS_URL = "https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin"
WEIGHTS_NAME = "motionbert_ft_h36m.bin"
ROOT = Path(__file__).resolve().parents[2]
WINDOW = 243
LEFT = [4, 5, 6, 11, 12, 13]
RIGHT = [1, 2, 3, 14, 15, 16]


def weights_path(download: bool = True) -> Path:
    for d in (ROOT / "third_party" / "weights" / "lifters", ROOT / "work" / "weights"):
        if (d / WEIGHTS_NAME).exists():
            return d / WEIGHTS_NAME
    dest = ROOT / "third_party" / "weights" / "lifters" / WEIGHTS_NAME
    if not download:
        raise FileNotFoundError(f"{WEIGHTS_NAME} not found; run `python -I -m squashvid.bodies --download`")
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.warning("downloading MotionBERT weights (170 MB) to %s", dest)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    tmp.rename(dest)
    return dest


def load_model(device: str = "cpu"):
    import torch

    from squashvid.bodies.dstformer import DSTformer

    ck = torch.load(weights_path(), map_location="cpu", weights_only=False)  # the released file is a plain pickle
    sd = {k.removeprefix("module."): v for k, v in ck["model_pos"].items()}
    model = DSTformer()
    model.load_state_dict(sd, strict=True)
    return model.eval().to(device)


def crop_scale(x: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """MotionBERT's input normalisation: the window's 2D bounding box is centred and its longer side mapped to [-1, 1]."""
    ok = conf > 0
    pts = x[ok]
    if len(pts) < 4:
        return np.zeros(x.shape[:-1] + (3,), np.float32)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    scale = max(hi[0] - lo[0], hi[1] - lo[1])
    if scale <= 0:
        return np.zeros(x.shape[:-1] + (3,), np.float32)
    xy = ((x - (lo + hi - scale) / 2) / scale - 0.5) * 2
    return np.clip(np.concatenate([xy, conf[..., None]], axis=-1), -1, 1).astype(np.float32)


def flip(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    a[..., 0] *= -1
    a[..., LEFT + RIGHT, :] = a[..., RIGHT + LEFT, :]
    return a


def lift_window(model, x2d: np.ndarray, conf: np.ndarray, device: str = "cpu") -> np.ndarray:
    """One window (T<=243, 17, 2) pixels + (T,17) confidence -> (T,17,3) root-relative, flip-averaged."""
    import torch

    inp = crop_scale(x2d, conf)
    with torch.no_grad():
        a = torch.from_numpy(inp)[None].to(device)
        b = torch.from_numpy(flip(inp))[None].to(device)
        out = (model(a) + torch.from_numpy(flip(model(b)[0].cpu().numpy())).to(device)[None]) / 2
    out = out[0].cpu().numpy().astype(np.float64)
    return out - out[:, :1]


def lift_segment(model, x2d: np.ndarray, conf: np.ndarray, device: str = "cpu", stride: int = 121) -> np.ndarray:
    """Lift a whole segment, blending overlapping windows with triangular weights so seams do not show.

    Each window is normalised on its own bounding box, so windows differ in unit; callers use the bone directions
    only, which do not depend on the unit."""
    n = len(x2d)
    if n <= WINDOW:
        return lift_window(model, x2d, conf, device)
    acc = np.zeros((n, 17, 3))
    wsum = np.zeros(n)
    starts = list(range(0, n - WINDOW + 1, stride))
    if starts[-1] != n - WINDOW:
        starts.append(n - WINDOW)
    tri = np.minimum(np.arange(1, WINDOW + 1), np.arange(WINDOW, 0, -1)).astype(np.float64)
    for s in starts:
        o = lift_window(model, x2d[s : s + WINDOW], conf[s : s + WINDOW], device)
        # Rescale to unit bone-sum so the blend mixes like with like.
        bs = np.linalg.norm(o[:, 1:] - o[:, [p for p in _PARENTS[1:]]], axis=2).sum(axis=1).mean()
        o = o / max(bs, 1e-9)
        acc[s : s + WINDOW] += o * tri[:, None, None]
        wsum[s : s + WINDOW] += tri
    return acc / wsum[:, None, None]


_PARENTS = (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)

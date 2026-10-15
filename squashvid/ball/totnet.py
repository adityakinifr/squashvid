"""TOTNet (AugustRushG/TOTNet): a 5-frame U-Net with 3D temporal convolutions and a spatial softmax.

It predicts the ball in the last of its five ImageNet-normalised RGB frames as
a softmax over every pixel, so it always names a location; we report the
probability mass within 2 px of the peak as the score and treat the ball as
visible above a threshold. Upstream ships tennis, badminton and table-tennis
(TTA) checkpoints trained with occlusion augmentation, the model's selling
point for a court where players and the door hide the ball.

Its pooling and upsampling sizes are baked in from the input shape, so the
network is built for whatever size it runs at (576x1024 works; the weights do
not depend on it). PyTorch MPS implements neither adaptive_max_pool3d nor
trilinear upsampling, so both are swapped for exact equivalents made of 2D
pooling, 2D bilinear resizing and an explicit temporal blend, which also keeps
the graph exportable to CoreML.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from squashvid.ball.base import Clip, Detector, plan_causal
from squashvid.ball.clips import THIRD_PARTY
from squashvid.ball.vendor import load_checkpoint, load_module, strip_prefix
from squashvid.ball.wasb import MEAN, STD

CHECKPOINTS = {
    sport: next((THIRD_PARTY / "TOTNet" / "weights").glob(f"TOTNet_{tag}_*/*_best.pth"), None)
    for sport, tag in (("tennis", "Tennis"), ("badminton", "Badminton"), ("tt", "TTA"))
} if (THIRD_PARTY / "TOTNet" / "weights").exists() else {}


def _bins(n_in: int, n_out: int) -> list[tuple[int, int]]:
    """PyTorch adaptive pooling bins: [floor(i*n_in/n_out), ceil((i+1)*n_in/n_out))."""
    return [((i * n_in) // n_out, -((-(i + 1) * n_in) // n_out)) for i in range(n_out)]


class AdaptiveMaxPool3dExact(nn.Module):
    """adaptive_max_pool3d for the case TOTNet uses: any temporal ratio, exact 2x (or 1x) spatial."""

    def __init__(self, output_size: tuple[int, int, int]):
        super().__init__()
        self.output_size = tuple(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n, h, w = x.shape
        to, ho, wo = self.output_size
        assert h % ho == 0 and w % wo == 0 and h // ho == w // wo, (x.shape, self.output_size)
        k = h // ho
        if k > 1:
            x = F.max_pool2d(x.reshape(b, c * n, h, w), k).reshape(b, c, n, ho, wo)
        if to == n:
            return x
        return torch.stack([x[:, :, s:e].amax(dim=2) for s, e in _bins(n, to)], dim=2)


class TrilinearExact(nn.Module):
    """nn.Upsample(size, mode='trilinear', align_corners=False) as bilinear 2D plus a fixed temporal blend."""

    def __init__(self, size: tuple[int, int, int]):
        super().__init__()
        self.size = tuple(size)

    @staticmethod
    def blend(n_in: int, n_out: int) -> torch.Tensor:
        m = torch.zeros(n_out, n_in)
        scale = n_in / n_out
        for o in range(n_out):
            src = max((o + 0.5) * scale - 0.5, 0.0)
            i0 = min(int(np.floor(src)), n_in - 1)
            i1 = min(i0 + 1, n_in - 1)
            lam = src - i0
            m[o, i0] += 1 - lam
            m[o, i1] += lam
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n, h, w = x.shape
        to, ho, wo = self.size
        if (ho, wo) != (h, w):
            x = F.interpolate(x.reshape(b, c * n, h, w), size=(ho, wo), mode="bilinear", align_corners=False).reshape(b, c, n, ho, wo)
        if to == n:
            return x
        m = self.blend(n, to).to(x.device, x.dtype)
        return torch.einsum("oi,bcihw->bcohw", m, x)


def _same_pad(k: int) -> tuple[int, int]:
    """PyTorch padding='same' for stride 1: the extra pixel of an even kernel goes after."""
    return (k - 1) // 2, k - 1 - (k - 1) // 2


class Conv3dAs2d(nn.Module):
    """A stride-1 Conv3d computed as one Conv2d per temporal tap, shifted and summed over frames.

    MPS runs Conv3d an order of magnitude slower than Conv2d and the Neural Engine has
    no 3D convolution, so this exact rewrite is what makes TOTNet usable on the M3.
    """

    def __init__(self, conv: nn.Conv3d):
        super().__init__()
        assert conv.stride == (1, 1, 1) and conv.dilation == (1, 1, 1) and conv.groups == 1
        self.weight = conv.weight
        self.bias = conv.bias
        kt, kh, kw = conv.kernel_size
        if conv.padding == "same":
            self.pt, ph, pw = _same_pad(kt), _same_pad(kh), _same_pad(kw)
        else:
            pt, ph_, pw_ = conv.padding
            self.pt, ph, pw = (pt, pt), (ph_, ph_), (pw_, pw_)
        self.pad2d = (pw[0], pw[1], ph[0], ph[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n, h, w = x.shape
        kt = self.weight.shape[2]
        n_out = n + self.pt[0] + self.pt[1] - kt + 1
        xf = F.pad(x.transpose(1, 2).reshape(b * n, c, h, w), self.pad2d)
        out = None
        for dt in range(kt):
            s = dt - self.pt[0]  # output frame t reads input frame t + s (zero outside [0, n))
            lo, hi = max(0, -s), min(n_out, n - s)
            if lo >= hi:
                continue
            y = F.conv2d(xf, self.weight[:, :, dt])
            y = y.reshape(b, n, *y.shape[1:])
            if out is None:
                out = y.new_zeros(b, n_out, *y.shape[2:])
            out[:, lo:hi] += y[:, lo + s : hi + s]
        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1, 1, 1)
        return out.transpose(1, 2)


def make_portable(net: nn.Module) -> nn.Module:
    """Swap the ops MPS lacks or runs slowly for the exact equivalents above, in place."""
    for name, mod in list(net.named_children()):
        if isinstance(mod, nn.AdaptiveMaxPool3d):
            setattr(net, name, AdaptiveMaxPool3dExact(mod.output_size))
        elif isinstance(mod, nn.Upsample) and mod.mode == "trilinear":
            assert not mod.align_corners and mod.size is not None
            setattr(net, name, TrilinearExact(mod.size))
        elif isinstance(mod, nn.Conv3d):
            setattr(net, name, Conv3dAs2d(mod))
        else:
            make_portable(mod)
    return net


class TOTNet(Detector):
    frames_in = 5
    merge = "pool"
    threshold = 0.1
    batch_default = {"default": 4, "native": 1}

    def __init__(self, checkpoint: str = "tennis", size="default", roi=None, threshold: float | None = None, **kw):
        self.name = f"totnet-{checkpoint}"
        if threshold is not None:
            self.threshold = threshold
        super().__init__(str(CHECKPOINTS[checkpoint]), size, roi, **kw)

    def build(self) -> nn.Module:
        mod = load_module("TOTNet/src/model/TOTNet.py", "totnet_model")
        ck = load_checkpoint(self.weights)
        cfg = ck["configs"]
        assert cfg["num_frames"] == self.frames_in and not cfg.get("bidirect"), "expects the causal 5-frame model"
        net = mod.TemporalConvNet(input_shape=self.geom.size, spatial_channels=cfg["num_channels"], num_frames=self.frames_in)
        net.load_state_dict(strip_prefix(ck["state_dict"]))
        return make_portable(net)

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        return (rgb.transpose(2, 0, 1).astype(np.float32) / 255.0 - MEAN) / STD

    def tensor(self, frames: list[np.ndarray]) -> np.ndarray:
        return np.stack(frames, axis=0)  # (N, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.geom.size
        return self.model(x).reshape(-1, 1, h, w)

    def plan(self, i0: int, i1: int, lo: int, hi: int) -> list[Clip]:
        return plan_causal(i0, i1, self.frames_in, lo)

    def decode(self, hm: np.ndarray):
        y, x = np.unravel_index(int(np.argmax(hm)), hm.shape)
        win = hm[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
        mass = float(win.sum())
        ys, xs = np.mgrid[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3][:, : win.shape[0], : win.shape[1]]
        cx, cy = float((xs * win).sum() / mass), float((ys * win).sum() / mass)
        return [(cx, cy, mass, mass)]

    def select(self, cands):
        return [(c[0][0], c[0][1], c[0][2], c[0][2] >= self.threshold) if c else (np.nan, np.nan, 0.0, False) for c in cands]

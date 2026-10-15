"""WASB (nttcom/WASB-SBDT) and BlurBall (cogsys-tuebingen/blurball): 3-frame HRNets.

Both take three ImageNet-normalised RGB frames (9 channels) and return three
logit heatmaps at input resolution. Upstream thresholds the sigmoid at 0.5,
takes heatmap-weighted blob centroids and lets an online tracker keep the
heaviest blob within 300 px of the last one. WASB tiles the video in steps of
three frames; BlurBall's README recommends one-frame steps with a 0.7
threshold, which pools three predictions per frame. BlurBall adds
squeeze-excitation to every basic block and was trained on table tennis with
the ball labelled at the blur centre, the convention our labels use. HRNet
has four branches down to 1/8 scale, so any input divisible by 8 works,
including native 576x1024.

Checkpoints: WASB's five sports (soccer, tennis, badminton, volleyball,
basketball) from its model zoo, plus BlurBall's table-tennis WASB retrain
(blur-centre labels) and BlurBall itself.
"""

from __future__ import annotations

import numpy as np
import torch

from squashvid.ball.base import Clip, Detector, plan_blocks
from squashvid.ball.clips import WEIGHTS
from squashvid.ball.vendor import AttrDict, load_checkpoint, load_module, strip_prefix

MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

CHECKPOINTS = {
    "wasb-tennis": WEIGHTS / "wasb" / "wasb_tennis.pth.tar",
    "wasb-badminton": WEIGHTS / "wasb" / "wasb_badminton.pth.tar",
    "wasb-soccer": WEIGHTS / "wasb" / "wasb_soccer.pth.tar",
    "wasb-volleyball": WEIGHTS / "wasb" / "wasb_volleyball.pth.tar",
    "wasb-basketball": WEIGHTS / "wasb" / "wasb_basketball.pth.tar",
    "wasb-tt": WEIGHTS / "blurball" / "wasb_midpoint_best",
    "blurball": WEIGHTS / "blurball" / "blurball_best",
}


def _stage(branches: int, block: str, channels: list[int], blocks: list[int]) -> dict:
    return {"NUM_MODULES": 1, "NUM_BRANCHES": branches, "BLOCK": block, "NUM_BLOCKS": blocks, "NUM_CHANNELS": channels, "FUSE_METHOD": "SUM"}


# configs/model/wasb.yaml and blurball.yaml (identical apart from the model name).
HRNET_CFG = {
    "frames_in": 3,
    "frames_out": 3,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "PRETRAINED_LAYERS": ["*"],
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": _stage(1, "BOTTLENECK", [32], [1]),
            "STAGE2": _stage(2, "BASIC", [16, 32], [2, 2]),
            "STAGE3": _stage(3, "BASIC", [16, 32, 64], [2, 2, 2]),
            "STAGE4": _stage(4, "BASIC", [16, 32, 64, 128], [2, 2, 2, 2]),
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
        },
        "INIT_WEIGHTS": True,
    },
}


class HRNetDetector(Detector):
    frames_in = 3
    merge = "pool"
    batch_default = {"default": 8, "native": 4}

    def __init__(self, checkpoint: str = "wasb-tennis", size="default", roi=None, step: int | None = None, threshold: float | None = None,
                 weights: str | None = None, name: str | None = None, **kw):
        """`weights` overrides the checkpoint file (a fine-tune of `checkpoint`); `name` then labels its outputs."""
        self.name = name or checkpoint
        self.blur = checkpoint == "blurball"
        self.step = step or (1 if self.blur else 3)
        self.threshold = threshold if threshold is not None else (0.7 if self.blur and self.step == 1 else 0.5)
        super().__init__(str(weights or CHECKPOINTS[checkpoint]), size, roi, **kw)
        if step is not None or threshold is not None:
            self.name = f"{self.name}-s{self.step}t{self.threshold:g}"

    def build(self) -> torch.nn.Module:
        cfg = AttrDict.wrap(HRNET_CFG)
        if self.blur:
            net = load_module("blurball/src/models/blurball.py", "blurball_model").BlurBall(cfg)
        else:
            net = load_module("WASB-SBDT/src/models/hrnet.py", "wasb_hrnet").HRNet(cfg)
        net.load_state_dict(strip_prefix(load_checkpoint(self.weights)["model_state_dict"]))
        return net

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        return (rgb.transpose(2, 0, 1).astype(np.float32) / 255.0 - MEAN) / STD

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[0].sigmoid()

    def plan(self, i0: int, i1: int, lo: int, hi: int) -> list[Clip]:
        return plan_blocks(i0, i1, self.frames_in, self.step, lo, hi)

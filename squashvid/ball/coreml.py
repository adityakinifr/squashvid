"""Run a detector's network as a Core ML program instead of through PyTorch on MPS.

On this M3 Pro, BlurBall at 576x1024 took 74 ms per call as an fp16 Core ML
program on the GPU against ~550 ms per call through torch MPS inside the
pipeline, with other jobs sharing the GPU in both cases (bench.py records the
load). The Neural Engine ('ALL') is quick at 288x512 but collapses at 576x1024
(1.3 s per call), so the GPU is the default. The program is traced at a fixed
input shape with batch 1, so a batch is run one sample at a time, and it is
cached under work/coreml/ per model and input size. TrackNetV3's per-run
median background is stacked onto the input in numpy, so it needs nothing
special here.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from squashvid.ball.clips import WORK

CACHE = WORK / "coreml"


def coremltools():
    """Import coremltools without leaving PyAV's Python log callback installed.

    coremltools imports torchvision, whose io module calls av.logging.set_level(ERROR).
    That routes FFmpeg's log lines through Python, and closing a frame-threaded HEVC
    decoder then deadlocks: a decoder thread logging at teardown waits for the GIL
    while the main thread holds it waiting for that thread (seen on the iPhone
    originals). Restoring the previous level (None: no callback) avoids it.
    """
    import av.logging

    level = av.logging.get_level()
    import coremltools as ct

    av.logging.set_level(level)
    return ct


def package_path(det) -> Path:
    """Keyed by checkpoint file and input size: step/threshold variants of one network share a program."""
    h, w = det.geom.size
    weights = Path(det.weights)
    stamp = f"_{weights.stat().st_mtime_ns}" if "ball_ft" in weights.parts else ""  # a retrained run must not reuse the old program
    return CACHE / f"{weights.name.split('.')[0]}{stamp}_{h}x{w}.mlpackage"


def example_input(det) -> torch.Tensor:
    frame = np.zeros((3, *det.geom.size), np.float32)
    if getattr(det, "median", False) is None:  # TrackNetV3 stacks a per-run background frame in front of the clip
        return torch.from_numpy(np.concatenate([frame] * (det.frames_in + 1))[None])
    return torch.from_numpy(det.tensor([frame] * det.frames_in)[None])


class _Traced(torch.nn.Module):
    """det.forward as a module, so tracing captures the weights of the CPU copy it runs on."""

    def __init__(self, det, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self._det = det

    def forward(self, x):
        return self._det.forward(x)


def export(det, path: Path | None = None) -> Path:
    ct = coremltools()

    path = path or package_path(det)
    if path.exists():
        return path
    x = example_input(det)
    cpu = copy.deepcopy(det.model).to("cpu", torch.float32).eval()
    saved, det.model = det.model, cpu  # det.forward reads self.model
    try:
        with torch.inference_mode(False), torch.no_grad():
            traced = torch.jit.trace(_Traced(det, cpu).eval(), x)
    finally:
        det.model = saved
    ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=tuple(x.shape))], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14)
    path.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(path))
    return path


def attach(det, units: str = "CPU_AND_GPU"):
    """Swap det's network call for the Core ML program; everything else (decode, warp, heatmap decoding, tracking) is unchanged."""
    ct = coremltools()

    model = ct.models.MLModel(str(export(det)), compute_units=getattr(ct.ComputeUnit, units))
    out = model.get_spec().description.output[0].name

    def forward(x: torch.Tensor) -> torch.Tensor:
        xs = x.detach().to("cpu", torch.float32).numpy()
        return torch.from_numpy(np.concatenate([model.predict({"x": xs[i : i + 1]})[out] for i in range(len(xs))]).astype(np.float32))

    det.forward = forward
    det.device, det.dtype = torch.device("cpu"), torch.float32
    det.model = None  # the torch weights are no longer used; free the GPU copy
    det.backend = f"coreml-{units}"
    return det

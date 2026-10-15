"""The Core ML backend reproduces the torch network and runs a batch one sample at a time."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from squashvid.ball import coreml as cm

# Skip without importing it here: the import itself is under test (it must go through cm.coremltools()).
pytestmark = pytest.mark.skipif(importlib.util.find_spec("coremltools") is None, reason="coremltools not installed")


class TinyDet:
    """Just the Detector surface coreml.py touches: weights, geom.size, frames_in, tensor, forward, model."""

    name = "tiny"
    weights = "/somewhere/tiny_best.pth.tar"
    frames_in = 2

    def __init__(self):
        torch.manual_seed(0)
        self.model = torch.nn.Sequential(torch.nn.Conv2d(6, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(4, 2, 1)).eval()
        self.geom = SimpleNamespace(size=(32, 64))
        self.device, self.dtype = torch.device("cpu"), torch.float32

    def tensor(self, frames):
        return np.concatenate(frames, axis=0)

    def forward(self, x):
        return self.model(x).sigmoid()


def test_coreml_matches_torch_on_a_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "CACHE", tmp_path)
    det = TinyDet()
    x = torch.rand(3, 6, 32, 64)  # batch of 3 although the program is traced at batch 1
    with torch.no_grad():
        ref = det.forward(x).numpy()
    cm.attach(det, "CPU_ONLY")
    out = det.forward(x).numpy()
    assert out.shape == ref.shape and det.model is None and det.backend == "coreml-CPU_ONLY"
    assert np.abs(out - ref).max() < 5e-3  # fp16 program
    assert (tmp_path / "tiny_best_32x64.mlpackage").exists()


def test_importing_coremltools_leaves_pyav_logging_off():
    import av.logging

    av.logging.set_level(None)
    cm.coremltools()
    assert av.logging.get_level() is None  # torchvision would set ERROR and install a GIL-taking log callback

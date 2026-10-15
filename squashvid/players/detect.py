"""Person detection with YOLO26m, kept only when the person stands on this court.

Detection runs at the native 1024 px width so small players behind the door
frame are still found (YOLO26n/s at 1024 lose one of them). The court filter
uses the box bottom-centre on the per-frame floor plane, which rejects people
beyond the walls, spectators behind the glass and most glass reflections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from squashvid.geom.camera import Camera
from squashvid.geom.court import CourtSpec
from squashvid.players.geometry import floor_xy, inside_court, stature_px

THIRD_PARTY = Path(os.path.expanduser("~/code/squashvid/third_party/players"))
WEIGHTS = ("yolo26m.pt", "yolo11m.pt")
COURT_MARGIN = 0.35
# Box height relative to an upright 1.45 m junior at the foot point: lunges and
# truncation shrink a box, but a person at the glass or a merged pair is far larger.
SIZE_RANGE = (0.3, 1.7)


def load_detector(device: str = "mps"):
    """YOLO26m if it downloads, else YOLO11m, cached under third_party/players."""
    from ultralytics import YOLO

    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for name in WEIGHTS:
        try:
            path = THIRD_PARTY / name
            model = YOLO(str(path) if path.exists() else name)
            if not path.exists() and Path(name).exists():
                Path(name).rename(path)
            model.to(device)
            return model
        except Exception as e:  # download or load failure: fall back to the next model
            last = e
    raise RuntimeError(f"no detector weights could be loaded: {last}")


@dataclass
class Detections:
    """Person boxes of one frame, as arrays."""

    xyxy: np.ndarray  # (N, 4)
    conf: np.ndarray  # (N,)

    def __len__(self) -> int:
        return len(self.conf)


def detect_batch(model, images: list[np.ndarray], conf: float = 0.15, device: str = "mps", half: bool = True, imgsz: int = 1024) -> list[Detections]:
    """Person boxes for a batch of BGR frames; batching roughly halves MPS time per frame."""
    results = model.predict(images, imgsz=imgsz, classes=[0], conf=conf, device=device, half=half, verbose=False)
    out = []
    for r in results:
        b = r.boxes.cpu().numpy()
        out.append(Detections(b.xyxy.astype(np.float32), b.conf.astype(np.float32)))
    return out


def foot_uv(xyxy: np.ndarray) -> np.ndarray:
    """Box bottom-centre, the fallback foot point."""
    xyxy = np.asarray(xyxy, np.float64).reshape(-1, 4)
    return np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]])


def court_filter(cam: Camera, xyxy: np.ndarray, spec: CourtSpec, margin: float = COURT_MARGIN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(on_court, size_ok, foot court xy) for each box.

    on_court: the box foot point lands on the floor polygon. size_ok: the box
    height is plausible for a junior standing there.
    """
    xyxy = np.asarray(xyxy, np.float64).reshape(-1, 4)
    xy = floor_xy(cam, foot_uv(xyxy), spec)
    on = inside_court(xy, spec, margin)
    size_ok = np.zeros(len(xyxy), bool)
    if on.any():
        expect = stature_px(cam, np.nan_to_num(xy[on]), 1.45)
        ratio = (xyxy[on, 3] - xyxy[on, 1]) / np.maximum(expect, 1.0)
        size_ok[on] = (ratio > SIZE_RANGE[0]) & (ratio < SIZE_RANGE[1])
    return on, size_ok, xy

"""Venue calibration: one court spec and one set of lens intrinsics shared by every clip filmed there.

A phone recording a match keeps its zoom and lens, and the court never moves,
so the focal length, distortion and court dimensions are fitted once from a
spread of frames and then held fixed while each frame's pose is tracked.

Court width is searched explicitly before the joint fit: matching only moves
model points along their normals, so a fit started from the wrong width
compensates with the pose and stays there (court 5 is 6.14 m wide, not the
WSF 6.40 m, and a fit started at 6.40 keeps 6.40).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from squashvid.calib.coarse import coarse_align
from squashvid.calib.lines import Maps, model_samples, pixel_scale, refine_joint, response_maps
from squashvid.geom import court
from squashvid.geom.camera import Camera
from squashvid.media.video import Video

FREE_SPEC = ("width", "length", "box_width", "box_depth")


@dataclass
class Venue:
    name: str
    spec: court.CourtSpec
    f: float
    cx: float
    cy: float
    dist: np.ndarray
    rms: float = float("nan")
    n_frames: int = 0
    notes: dict = field(default_factory=dict)
    image_size: tuple[int, int] | None = None  # (width, height) the intrinsics belong to

    @property
    def size(self) -> tuple[int, int]:
        """Frame size, assumed centred on the principal point when the file predates image_size."""
        return self.image_size or (int(round(2 * self.cx)), int(round(2 * self.cy)))

    @property
    def pixel_scale(self) -> float:
        return pixel_scale(self.size[0])

    def camera(self, rvec, tvec) -> Camera:
        return Camera(self.f, self.cx, self.cy, np.asarray(rvec, np.float64), np.asarray(tvec, np.float64), self.dist.copy())

    def resized(self, width: int, height: int) -> "Venue":
        """The same venue seen in a uniformly rescaled copy of the footage (poses are unchanged)."""
        w, h = self.size
        s = width / w
        if abs(height / h - s) > 1e-3:
            raise ValueError(f"{w}x{h} -> {width}x{height} is not a uniform rescale")
        # Pixel centres map as (x + 0.5) * s - 0.5, the convention resize uses.
        return Venue(self.name, self.spec, self.f * s, (self.cx + 0.5) * s - 0.5, (self.cy + 0.5) * s - 0.5, self.dist.copy(), self.rms * s, self.n_frames, dict(self.notes), (width, height))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "spec": self.spec.to_dict(),
            "f": self.f,
            "cx": self.cx,
            "cy": self.cy,
            "dist": [float(x) for x in self.dist],
            "rms": self.rms,
            "n_frames": self.n_frames,
            "notes": self.notes,
            "image_size": list(self.size),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Venue":
        size = tuple(d["image_size"]) if d.get("image_size") else None
        return cls(d["name"], court.CourtSpec(**d["spec"]), d["f"], d["cx"], d["cy"], np.asarray(d["dist"], np.float64), d.get("rms", float("nan")), d.get("n_frames", 0), d.get("notes", {}), size)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Venue":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class VenueFrame:
    source: str
    t: float
    maps: Maps
    camera: Camera


def collect_frames(sources: dict[str, tuple[Video, list[float]]], seed: Camera, spec: court.CourtSpec) -> list[VenueFrame]:
    """Decode the requested frames and bring each camera into the basin of its true pose."""
    samples = model_samples(spec=spec)
    out = []
    for name, (video, times) in sources.items():
        for t in times:
            _, pts, img = video.frame_at(t)
            maps = response_maps(img)
            start = coarse_align(seed, maps, samples, spec).camera
            fit = refine_joint([start], [maps], samples, radii=(10.0, 6.0, 4.0, 3.0), spec=spec)
            out.append(VenueFrame(name, pts, maps, fit.cameras[0]))
    return out


def scan_width(frames: list[VenueFrame], spec: court.CourtSpec, widths: np.ndarray, max_frames: int = 6, lines: set[str] | None = None) -> tuple[float, dict[float, int]]:
    """Pick the court width with the most line inliers after a pose+focal fit at each candidate.

    lines restricts the fit to those court lines (for held-out evaluation).
    """
    pick = frames if len(frames) <= max_frames else [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
    cams = [f.camera for f in pick]
    maps = [f.maps for f in pick]
    support = {}
    for w in widths:
        sp = spec.with_(width=float(w))
        samples = model_samples(spec=sp)
        if lines is not None:
            samples = samples.subset(lines)
        r = refine_joint(cams, maps, samples, radii=(8.0, 5.0, 3.0, 2.0), free="pose+f", spec=sp)
        support[float(w)] = sum(fr.n_inliers for fr in r.frames)
    best = max(support, key=support.get)
    return best, support


def calibrate_venue(
    name: str,
    sources: dict[str, tuple[Video, list[float]]],
    seed: Camera,
    base: court.CourtSpec = court.WSF,
    widths: np.ndarray | None = None,
    free_spec: tuple[str, ...] = FREE_SPEC,
) -> tuple[Venue, list[VenueFrame]]:
    widths = np.round(np.arange(base.width - 0.46, base.width + 0.22, 0.04), 3) if widths is None else widths
    frames = collect_frames(sources, seed, base)
    sizes = {f.maps.shape for f in frames}
    if len(sizes) != 1:
        raise ValueError(f"venue frames differ in size: {sorted(sizes)}; calibrate each resolution separately")
    (height, width_px), = sizes
    width, support = scan_width(frames, base, widths)
    spec = base.with_(width=width)
    samples = model_samples(spec=spec)
    # Re-seat every frame at the scanned width before the joint fit.
    for fr in frames:
        fr.camera = refine_joint([fr.camera], [fr.maps], samples, radii=(6.0, 4.0, 3.0), free="pose", spec=spec).cameras[0]
    joint = refine_joint([f.camera for f in frames], [f.maps for f in frames], samples, radii=(4.0, 3.0, 2.0), free="pose+f+k", spec=spec, free_spec=free_spec)
    for fr, cam in zip(frames, joint.cameras):
        fr.camera = cam
    c = joint.cameras[0]
    venue = Venue(
        name,
        joint.spec,
        float(c.f),
        float(c.cx),
        float(c.cy),
        np.asarray(c.dist, np.float64).copy(),
        joint.rms,
        len(frames),
        {"width_scan": {f"{k:.2f}": v for k, v in support.items()}, "frame_rms": [round(r.rms, 3) for r in joint.frames]},
        (width_px, height),
    )
    return venue, frames

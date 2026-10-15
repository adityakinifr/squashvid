"""Held-out accuracy for the court calibration.

Fit residuals flatter a model that has a free parameter for every quirk, so
accuracy is reported only on evidence the fit did not see: whole groups of
court lines left out of the fit, a whole clip left out of the venue fit, and
hand-clicked landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from squashvid.calib.lines import Maps, ModelSamples, _project_with_normals, match_edges, match_lines, model_samples, refine_joint
from squashvid.calib.venue import FREE_SPEC, VenueFrame, scan_width
from squashvid.geom import court
from squashvid.geom.camera import Camera

LINE_GROUPS: dict[str, set[str]] = {
    "side_out": {"side_out_left", "side_out_right"},
    "front_paint": {"tin", "service_line", "front_out"},
    "floor_paint": {"short_line", "half_court", "box_back_left", "box_inner_left", "box_back_right", "box_inner_right"},
    "junctions": {"front_floor_edge", "left_floor_edge", "right_floor_edge", "front_left_corner", "front_right_corner"},
}

# Which court dimensions remain observable once a line group is withheld.
_FREE_WITHOUT = {
    "side_out": FREE_SPEC,
    "front_paint": FREE_SPEC,
    "floor_paint": ("width", "length"),
    "junctions": FREE_SPEC,  # the side-wall out lines still place the walls
}


@dataclass
class ErrorStats:
    n: int
    n_visible: int
    rms: float
    median: float
    p90: float
    bias: float

    @classmethod
    def of(cls, err: np.ndarray, n_visible: int) -> "ErrorStats":
        if not len(err):
            return cls(0, n_visible, np.nan, np.nan, np.nan, np.nan)
        a = np.abs(err)
        return cls(len(err), n_visible, float(np.sqrt(np.mean(err**2))), float(np.median(a)), float(np.percentile(a, 90)), float(np.mean(err)))

    def __str__(self) -> str:
        return f"rms {self.rms:.2f} px, median {self.median:.2f}, p90 {self.p90:.2f}, bias {self.bias:+.2f} ({self.n}/{self.n_visible} matched)"


def line_errors(cam: Camera, maps: Maps, samples: ModelSamples, spec: court.CourtSpec, radius: float = 3.0) -> tuple[np.ndarray, int]:
    """Signed normal distance (frame px) from each projected sample to the line found within radius (reference px)."""
    uv, normals, keep = _project_with_normals(cam, samples, maps.shape, spec=spec)
    radius = radius * maps.scale
    idx = np.flatnonzero(keep)
    edge = samples.is_edge[idx]
    out = []
    if (~edge).any():
        off, _, ok = match_lines(maps.paint, uv[idx[~edge]], normals[idx[~edge]], radius)
        out.append(off[ok])
    if edge.any():
        off, _, ok = match_edges(maps, uv[idx[edge]], normals[idx[edge]], radius)
        out.append(off[ok])
    return (np.concatenate(out) if out else np.zeros(0)), len(idx)


def _fit(frames: list[VenueFrame], base: court.CourtSpec, lines: set[str] | None, free_spec: tuple[str, ...], widths: np.ndarray | None):
    widths = np.round(np.arange(base.width - 0.46, base.width + 0.22, 0.04), 3) if widths is None else widths
    if "width" in free_spec:
        width, _ = scan_width(frames, base, widths, lines=lines)
        spec = base.with_(width=width)
    else:
        spec = base
    samples = model_samples(spec=spec)
    if lines is not None:
        samples = samples.subset(lines)
    cams = [refine_joint([f.camera], [f.maps], samples, radii=(6.0, 4.0, 3.0), spec=spec).cameras[0] for f in frames]
    return refine_joint(cams, [f.maps for f in frames], samples, radii=(4.0, 3.0, 2.0), free="pose+f+k", spec=spec, free_spec=free_spec)


def holdout_lines(frames: list[VenueFrame], base: court.CourtSpec, groups: dict[str, set[str]] = LINE_GROUPS, widths: np.ndarray | None = None) -> dict[str, tuple[court.CourtSpec, ErrorStats]]:
    """Refit without each line group in turn and measure that group against the refit cameras."""
    everything = set(model_samples(spec=base).names)
    out = {}
    for name, group in groups.items():
        free = _FREE_WITHOUT.get(name, FREE_SPEC)
        fit = _fit(frames, base, everything - group, free, widths)
        test = model_samples(spec=fit.spec).subset(group)
        errs, vis = zip(*(line_errors(c, f.maps, test, fit.spec) for c, f in zip(fit.cameras, frames)))
        out[name] = (fit.spec, ErrorStats.of(np.concatenate(errs), int(sum(vis))))
    return out


def holdout_source(frames: list[VenueFrame], base: court.CourtSpec, source: str, widths: np.ndarray | None = None) -> tuple[court.CourtSpec, Camera, ErrorStats]:
    """Calibrate the venue without one clip, then track that clip's frames with pose only and measure all lines."""
    train = [f for f in frames if f.source != source]
    test = [f for f in frames if f.source == source]
    fit = _fit(train, base, None, FREE_SPEC, widths)
    ref = fit.cameras[0]
    samples = model_samples(spec=fit.spec)
    errs, vis = [], 0
    for f in test:
        start = f.camera.copy(f=ref.f, dist=ref.dist.copy())
        cam = refine_joint([start], [f.maps], samples, radii=(4.0, 3.0, 2.0), spec=fit.spec).cameras[0]
        e, v = line_errors(cam, f.maps, samples, fit.spec)
        errs.append(e)
        vis += v
    return fit.spec, ref, ErrorStats.of(np.concatenate(errs), vis)


def click_errors(cam: Camera, spec: court.CourtSpec, clicks: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Pixel distance between each hand-clicked landmark and its projection."""
    lm = court.landmarks(spec)
    names = [k for k in clicks if k in lm]
    world = np.array([lm[k] for k in names])
    uv = cam.project(world)
    return {k: float(np.linalg.norm(uv[i] - np.asarray(clicks[k]))) for i, k in enumerate(names)}

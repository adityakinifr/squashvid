"""Court-line refinement: fit cameras (and the court itself) so projected markings sit on the image's lines.

Two kinds of model line are matched differently. Painted markings are thin dark
stripes, located at the centre of a morphological black-hat's response: a line
several pixels wide gives a flat-topped response, so its highest sample can sit
anywhere across it and the centre of the response above half its peak is used
instead. Surface junctions (the front corners, the wall-floor joins) are
brightness steps, found as peaks of the image gradient taken along the
projected line's normal.

Matching only ever moves a sample along its line's normal, so a painted line
says nothing about where it ends. The court's width and length therefore come
from the junction edges; the painted lines pin heights and floor positions.

Pixel settings (kernel sizes, search radii, thresholds) are stated for a
1024-pixel-wide frame and scaled to the frame being matched, so the same
settings hold for a WhatsApp copy and for the 1080p or 4K original. Residuals
and reported errors are in the frame's own pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from squashvid.geom import court
from squashvid.geom.camera import Camera

REFERENCE_WIDTH = 1024  # frame width the pixel settings below are stated for
BLACKHAT_KERNEL = 9  # reference px
PAINT_MIN_PEAK = 12.0  # black-hat grey levels
EDGE_MIN_PEAK = 0.8  # grey levels per reference px; the front corners are a 5-7 level shading step
EDGE_WEIGHT = 0.7  # junction edges are softer than paint; weight their residuals down a little


@dataclass
class Maps:
    """Per-frame response images used for matching."""

    paint: np.ndarray  # black-hat on luminance
    gx: np.ndarray  # luminance gradient, grey levels per reference px
    gy: np.ndarray
    scale: float = 1.0  # frame px per reference px

    @property
    def shape(self) -> tuple[int, int]:
        return self.paint.shape


def pixel_scale(width: int) -> float:
    """Frame px per reference px for a frame of this width."""
    return width / REFERENCE_WIDTH


def line_response(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Thin-dark-structure strength: morphological black-hat on luminance.

    Court markings are narrower than the kernel near and far, so they respond;
    hinges, the door handle and players' bodies are wider and mostly do not.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    k = int(round(BLACKHAT_KERNEL * scale)) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray.astype(np.float32), cv2.MORPH_BLACKHAT, ker)


def response_maps(img: np.ndarray) -> Maps:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = pixel_scale(gray.shape[1])
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), s)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) * (s / 8.0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) * (s / 8.0)
    return Maps(line_response(gray, s), gx, gy, s)


@dataclass
class ModelSamples:
    """Samples along court lines, kept as (line, fraction) so they follow changes to the court spec."""

    names: list[str]
    kinds: list[str]
    line_ids: np.ndarray  # (N,)
    t: np.ndarray  # (N,) fraction along the line from a to b
    spec: court.CourtSpec = field(default_factory=lambda: court.WSF)

    def points(self, spec: court.CourtSpec | None = None) -> np.ndarray:
        lines = {ln.name: ln for ln in court.court_lines(spec or self.spec)}
        a = np.array([lines[n].a for n in self.names], dtype=np.float64)
        b = np.array([lines[n].b for n in self.names], dtype=np.float64)
        return a[self.line_ids] + (b - a)[self.line_ids] * self.t[:, None]

    @property
    def is_edge(self) -> np.ndarray:
        return np.array([k == "edge" for k in self.kinds])[self.line_ids]

    def subset(self, names: set[str]) -> "ModelSamples":
        keep_lines = [i for i, n in enumerate(self.names) if n in names]
        remap = -np.ones(len(self.names), dtype=int)
        remap[keep_lines] = np.arange(len(keep_lines))
        sel = np.isin(self.line_ids, keep_lines)
        return ModelSamples(
            [self.names[i] for i in keep_lines],
            [self.kinds[i] for i in keep_lines],
            remap[self.line_ids[sel]],
            self.t[sel],
            self.spec,
        )


def model_samples(lines: list[court.CourtLine] | None = None, step: float = 0.05, spec: court.CourtSpec = court.WSF) -> ModelSamples:
    lines = court.court_lines(spec) if lines is None else lines
    ids, ts = [], []
    for k, line in enumerate(lines):
        n = max(2, int(np.ceil(np.linalg.norm(np.subtract(line.b, line.a)) / step)) + 1)
        ts.append(np.linspace(0.0, 1.0, n))
        ids.append(np.full(n, k))
    return ModelSamples([ln.name for ln in lines], [ln.kind for ln in lines], np.concatenate(ids), np.concatenate(ts), spec)


def _project_with_normals(cam: Camera, samples: ModelSamples, shape: tuple[int, int], margin: int = 6, mask: np.ndarray | None = None, spec: court.CourtSpec | None = None):
    """Project samples; keep those in view and return per-point image normals of their line."""
    h, w = shape
    pts = samples.points(spec)
    uv = cam.project(pts)
    keep = cam.in_field(pts, w, h)
    # Normal of each projected line at each sample, from neighbouring samples on the same line.
    normals = np.zeros_like(uv)
    for k in np.unique(samples.line_ids):
        sel = np.flatnonzero(samples.line_ids == k)
        if len(sel) < 2:
            keep[sel] = False
            continue
        tang = np.gradient(uv[sel], axis=0)
        n = np.column_stack([-tang[:, 1], tang[:, 0]])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        bad = norm[:, 0] < 1e-6
        normals[sel] = n / np.where(norm < 1e-6, 1.0, norm)
        keep[sel[bad]] = False
    keep &= (uv[:, 0] > margin) & (uv[:, 0] < w - margin) & (uv[:, 1] > margin) & (uv[:, 1] < h - margin)
    if mask is not None:
        ui = np.clip(uv[:, 0].round().astype(int), 0, w - 1)
        vi = np.clip(uv[:, 1].round().astype(int), 0, h - 1)
        keep &= mask[vi, ui] == 0
    return uv, normals, keep


def _peaks(vals: np.ndarray, offs: np.ndarray, min_peak: float):
    k = np.argmax(vals, axis=1)
    rows = np.arange(len(k))
    peak = vals[rows, k]
    interior = (k > 0) & (k < len(offs) - 1)
    kk = np.clip(k, 1, len(offs) - 2)
    a, b, c = vals[rows, kk - 1], vals[rows, kk], vals[rows, kk + 1]
    denom = a - 2 * b + c
    safe = np.where(np.abs(denom) > 1e-6, denom, 1.0)
    sub = np.where(np.abs(denom) > 1e-6, 0.5 * (a - c) / safe, 0.0)
    off = offs[kk] + np.clip(sub, -0.5, 0.5)
    return off, peak, interior & (peak >= min_peak)


def _sample_along(img: np.ndarray, uv: np.ndarray, normals: np.ndarray, offs: np.ndarray) -> np.ndarray:
    xs = (uv[:, None, 0] + offs[None, :] * normals[:, None, 0]).astype(np.float32)
    ys = (uv[:, None, 1] + offs[None, :] * normals[:, None, 1]).astype(np.float32)
    return cv2.remap(img, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _half_max_centres(vals: np.ndarray, offs: np.ndarray, inner: np.ndarray, min_peak: float):
    """Peak within the inner window, then the centroid of the contiguous run above half that peak.

    A run cut off by the sampled window has no known centre and is rejected.
    """
    n = vals.shape[1]
    k = np.argmax(np.where(inner[None, :], vals, -np.inf), axis=1)
    peak = vals[np.arange(len(k)), k]
    half = 0.5 * peak[:, None]
    idx = np.arange(n)[None, :]
    below = vals < half
    left = np.where(below & (idx < k[:, None]), idx, -1).max(axis=1)
    right = np.where(below & (idx > k[:, None]), idx, n).min(axis=1)
    w = np.where((idx > left[:, None]) & (idx < right[:, None]), vals - half, 0.0)
    off = (w * offs[None, :]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)
    ii = np.flatnonzero(inner)
    ok = (k > ii[0]) & (k < ii[-1]) & (left >= 0) & (right < n) & (peak >= min_peak)
    return off.astype(np.float32), peak, ok


def match_lines(resp: np.ndarray, uv: np.ndarray, normals: np.ndarray, radius: float, min_peak: float = PAINT_MIN_PEAK):
    """Find the paint line within ±radius px along each normal; return its centre offset (px), peak, validity.

    The profile is sampled to ±2·radius so a wide line's whole response is seen.
    """
    offs = np.arange(-2 * radius, 2 * radius + 1e-6, 1.0, dtype=np.float32)
    inner = np.abs(offs) <= radius + 1e-6
    return _half_max_centres(_sample_along(resp, uv, normals, offs), offs, inner, min_peak)


def match_edges(maps: Maps, uv: np.ndarray, normals: np.ndarray, radius: float, min_peak: float = EDGE_MIN_PEAK):
    """Search ±radius px along each normal for the strongest brightness step across it."""
    offs = np.arange(-radius, radius + 1e-6, 1.0, dtype=np.float32)
    gx = _sample_along(maps.gx, uv, normals, offs)
    gy = _sample_along(maps.gy, uv, normals, offs)
    vals = np.abs(gx * normals[:, None, 0] + gy * normals[:, None, 1])
    return _peaks(vals, offs, min_peak)


@dataclass
class Matches:
    idx: np.ndarray  # sample indices
    targets: np.ndarray  # (M, 2) matched image points
    normals: np.ndarray  # (M, 2)
    weights: np.ndarray  # (M,)
    n_visible: int


def match_frame(cam: Camera, maps: Maps, samples: ModelSamples, radius: float, mask: np.ndarray | None = None, spec: court.CourtSpec | None = None) -> Matches:
    """Match every visible sample within radius frame px of its projection."""
    uv, normals, keep = _project_with_normals(cam, samples, maps.shape, mask=mask, spec=spec)
    idx = np.flatnonzero(keep)
    edge = samples.is_edge[idx]
    off = np.zeros(len(idx), np.float32)
    peak = np.zeros(len(idx), np.float32)
    ok = np.zeros(len(idx), bool)
    if (~edge).any():
        off[~edge], peak[~edge], ok[~edge] = match_lines(maps.paint, uv[idx[~edge]], normals[idx[~edge]], radius)
    if edge.any():
        off[edge], peak[edge], ok[edge] = match_edges(maps, uv[idx[edge]], normals[idx[edge]], radius)
    w = np.where(edge, EDGE_WEIGHT * np.sqrt(np.clip(peak / 3.0, 0.25, 1.5)), np.sqrt(np.clip(peak / 40.0, 0.25, 1.5)))
    m = idx[ok]
    return Matches(m, uv[m] + off[ok, None] * normals[m], normals[m], w[ok], len(idx))


@dataclass
class FitResult:
    camera: Camera
    rms: float  # residual RMS over inliers, px
    n_visible: int
    n_matched: int
    n_inliers: int

    @property
    def match_frac(self) -> float:
        return self.n_matched / max(1, self.n_visible)


INTRINSICS = ("f", "k1", "k2")


def _intr_get(cam: Camera, name: str) -> float:
    d = np.asarray(cam.dist, np.float64).reshape(-1)
    return {"f": cam.f, "k1": d[0], "k2": d[1]}[name]


def _intr_set(cam: Camera, values: dict[str, float]) -> Camera:
    c = cam.copy()
    d = c.dist.copy()
    if "f" in values:
        c.f = float(values["f"])
    if "k1" in values:
        d[0] = values["k1"]
    if "k2" in values:
        d[1] = values["k2"]
    c.dist = d
    return c


def _free_intrinsics(free: str) -> tuple[str, ...]:
    out = []
    if "f" in free.split("+"):
        out.append("f")
    if "k" in free.split("+"):
        out += ["k1", "k2"]
    return tuple(out)


@dataclass
class JointResult:
    cameras: list[Camera]
    spec: court.CourtSpec
    frames: list[FitResult]

    @property
    def rms(self) -> float:
        num = sum(r.rms**2 * r.n_inliers for r in self.frames if np.isfinite(r.rms))
        den = sum(r.n_inliers for r in self.frames if np.isfinite(r.rms))
        return float(np.sqrt(num / den)) if den else np.inf


def refine_joint(
    cams: list[Camera],
    maps: list[Maps],
    samples: ModelSamples,
    radii=(8.0, 5.0, 3.0),
    free: str = "pose",
    spec: court.CourtSpec | None = None,
    free_spec: tuple[str, ...] = (),
    masks: list[np.ndarray | None] | None = None,
    prior: Camera | None = None,
    prior_weight: float = 0.0,
) -> JointResult:
    """Fit per-frame poses plus shared intrinsics and court dimensions to several frames at once.

    free selects shared intrinsics: "pose", "pose+f" or "pose+f+k". free_spec
    names CourtSpec fields to estimate (e.g. "width", "length", "box_width");
    line heights stay fixed and set the scale. prior pulls a single frame's pose
    toward another camera (pixel-equivalent units scaled by prior_weight).

    radii are reference px. The fit weighs every frame in reference px, so
    frames of different sizes count equally; FitResult.rms is in frame px.
    """
    spec = spec or samples.spec
    masks = masks or [None] * len(cams)
    intr_names = _free_intrinsics(free)
    cur = [c.copy() for c in cams]
    results = [FitResult(c, np.inf, 0, 0, 0) for c in cur]
    for radius in radii:
        matches = [match_frame(c, m, samples, radius * m.scale, mk, spec) for c, m, mk in zip(cur, maps, masks)]
        usable = [len(mt.idx) >= 12 for mt in matches]
        if not any(usable):
            return JointResult(cur, spec, [FitResult(c, np.inf, mt.n_visible, len(mt.idx), 0) for c, mt in zip(cur, matches)])
        frame_ids = [i for i, u in enumerate(usable) if u]
        n_i, n_s = len(intr_names), len(free_spec)
        x0 = np.concatenate(
            [
                [_intr_get(cur[0], n) for n in intr_names],
                [getattr(spec, n) for n in free_spec],
                *[np.concatenate([cur[i].rvec, cur[i].tvec]) for i in frame_ids],
            ]
        )
        prior_x = None if prior is None else np.concatenate([prior.rvec, prior.tvec])

        def unpack(x):
            intr = dict(zip(intr_names, x[:n_i]))
            sp = spec.with_(**dict(zip(free_spec, x[n_i : n_i + n_s]))) if n_s else spec
            out = {}
            for j, i in enumerate(frame_ids):
                p = x[n_i + n_s + 6 * j : n_i + n_s + 6 * j + 6]
                out[i] = _intr_set(cur[i], intr).copy(rvec=p[:3].copy(), tvec=p[3:].copy())
            return out, sp

        def resid(x):
            cs, sp = unpack(x)
            pts = samples.points(sp) if n_s else base_pts
            r = []
            for i in frame_ids:
                mt = matches[i]
                r.append(np.sum((cs[i].project(pts[mt.idx]) - mt.targets) * mt.normals, axis=1) * (mt.weights / maps[i].scale))
                if prior_x is not None and prior_weight > 0:
                    r.append(prior_weight * (np.concatenate([cs[i].rvec, cs[i].tvec]) - prior_x))
            return np.concatenate(r)

        base_pts = samples.points(spec)
        sol = least_squares(resid, x0, loss="huber", f_scale=1.0, x_scale="jac", max_nfev=80 if n_s or n_i else 60)
        cams_new, spec = unpack(sol.x)
        pts = samples.points(spec)
        results = []
        for i, c in enumerate(cur):
            mt = matches[i]
            if i not in cams_new:
                results.append(FitResult(c, np.inf, mt.n_visible, len(mt.idx), 0))
                continue
            cur[i] = cams_new[i]
            r = np.abs(np.sum((cur[i].project(pts[mt.idx]) - mt.targets) * mt.normals, axis=1))
            inl = r < max(2.0, radius / 2) * maps[i].scale
            rms = float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else np.inf
            results.append(FitResult(cur[i], rms, mt.n_visible, len(mt.idx), int(inl.sum())))
    return JointResult(cur, spec, results)


def refine(
    cam: Camera,
    resp: Maps | np.ndarray,
    samples: ModelSamples,
    radii=(8.0, 5.0, 3.0),
    free: str = "pose",
    mask: np.ndarray | None = None,
    prior: Camera | None = None,
    prior_weight: float = 0.0,
    spec: court.CourtSpec | None = None,
) -> FitResult:
    """Single-frame refinement with the court held fixed."""
    maps = resp if isinstance(resp, Maps) else Maps(resp, np.zeros_like(resp), np.zeros_like(resp), pixel_scale(resp.shape[1]))
    out = refine_joint([cam], [maps], samples, radii, free, spec, (), [mask], prior, prior_weight)
    return out.frames[0]


def per_line_offsets(cam: Camera, maps: Maps, samples: ModelSamples, radius: float = 4.0, spec: court.CourtSpec | None = None) -> dict[str, tuple[int, int, float]]:
    """Diagnostic: for each line, (matched, visible, median signed offset in frame px); radius is reference px."""
    uv, normals, keep = _project_with_normals(cam, samples, maps.shape, spec=spec)
    radius = radius * maps.scale
    out = {}
    for k, name in enumerate(samples.names):
        sel = np.flatnonzero(keep & (samples.line_ids == k))
        if not len(sel):
            continue
        if samples.kinds[k] == "edge":
            off, _, ok = match_edges(maps, uv[sel], normals[sel], radius)
        else:
            off, _, ok = match_lines(maps.paint, uv[sel], normals[sel], radius)
        out[name] = (int(ok.sum()), len(sel), float(np.median(off[ok])) if ok.any() else float("nan"))
    return out

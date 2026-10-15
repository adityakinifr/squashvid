"""Fine-tune BlurBall on hand labels and measure it with whole-window hold-out.

    python -m squashvid.ball.finetune snapshot            # freeze the current labels (work/ball_ft/labels_snapshot.parquet)
    python -m squashvid.ball.finetune cache               # decode the labelled windows once at the model's input size
    python -m squashvid.ball.finetune train RUN --test IMG_4091-r12 IMG_4089-r3     # one run, validation tail from the training windows
    python -m squashvid.ball.finetune train final --all --epochs 8                  # train on every eligible window
    python -m squashvid.ball.finetune cv                  # three folds: train, predict the held-out windows, pooled report

Use `blurball-ft:<run>@576x1024` anywhere a model spec is accepted (predict.py, bench.py, --coreml).

Inputs match inference exactly: the cache holds what `Detector.run` feeds the network (the SDR-tone-mapped
decode, `Geometry.warp` to 576x1024, RGB, ImageNet normalisation), stacked three consecutive frames at a time.
Targets follow BlurBall's `gen_heatmap` (sigma 2.5, min_value 0.7, cut at 0.5) on the model's pixel grid, one
per frame of the stack; a frame labelled `out` has an empty target and a frame nobody labelled is masked from
the loss. The loss is BlurBall's weighted BCE. Windows are never split by frame: adjacent frames are
near-duplicates, so folds hold out whole windows and early stopping uses the last 15% of each training window.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from squashvid.ball.base import blobs, resolve_geometry
from squashvid.ball.clips import LABELS, WORK, ball_dir, video

FT = WORK / "ball_ft"
SNAPSHOT = FT / "labels_snapshot_v2.parquet"  # v1 (labels_snapshot.parquet) was taken while labelling was unfinished; v2 is final
CACHE = FT / "cache"
FRAME_HW = (1080, 1920)
MODEL_HW = (576, 1024)
SIGMA, MIN_VALUE = 2.5, 0.7  # configs/dataloader/default.yaml heatmap block
MIN_LABELS = 100  # a window with fewer labels is not used for training or testing
VAL_FRAC = 0.15
TOLS = (4.0, 7.5, 15.0)
LABELLER = "aditya"

# Whole-window folds from the brief. A window joins its fold's test set only if it has MIN_LABELS labels.
FOLD_TESTS = {"A": ["IMG_4091-r12", "IMG_4089-r3"], "B": ["IMG_4091-r16", "IMG_4089-r11"], "C": ["IMG_4090-r18", "IMG_4090-r9"]}


# --- labels and splits (pure) --------------------------------------------------------------------------------------------
def snapshot(root: Path = LABELS, labeller: str = LABELLER, out: Path = SNAPSHOT) -> pd.DataFrame:
    """Freeze the labeller's current labels (latest record per frame, 'cleared' dropped) with their window id."""
    from squashvid.ball import labels as L

    wins = json.loads((root / "windows.json").read_text())
    rows = []
    for stem in sorted({w["stem"] for w in wins}):
        d = L.load(stem, root, labeller)
        rows.append(d[d.source != "cleared"].assign(stem=stem))
    d = pd.concat(rows, ignore_index=True)
    d["window"] = None
    for w in wins:
        d.loc[(d.stem == w["stem"]) & (d.frame >= w["i0"]) & (d.frame <= w["i1"]), "window"] = w["id"]
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out, index=False)
    return d


def window_counts(snap: pd.DataFrame) -> dict[str, int]:
    return snap.dropna(subset=["window"]).groupby("window").size().to_dict()


def eligible(counts: dict[str, int], min_labels: int = MIN_LABELS) -> list[str]:
    return sorted(w for w, n in counts.items() if n >= min_labels)


def make_folds(ids: list[str], tests: dict[str, list[str]] = FOLD_TESTS) -> dict[str, tuple[list[str], list[str]]]:
    """{fold: (train windows, test windows)}: each fold holds out whole windows; a fold with no eligible test window is dropped."""
    out = {}
    for name, held in tests.items():
        test = [w for w in held if w in ids]
        if test:
            out[name] = ([w for w in ids if w not in test], test)
    return out


def val_split(frames: np.ndarray, frac: float = VAL_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """(train frames, validation frames): the last `frac` of a window as one contiguous block.

    The two frames just before the block are dropped from training so that no 3-frame stack straddles the cut.
    """
    f = np.sort(np.asarray(frames))
    k = max(1, int(round(len(f) * frac)))
    return f[: max(0, len(f) - k - 2)], f[len(f) - k :]


# --- geometry and targets (pure) -----------------------------------------------------------------------------------------
def model_scale(frame_hw: tuple[int, int] = FRAME_HW, model_hw: tuple[int, int] = MODEL_HW) -> tuple[float, float]:
    return frame_hw[1] / model_hw[1], frame_hw[0] / model_hw[0]


def to_model(x: float, y: float, scale: tuple[float, float] | None = None) -> tuple[float, float]:
    """Full-frame pixel (centres at integers) to model pixel: the inverse of `Geometry.to_frame`."""
    sx, sy = scale or model_scale()
    return (x + 0.5) / sx - 0.5, (y + 0.5) / sy - 0.5


def heatmap(hw: tuple[int, int], xy: tuple[float, float] | None, sigma: float = SIGMA, min_value: float = MIN_VALUE) -> np.ndarray:
    """BlurBall's `gen_heatmap` on a pixel grid with centres at integers; None gives an empty target."""
    h, w = hw
    out = np.zeros((h, w), np.float32)
    if xy is None or not np.isfinite(xy).all():
        return out
    cx, cy = xy
    r = int(np.ceil(3 * sigma)) + 1
    x0, x1 = max(0, int(np.floor(cx)) - r), min(w, int(np.ceil(cx)) + r + 1)
    y0, y1 = max(0, int(np.floor(cy)) - r), min(h, int(np.ceil(cy)) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return out
    ys, xs = np.mgrid[y0:y1, x0:x1]
    hm = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / sigma**2) / np.exp(-1.0) * min_value
    hm[hm < 0.5] = 0
    out[y0:y1, x0:x1] = np.minimum(hm, 1.0)
    return out


def hflip(stack: np.ndarray, points: list[tuple[float, float] | None]) -> tuple[np.ndarray, list[tuple[float, float] | None]]:
    """Mirror a (T, H, W, C) stack and its model-pixel points. The court is left-right symmetric."""
    w = stack.shape[2]
    return stack[:, :, ::-1], [None if p is None else (w - 1 - p[0], p[1]) for p in points]


def affine(scale: float, shift: tuple[float, float], hw: tuple[int, int]) -> np.ndarray:
    """2x3 matrix: zoom about the image centre by `scale`, then shift in pixels."""
    h, w = hw
    cx, cy = (w - 1) / 2, (h - 1) / 2
    return np.array([[scale, 0, cx - scale * cx + shift[0]], [0, scale, cy - scale * cy + shift[1]]], np.float64)


def apply_affine(m: np.ndarray, p: tuple[float, float] | None) -> tuple[float, float] | None:
    return None if p is None else (float(m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2]), float(m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2]))


def photometric(stack: np.ndarray, brightness: float, contrast: float, gamma: float) -> np.ndarray:
    """Same brightness/contrast/gamma on every frame of a uint8 stack (so the ball's motion cue is untouched)."""
    x = stack.astype(np.float32) / 255.0
    x = np.clip((x - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0) ** gamma
    return (x * 255.0 + 0.5).astype(np.uint8)


def pick_crop(rng: np.random.Generator, hw: tuple[int, int], crop: tuple[int, int], points: list[tuple[float, float] | None], p_ball: float = 0.7) -> tuple[int, int]:
    """Top-left (x, y) of a crop; with probability p_ball it contains one of the visible points (margin 16 px)."""
    h, w = hw
    ch, cw = crop
    vis = [p for p in points if p is not None]
    if vis and rng.random() < p_ball:
        px, py = vis[rng.integers(len(vis))]
        m = 16
        lo_x, hi_x = int(np.ceil(px + m - cw + 1)), int(np.floor(px - m))
        lo_y, hi_y = int(np.ceil(py + m - ch + 1)), int(np.floor(py - m))
        x0 = int(rng.integers(max(0, lo_x), min(w - cw, max(hi_x, 0)) + 1)) if max(0, lo_x) <= min(w - cw, hi_x) else int(np.clip(px - cw / 2, 0, w - cw))
        y0 = int(rng.integers(max(0, lo_y), min(h - ch, max(hi_y, 0)) + 1)) if max(0, lo_y) <= min(h - ch, hi_y) else int(np.clip(py - ch / 2, 0, h - ch))
        return x0, y0
    return int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))


# --- cache ---------------------------------------------------------------------------------------------------------------
@dataclass
class WindowCache:
    """Decoded 576x1024 RGB frames lo..hi of a window as a uint8 memmap, plus its labels in full-frame pixels."""

    wid: str
    stem: str
    lo: int
    hi: int
    labels: dict[int, tuple[str, float, float]]  # frame -> (visibility, x, y)

    @property
    def path(self) -> Path:
        return CACHE / f"{self.wid}.u8"

    def frames(self) -> np.ndarray:
        return np.memmap(self.path, dtype=np.uint8, mode="r", shape=(self.hi - self.lo + 1, *MODEL_HW, 3))

    def has(self, f: int) -> bool:
        return self.lo <= f <= self.hi


def window_caches(snap: pd.DataFrame, wins: list[dict], ids: list[str] | None = None) -> dict[str, WindowCache]:
    out = {}
    for w in wins:
        if ids is not None and w["id"] not in ids:
            continue
        d = snap[snap.window == w["id"]]
        if not len(d):
            continue
        lab = {int(r.frame): (r.visibility, float(r.x) if r.visibility == "visible" else np.nan, float(r.y) if r.visibility == "visible" else np.nan) for r in d.itertuples()}
        out[w["id"]] = WindowCache(w["id"], w["stem"], max(0, min(lab) - 2), max(lab) + 2, lab)
    return out


def build_cache(wc: WindowCache, force: bool = False) -> Path:
    """Decode frames lo..hi exactly as `Detector.run` does and store them; resumes by skipping a finished file."""
    n = wc.hi - wc.lo + 1
    expect = n * MODEL_HW[0] * MODEL_HW[1] * 3
    if wc.path.exists() and wc.path.stat().st_size == expect and not force:
        return wc.path
    v = video(wc.stem)
    geom = resolve_geometry(FRAME_HW, f"{MODEL_HW[0]}x{MODEL_HW[1]}", None)
    ts = v.timestamps
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = wc.path.with_suffix(".part")
    mm = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(n, *MODEL_HW, 3))
    got = np.zeros(n, bool)
    for idx, _, bgr in v.frames(float(ts[wc.lo]), float(ts[wc.hi])):
        if wc.lo <= idx <= wc.hi:
            mm[idx - wc.lo] = cv2.cvtColor(geom.warp(bgr), cv2.COLOR_BGR2RGB)
            got[idx - wc.lo] = True
    for i in np.flatnonzero(~got):  # a frame the decoder skipped: reuse the nearest earlier one, as Detector._frame does
        j = i - 1
        while j >= 0 and not got[j]:
            j -= 1
        j = j if j >= 0 else int(np.flatnonzero(got)[0])
        mm[i] = mm[j]
    mm.flush()
    del mm
    os.replace(tmp, wc.path)
    return wc.path


class CachedVideo:
    """A drop-in for `Video` that replays a window's cache, so the detector sees the same pixels without decoding.

    Frames are served as BGR 576x1024; `Geometry.warp` leaves an image already at the model size untouched.
    """

    def __init__(self, stem: str, caches: list[WindowCache]):
        self.stem = stem
        self._ts = video(stem).timestamps
        self._mm = {c.wid: (c, c.frames()) for c in caches}

    @property
    def timestamps(self) -> np.ndarray:
        return self._ts

    def frames(self, start: float, end: float):
        lo, hi = int(np.searchsorted(self._ts, start - 1e-6)), int(np.searchsorted(self._ts, end + 1e-6)) - 1
        for i in range(lo, hi + 1):
            for c, mm in self._mm.values():
                if c.has(i):
                    yield i, float(self._ts[i]), np.ascontiguousarray(mm[i - c.lo][:, :, ::-1])
                    break


# --- dataset -------------------------------------------------------------------------------------------------------------
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
FRAMES_IN = 3


def normalise(stack: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) uint8 RGB to the network's (3T, H, W) float32 input, as `HRNetDetector.preprocess` + `tensor` do."""
    x = stack.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).reshape(-1, *stack.shape[1:3])


def stack_items(caches: dict[str, WindowCache], frames: dict[str, np.ndarray]) -> list[tuple[str, int]]:
    """(window, start) for every 3-frame stack lying inside the cache whose frames come from `frames` and has a labelled one."""
    items = []
    for wid, fs in frames.items():
        c, allowed = caches[wid], set(int(f) for f in fs)
        for s in range(c.lo, c.hi - FRAMES_IN + 2):
            ks = [s + k for k in range(FRAMES_IN)]
            if all(f in allowed or f not in c.labels for f in ks) and any(f in c.labels and f in allowed for f in ks):
                items.append((wid, s))
    return items


class StackDataset(torch.utils.data.Dataset):
    """Three consecutive frames to three heatmap targets and a per-frame mask (0 where the frame is unlabelled)."""

    def __init__(self, caches: dict[str, WindowCache], items: list[tuple[str, int]], allowed: dict[str, np.ndarray], train: bool, crop: tuple[int, int] | None = (288, 512), seed: int = 0,
                 sigma: float = SIGMA):
        self.caches, self.items, self.train, self.crop, self.seed, self.sigma = caches, items, train, crop, seed, sigma
        self.allowed = {k: set(int(f) for f in v) for k, v in allowed.items()}
        self._mm: dict[str, np.ndarray] = {}
        self._epoch = 0

    def set_epoch(self, e: int) -> None:
        self._epoch = e

    def __len__(self) -> int:
        return len(self.items)

    def _frames(self, wid: str) -> np.ndarray:
        if wid not in self._mm:
            self._mm[wid] = self.caches[wid].frames()
        return self._mm[wid]

    def __getitem__(self, i: int):
        wid, s = self.items[i]
        c = self.caches[wid]
        rng = np.random.default_rng((self.seed, self._epoch, i))
        stack = np.stack([self._frames(wid)[s + k - c.lo] for k in range(FRAMES_IN)])
        scale = model_scale()
        pts, mask = [], []
        for k in range(FRAMES_IN):
            lab = c.labels.get(s + k)
            use = lab is not None and (s + k) in self.allowed[wid]
            mask.append(1.0 if use else 0.0)
            pts.append(to_model(lab[1], lab[2], scale) if use and lab[0] == "visible" else None)
        hw = stack.shape[1:3]
        if self.train:
            if rng.random() < 0.5:
                stack, pts = hflip(stack, pts)
            stack = np.ascontiguousarray(stack)
            if rng.random() < 0.7:
                m = affine(float(rng.uniform(0.9, 1.1)), (float(rng.uniform(-0.04, 0.04) * hw[1]), float(rng.uniform(-0.04, 0.04) * hw[0])), hw)
                stack = np.stack([cv2.warpAffine(f, m, (hw[1], hw[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for f in stack])
                pts = [apply_affine(m, p) for p in pts]
            if rng.random() < 0.8:
                stack = photometric(stack, float(rng.uniform(-0.12, 0.12)), float(rng.uniform(0.75, 1.25)), float(rng.uniform(0.75, 1.3)))
        if self.crop and tuple(self.crop) != tuple(hw):
            ch, cw = self.crop
            x0, y0 = pick_crop(rng, hw, self.crop, pts) if self.train else ((hw[1] - cw) // 2, (hw[0] - ch) // 2)
            stack = stack[:, y0 : y0 + ch, x0 : x0 + cw]
            pts = [None if p is None else (p[0] - x0, p[1] - y0) for p in pts]
            hw = (ch, cw)
        pts = [p if p is not None and 0 <= p[0] <= hw[1] - 1 and 0 <= p[1] <= hw[0] - 1 else None for p in pts]
        target = np.stack([heatmap(hw, p, self.sigma) for p in pts])
        return torch.from_numpy(normalise(np.ascontiguousarray(stack))), torch.from_numpy(target), torch.tensor(mask, dtype=torch.float32)


# --- model, loss, validation ---------------------------------------------------------------------------------------------
def wbce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BlurBall's weighted BCE (`losses/wbce.py`, scale 0), averaged over the pixels of the labelled frames only."""
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    loss = -(((1 - p) ** 2) * target * torch.log(p) + (p**2) * (1 - target) * torch.log(1 - p))
    m = mask[:, :, None, None]
    return (loss * m).sum() / (m.sum() * loss.shape[2] * loss.shape[3]).clamp_min(1.0)


def build_model(weights: str | Path | None = None, device: str = "cpu") -> torch.nn.Module:
    from squashvid.ball.wasb import HRNetDetector

    det = HRNetDetector("blurball", size=f"{MODEL_HW[0]}x{MODEL_HW[1]}", device="cpu", frame_hw=FRAME_HW, weights=str(weights) if weights else None)
    return det.model.to(device).train()


def save_checkpoint(model: torch.nn.Module, path: Path, meta: dict) -> None:
    """A BlurBall checkpoint in the format `vendor.load_checkpoint` reads (plain tensors and python scalars only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: (v if isinstance(v, (int, float, str, bool)) else json.dumps(v)) for k, v in meta.items()}
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "meta": clean}, path)


@torch.no_grad()
def validate(model: torch.nn.Module, caches: dict[str, WindowCache], val: dict[str, np.ndarray], device: str, thr: float = 0.7, tol: float = 7.5, batch: int = 2) -> dict:
    """Full-frame loss and P/R/F1 on validation frames: centre output of the (f-1, f, f+1) stack, best blob by mass."""
    model.eval()
    scale, tp = model_scale(), [0, 0, 0]  # tp, fp, fn
    loss_sum, n_loss = 0.0, 0
    jobs = [(wid, int(f)) for wid, fs in val.items() for f in fs if int(f) in caches[wid].labels and caches[wid].lo + 1 <= int(f) <= caches[wid].hi - 1]
    for b in range(0, len(jobs), batch):
        chunk = jobs[b : b + batch]
        xs, ts = [], []
        for wid, f in chunk:
            c = caches[wid]
            mm = c.frames()
            xs.append(normalise(np.stack([mm[f + k - 1 - c.lo] for k in range(3)])))
            lab = c.labels[f]
            ts.append(heatmap(MODEL_HW, to_model(lab[1], lab[2], scale) if lab[0] == "visible" else None))
        out = model(torch.from_numpy(np.stack(xs)).to(device))[0]
        t = torch.from_numpy(np.stack(ts)).to(device)
        p = out[:, 1].sigmoid().clamp(1e-4, 1 - 1e-4)
        loss_sum += float((-(((1 - p) ** 2) * t * torch.log(p) + (p**2) * (1 - t) * torch.log(1 - p))).mean(dim=(1, 2)).sum())
        n_loss += len(chunk)
        hm = p.cpu().numpy()
        for (wid, f), h in zip(chunk, hm):
            lab = caches[wid].labels[f]
            cs = blobs(h, thr)
            best = max(cs, key=lambda q: q[3]) if cs else None
            if best is not None:
                px, py = (best[0] + 0.5) * scale[0] - 0.5, (best[1] + 0.5) * scale[1] - 0.5
            if lab[0] == "visible":
                hit = best is not None and np.hypot(px - lab[1], py - lab[2]) <= tol
                tp[0] += int(hit)
                tp[1] += int(best is not None and not hit)
                tp[2] += int(not hit)
            else:
                tp[1] += int(best is not None)
    a, fp, fn = tp
    pr, rc = a / (a + fp) if a + fp else 0.0, a / (a + fn) if a + fn else 0.0
    model.train()
    if device == "mps":
        torch.mps.empty_cache()
    return {"loss": loss_sum / max(1, n_loss), "precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "n": len(jobs)}


def freeze_bn(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def train(run: str, train_ids: list[str], snap: pd.DataFrame, wins: list[dict], epochs: int = 12, lr: float = 1e-4, batch: int = 8, crop: tuple[int, int] | None = (288, 512),
          patience: int = 4, use_val: bool = True, workers: int = 3, sigma: float = SIGMA, device: str | None = None, seed: int = 0, max_minutes: float | None = None,
          steps_per_epoch: int | None = None) -> dict:
    """Fine-tune BlurBall on `train_ids`; with use_val the last 15% of each window is held back for early stopping."""
    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    if dev == "cpu":
        print("WARNING: MPS unavailable, training on CPU (slow)", flush=True)
    torch.manual_seed(seed)
    caches = window_caches(snap, wins, train_ids)
    tr_frames, va_frames = {}, {}
    for wid, c in caches.items():
        fs = np.array(sorted(c.labels))
        tr_frames[wid], va_frames[wid] = val_split(fs) if use_val else (fs, fs[:0])
    items = stack_items(caches, tr_frames)
    ds = StackDataset(caches, items, tr_frames, True, crop, seed, sigma)
    out_dir = FT / run
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    total = epochs * (steps_per_epoch or max(1, len(items) // batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total + 1, pct_start=0.1, anneal_strategy="cos", div_factor=5, final_div_factor=10)
    log = {"run": run, "device": dev, "train_windows": train_ids, "n_stacks": len(items), "crop": list(crop) if crop else None, "lr": lr, "batch": batch, "epochs": []}
    base = validate(model, caches, va_frames, dev) if use_val else None
    log["zero_shot_val"] = base
    print(f"[{run}] {len(items)} stacks, crop {crop}, zero-shot val {base}", flush=True)
    best, bad, t_start = (base["f1"] if base else -1.0, base["loss"] if base else 9.0), 0, time.time()  # early stopping follows validation loss: F1 on ~170 frames moves by single detections
    best_epoch = 0
    if use_val:
        save_checkpoint(model, out_dir / f"{run}.pt", {"epoch": 0, "val_f1": base["f1"], "note": "zero-shot"})
    step = 0
    for ep in range(1, epochs + 1):
        ds.set_epoch(ep)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers, persistent_workers=False,
                                         generator=torch.Generator().manual_seed(seed * 1000 + ep))
        model.train()
        freeze_bn(model)
        t0, run_loss, n = time.time(), 0.0, 0
        for bi, (x, t, m) in enumerate(dl):
            if steps_per_epoch and bi >= steps_per_epoch:
                break
            x, t, m = x.to(dev), t.to(dev), m.to(dev)
            loss = wbce(model(x)[0], t, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_loss += float(loss)
            n += 1
        rec = {"epoch": ep, "train_loss": run_loss / max(1, n), "epoch_s": round(time.time() - t0, 1), "lr": sched.get_last_lr()[0]}
        if use_val:
            v = validate(model, caches, va_frames, dev)
            rec["val"] = v
            if v["loss"] < best[1]:
                best, bad, best_epoch = (v["f1"], v["loss"]), 0, ep
                save_checkpoint(model, out_dir / f"{run}.pt", {"epoch": ep, "val_f1": v["f1"], "val_loss": v["loss"]})
            else:
                bad += 1
        log["epochs"].append(rec)
        print(f"[{run}] {json.dumps(rec)}", flush=True)
        (out_dir / "log.json").write_text(json.dumps(log, indent=1, default=float))
        if use_val and bad >= patience:
            print(f"[{run}] early stop: no validation gain for {patience} epochs", flush=True)
            break
        if max_minutes and (time.time() - t_start) / 60 > max_minutes:
            print(f"[{run}] time budget reached", flush=True)
            break
    if not use_val:
        save_checkpoint(model, out_dir / f"{run}.pt", {"epoch": len(log["epochs"]), "note": "last epoch, no validation"})
    log["best_epoch"] = best_epoch if use_val else len(log["epochs"])
    log["best_val"] = {"f1": best[0], "loss": best[1]} if use_val else None
    log["wall_s"] = round(time.time() - t_start, 1)
    (out_dir / "log.json").write_text(json.dumps(log, indent=1, default=float))
    return log


# --- scoring on held-out windows ----------------------------------------------------------------------------------------
def window_stem(wins: list[dict]) -> dict[str, dict]:
    return {w["id"]: w for w in wins}


def predict_windows(run: str, wids: list[str], wins: list[dict], coreml: str | None = "CPU_AND_GPU", threshold: float | None = None) -> dict[str, str]:
    """Run `blurball-ft:<run>@576x1024` over whole windows through predict.py's own path (real decode, Core ML); returns {window: variant}."""
    from squashvid.ball import predict

    by_id, out = window_stem(wins), {}
    for wid in wids:
        w = by_id[wid]
        kw = {"threshold": threshold} if threshold is not None else {}
        _, variant, _ = predict.run_one(f"{registry_prefix()}{run}@{MODEL_HW[0]}x{MODEL_HW[1]}", w["stem"], w["start"], w["end"], coreml=coreml, **kw)
        out[wid] = variant
    return out


def registry_prefix() -> str:
    from squashvid.ball.registry import FINETUNED

    return FINETUNED


def load_variant(stem: str, variant: str) -> pd.DataFrame:
    return pd.read_parquet(ball_dir(stem) / f"{variant}.parquet")


def score_windows(snap: pd.DataFrame, preds: dict[str, pd.DataFrame], wins: list[dict], tol: float, threshold: float | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Pool per-frame matches over windows: (matches, label region per frame, predicted region per frame)."""
    from squashvid.ball.evaluate import match
    from squashvid.ball.regions import label_regions

    by_id = window_stem(wins)
    ms, gr, pr = [], [], []
    for wid, pred in preds.items():
        stem = by_id[wid]["stem"]
        gt = snap[snap.window == wid].sort_values("frame")
        m = match(gt, pred, tol, threshold)
        ms.append(m.assign(window=wid))
        gr.append(label_regions(stem, gt.frame.to_numpy(), gt[["x", "y"]].to_numpy(float)))
        vis = m.pred_visible.to_numpy()
        reg = np.full(len(m), None, dtype=object)
        if vis.any():
            reg[vis] = label_regions(stem, m.frame.to_numpy()[vis], m[["pred_x", "pred_y"]].to_numpy(float)[vis])
        pr.append(reg)
    return pd.concat(ms, ignore_index=True), np.concatenate(gr), np.concatenate(pr)


def pooled_tables(snap: pd.DataFrame, preds: dict[str, pd.DataFrame], wins: list[dict], tols=TOLS, threshold: float | None = None) -> dict:
    """{'tol7.5': by-region table records, ...} plus per-window P/R/F1 at 7.5 px, over exactly the labelled frames of `preds`' windows."""
    from squashvid.ball.evaluate import by_region, prf
    from squashvid.ball.regions import REGIONS

    out: dict = {}
    for tol in tols:
        m, gr, pr = score_windows(snap, preds, wins, tol, threshold)
        out[f"tol{tol:g}"] = by_region(m, gr, pr, REGIONS).replace({np.nan: None}).to_dict(orient="records")
        if tol == 7.5:
            out["per_window"] = {w: {**prf(*(int(g[k].sum()) for k in ("tp", "fp", "fn"))), "n": int(len(g))} for w, g in m.groupby("window")}
            tp = m.dist[m.tp == 1]
            out["median_err_px_tp"] = float(tp.median()) if len(tp) else None
    return out


# --- shoe suppression ------------------------------------------------------------------------------------------------------
KNEE_ANKLE = (13, 14, 15, 16)  # COCO keypoints: knees and ankles


def foot_zones(players: pd.DataFrame, frame: int, drop: float = 0.72) -> list[tuple[float, float, float, float]]:
    """Per-player boxes (x0, y0, x1, y1, 1080p px) of the lower legs and feet: below `drop` of the player's box height.

    The box is widened by 15% of its width each side because shoes land outside the torso's span.
    """
    rows = players[(players.frame == frame) & players.detected]
    zones = []
    for r in rows.itertuples():
        h, w = r.y2 - r.y1, r.x2 - r.x1
        zones.append((r.x1 - 0.15 * w, r.y1 + drop * h, r.x2 + 0.15 * w, r.y2 + 0.05 * h))
    return zones


def in_zone(x: float, y: float, zones) -> bool:
    return any(z[0] <= x <= z[2] and z[1] <= y <= z[3] for z in zones)


def suppress_shoes(pred: pd.DataFrame, players: pd.DataFrame, drop: float = 0.72, link_px: float = 120.0, reach: int = 2) -> pd.DataFrame:
    """Drop detections inside a player's lower-leg/foot zone unless a ball-like track passes through them.

    A detection is consistent with the ball track when a visible detection within `reach` frames, itself outside every
    foot zone, lies within `link_px` per frame of elapsed time of it. A shoe that sits in the zone for many frames has
    neighbours only inside the zone, so it fails; a ball crossing the zone arrives from outside it.
    """
    pred = pred.sort_values("frame").reset_index(drop=True)
    vis = pred[pred.visible & np.isfinite(pred.x)]
    zones = {int(f): foot_zones(players, int(f), drop) for f in vis.frame}
    inside = {int(r.frame): in_zone(r.x, r.y, zones[int(r.frame)]) for r in vis.itertuples()}
    pts = {int(r.frame): (r.x, r.y) for r in vis.itertuples()}
    keep = pred.visible.copy()
    for r in vis.itertuples():
        f = int(r.frame)
        if not inside[f]:
            continue
        ok = False
        for d in range(1, reach + 1):
            for g in (f - d, f + d):
                if g in pts and not inside[g] and np.hypot(pts[g][0] - r.x, pts[g][1] - r.y) <= link_px * d:
                    ok = True
        if not ok:
            keep[pred.frame == f] = False
    out = pred.copy()
    out["visible"] = keep
    out.loc[~keep & pred.visible, ["x", "y", "score"]] = [np.nan, np.nan, 0.0]
    return out


def threshold_run(run: str, wids: list[str], snap: pd.DataFrame, wins: list[dict], thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), coreml: str = "CPU_AND_GPU") -> dict[float, dict[str, pd.DataFrame]]:
    """Re-run the detector at several blob thresholds on the cached frames (no decoding): {threshold: {window: predictions}}.

    The cache holds the pixels `Detector.run` would have fed the network, so this is the same detector at a different
    threshold; only the first and last two frames of a cached span see replicated rather than real neighbours.
    """
    from squashvid.ball import coreml as cm
    from squashvid.ball import registry

    caches = window_caches(snap, wins, wids)
    by_id = window_stem(wins)
    det = registry.build(f"{registry_prefix()}{run}@{MODEL_HW[0]}x{MODEL_HW[1]}", frame_hw=FRAME_HW, device="cpu", half=False)
    cm.attach(det, coreml)
    out: dict[float, dict[str, pd.DataFrame]] = {t: {} for t in thresholds}
    ckpt = FT / run / f"{run}.pt"
    for wid in wids:
        c = caches[wid]
        cv = CachedVideo(by_id[wid]["stem"], [c])
        ts = cv.timestamps
        for t in thresholds:
            f = FT / "sweep_cache" / f"{run}_{int(ckpt.stat().st_mtime)}_{wid}_{t:g}.parquet"  # keyed by checkpoint time so a retrain is never served stale
            if f.exists():
                out[t][wid] = pd.read_parquet(f)
                continue
            det.threshold = t
            out[t][wid] = det.run(cv, float(ts[c.lo]), float(ts[c.hi]))
            f.parent.mkdir(parents=True, exist_ok=True)
            out[t][wid].to_parquet(f, index=False)
    return out


def f1_of(table: list[dict]) -> float:
    row = next(r for r in table if r["region"] == "all")
    return float(row["f1"])


def cv_report(runs: dict[str, str], snap: pd.DataFrame, wins: list[dict], baseline: str = "blurball@576x1024-coreml", out: Path = FT / "cv_report.json") -> dict:
    """Predict each fold's held-out windows with its own model, pool the folds, and compare with the zero-shot baseline on the same frames."""
    by_id = window_stem(wins)
    folds = make_folds(eligible(window_counts(snap)))
    base, ft, variants = {}, {}, {}
    for name, (_, test) in folds.items():
        run = runs[name]
        v = predict_windows(run, test, wins)
        for wid in test:
            ft[wid] = load_variant(by_id[wid]["stem"], v[wid])
            base[wid] = load_variant(by_id[wid]["stem"], baseline)
        variants[name] = v
    rep: dict = {"folds": {n: {"train": tr, "test": te, "run": runs[n]} for n, (tr, te) in folds.items()}, "variants": variants,
                 "counts": window_counts(snap), "baseline": pooled_tables(snap, base, wins), "finetuned": pooled_tables(snap, ft, wins)}
    rep["per_fold"] = {}
    for name, (_, test) in folds.items():
        rep["per_fold"][name] = {"baseline": pooled_tables(snap, {w: base[w] for w in test}, wins, tols=(7.5,)), "finetuned": pooled_tables(snap, {w: ft[w] for w in test}, wins, tols=(7.5,))}
    out.write_text(json.dumps(rep, indent=1, default=float))
    return rep


def row_all(table: list[dict]) -> dict:
    return next(r for r in table if r["region"] == "all")


def pooled_sweep(snap: pd.DataFrame, preds: dict[str, pd.DataFrame], tol: float = 7.5) -> pd.DataFrame:
    """`evaluate.sweep` per window (post-hoc score thresholds on one prediction set), counts summed over windows."""
    from squashvid.ball.evaluate import prf, sweep

    parts = [sweep(snap[snap.window == w].sort_values("frame"), p, tol).set_index("threshold")[["tp", "fp", "fn"]] for w, p in preds.items()]
    tot = sum(parts)
    return pd.DataFrame([{"threshold": t, **prf(int(r.tp), int(r.fp), int(r.fn))} for t, r in tot.iterrows()])


def sweep_and_shoes(runs: dict[str, str], snap: pd.DataFrame, wins: list[dict], thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8), out: Path = FT / "sweep_report.json") -> dict:
    """Pooled held-out predictions at several detector thresholds, the best by F1, and the shoe filter on that best."""
    by_id = window_stem(wins)
    folds = make_folds(eligible(window_counts(snap)))
    per_thr: dict[float, dict[str, pd.DataFrame]] = {t: {} for t in thresholds}
    for name, (_, test) in folds.items():
        res = threshold_run(runs[name], test, snap, wins, thresholds)
        for t in thresholds:
            per_thr[t].update(res[t])
    rep: dict = {"thresholds": {}}
    for t in thresholds:
        tab = pooled_tables(snap, per_thr[t], wins)
        rep["thresholds"][f"{t:g}"] = {tol: row_all(tab[tol]) for tol in ("tol4", "tol7.5", "tol15")}
    best = max(thresholds, key=lambda t: rep["thresholds"][f"{t:g}"]["tol7.5"]["f1"])
    rep["best_threshold"] = best
    rep["best_tables"] = pooled_tables(snap, per_thr[best], wins)
    low = min(thresholds)
    rep["posthoc_sweep_from_lowest"] = pooled_sweep(snap, per_thr[low]).replace({np.nan: None}).to_dict(orient="records")
    rep["posthoc_sweep_threshold_of_run"] = low
    shoes = {}
    for drop, link in ((0.72, 120.0), (0.6, 120.0), (0.72, 60.0), (0.85, 120.0)):
        filt = {}
        for wid, pred in per_thr[best].items():
            pl = pd.read_parquet(WORK / by_id[wid]["stem"] / "players.parquet")
            filt[wid] = suppress_shoes(pred, pl, drop, link)
        tab = pooled_tables(snap, filt, wins)
        shoes[f"drop{drop}_link{link:g}"] = {"all": {tol: row_all(tab[tol]) for tol in ("tol4", "tol7.5", "tol15")}, "tables7.5": tab["tol7.5"],
                                             "removed": int(sum(int(per_thr[best][w].visible.sum() - f.visible.sum()) for w, f in filt.items()))}
    rep["shoe_filter"] = shoes
    out.write_text(json.dumps(rep, indent=1, default=float))
    for t, p in per_thr[best].items():
        p.to_parquet(FT / f"best_thr_preds_{t}.parquet", index=False)
    return rep


# --- command line --------------------------------------------------------------------------------------------------------
def _load() -> tuple[pd.DataFrame, list[dict]]:
    return pd.read_parquet(SNAPSHOT), json.loads((LABELS / "windows.json").read_text())


def train_fold(name: str, snap: pd.DataFrame, wins: list[dict], **kw) -> dict:
    train_ids, test_ids = make_folds(eligible(window_counts(snap)))[name]
    for wc in window_caches(snap, wins, train_ids + test_ids).values():
        build_cache(wc)
    return train(f"fold{name}", train_ids, snap, wins, **kw)


def _leave() -> None:
    """Exit now: after a long MPS run the interpreter's own teardown has hung with everything already written."""
    import sys

    sys.stdout.flush()
    os._exit(0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m squashvid.ball.finetune")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="freeze the current labels")
    s.add_argument("--force", action="store_true")
    sub.add_parser("cache", help="decode the eligible windows to work/ball_ft/cache")
    r = sub.add_parser("report", help="held-out predictions of fold models A, B, C, pooled against the baseline, threshold sweep, shoe filter")
    r.add_argument("--runs", nargs=3, default=["foldA", "foldB", "foldC"])
    for name in ("train", "fold"):
        t = sub.add_parser(name)
        t.add_argument("run" if name == "train" else "fold", help="run name" if name == "train" else "A, B or C")
        if name == "train":
            t.add_argument("--all", action="store_true", help="train on every eligible window, no validation (use with --epochs)")
            t.add_argument("--test", nargs="*", default=[], help="windows to hold out entirely")
        t.add_argument("--epochs", type=int, default=12)
        t.add_argument("--lr", type=float, default=1e-4)
        t.add_argument("--batch", type=int, default=8)
        t.add_argument("--crop", default="288x512", help="HxW training crop, or 'full'")
        t.add_argument("--patience", type=int, default=4)
        t.add_argument("--workers", type=int, default=3)
        t.add_argument("--sigma", type=float, default=SIGMA)
        t.add_argument("--max-minutes", type=float, default=None)
    a = p.parse_args(argv)
    if a.cmd == "snapshot":
        if SNAPSHOT.exists() and not a.force:
            raise SystemExit(f"{SNAPSHOT} exists; --force replaces it")
        d = snapshot()
        print(json.dumps(window_counts(d)))
        return 0
    snap, wins = _load()
    ids = eligible(window_counts(snap))
    if a.cmd == "cache":
        for wc in window_caches(snap, wins, ids).values():
            print(wc.wid, build_cache(wc), flush=True)
        return 0
    if a.cmd == "report":
        runs = dict(zip("ABC", a.runs))
        cv_report(runs, snap, wins)
        rep = sweep_and_shoes(runs, snap, wins)
        print(json.dumps({"best_threshold": rep["best_threshold"], "thresholds": rep["thresholds"]}, indent=1))
        return 0
    kw = dict(epochs=a.epochs, lr=a.lr, batch=a.batch, crop=None if a.crop == "full" else tuple(int(v) for v in a.crop.split("x")), patience=a.patience, workers=a.workers,
              sigma=a.sigma, max_minutes=a.max_minutes)
    if a.cmd == "fold":
        train_fold(a.fold, snap, wins, **kw)
        _leave()
        return 0
    held = set(a.test)
    train_ids = [w for w in ids if w not in held]
    for wc in window_caches(snap, wins, train_ids).values():
        build_cache(wc)
    train(a.run, train_ids, snap, wins, use_val=not a.all, **kw)
    _leave()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

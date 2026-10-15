"""Run detector variants over the labelling windows and save one parquet per variant per clip.

    python -m squashvid.ball.predict blurball@576x1024 wasb-tennis [--windows IMG_4091-r12 ...] [--copies] [--roi front]
    python -m squashvid.ball.predict wasb-tennis --labelled claude-self   # only the labelled stretch of each window
    python -m squashvid.ball.predict blurball@576x1024 --coreml           # network as a Core ML program on the GPU
    python -m squashvid.ball.predict blurball-ft:final@576x1024 --coreml --full   # whole originals, in chunks

Outputs go to work/<stem>/ball/<variant>.parquet (frame, t, x, y, score, visible)
in that clip's own pixels, merged with earlier runs of the same variant so
windows can be added one at a time. With --copies each window also runs on the
WhatsApp copy over the same time span (written under the copy's stem), and the
copy-to-original frame match for the window is saved for evaluation.

Sizes: default (288x512), native (1:1), x2, or an explicit HxW such as 576x1024
(an original shrunk to exactly the copy's resolution, which isolates what
compression costs). --roi front crops a 576x1024 window at the original's full
resolution around the front wall and front court (on a copy: the same area,
enlarged to 576x1024). HLG originals arrive as SDR from media.video; variants
ending '-tone' (histogram curve, tone.py) and '-hlgraw' (raw HLG code values)
are kept from the first evaluation, before that conversion existed.
--coreml writes a separate '-coreml' variant (see coreml.py for why and how fast).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from squashvid.ball import registry
from squashvid.ball.clips import COPIES, COPY_SCALE, LABELS, ball_dir, is_original, video

FRONT_HW = (576, 1024)  # the front crop in original pixels
LABEL_PAD = 8  # frames of context either side of a labelled stretch (trackers and 3-frame inputs need lead-in)
CHUNK_S = 60.0  # whole-clip runs are saved a chunk at a time, so an interrupted run keeps what it finished
CHUNK_LEAD_S = 1.0  # each chunk runs this far past both ends and drops the extra, so no frame is predicted without context


def front_roi(original: str, frame: int) -> tuple[int, int, int, int]:
    """576x1024 original-pixel crop centred across the front wall, starting ~110 px above its top edge."""
    from squashvid.ball.regions import camera_at, venue

    cam = camera_at(original, frame)
    s = venue(original).spec
    W, L, z = s.width, s.length, s.front_out_z
    uv = cam.project(np.array([[0, L, 0], [W, L, 0], [0, L, z], [W, L, z]], float))
    cx, top = float(uv[:, 0].mean()), float(uv[2:, 1].min())
    h, w = FRONT_HW
    x0 = int(np.clip(round(cx - w / 2), 0, 1920 - w))
    y0 = int(np.clip(round(top - 110), 0, 1080 - h))
    return x0, y0, x0 + w, y0 + h


def scale_roi(roi: tuple[int, int, int, int], s: float) -> tuple[int, int, int, int]:
    return tuple(int(round(v / s)) for v in roi)


def merge_save(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old[~old.frame.isin(df.frame)], df]).sort_values("frame")
    df.reset_index(drop=True).to_parquet(path, index=False)
    return path


def labelled_span(w: dict, labeller: str, root: Path = LABELS) -> tuple[float, float] | None:
    """The part of a window that `labeller` has labelled, padded, as (start, end) seconds on the original."""
    from squashvid.ball import labels as L

    lab = L.load(w["stem"], root, labeller)
    f = lab.frame[(lab.frame >= w["i0"]) & (lab.frame <= w["i1"])] if len(lab) else lab.frame
    if not len(f):
        return None
    ts = video(w["stem"]).timestamps
    return float(ts[max(w["i0"], int(f.min()) - LABEL_PAD)]), float(ts[min(w["i1"], int(f.max()) + LABEL_PAD)])


def chunks(duration: float, size: float = CHUNK_S, lead: float = CHUNK_LEAD_S) -> list[tuple[float, float, float, float]]:
    """(run_from, keep_from, keep_to, run_to) spans tiling [0, duration]; frames are kept on [keep_from, keep_to).

    The last chunk keeps up to and including duration."""
    out, t = [], 0.0
    while t < duration:
        end = min(duration, t + size)
        out.append((max(0.0, t - lead), t, end, min(duration, end + lead)))
        t = end
    return out


def run_one(spec: str, stem: str, start: float, end: float, roi=None, suffix: str = "", roi_name: str | None = None,
            coreml: str | None = None, keep: tuple[float, float] | None = None, **kw) -> tuple[pd.DataFrame, str, dict]:
    """One variant over one span; a named roi ('front') replaces the pixel box in the variant name so windows merge.

    coreml names Core ML compute units (e.g. CPU_AND_GPU) to run the network as a Core ML program; the variant gets a
    '-coreml' suffix so its output never overwrites the torch run it is compared against. With keep=(t0, t1), frames
    outside [t0, t1) are context only: predicted, then dropped before saving."""
    v = video(stem)
    hw = (1080, 1920) if is_original(stem) else (576, 1024)
    if coreml:
        kw = {**kw, "device": "cpu", "half": False}
    det = registry.build(spec, roi=roi, frame_hw=hw, **kw)
    if coreml:
        from squashvid.ball import coreml as cm

        cm.attach(det, coreml)
        suffix += "-coreml"
    t0 = time.perf_counter()
    df = det.run(v, start, end)
    wall = time.perf_counter() - t0
    if keep is not None:
        inside = (df.t >= keep[0]) & (df.t < keep[1])
        context = set(df.frame[~inside])
        df = df[inside].reset_index(drop=True)
        if getattr(det, "candidates", None):
            det.candidates = {f: c for f, c in det.candidates.items() if f not in context}
    variant = (det.variant.split("-roi")[0] + f"-{roi_name}" if roi and roi_name else det.variant) + suffix
    st = det.stats
    stats = dict(variant=variant, stem=stem, start=start, end=end, frames=len(df), wall_s=round(wall, 2), infer_s=round(st.infer_s, 2),
                 decode_s=round(st.decode_s, 2), forwards=st.forwards, ms_per_frame=round(1e3 * wall / max(1, len(df)), 1), visible=int(df.visible.sum()),
                 backend=getattr(det, "backend", f"torch-{det.device.type}"))
    merge_save(df, ball_dir(stem) / f"{variant}.parquet")
    save_candidates(det, ball_dir(stem) / "cands" / f"{variant}.parquet")
    return df, variant, stats


def save_candidates(det, path: Path) -> Path | None:
    """All heatmap blobs per frame (frame, x, y, peak, mass) in frame pixels; a frame with none gets no row."""
    cands = getattr(det, "candidates", None)
    if not cands:
        return None
    rows = [(f, x, y, p, m) for f, cs in cands.items() for x, y, p, m in cs]
    df = pd.DataFrame(rows, columns=["frame", "x", "y", "peak", "mass"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old[~old.frame.isin(list(cands))], df]).sort_values("frame")
    df.reset_index(drop=True).to_parquet(path, index=False)
    return path


def save_alignment(original: str, start: float, end: float) -> Path:
    from squashvid.ball.align import match_window

    m = match_window(original, start, end).rename(columns={"copy_frame": "frame", "frame": "orig_frame", "t": "orig_t"})
    return merge_save(m, ball_dir(COPIES[original]) / "orig_frames.parquet")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m squashvid.ball.predict")
    p.add_argument("specs", nargs="+", help="model@size, e.g. blurball@576x1024")
    p.add_argument("--windows", nargs="*", default=None, help="window ids from work/labels/windows.json (default: all)")
    p.add_argument("--copies", action="store_true", help="also run on the WhatsApp copy over the same time span")
    p.add_argument("--copies-only", action="store_true")
    p.add_argument("--copy-spec", default=None, help="size for the copy run, e.g. native (default: same spec)")
    p.add_argument("--roi", choices=["full", "front"], default="full")
    p.add_argument("--half", action="store_true", help="fp16 on MPS")
    p.add_argument("--coreml", nargs="?", const="CPU_AND_GPU", default=None, metavar="UNITS",
                   help="run the network as a Core ML program (default units CPU_AND_GPU; ALL adds the Neural Engine)")
    p.add_argument("--labelled", default=None, metavar="LABELLER", help="run only where this labeller has labels (padded)")
    p.add_argument("--full", nargs="*", default=None, metavar="STEM",
                   help="run over whole originals instead of the windows (default: every clip that has windows)")
    p.add_argument("--log", type=Path, default=LABELS.parent / "ball_runs.jsonl")
    a = p.parse_args(argv)

    wins = json.loads((LABELS / "windows.json").read_text())
    if a.full is not None:
        for stem in a.full or sorted({w["stem"] for w in wins}):
            duration = float(video(stem).timestamps[-1]) + 1e-3  # so the last frame falls inside [keep_from, keep_to)
            for run_from, keep_from, keep_to, run_to in chunks(duration):
                for spec in a.specs:
                    _, variant, stats = run_one(spec, stem, run_from, run_to, half=a.half, coreml=a.coreml, keep=(keep_from, keep_to))
                    stats.update(window="full", roi="full", ts=time.time(), load=list(np.round(__import__("os").getloadavg(), 1)))
                    with a.log.open("a") as fh:
                        fh.write(json.dumps(stats) + "\n")
                    print(json.dumps(stats), flush=True)
        return 0
    if a.windows:
        wins = [w for w in wins if w["id"] in a.windows]
    for w in wins:
        orig = w["stem"]
        if a.labelled:
            span = labelled_span(w, a.labelled)
            if span is None:
                continue
            w = {**w, "start": span[0], "end": span[1]}
        roi = front_roi(orig, (w["i0"] + w["i1"]) // 2) if a.roi == "front" else None
        for spec in a.specs:
            jobs = []
            if not a.copies_only:
                jobs.append((spec, orig, roi))
            if a.copies or a.copies_only:
                cspec = spec if not a.copy_spec else f"{registry.parse(spec)[0]}@{a.copy_spec}"
                jobs.append((cspec, COPIES[orig], scale_roi(roi, COPY_SCALE) if roi else None))
            for sp, stem, r in jobs:
                _, variant, stats = run_one(sp, stem, w["start"], w["end"], roi=r, roi_name=a.roi if r else None,
                                           half=a.half, coreml=a.coreml)
                stats.update(window=w["id"], roi=a.roi, ts=time.time(), load=list(np.round(__import__("os").getloadavg(), 1)))
                with a.log.open("a") as fh:
                    fh.write(json.dumps(stats) + "\n")
                print(json.dumps(stats), flush=True)
        if a.copies or a.copies_only:
            save_alignment(orig, w["start"], w["end"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

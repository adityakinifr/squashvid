"""Local ball-labelling app: full-resolution frames, model pre-labels, keyboard-driven, autosaving.

    python -m squashvid.ball.label_app --port 8770 [--labeller name]    # then open http://127.0.0.1:8770

(8765 is taken on this laptop by a long-running tpg.py server, hence 8770.)

Labelling works through rally windows (work/labels/windows.json, written by
windows.py) one consecutive frame at a time, because the ball's track across
neighbouring frames is what makes a 2 px blur findable. Frames are decoded once
per window in order (random access into long-GOP HEVC is slow) and cached as
JPEGs under work/labels/frames/. Every keystroke that changes a label appends a
record to work/labels/<stem>.jsonl, so nothing is ever lost and corrections keep
their history. Pre-labels are whatever model outputs exist in work/<stem>/ball/,
plus other labellers' labels as their own layer ("labels:<name>"), so each
person's labels stay independent and progress counts only their own. HLG
originals arrive already converted to SDR BT.709 by media.video, which changes
how a frame looks but not its pixel coordinates.
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel

from squashvid.ball import labels as L
from squashvid.ball.clips import LABELS, ball_dir, video

STATIC = Path(__file__).with_name("static")
WINDOWS = LABELS / "windows.json"
TARGET_FRAMES = 2700
# Layers listed first and switched on when a window opens; every other model output is listed but off, so the
# frame is not buried under a dozen markers. The first is the best zero-shot model on the self-label check.
DEFAULT_LAYERS = ("blurball@576x1024-coreml", "blurball@576x1024", "blurball@default")
# Outputs computed on decodes that media.video no longer produces for the originals (the fitted tone curve, raw HLG)
# stay listed for comparison but after every current-decode output, so the 1-9 keys reach current models first.
LEGACY_DECODE = ("-tone", "-hlgraw")
JPEG_QUALITY = 92


class LabelIn(BaseModel):
    stem: str
    frame: int
    t: float
    x: float | None = None
    y: float | None = None
    visibility: str
    source: str = "user"
    event: str | None = None
    prelabel: str | None = None


def load_windows(path: Path = WINDOWS) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


class FrameStore:
    """JPEG frames on disk, filled by one sequential decode per window on first request."""

    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()
        self.jobs: dict[str, threading.Thread] = {}
        self.ready = threading.Condition()

    def path(self, stem: str, frame: int) -> Path:
        return self.root / stem / f"{frame:06d}.jpg"

    def _decode(self, key: str, stem: str, i0: int, i1: int) -> None:
        v = video(stem)
        ts = v.timestamps
        out = self.root / stem
        out.mkdir(parents=True, exist_ok=True)
        try:
            for idx, _, img in v.frames(float(ts[i0]), float(ts[i1])):
                p = self.path(stem, idx)
                if not p.exists():
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    tmp = p.with_suffix(".tmp")
                    tmp.write_bytes(buf.tobytes())
                    tmp.replace(p)
                with self.ready:
                    self.ready.notify_all()
        finally:
            with self.lock:
                self.jobs.pop(key, None)
            with self.ready:
                self.ready.notify_all()

    def ensure(self, stem: str, i0: int, i1: int) -> None:
        """Start decoding [i0, i1] in the background unless its frames are cached or already decoding."""
        if self.path(stem, i0).exists() and self.path(stem, i1).exists():
            return
        key = f"{stem}:{i0}:{i1}"
        with self.lock:
            if key not in self.jobs:
                th = threading.Thread(target=self._decode, args=(key, stem, i0, i1), daemon=True)
                self.jobs[key] = th
                th.start()

    def get(self, stem: str, frame: int, window: tuple[int, int] | None, timeout: float = 60.0) -> bytes | None:
        p = self.path(stem, frame)
        if not p.exists():
            lo, hi = window if window else (frame, frame + 60)
            self.ensure(stem, max(lo, frame - 2), hi)
            with self.ready:
                self.ready.wait_for(p.exists, timeout=timeout)
        return p.read_bytes() if p.exists() else None


def prelabels(stem: str, i0: int, i1: int, others: pd.DataFrame | None = None) -> dict[str, list]:
    """Every model output for this clip, restricted to the window, as {variant: [[frame, x, y, score, visible], ...]},
    plus other labellers' positions as 'labels:<name>' layers."""
    out = {}
    if others is not None and len(others) and "x" in others:
        o = others[(others.frame >= i0) & (others.frame <= i1) & others.x.notna()]
        for name, g in o.groupby("labeller"):
            out[f"labels:{name}"] = [[int(r.frame), float(r.x), float(r.y), 1.0, True] for r in g.itertuples()]
    d = ball_dir(stem)
    rank = {name: i for i, name in enumerate(DEFAULT_LAYERS)}

    def order(q: Path):
        return rank.get(q.stem, len(rank)), any(tag in q.stem for tag in LEGACY_DECODE), q.stem

    for path in sorted(d.glob("*.parquet"), key=order) if d.exists() else []:
        df = pd.read_parquet(path, columns=["frame", "x", "y", "score", "visible"])
        df = df[(df.frame >= i0) & (df.frame <= i1)]
        if len(df):
            rows = df.replace({np.nan: None}).to_numpy().tolist()
            out[path.stem] = [[int(r[0]), r[1], r[2], None if r[3] is None else round(float(r[3]), 3), bool(r[4])] for r in rows]
    return out


def create_app(labeller: str = "user", root: Path = LABELS, windows_path: Path | None = None, target: int = TARGET_FRAMES):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, Response

    windows_path = windows_path or root / "windows.json"
    store = FrameStore(root / "frames")
    app = FastAPI(title="squashvid ball labels")

    def windows() -> list[dict]:
        return load_windows(windows_path)

    def window(wid: str) -> dict:
        w = next((w for w in windows() if w["id"] == wid), None)
        if w is None:
            raise HTTPException(404, f"no window {wid}")
        return w

    @app.get("/")
    def index():
        return FileResponse(STATIC / "label.html")

    @app.get("/api/windows")
    def list_windows():
        out, done = [], 0
        cache: dict[str, pd.DataFrame] = {}
        for w in windows():
            df = cache.setdefault(w["stem"], L.load(w["stem"], root, labeller))
            n = int(((df.frame >= w["i0"]) & (df.frame <= w["i1"])).sum()) if len(df) else 0
            done += n
            out.append({**w, "n_frames": w["i1"] - w["i0"] + 1, "n_labelled": n})
        return {"labeller": labeller, "target": target, "labelled": done, "windows": out}

    @app.get("/api/window/{wid}")
    def get_window(wid: str):
        w = window(wid)
        stem, i0, i1 = w["stem"], w["i0"], w["i1"]
        store.ensure(stem, i0, i1)
        ts = video(stem).timestamps
        df = L.load(stem, root, labeller)
        lab = df[(df.frame >= i0) & (df.frame <= i1)] if len(df) else df
        return {
            **w,
            "t": [round(float(ts[i]), 4) for i in range(i0, i1 + 1)],
            "labels": json.loads(lab.to_json(orient="records")) if len(lab) else [],
            "prelabels": prelabels(stem, i0, i1, other_labels(stem)),
            "default_layers": list(DEFAULT_LAYERS),
        }

    def other_labels(stem: str) -> pd.DataFrame:
        """Latest label per (labeller, frame) for everyone but this labeller."""
        names = {r.get("labeller") for r in L.read_log(stem, root)} - {labeller, None}
        parts = [L.load(stem, root, n).assign(labeller=n) for n in sorted(names)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    @app.get("/api/frame/{wid}/{frame}.jpg")
    def get_frame(wid: str, frame: int):
        w = window(wid)
        if not w["i0"] <= frame <= w["i1"]:
            raise HTTPException(404, "frame outside window")
        data = store.get(w["stem"], frame, (w["i0"], w["i1"]))
        if data is None:
            raise HTTPException(504, "decode timed out")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.post("/api/label")
    def post_label(lab: LabelIn):
        try:
            rec = L.Label(lab.frame, lab.t, lab.x, lab.y, lab.visibility, lab.source, labeller, event=lab.event, prelabel=lab.prelabel)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        L.append(lab.stem, rec, root)
        return {"ok": True, "ts": rec.ts}

    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m squashvid.ball.label_app")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--labeller", default="user")
    p.add_argument("--root", type=Path, default=LABELS, help="labels directory (jsonl logs, windows.json, frame cache)")
    p.add_argument("--target", type=int, default=TARGET_FRAMES)
    a = p.parse_args(argv)
    import uvicorn

    uvicorn.run(create_app(a.labeller, a.root, target=a.target), host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

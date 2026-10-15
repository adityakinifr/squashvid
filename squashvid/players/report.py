"""Per-clip measurements and check pictures for the player stream (pilot, unvalidated).

Writes into work/<stem>/: players_report.json, topview_rally.png,
overlay_rally.mp4, overlay_check.jpg and crossings/ (one contact sheet per
crossing event, plus stacked pages for review).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from squashvid.calib.venue import Venue
from squashvid.media.video import Video
from squashvid.players import metrics as M
from squashvid.players import render as R
from squashvid.players.geometry import CameraTrack, px_scale
from squashvid.players.occluders import OccluderTrack, hardware_bands
from squashvid.players.tracklets import box_gap

EVENTS_PER_PAGE = 6
PAD_S = 1.0  # colour check window either side of an event


def predicted_boxes(w: pd.DataFrame, cams: CameraTrack, statures: dict[str, float]) -> dict[str, np.ndarray]:
    """Per player and frame: the detection box, else the smoothed court position projected at the player's stature."""
    out = {}
    for p in ("A", "B"):
        box = w[[f"{p}_x1", f"{p}_y1", f"{p}_x2", f"{p}_y2"]].to_numpy(np.float64).copy()
        det = w[f"{p}_detected"].fillna(False).to_numpy(bool)
        box[~det] = np.nan
        xy = w[[f"{p}_sx", f"{p}_sy"]].to_numpy()
        for i in np.flatnonzero(~det & np.isfinite(xy).all(1)):
            cam = cams(int(w.index[i]))
            uv = cam.project(np.array([[xy[i, 0], xy[i, 1], 0.0], [xy[i, 0], xy[i, 1], statures[p]]]))
            h = uv[0, 1] - uv[1, 1]
            box[i] = [uv[0, 0] - 0.22 * h, uv[1, 1], uv[0, 0] + 0.22 * h, uv[0, 1]]
        out[p] = box
    return out


def colour_check(table: pd.DataFrame, ev: M.Event) -> dict[str, dict[str, float]]:
    """Mean colour evidence of each player's detections in the second before and after an event.

    Positive means light-blue shirt, so A should stay positive and B negative.
    """
    out = {}
    for p in ("A", "B"):
        tp = table[(table.player == p) & table.detected & (table.llr != 0)]
        before = tp[(tp.t >= ev.t0 - PAD_S) & (tp.t < ev.t0)].llr
        after = tp[(tp.t > ev.t1) & (tp.t <= ev.t1 + PAD_S)].llr
        out[p] = {"before": round(float(before.mean()), 2) if len(before) else None, "after": round(float(after.mean()), 2) if len(after) else None}
    return out


def suspect(check: dict[str, dict[str, float]]) -> bool:
    a, b = check["A"]["after"], check["B"]["after"]
    return (a is not None and a < -1.0) or (b is not None and b > 1.0)


def report(video: Video, venue: Venue, cams: CameraTrack, work: Path, rally: tuple[float, float] | None = None, sheets: bool = True) -> dict:
    spec = venue.spec
    table = pd.read_parquet(work / "players.parquet")
    summary = json.loads((work / "players_summary.json").read_text())
    occ = OccluderTrack.load(work / "occluders.npz")
    hips = summary["hip_height_m"]
    statures = {p: hips[p] / 0.53 for p in ("A", "B")}
    w = M.wide(table).sort_index()
    pred = predicted_boxes(w, cams, statures)

    # Where lost players are, in the image and on court.
    bands = hardware_bands(occ, video.info.width)
    image_u, overlap = {}, {}
    for p, q in (("A", "B"), ("B", "A")):
        u = (pred[p][:, 0] + pred[p][:, 2]) / 2
        ov = np.array([np.isfinite(a).all() and np.isfinite(b).all() and box_gap(a, b) < 0 for a, b in zip(pred[p], pred[q])])
        rows = table[table.player == p].frame.to_numpy()
        pos = {f: i for i, f in enumerate(w.index.to_numpy())}
        sel = np.array([pos[f] for f in rows])
        image_u[p], overlap[p] = u[sel], ov[sel]

    rally_list = M.rallies(table)
    if rally is None and rally_list:
        rally = max(rally_list, key=lambda r: r[1] - r[0])
    out = {
        "label": "pilot, unvalidated",
        "coverage": M.coverage(table),
        "losses": M.losses(table, spec, image_u, bands, overlap),
        "hardware_bands_px": bands,
        "jitter": M.jitter(table, spec),
        "movement_clip": M.movement(table, spec),
        "rallies_heuristic": rally_list,
        "rally": rally,
        "movement_rally": M.movement(table, spec, *rally) if rally else None,
        "statures_m": {p: round(s, 2) for p, s in statures.items()},
    }

    events = M.crossing_events(w, pred, M.CONTACT_PX * px_scale(video.info.width))
    ev_rows = []
    for k, ev in enumerate(events):
        chk = colour_check(table, ev)
        ev_rows.append({"id": k, "start": ev.start, "end": ev.end, "t0": round(ev.t0, 2), "t1": round(ev.t1, 2), "kind": ev.kind, "colour": chk, "suspect": suspect(chk)})
    out["events"] = {"n": len(events), "n_overlap": sum(e.kind == "overlap" for e in events), "n_hidden": sum(e.kind == "hidden" for e in events), "n_suspect_colour": sum(r["suspect"] for r in ev_rows)}
    (work / "crossings.json").write_text(json.dumps(ev_rows, indent=1))

    consumers: list = [OverlayCheck(video, w, pred)]
    if rally:
        R.topview_png(table, spec, rally[0], rally[1], work / "topview_rally.png", f"{video.path.stem}: player trajectories, one rally", out["movement_rally"])
        overlay = R.OverlayVideo(video, w, pred, spec, rally[0], rally[1], work / "overlay_rally.mp4")
        consumers.append(overlay)
    sheets_list = []
    if sheets:
        for ev, r in zip(events, ev_rows):
            c = r["colour"]
            title = f"#{r['id']} {ev.kind} t={ev.t0:.2f}-{ev.t1:.2f}s  colour A {c['A']['before']}->{c['A']['after']}  B {c['B']['before']}->{c['B']['after']}{'  SUSPECT' if r['suspect'] else ''}"
            sheets_list.append(R.ContactSheet(video, w, pred, ev, title))
    consumers.extend(sheets_list)
    R.decode_pass(video, consumers)
    consumers[0].save(work / "overlay_check.jpg")
    if rally:
        overlay.close()
    if sheets:
        _write_sheets(sheets_list, ev_rows, work / "crossings")
    (work / "players_report.json").write_text(json.dumps(out, indent=1, default=float))
    return out


class OverlayCheck:
    """Twelve evenly spaced frames with IDs drawn, as one grid image (a decode_pass consumer)."""

    def __init__(self, video: Video, w: pd.DataFrame, pred: dict[str, np.ndarray], n: int = 12):
        self.w, self.pred, self.ts = w, pred, video.timestamps
        self.pos = {f: i for i, f in enumerate(w.index.to_numpy())}
        self.want = [int(f) for f in w.index.to_numpy()[np.linspace(0, len(w) - 1, n + 2).round().astype(int)[1:-1]]]
        self.tiles: dict[int, np.ndarray] = {}

    def frames(self) -> set[int]:
        return set(self.want)

    def add(self, idx: int, img: np.ndarray) -> None:
        s = 512 / img.shape[1]
        img = cv2.resize(img, (512, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        R.draw_players(img, R.frame_rows(self.w, idx, self.pred, self.pos), s)
        cv2.putText(img, f"t={self.ts[idx]:.1f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        self.tiles[idx] = img

    def save(self, path: Path) -> None:
        tiles = [self.tiles[f] for f in self.want if f in self.tiles]
        rows = [np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
        if rows:
            cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def _write_sheets(sheets: list[R.ContactSheet], rows: list[dict], folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    page: list[np.ndarray] = []
    for sheet, r in zip(sheets, rows):
        img = sheet.image()
        cv2.imwrite(str(folder / f"event_{r['id']:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        page.append(img)
        if len(page) == EVENTS_PER_PAGE:
            cv2.imwrite(str(folder / f"page_{r['id'] // EVENTS_PER_PAGE:02d}.jpg"), R.stack_sheets(page), [cv2.IMWRITE_JPEG_QUALITY, 80])
            page = []
    if page:
        cv2.imwrite(str(folder / f"page_{rows[-1]['id'] // EVENTS_PER_PAGE:02d}.jpg"), R.stack_sheets(page), [cv2.IMWRITE_JPEG_QUALITY, 80])

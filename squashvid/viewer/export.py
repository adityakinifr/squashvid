"""Per-clip viewer export: work/<stem>/viewer/{meta.json, <array>.bin}.

Everything is keyed by frame index (the index into the clip's presentation timestamps, `t`). Dense per-frame data
is stored as little-endian binary arrays described in meta.json["arrays"] (dtype, shape, scale, missing sentinel;
a value v decodes to v / scale, and the sentinel decodes to NaN), which the browser reads as typed arrays.
Sparse data (court lines, rallies, events, stats, insights) is plain JSON inside meta.json.

Every upstream file is optional: a missing input leaves its arrays filled with the sentinel and its meta key
empty, and `meta["sources"]` records what was found.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
I16_MISSING = -32768
PLAYERS = ("A", "B")
N_KPT = 17
BALL_STATE = {"none": 0, "detected": 1, "linked": 2, "interpolated": 3}
# Raw detector outputs used when no linked ball_track.parquet exists, best first.
RAW_BALL = ("blurball-ft-final@576x1024-coreml", "blurball-ft-foldA@576x1024-coreml", "blurball@576x1024-coreml")
EVENT_FIELDS = ("t", "frame", "kind", "u", "v", "x", "y", "z", "sigma_a", "sigma_b", "evidence", "confidence", "player")
AUDIO_KINDS = ("racket", "front_wall", "floor", "side_wall")


def _work(stem: str, root: Path | None) -> Path:
    if root is not None:
        return Path(root) / stem
    from squashvid.ball.clips import work_dir

    return work_dir(stem)


def clean(x):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def quantise(a: np.ndarray, scale: float, dtype=np.int16) -> np.ndarray:
    """Float array -> integer array of round(a * scale); NaN/inf/out-of-range become the sentinel."""
    info = np.iinfo(dtype)
    sentinel = info.min if info.min < 0 else info.max
    a = np.asarray(a, np.float64)
    q = np.round(a * scale)
    bad = ~np.isfinite(q) | (q <= info.min) | (q >= info.max)
    return np.where(bad, sentinel, q).astype(dtype)


def dequantise(q: np.ndarray, scale: float) -> np.ndarray:
    info = np.iinfo(q.dtype)
    sentinel = info.min if info.min < 0 else info.max
    return np.where(q == sentinel, np.nan, q / scale)


def frame_of_time(ts: np.ndarray, t) -> np.ndarray:
    """Index of the frame showing at media time t: the last frame whose PTS is <= t (0 before the first)."""
    ts = np.asarray(ts, np.float64)
    return np.clip(np.searchsorted(ts, np.asarray(t, np.float64) + 1e-6, side="right") - 1, 0, len(ts) - 1)


def court_payload(spec) -> dict:
    from squashvid.geom import court

    lines = []
    for ln in court.court_lines(spec):
        lines.append({"name": ln.name, "surface": ln.surface, "kind": ln.kind, "pts": np.round(ln.sample(0.25), 4).tolist()})
    return {
        "spec": spec.to_dict(),
        "derived": {
            "short_y": spec.short_y, "box_back_y": spec.box_back_y, "box_inner_left_x": spec.box_inner_left_x,
            "box_inner_right_x": spec.box_inner_right_x, "front_out_z": spec.front_out_z, "service_z": spec.service_z,
            "tin_z": spec.tin_z, "back_out_z": spec.back_out_z,
        },
        "lines": lines,
    }


class _Out:
    def __init__(self, folder: Path):
        self.folder = folder
        self.arrays: dict[str, dict] = {}

    def put(self, name: str, a: np.ndarray, scale: float = 1.0):
        a = np.ascontiguousarray(a)
        (self.folder / f"{name}.bin").write_bytes(a.astype(a.dtype.newbyteorder("<")).tobytes())
        info = np.iinfo(a.dtype) if a.dtype.kind in "iu" else None
        self.arrays[name] = {
            "file": f"{name}.bin", "dtype": a.dtype.name, "shape": list(a.shape), "scale": scale,
            "missing": (int(info.min if info.min < 0 else info.max) if info else None),
        }


def _read(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path) if path.exists() else None
    except Exception:  # unreadable upstream file: treat as absent, the viewer shows what it has
        return None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except Exception:
        return None


def _by_frame(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df[(df.frame >= 0) & (df.frame < n)]
    return df.drop_duplicates("frame", keep="last").set_index("frame")


def build_camera(df: pd.DataFrame, venue, n: int) -> np.ndarray:
    """(n, 9) float32: f, cx, cy, rvec(3), tvec(3) per frame, from the smoothed pose; NaN where no pose exists."""
    out = np.full((n, 9), np.nan, np.float32)
    d = _by_frame(df, n)
    cam0 = venue.camera(np.zeros(3), np.zeros(3))
    idx = d.index.to_numpy()
    out[idx, 0], out[idx, 1], out[idx, 2] = cam0.f, cam0.cx, cam0.cy
    out[idx, 3:6] = d[["sr0", "sr1", "sr2"]].to_numpy()
    out[idx, 6:9] = d[["st0", "st1", "st2"]].to_numpy()
    if len(idx):  # the tracked pose is held outside its range, as regions.camera_at does
        out[: idx.min()] = out[idx.min()]
        out[idx.max() + 1:] = out[idx.max()]
    return out


def players_arrays(df: pd.DataFrame, n: int) -> dict[str, np.ndarray]:
    """Quantised per-player arrays: pos mm (n,2), box px*8 (n,4), kp px*8 (n,17,2), ks/conf/status u8."""
    out: dict[str, np.ndarray] = {}
    for p in PLAYERS:
        sub = df[df.player == p]
        d = _by_frame(sub, n) if len(sub) else sub.set_index("frame")
        idx = d.index.to_numpy(dtype=int)
        pos = np.full((n, 2), np.nan)
        box = np.full((n, 4), np.nan)
        kp = np.full((n, N_KPT, 2), np.nan)
        ks = np.zeros((n, N_KPT), np.uint8)
        conf = np.zeros(n, np.uint8)
        status = np.zeros(n, np.uint8)  # 0 none, 1 interpolated, 2 detected
        if len(idx):
            pos[idx] = d[["sx", "sy"]].to_numpy(float)
            box[idx] = d[["x1", "y1", "x2", "y2"]].to_numpy(float)
            for i, (k, s) in enumerate(zip(d["kpts"], d["kscore"], strict=False)):
                if k is not None and len(k) == 2 * N_KPT:
                    kp[idx[i]] = np.asarray(k, float).reshape(N_KPT, 2)
                    ks[idx[i]] = np.clip(np.round(np.asarray(s, float) * 255), 0, 255) if s is not None and len(s) == N_KPT else 0
            conf[idx] = np.clip(np.round(d["confidence"].fillna(0).to_numpy(float) * 255), 0, 255)
            has = np.isfinite(pos[idx]).all(axis=1)
            det = d["detected"].to_numpy(bool)
            status[idx] = np.where(has, np.where(det, 2, 1), 0)
        out[f"{p}_pos"] = quantise(pos, 1000)
        out[f"{p}_box"] = quantise(box, 8)
        out[f"{p}_kp"] = quantise(kp, 8)
        out[f"{p}_ks"] = ks
        out[f"{p}_conf"] = conf
        out[f"{p}_status"] = status
    return out


def load_ball(work: Path, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    """(uv (n,2) px, state (n,) u8, score (n,) f32, source name): ball_track.parquet, else the best raw detector output."""
    uv = np.full((n, 2), np.nan)
    state = np.zeros(n, np.uint8)
    score = np.zeros(n, np.float32)
    df = _read(work / "ball_track.parquet")
    if df is not None and len(df):
        d = _by_frame(df, n)
        i = d.index.to_numpy(dtype=int)
        uv[i] = d[["u", "v"]].to_numpy(float)
        st = d["state"].map(BALL_STATE).fillna(0).to_numpy(np.uint8) if "state" in d else np.full(len(d), 1, np.uint8)
        state[i] = np.where(np.isfinite(uv[i]).all(axis=1), st, 0)
        if "score" in d:
            score[i] = d["score"].fillna(0).to_numpy(np.float32)
        return uv, state, score, "ball_track"
    for name in RAW_BALL:
        raw = _read(work / "ball" / f"{name}.parquet")
        if raw is not None and len(raw):
            d = _by_frame(raw, n)
            i = d.index.to_numpy(dtype=int)
            vis = d["visible"].to_numpy(bool) & np.isfinite(d[["x", "y"]].to_numpy(float)).all(axis=1)
            uv[i[vis]] = d.loc[vis, ["x", "y"]].to_numpy(float)
            state[i[vis]] = BALL_STATE["detected"]
            score[i] = d["score"].fillna(0).to_numpy(np.float32)
            return uv, state, score, f"raw:{name}"
    return uv, state, score, None


def ball_floor(cam: np.ndarray, uv: np.ndarray, venue) -> np.ndarray:
    """Where each ball pixel's ray meets the floor (x, y metres): the ball's shadow if it were on the floor; NaN if none."""
    from squashvid.geom.camera import Camera

    out = np.full((len(uv), 2), np.nan)
    dist = venue.camera(np.zeros(3), np.zeros(3)).dist
    for i in np.flatnonzero(np.isfinite(uv).all(axis=1) & np.isfinite(cam).all(axis=1)):
        c = Camera(float(cam[i, 0]), float(cam[i, 1]), float(cam[i, 2]), cam[i, 3:6].astype(np.float64), cam[i, 6:9].astype(np.float64), dist)
        p = c.to_plane(uv[i : i + 1], "floor", spec=venue.spec)[0]
        out[i] = p[:2]
    return out


def table_records(df: pd.DataFrame | None, fields=None) -> list[dict]:
    if df is None or not len(df):
        return []
    cols = [c for c in (fields or df.columns) if c in df.columns]
    return clean(df[cols].to_dict(orient="records"))


HITTER_MAX_S = 3.0  # a ball event belongs to the latest shot no more than this long before it


def attribute_hitters(events: list[dict], hits: pd.DataFrame | None, max_s: float = HITTER_MAX_S) -> None:
    """Set `hitter` on each ball event to the player whose shot most recently preceded it, as the analysis stage does."""
    if hits is None or not len(hits) or "hitter" not in hits:
        return
    h = hits.dropna(subset=["t"]).sort_values("t")
    ht, who = h["t"].to_numpy(float), h["hitter"].to_numpy()
    for e in events:
        if e.get("t") is None:
            continue
        i = int(np.searchsorted(ht, e["t"], side="right")) - 1
        if i >= 0 and e["t"] - ht[i] <= max_s and isinstance(who[i], str):
            e["hitter"] = who[i]


def load_events(work: Path, ts: np.ndarray) -> list[dict]:
    ev = _read(work / "ball_events.parquet")
    hits = _read(work / "analysis" / "hits.parquet")
    out = []
    for r in table_records(ev, EVENT_FIELDS):
        r.setdefault("source", "ball")
        if r.get("frame") is None and r.get("t") is not None:
            r["frame"] = int(frame_of_time(ts, r["t"]))
        out.append(r)
    attribute_hitters(out, hits)
    on = _read(work / "audio_onsets.parquet")
    if on is not None and len(on):
        on = on[on["kind"].isin(AUDIO_KINDS)]
        for t, kind, rally in zip(on["t_peak"] if "t_peak" in on else on["t"], on["kind"], on.get("rally", [None] * len(on)), strict=False):
            out.append({"t": float(t), "frame": int(frame_of_time(ts, t)), "kind": f"audio_{kind}", "source": "audio", "rally": clean(rally)})
    for r in table_records(hits, ("t", "frame", "hitter", "rally", "shot_index", "situation")):
        if r.get("t") is not None:
            out.append({"t": r["t"], "frame": int(frame_of_time(ts, r["t"])), "kind": "hit", "source": "hits", "player": r.get("hitter"),
                        "rally": r.get("rally"), "shot_index": r.get("shot_index")})
    out.sort(key=lambda r: r["t"] if r.get("t") is not None else 0.0)
    return out


def load_rallies(work: Path) -> tuple[list[dict], str | None]:
    for rel in ("analysis/rallies.parquet", "audio_rallies.parquet"):
        df = _read(work / rel)
        if df is not None and len(df):
            return table_records(df), rel
    return [], None


def export(stem: str, root: Path | None = None, timestamps: np.ndarray | None = None, out_dir: Path | None = None,
           work_dir: Path | None = None) -> dict:
    """Write work/<stem>/viewer/. `timestamps` defaults to the original's PTS, else the proxy's, else the pose table's t."""
    work = Path(work_dir) if work_dir else _work(stem, root)
    folder = out_dir or work / "viewer"
    folder.mkdir(parents=True, exist_ok=True)
    sources: dict[str, bool] = {}

    cam_df = _read(work / "camera.parquet")
    players_df = _read(work / "players.parquet")
    ts = timestamps
    if ts is None and (work / "proxy.mp4").exists():  # what the browser plays; equals the original's PTS (proxy.check_pts)
        from squashvid.viewer.proxy import pts_of

        ts = pts_of(work / "proxy.mp4")
    if ts is None and root is None:
        try:
            from squashvid.ball.clips import video

            ts = video(stem).timestamps
        except Exception:
            ts = None
    if ts is None:
        for df in (cam_df, players_df):
            if df is not None and len(df):
                d = df.drop_duplicates("frame").sort_values("frame")
                ts = np.interp(np.arange(int(d.frame.max()) + 1), d.frame, d.t)
                break
    if ts is None:
        raise FileNotFoundError(f"no timestamps for {stem}: need the original video, a proxy, or camera/players tables")
    ts = np.asarray(ts, np.float64)
    n = len(ts)

    out = _Out(folder)
    out.put("t", ts)

    spec = None
    venue = None
    try:
        from squashvid.ball.regions import venue as _venue

        if root is None:
            venue = _venue(stem)
            spec = venue.spec
    except Exception:
        venue = None
    if spec is None:
        from squashvid.geom import court

        spec = court.WSF
    sources["venue"] = venue is not None

    cam = None
    if cam_df is not None and venue is not None and "sr0" in cam_df:
        cam = build_camera(cam_df, venue, n)
        out.put("cam", cam)
    sources["camera"] = cam is not None

    if players_df is not None and len(players_df):
        for name, a in players_arrays(players_df, n).items():
            out.put(name, a, scale={"pos": 1000, "box": 8, "kp": 8, "ks": 255, "conf": 255}.get(name.split("_")[1], 1))
    sources["players"] = players_df is not None and len(players_df) > 0

    uv, state, score, ball_src = load_ball(work, n)
    out.put("ball_uv", quantise(uv, 8), 8)
    out.put("ball_state", state)
    out.put("ball_score", np.clip(np.round(score * 255), 0, 255).astype(np.uint8), 255)
    sources["ball"] = ball_src
    if cam is not None and venue is not None and ball_src:
        out.put("ball_floor", quantise(ball_floor(cam, uv, venue), 1000), 1000)

    b3 = _read(work / "ball_3d.parquet")
    sources["ball_3d"] = b3 is not None and len(b3) > 0
    if sources["ball_3d"]:
        d = _by_frame(b3, n)
        i = d.index.to_numpy(dtype=int)
        xyz = np.full((n, 3), np.nan)
        xyz[i] = d[["x", "y", "z"]].to_numpy(float)
        out.put("ball_xyz", quantise(xyz, 1000), 1000)
        anchored = np.zeros(n, np.uint8)
        if "anchored" in d:
            anchored[i] = d["anchored"].fillna(False).to_numpy(bool)
        out.put("ball_anchored", anchored)

    rallies, rallies_src = load_rallies(work)
    events = load_events(work, ts)
    sources["rallies"] = rallies_src
    sources["events"] = _read(work / "ball_events.parquet") is not None
    sources["audio_events"] = _read(work / "audio_onsets.parquet") is not None

    a = work / "analysis"
    hits = table_records(_read(a / "hits.parquet"))
    stats = _read_json(a / "stats.json")
    insights = _read_json(a / "insights.json")
    report = (a / "report.md").read_text() if (a / "report.md").exists() else None
    sources.update(hits=bool(hits), stats=stats is not None, insights=insights is not None, report=report is not None)

    summary = _read_json(work / "players_summary.json") or {}
    meta = {
        "schema_version": SCHEMA_VERSION,
        "stem": stem,
        "n_frames": n,
        "duration": float(ts[-1]) if n else 0.0,
        "fps": float((n - 1) / (ts[-1] - ts[0])) if n > 1 and ts[-1] > ts[0] else None,
        "sources": sources,
        "proxy": (work / "proxy.mp4").exists(),
        "court": court_payload(spec),
        "camera_dist": clean(venue.camera(np.zeros(3), np.zeros(3)).dist.tolist()) if venue is not None else [0, 0, 0, 0, 0],
        "players": {"ids": list(PLAYERS), "hip_height_m": clean(summary.get("hip_height_m", {}))},
        "arrays": out.arrays,
        "rallies": rallies,
        "events": events,
        "hits": hits,
        "stats": clean(stats),
        "insights": clean(insights),
        "report": report,
    }
    (folder / "meta.json").write_text(json.dumps(clean(meta), separators=(",", ":")))
    return {"stem": stem, "frames": n, "bytes": sum(p.stat().st_size for p in folder.iterdir()), "sources": sources}

"""python -I -m squashvid.balltrack <stem> [--variant V]: link candidates, find events, write parquet."""

from __future__ import annotations

import argparse
import time

from squashvid.balltrack import data, pipeline

TRACK_COLS = ["frame", "t", "u", "v", "state", "score", "track_id", "segment_id"]


def run(stem: str, variant: str = "blurball-ft-final", params: pipeline.Params | None = None) -> dict:
    params = params or pipeline.Params()
    t0 = time.time()
    T = data.timestamps(stem)
    cands = data.clip_candidates(stem, variant)
    n = len(T)
    track = pipeline.build_track(cands, T, n, params)
    fr = track.frames.copy()
    fr["t"] = T[fr.frame.to_numpy().astype(int)]
    fr[TRACK_COLS].to_parquet(data.output_path(stem, "ball_track.parquet"), index=False)
    ev = pipeline.build_events(stem, track, T, params)
    ev.to_parquet(data.output_path(stem, "ball_events.parquet"), index=False)
    summary = {"stem": stem, "frames": n, "track_rows": len(fr), "states": fr.state.value_counts().to_dict(),
               "events": ev.kind.value_counts().to_dict(), "seconds": round(time.time() - t0, 1)}
    try:
        from squashvid.balltrack import traj3d
    except ImportError:
        return summary
    b3 = traj3d.fit(stem, track, ev, T)
    b3.to_parquet(data.output_path(stem, "ball_3d.parquet"), index=False)
    summary["ball_3d_rows"] = len(b3)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="squashvid.balltrack")
    ap.add_argument("stem")
    ap.add_argument("--variant", default="blurball-ft-final")
    a = ap.parse_args()
    print(run(a.stem, a.variant))


if __name__ == "__main__":
    main()

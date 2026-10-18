"""One command from a new video to everything the viewer needs.

    python -I -m squashvid.process <video> [--stem NAME] [--venue VENUE_JSON] [--from STAGE] [--only STAGE]

Stages run in order and each skips itself when its outputs already exist, unless
--from (that stage and every later one) or --only (just that one) forces it. The
video is recorded in work/clips.json so the per-stage modules can find it by stem.
--venue pins a venue (another court or mount) by copying it to work/<stem>/venue.json;
without it a clip uses its pinned venue, else the court-5 venue of its frame size.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from squashvid.ball import clips

BALL_SPEC = "blurball-ft:final@576x1024"
BALL_VARIANT = "blurball-ft-final@576x1024-coreml"


@dataclass
class Stage:
    name: str
    outputs: Callable[[], list[Path]]  # files whose presence means the stage is done
    run: Callable[[], dict | None]
    complete: Callable[[], bool] = lambda: True  # extra check for outputs that can be partial (ball saves per chunk)

    def done(self) -> bool:
        return all(p.exists() for p in self.outputs()) and self.complete()


def plan(stages: list[Stage], start: str | None = None, only: str | None = None) -> list[tuple[Stage, bool]]:
    """(stage, will_run) for every stage: --only forces one and skips the rest, --from forces it and all later ones."""
    names = [s.name for s in stages]
    for n in (start, only):
        if n is not None and n not in names:
            raise ValueError(f"unknown stage {n!r}; stages are {', '.join(names)}")
    if start and only:
        raise ValueError("give --from or --only, not both")
    first = names.index(start) if start else 0
    return [(s, s.name == only if only else (i >= first if start else not s.done())) for i, s in enumerate(stages)]


def execute(stages: list[Stage], start: str | None = None, only: str | None = None, out=print) -> list[dict]:
    """Run the planned stages in order; a failing stage stops the run with its name attached."""
    rows = []
    for stage, go in plan(stages, start, only):
        if not go:
            out(f"[{stage.name}] skipped (outputs exist)" if only is None else f"[{stage.name}] not selected")
            rows.append({"stage": stage.name, "status": "skipped", "seconds": 0.0})
            continue
        out(f"[{stage.name}] running")
        t0 = time.perf_counter()
        try:
            info = stage.run() or {}
        except Exception as e:
            raise RuntimeError(f"stage {stage.name!r} failed after {time.perf_counter() - t0:.0f}s: {type(e).__name__}: {e}") from e
        dt = time.perf_counter() - t0
        missing = [p for p in stage.outputs() if not p.exists()]
        if missing:
            raise RuntimeError(f"stage {stage.name!r} finished but did not write {', '.join(map(str, missing))}")
        out(f"[{stage.name}] done in {dt:.1f}s -> {', '.join(str(p) for p in stage.outputs())}")
        rows.append({"stage": stage.name, "status": "ran", "seconds": round(dt, 1), **info})
    return rows


def build_stages(stem: str, video_path: Path, venue_json: Path, work: Path | None = None) -> list[Stage]:
    w = (work or clips.WORK) / stem

    def register() -> dict:
        from squashvid.calib.venue import Venue
        from squashvid.calib.register import register_video
        from squashvid.media.video import Video

        t0 = time.perf_counter()

        def progress(idx, t, fr) -> None:
            if idx % 600 == 0:
                print(f"  register t={t:7.1f}s frame={idx} match={fr.match_frac:.2f} {idx / max(1e-9, time.perf_counter() - t0):.1f} fps", flush=True)

        s = register_video(Video(video_path), Venue.load(venue_json), w, progress)
        if s["match_frac_median"] < 0.5:
            print(f"  WARNING: median match_frac {s['match_frac_median']} is low; is this court 5, framed like the other clips?", flush=True)
        return s

    def players() -> dict:
        from squashvid.players.__main__ import run

        run(video_path, stem=stem, venue_json=str(venue_json), work=w)
        return {}

    def audio() -> dict:
        from squashvid.audio.__main__ import main

        main([str(video_path), "--work", str(w)])
        return {}

    def ball() -> dict:
        from squashvid.ball.predict import run_full

        stats = run_full(stem, [BALL_SPEC], coreml="CPU_AND_GPU")
        return {"chunks": len(stats), "frames": sum(s["frames"] for s in stats)}

    def ball_complete() -> bool:
        import pandas as pd

        df = pd.read_parquet(w / "ball" / f"{BALL_VARIANT}.parquet", columns=["frame"])
        return int(df.frame.max()) >= len(clips.video(stem).timestamps) - 2

    def balltrack() -> dict:
        from squashvid.balltrack.__main__ import run

        return run(stem)

    def analysis() -> dict:
        from squashvid.analysis import run

        return run.run(stem)

    def bodies() -> dict:
        from squashvid.bodies.run import run

        s = run(stem, device="mps")
        return {"stature_m": {p: d.get("stature_m_p90") for p, d in s["players"].items()}}

    def viewer() -> dict:
        from squashvid.viewer import proxy
        from squashvid.viewer.export import export

        return {"proxy": proxy.make_proxy(stem), "export": export(stem)}

    return [
        Stage("register", lambda: [w / "camera.parquet", w / "camera_summary.json"], register),
        Stage("players", lambda: [w / "detections.parquet", w / "players.parquet", w / "players_summary.json"], players),
        Stage("audio", lambda: [w / "audio_onsets.parquet", w / "audio_rallies.parquet"], audio),
        Stage("ball", lambda: [w / "ball" / f"{BALL_VARIANT}.parquet"], ball, ball_complete),
        Stage("balltrack", lambda: [w / "ball_track.parquet", w / "ball_events.parquet"], balltrack),
        Stage("analysis", lambda: [w / "analysis" / "report.md", w / "analysis" / "stats.json"], analysis),
        Stage("bodies", lambda: [w / "bodies.parquet"], bodies),
        Stage("viewer", lambda: [w / "proxy.mp4", w / "viewer" / "meta.json"], viewer),
    ]


def pin_venue(stem: str, venue_json: Path, size: tuple[int, int], work: Path | None = None) -> Path:
    """Copy venue_json to <work>/<stem>/venue.json after checking its intrinsics belong to this frame size."""
    from squashvid.calib.venue import Venue

    venue = Venue.load(venue_json)
    if tuple(venue.size) != tuple(size):
        raise ValueError(f"{venue_json} is calibrated for {venue.size[0]}x{venue.size[1]}, the video is {size[0]}x{size[1]}")
    return venue.save((work or clips.WORK) / stem / clips.VENUE_SIDECAR)


def resolve_venue(stem: str, size: tuple[int, int], work: Path | None = None) -> Path:
    """The clip's pinned venue, else the court-5 venue of its frame size."""
    pinned = (work or clips.WORK) / stem / clips.VENUE_SIDECAR
    return pinned if pinned.exists() else clips.venue_for_size(*size, work)


STAGE_NAMES = ("register", "players", "audio", "ball", "balltrack", "analysis", "bodies", "viewer")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -I -m squashvid.process", description=__doc__.split("\n\n")[0])
    p.add_argument("video")
    p.add_argument("--stem", default=None, help="name for the work directory (default: the file name without extension)")
    p.add_argument("--venue", default=None, help="venue file for a court other than court 5; pinned to the clip for later runs")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--from", dest="start", choices=STAGE_NAMES, help="rerun this stage and every later one")
    g.add_argument("--only", choices=STAGE_NAMES, help="rerun just this stage")
    a = p.parse_args(argv)

    from squashvid.media.video import probe

    video_path = Path(a.video).expanduser().resolve()
    if not video_path.exists():
        p.error(f"no such file: {video_path}")
    stem = a.stem or video_path.stem
    info = probe(video_path)
    try:
        venue_json = pin_venue(stem, Path(a.venue).expanduser(), (info.width, info.height)) if a.venue else resolve_venue(stem, (info.width, info.height))
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if stem in clips.CLIPS and clips.clip_path(stem).resolve() != video_path:
        p.error(f"stem {stem!r} is a built-in clip; pass --stem to name this one")
    if stem not in clips.CLIPS:
        print(f"registered {stem} -> {video_path} in {clips.register(stem, video_path)}")
    print(f"{stem}: {info.width}x{info.height}, {info.duration:.1f}s, venue {venue_json.name}; outputs in {clips.work_dir(stem)}", flush=True)

    t0 = time.perf_counter()
    rows = execute(build_stages(stem, video_path, venue_json), a.start, a.only)
    print(f"\n{'stage':<10}{'status':<9}{'seconds':>9}")
    for r in rows:
        print(f"{r['stage']:<10}{r['status']:<9}{r['seconds']:>9.1f}")
    print(f"{'total':<19}{time.perf_counter() - t0:>9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

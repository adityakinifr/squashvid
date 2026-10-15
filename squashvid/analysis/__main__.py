"""python -I -m squashvid.analysis <stem>|--all [--no-ball] [--pooled]"""

from __future__ import annotations

import argparse
import json
import logging

from squashvid.analysis.io import CLIPS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="squashvid.analysis", description="Rallies, hits, statistics and coaching insights for a clip.")
    ap.add_argument("stem", nargs="?", help="clip stem, e.g. IMG_4091")
    ap.add_argument("--all", action="store_true", help="run all three clips")
    ap.add_argument("--no-ball", action="store_true", help="ignore ball events even if present")
    ap.add_argument("--no-review", action="store_true", help="skip the timeline plot")
    ap.add_argument("--pooled", action="store_true", help="with --all, also write a report pooling the clips (assumes A and B are the same people in every clip)")
    ap.add_argument("--tag", default="", help="suffix for review plots, e.g. ball")
    args = ap.parse_args(argv)
    if not args.stem and not args.all:
        ap.error("give a clip stem or --all")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from squashvid.analysis import run

    stems = list(CLIPS) if args.all else [args.stem]
    out = [run.run(s, use_ball=not args.no_ball, review=not args.no_review, tag=args.tag) for s in stems]
    if args.all and args.pooled:
        out.append(run.run_pooled(stems, use_ball=not args.no_ball))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

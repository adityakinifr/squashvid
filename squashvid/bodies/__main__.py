from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -I -m squashvid.bodies", description="Lift players' 2D keypoints to metric 3D skeletons.")
    ap.add_argument("stems", nargs="*", help="clip stems under work/")
    ap.add_argument("--device", default="cpu", help="torch device (cpu, mps)")
    ap.add_argument("--download", action="store_true", help="fetch the MotionBERT weights and exit")
    ap.add_argument("--validate", action="store_true", help="also print validation numbers (writes work/<stem>/bodies_validation.json)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from squashvid.bodies import lift

    if a.download:
        print(lift.weights_path())
        return 0
    if not a.stems:
        ap.error("give at least one stem")
    from squashvid.bodies import run

    for stem in a.stems:
        s = run.run(stem, device=a.device)
        print(stem, {p: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != "bone_m"} for p, d in s["players"].items()})
        if a.validate:
            from squashvid.bodies import validate

            validate.report(stem)
    return 0


raise SystemExit(main())

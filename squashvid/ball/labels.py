"""Ball labels: an append-only JSONL log per clip, replayed so the latest record per frame wins.

Appending (never rewriting) makes autosave crash-proof and keeps every
correction as history. Positions are full-frame pixels with pixel centres at
integers, placed at the centre of the blur streak (BlurBall's convention).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from squashvid.ball.clips import LABELS

VISIBILITY = ("visible", "occluded", "out")
SOURCES = ("user", "accepted-prelabel", "self", "cleared")


@dataclass
class Label:
    frame: int
    t: float
    x: float | None
    y: float | None
    visibility: str
    source: str = "user"
    labeller: str = "user"
    ts: float = field(default_factory=time.time)
    event: str | None = None  # "racket" for a racket contact in this frame
    prelabel: str | None = None  # model variant whose point was accepted

    def __post_init__(self):
        if self.visibility not in VISIBILITY:
            raise ValueError(f"visibility must be one of {VISIBILITY}, got {self.visibility!r}")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.visibility == "out":
            self.x = self.y = None
        elif self.x is None or self.y is None:
            raise ValueError("visible and occluded labels need a position (best guess when occluded)")


def label_path(stem: str, root: Path = LABELS) -> Path:
    return root / f"{stem}.jsonl"


def append(stem: str, label: Label, root: Path = LABELS) -> Path:
    path = label_path(stem, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(asdict(label)) + "\n")
    return path


def read_log(stem: str, root: Path = LABELS) -> list[dict]:
    path = label_path(stem, root)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn final line from a crash mid-write
    return out


def load(stem: str, root: Path = LABELS, labeller: str | None = None) -> pd.DataFrame:
    """Current labels, one row per frame (latest record wins; 'cleared' records delete)."""
    latest: dict[int, dict] = {}
    for rec in read_log(stem, root):
        if labeller and rec.get("labeller") != labeller:
            continue
        latest[int(rec["frame"])] = rec
    rows = [r for r in latest.values() if r.get("source") != "cleared"]
    cols = ["frame", "t", "x", "y", "visibility", "source", "labeller", "ts", "event", "prelabel"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values("frame").reset_index(drop=True)


def load_all(root: Path = LABELS, labeller: str | None = None) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*.jsonl")):
        df = load(path.stem, root, labeller)
        if len(df):
            frames.append(df.assign(stem=path.stem))
    # Rebuilt from records: concatenating frames whose event/prelabel columns are all empty is deprecated in pandas.
    return pd.DataFrame([r for f in frames for r in f.to_dict("records")]) if frames else pd.DataFrame()

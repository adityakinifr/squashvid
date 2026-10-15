"""Run the audio hit-timing stream on one clip.

    python -m squashvid.audio ~/Downloads/deepti-originals/IMG_4091.MOV   # writes work/IMG_4091/audio_*.parquet + plot
    python -m squashvid.audio ~/Downloads/<stem>.mp4                  # the WhatsApp copies work the same way
    python -m squashvid.audio ~/Downloads/<stem>.mp4 --plot-start 75  # choose the 30 s plot window
    python -m squashvid.audio ~/Downloads/<stem>.mp4 --work work/scratch
    python -m squashvid.audio <video> --retype                         # after editing classify.py: skip detection

If work/<stem>/hit_labels.csv exists, onsets are also scored against it and the
result is written to work/<stem>/audio_eval.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.path.expanduser("~/code/squashvid"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m squashvid.audio")
    p.add_argument("video")
    p.add_argument("--work", default=None, help="output directory (default work/<stem>)")
    p.add_argument("--plot-start", type=float, default=None, help="start of the 30 s plot window (default: busiest rally window)")
    p.add_argument("--offset-ms", type=float, default=None, help="audio minus video offset for scoring (default: estimated from high-confidence labels)")
    p.add_argument("--band-hi", type=float, default=None, help="upper edge of the detection band in Hz (default: from the codec's usable bandwidth)")
    p.add_argument("--retype", action="store_true", help="reuse the saved audio_onsets.parquet: re-run typing, rallies, scoring and the plot only")
    a = p.parse_args(argv)

    import numpy as np
    import pandas as pd
    import soundfile as sf

    from squashvid.audio import classify, evaluate, onsets, plot, rallies
    from squashvid.media.video import extract_audio

    video = Path(a.video).expanduser()
    work = Path(a.work) if a.work else ROOT / "work" / video.stem
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "audio_stereo.wav"
    if not wav.exists():
        extract_audio(video, wav, rate=None, mono=False)
    st, sr = sf.read(wav)
    st = st if st.ndim == 2 else np.stack([st, st], axis=1)
    y = st.mean(axis=1)

    t0 = time.time()
    params = onsets.band_for(y, sr) if a.band_hi is None else onsets.OnsetParams(band=(2000.0, a.band_hi))
    saved = work / "audio_onsets.parquet"
    if a.retype and saved.exists():
        df = pd.read_parquet(saved).drop(columns=["kind", "rally"], errors="ignore")
    else:
        df = onsets.detect(y, sr, params, stereo=st)
    df = classify.classify(df)
    hits = df.loc[df.kind == "racket", "t"].to_numpy()
    rl = rallies.segment(df.loc[classify.rally_impacts(df), "t"].to_numpy())
    df["rally"] = rallies.assign(df, rl)
    df = classify.mark_other_court(df)
    rl["n_onsets"] = [int((df.rally == r).sum()) for r in rl.rally]
    rl["n_racket"] = [int(((df.rally == r) & (df.kind == "racket")).sum()) for r in rl.rally]
    df.to_parquet(work / "audio_onsets.parquet", index=False)
    rl.to_parquet(work / "audio_rallies.parquet", index=False)
    print(f"{video.stem}: {len(df)} onsets, {len(hits)} racket candidates, {len(rl)} rallies in {time.time() - t0:.1f}s "
          f"({len(y) / sr:.0f}s audio at {sr} Hz, band {params.band[0]:.0f}-{params.band[1]:.0f} Hz)")
    print(df.kind.value_counts().to_string())

    labels_path = work / "hit_labels.csv"
    labels, offset = None, 0.0
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        report = evaluate.evaluate_clip(df, rl, labels, offset_ms=a.offset_ms)
        offset = report["offset_used_ms"] / 1e3
        labels["t_audio"] = labels["t"] + labels["travel_s"] + offset
        (work / "audio_eval.json").write_text(json.dumps(report, indent=1, default=float))
        print(json.dumps({k: v for k, v in report.items() if k != "per_label"}, indent=1, default=float))

    start = a.plot_start if a.plot_start is not None else plot.busiest_window(rl, len(y) / sr)
    png = plot.plot_window(y, sr, df, rl, start, work / "audio_plot.png", labels=labels, p=params,
                           title=f"{video.stem}  {start:.0f}-{start + 30:.0f} s  (marker size = strength; ticks at the bottom = visual labels)")
    print("wrote", work / "audio_onsets.parquet", work / "audio_rallies.parquet", png)
    return 0


if __name__ == "__main__":
    sys.exit(main())

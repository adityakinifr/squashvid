"""Speed of each detector on this machine: network-only ms/frame on MPS and CoreML, and a sustained end-to-end run.

    python -m squashvid.ball.bench nets blurball@default blurball@576x1024 --coreml
    python -m squashvid.ball.bench sustained blurball@576x1024 IMG_4091 --minutes 15 --coreml

ms/frame divides the time per network call by the frames each call labels:
WASB tiles three frames per call, BlurBall and ensembled TrackNetV3 make one
call per frame (three and eight overlapping predictions pooled), TOTNet one
call per frame. CoreML models are traced at a fixed input shape, converted as
fp16 ML programs and run with compute units ALL (the Neural Engine where the
ops allow) and CPU_AND_GPU. Every result records the load average, because the
laptop is shared and a busy CPU or GPU inflates every number here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from squashvid.ball import registry
from squashvid.ball.clips import WORK, is_original, video

BENCH = WORK / "bench"


def frames_per_call(det) -> float:
    clips = det.plan(1000, 1999, 0, 10**6)
    return 1000 / len(clips)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def input_shape(det, batch: int = 1) -> tuple[int, ...]:
    h, w = det.geom.size
    dummy = [np.zeros((3, h, w), np.float32)] * det.frames_in
    if hasattr(det, "median") and det.median is None:  # TrackNetV3's background frame is set per run by prepare()
        det.median = dummy[0]
    return (batch, *det.tensor(dummy).shape)


def bench_torch(spec: str, frame_hw=(1080, 1920), half: bool = False, iters: int = 10, batch: int = 1, device: str | None = None) -> dict:
    det = registry.build(spec, frame_hw=frame_hw, half=half, device=device)
    x = torch.randn(input_shape(det, batch), device=det.device, dtype=det.dtype)
    with torch.inference_mode():
        for _ in range(2):
            det.forward(x)
        _sync(det.device)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            det.forward(x)
            _sync(det.device)
            times.append(time.perf_counter() - t0)
    call_ms = 1e3 * float(np.median(times)) / batch
    fpc = frames_per_call(det)
    return {"spec": spec, "backend": f"torch-{det.device.type}", "dtype": "fp16" if half else "fp32", "input": list(det.geom.size), "batch": batch,
            "ms_per_call": round(call_ms, 1), "frames_per_call": round(fpc, 2), "ms_per_frame": round(call_ms / fpc, 1),
            "iters": iters, "load": [round(v, 1) for v in os.getloadavg()]}


def export_coreml(spec: str, frame_hw=(1080, 1920)) -> tuple[Path, object]:
    from squashvid.ball.coreml import coremltools

    ct = coremltools()

    det = registry.build(spec, frame_hw=frame_hw, device="cpu")
    shape = input_shape(det)

    class Wrap(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d
            self.model = d.model

        def forward(self, x):
            return self.d.forward(x)

    traced = torch.jit.trace(Wrap(det).eval(), torch.randn(shape))
    ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=shape)], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14)
    path = BENCH / f"{spec.replace('@', '_')}.mlpackage"
    path.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(path))
    return path, det


def bench_coreml(spec: str, frame_hw=(1080, 1920), units: tuple[str, ...] = ("ALL", "CPU_AND_GPU"), iters: int = 10) -> list[dict]:
    from squashvid.ball.coreml import coremltools

    ct = coremltools()

    path, det = export_coreml(spec, frame_hw)
    shape = input_shape(det)
    x = {"x": np.random.randn(*shape).astype(np.float32)}
    out = []
    for u in units:
        t0 = time.perf_counter()
        m = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, u))
        load_s = time.perf_counter() - t0
        for _ in range(2):
            m.predict(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            m.predict(x)
            times.append(time.perf_counter() - t0)
        call_ms = 1e3 * float(np.median(times))
        fpc = frames_per_call(det)
        out.append({"spec": spec, "backend": f"coreml-{u}", "dtype": "fp16", "input": list(det.geom.size), "batch": 1, "ms_per_call": round(call_ms, 1),
                    "frames_per_call": round(fpc, 2), "ms_per_frame": round(call_ms / fpc, 1), "load_model_s": round(load_s, 1), "iters": iters,
                    "load": [round(v, 1) for v in os.getloadavg()]})
    return out


def sustained(spec: str, stem: str, minutes: float = 15.0, chunk_s: float = 10.0, log: Path | None = None,
              coreml: str | None = None) -> Path:
    """Run the full pipeline (decode incl. HLG->SDR, network, decode heatmaps, track) over the clip, looping, logging fps
    at least once a minute (each record covers whole 10 s chunks, so intervals run a little over 60 s)."""
    v = video(stem)
    hw = (1080, 1920) if is_original(stem) else (576, 1024)
    det = registry.build(spec, frame_hw=hw, **({"device": "cpu"} if coreml else {}))
    if coreml:
        from squashvid.ball import coreml as cm

        cm.attach(det, coreml)
    backend = getattr(det, "backend", f"torch-{det.device.type}")
    log = log or BENCH / f"sustained_{spec.replace('@', '_')}_{stem}{'_coreml' if coreml else ''}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    dur = float(v.timestamps[-1])
    t_start = time.perf_counter()
    minute, frames, t_min = 0, 0, t_start
    pos = 0.0
    def record(now: float, **extra) -> None:
        st = det.stats
        rec = {"minute": minute + 1, "backend": backend, "frames": frames, "fps": round(frames / (now - t_min), 2), "elapsed_s": round(now - t_start, 1),
               "load": [round(x, 1) for x in os.getloadavg()], "last_chunk": {"decode_s": round(st.decode_s, 2), "infer_s": round(st.infer_s, 2), "post_s": round(st.post_s, 2), "frames": st.frames}, **extra}
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(json.dumps(rec), flush=True)

    with log.open("w") as fh:
        while time.perf_counter() - t_start < 60 * minutes:
            end = min(pos + chunk_s, dur)
            df = det.run(v, pos, end)
            frames += len(df)
            pos = 0.0 if end >= dur else end
            now = time.perf_counter()
            if now - t_min >= 60:
                record(now)
                minute, frames, t_min = minute + 1, 0, now
        if frames:  # the chunks after the last full interval, so the log accounts for every frame processed
            record(time.perf_counter(), partial=True)
    return log


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m squashvid.ball.bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("nets")
    n.add_argument("specs", nargs="+")
    n.add_argument("--frame", default="1080x1920", help="frame HxW the sizes refer to")
    n.add_argument("--half", action="store_true")
    n.add_argument("--coreml", action="store_true")
    n.add_argument("--iters", type=int, default=10)
    s = sub.add_parser("sustained")
    s.add_argument("spec")
    s.add_argument("stem")
    s.add_argument("--minutes", type=float, default=15.0)
    s.add_argument("--coreml", nargs="?", const="CPU_AND_GPU", default=None, metavar="UNITS", help="network as a Core ML program (coreml.py)")
    a = p.parse_args(argv)
    BENCH.mkdir(parents=True, exist_ok=True)
    if a.cmd == "nets":
        hw = tuple(int(v) for v in a.frame.split("x"))
        with (BENCH / "nets.jsonl").open("a") as fh:
            for spec in a.specs:
                results = [bench_torch(spec, hw, a.half, a.iters)]
                if a.coreml:
                    try:
                        results += bench_coreml(spec, hw, iters=a.iters)
                    except Exception as e:  # conversion failures are a result too
                        results.append({"spec": spec, "backend": "coreml", "error": f"{type(e).__name__}: {e}"[:300]})
                for r in results:
                    r["ts"] = time.time()
                    fh.write(json.dumps(r) + "\n")
                    print(json.dumps(r), flush=True)
    else:
        sustained(a.spec, a.stem, a.minutes, coreml=a.coreml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

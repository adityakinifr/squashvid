"""Build any detector variant from a short spec string.

Spec: `<model>@<size>`, where model is one of MODELS and size is `default`
(288x512, what the papers train at), `native` (1:1 pixels, 576x1024 for the
full frame) or `x2` (2x upscale, meant for crops). A crop is passed as `roi`;
predict.py derives the front-wall crop from the camera pose.
"""

from __future__ import annotations

from typing import Callable

from squashvid.ball.base import Detector


def _tracknet(**kw) -> Detector:
    from squashvid.ball.tracknet import TrackNetV3

    return TrackNetV3(**kw)


def _tracknet_inp(**kw) -> Detector:
    from squashvid.ball.tracknet import TrackNetV3

    return TrackNetV3(inpaint=True, **kw)


def _tracknet_fast(**kw) -> Detector:
    from squashvid.ball.tracknet import TrackNetV3

    return TrackNetV3(ensemble=False, **kw)


def _hrnet(ckpt: str) -> Callable[..., Detector]:
    def make(**kw) -> Detector:
        from squashvid.ball.wasb import HRNetDetector

        return HRNetDetector(ckpt, **kw)

    return make


def _totnet(ckpt: str) -> Callable[..., Detector]:
    def make(**kw) -> Detector:
        from squashvid.ball.totnet import TOTNet

        return TOTNet(ckpt, **kw)

    return make


MODELS: dict[str, Callable[..., Detector]] = {
    "tracknetv3": _tracknet,
    "tracknetv3-inp": _tracknet_inp,
    "tracknetv3-fast": _tracknet_fast,
    **{f"wasb-{s}": _hrnet(f"wasb-{s}") for s in ("tennis", "badminton", "soccer", "volleyball", "basketball", "tt")},
    "blurball": _hrnet("blurball"),
    **{f"totnet-{s}": _totnet(s) for s in ("tennis", "badminton", "tt")},
}


FINETUNED_THRESHOLD = 0.5
FINETUNED = "blurball-ft:"  # blurball-ft:<run>@576x1024 loads work/ball_ft/<run>/<run>.pt (finetune.py)


def finetuned(run: str) -> Callable[..., Detector]:
    """BlurBall with the weights of a fine-tune run; its outputs are named blurball-ft-<run>@<size>."""

    def make(**kw) -> Detector:
        from squashvid.ball.clips import WORK
        from squashvid.ball.wasb import HRNetDetector

        path = WORK / "ball_ft" / run / f"{run}.pt"
        if not path.exists():
            raise FileNotFoundError(f"no fine-tuned weights at {path}")
        det = HRNetDetector("blurball", weights=str(path), name=f"blurball-ft-{run}", **kw)
        if kw.get("threshold") is None:
            det.threshold = FINETUNED_THRESHOLD  # the held-out sweep's best; the pretrained 0.7 is too strict for these heatmaps
        return det

    return make


def parse(spec: str) -> tuple[str, str]:
    model, _, size = spec.partition("@")
    if model.startswith(FINETUNED):
        return model, size or "default"
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
    return model, size or "default"


def build(spec: str, roi: tuple[int, int, int, int] | None = None, **kw) -> Detector:
    model, size = parse(spec)
    make = finetuned(model[len(FINETUNED):]) if model.startswith(FINETUNED) else MODELS[model]
    return make(size=size, roi=roi, **kw)

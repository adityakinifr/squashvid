"""Load third-party model code and checkpoints without putting their repos on sys.path.

TrackNetV3, WASB, BlurBall and TOTNet each ship top-level packages called
`model`, `models` or `utils`, so importing them through sys.path would collide.
Each module is loaded from its file under a private name instead, and every
checkpoint goes through `torch.load(weights_only=True)`.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType

import torch

from squashvid.ball.clips import THIRD_PARTY


def load_module(rel_path: str, name: str) -> ModuleType:
    """Import third_party/<rel_path> as a module called `name` (cached in sys.modules)."""
    key = f"_sv_vendor_{name}"
    if key in sys.modules:
        return sys.modules[key]
    path = THIRD_PARTY / rel_path
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: clone the repo into third_party/ (see squashvid/ball/README in the report)")
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


class AttrDict(dict):
    """dict with attribute access, enough for HRNet configs that mix cfg['x'] and cfg.x."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    @classmethod
    def wrap(cls, obj):
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj


def _easydict_shim() -> type:
    """A stand-in for easydict.EasyDict so TOTNet checkpoints load under weights_only=True.

    TOTNet pickles its training config as an EasyDict next to the weights. Allow-listing a
    plain dict subclass under that name keeps the safe unpickler (no arbitrary code) while
    letting the config deserialise.
    """
    if "easydict" in sys.modules and hasattr(sys.modules["easydict"], "EasyDict"):
        return sys.modules["easydict"].EasyDict
    mod = types.ModuleType("easydict")

    class EasyDict(dict):
        def __setstate__(self, state):
            self.update(state)

    EasyDict.__module__ = "easydict"
    EasyDict.__qualname__ = "EasyDict"  # the safe unpickler matches module + qualname
    mod.EasyDict = EasyDict
    sys.modules["easydict"] = mod
    return EasyDict


def load_checkpoint(path: str | Path) -> dict:
    with torch.serialization.safe_globals([_easydict_shim()]):
        return torch.load(str(path), map_location="cpu", weights_only=True)


def strip_prefix(state: dict, prefix: str = "module.") -> dict:
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}

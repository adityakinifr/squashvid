"""What the 3D figures wear and which hand holds the racket: colours read from the video at the pose keypoints."""

import numpy as np
import pandas as pd

from squashvid.viewer import appearance as A
from squashvid.viewer import figure_model as F

H, W = 240, 320
SHIRT, SHORTS, WALL = (40, 90, 200), (20, 120, 40), (210, 210, 210)


def pose():
    """COCO keypoints (pixels) of a standing figure facing the camera, all sure."""
    k = np.zeros((17, 2))
    k[A.L_SH], k[A.R_SH] = (180, 60), (140, 60)
    k[A.L_HIP], k[A.R_HIP] = (175, 130), (145, 130)
    k[A.L_KN], k[A.R_KN] = (175, 180), (145, 180)
    k[A.L_WR], k[A.R_WR] = (200, 120), (120, 120)
    return k, np.ones(17)


def frame(k):
    """A BGR frame: wall, a shirt over the torso and shorts over the thighs, like the figure in `pose`."""
    import cv2

    img = np.zeros((H, W, 3), np.uint8)
    img[:] = WALL
    cv2.fillPoly(img, [np.round(k[[A.L_SH, A.R_SH, A.R_HIP, A.L_HIP]]).astype(np.int32)], SHIRT)
    cv2.rectangle(img, (138, 125), (182, 172), SHORTS, -1)
    return img[..., ::-1].copy()


def players(frames, player="A"):
    k, s = pose()
    return pd.DataFrame({"frame": frames, "player": player, "kpts": [k.ravel()] * len(frames), "kscore": [s] * len(frames)})


def test_the_regions_stay_on_the_clothes():
    k, s = pose()
    rgb = frame(k)[..., ::-1]
    reg = A.regions(k, s, (H, W))
    assert len(reg["shirt"]) == 1 and len(reg["shorts"]) == 2
    assert (rgb[reg["shirt"][0]] == SHIRT).all()
    for m in reg["shorts"]:
        assert (rgb[m] == SHORTS).all()


def test_no_regions_without_a_sure_torso():
    k, s = pose()
    s[A.L_HIP] = 0.2
    assert all(not v for v in A.regions(k, s, (H, W)).values())


def test_no_shorts_without_the_knees():
    k, s = pose()
    s[A.R_KN] = 0.2
    reg = A.regions(k, s, (H, W))
    assert reg["shirt"] and not reg["shorts"]


def test_the_estimate_keeps_the_colours_order_and_lifts_a_dim_frame_to_white():
    k, _ = pose()
    frames = {f: frame(k) for f in range(A.MIN_FRAMES)}
    look = A.estimate(frames, players(list(frames)), "A")
    assert set(look) == {"shirt", "shorts"}
    shirt, shorts = np.array(look["shirt"]), np.array(look["shorts"])
    assert shirt.argmax() == 2 and shorts.argmax() == 1  # blue stays blue, green stays green
    assert (shirt >= SHIRT).all() and (shorts >= SHORTS).all()  # walls at 210 are lifted to WHITE


def test_too_few_frames_say_nothing():
    k, _ = pose()
    frames = {f: frame(k) for f in range(A.MIN_FRAMES - 1)}
    assert A.estimate(frames, players(list(frames)), "A") == {}


def test_racket_hand_follows_the_nearer_wrist_and_defaults_to_right(tmp_path):
    k, s = pose()
    n = A.MIN_VOTES + 2
    rows = [{"frame": f, "player": p, "kpts": k.ravel(), "kscore": s} for f in range(n) for p in ("A", "B")]
    pd.DataFrame(rows).to_parquet(tmp_path / "players.parquet")
    lw, rw = k[A.L_WR], k[A.R_WR]
    ev = [{"t": f / 30, "frame": float(f), "kind": "racket", "u": lw[0] + 5, "v": lw[1], "player": "A"} for f in range(n)]
    ev += [{"t": f / 30, "frame": float(f), "kind": "racket", "u": rw[0] - 5, "v": rw[1], "player": "B"} for f in range(3)]
    pd.DataFrame(ev).to_parquet(tmp_path / "ball_events.parquet")
    assert A.racket_hand(tmp_path) == {"A": "left", "B": "right"}  # B: right, but with too few votes it is the default anyway
    assert A.racket_hand(tmp_path / "missing") == {"A": "right", "B": "right"}


def test_the_figure_masks_mark_the_shirt_shorts_and_skin_and_leave_the_sandals_and_hair():
    n = 200
    body = np.full((n, n, 3), [0.55, 0.4, 0.33])  # skin: brown, saturated
    body[10:50] = [0.3, 0.29, 0.26]  # shorts: dull, in the top band
    body[80:150, 40:160] = [0.9, 0.45, 0.2]  # shirt: bright, saturated orange
    body[180:, :20] = [0.9, 0.45, 0.2]  # a palm in a bottom corner is orange too, but skin
    body[170:, 100:140] = [0.2, 0.2, 0.21]  # a sandal: dark grey
    head = np.full((n, n, 3), [0.55, 0.4, 0.33])
    head[:60] = [0.13, 0.11, 0.1]  # hair: dark, at the top
    mb, mh = (m // F.LABEL_STEP for m in F.masks(body, head))
    assert (mb[90:140, 50:150] == F.SHIRT).all()
    assert (mb[15:45, 20:180] == F.SHORTS).all()
    assert (mb[60:70, 10:190] == F.SKIN).all() and (mb[182:192, 8:15] == F.SKIN).all()
    assert (mb[178:, 108:132] == 0).all()
    assert (mh[10:45, 10:190] == 0).all() and (mh[80:110, 60:140] == F.SKIN).all()


def test_racket_hand_without_located_impacts_is_the_default(tmp_path):
    k, s = pose()
    pd.DataFrame([{"frame": 0, "player": "A", "kpts": k.ravel(), "kscore": s}]).to_parquet(tmp_path / "players.parquet")
    pd.DataFrame({"t": [0.0], "kind": ["racket"]}).to_parquet(tmp_path / "ball_events.parquet")
    assert A.racket_hand(tmp_path) == {"A": "right", "B": "right"}

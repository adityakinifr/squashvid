"""Ball detection scoring: strict per-frame matching, region binning, threshold sweeps, agreement, pixel mapping."""

import numpy as np
import pandas as pd
import pytest

from squashvid.ball.align import best_match
from squashvid.ball.clips import copy_to_original_px
from squashvid.ball.evaluate import agreement, by_region, match, prf, sweep
from squashvid.ball.tone import Tone, match_curve

GT = pd.DataFrame(
    {
        "frame": [0, 1, 2, 3, 4, 5],
        "x": [100.0, 200.0, 300.0, np.nan, 500.0, 600.0],
        "y": [100.0, 100.0, 100.0, np.nan, 100.0, 100.0],
        "visibility": ["visible", "visible", "visible", "out", "visible", "occluded"],
    }
)
PRED = pd.DataFrame(
    {
        "frame": [0, 1, 2, 3, 4, 5],
        "x": [102.0, 230.0, np.nan, 50.0, 506.0, 601.0],
        "y": [101.0, 100.0, np.nan, 50.0, 100.0, 100.0],
        "score": [0.9, 0.8, 0.0, 0.6, 0.3, 0.9],
        "visible": [True, True, False, True, True, True],
    }
)


def test_strict_matching_outcomes():
    m = match(GT, PRED, tol=4.0).set_index("frame")
    assert m.loc[0, ["tp", "fp", "fn"]].tolist() == [1, 0, 0]  # within tolerance
    assert m.loc[1, ["tp", "fp", "fn"]].tolist() == [0, 1, 1]  # found, but in the wrong place: both errors
    assert m.loc[2, ["tp", "fp", "fn"]].tolist() == [0, 0, 1]  # missed
    assert m.loc[3, ["tp", "fp", "fn"]].tolist() == [0, 1, 0]  # ball out, model fired
    assert m.loc[4, ["tp", "fp", "fn"]].tolist() == [0, 1, 1]  # 6 px off at a 4 px tolerance
    assert m.loc[5, ["tp", "fp", "fn"]].tolist() == [0, 0, 0] and m.loc[5, "occ_hit"] == 1  # occluded: recall only


def test_wider_tolerance_only_converts_errors_to_hits():
    a, b = match(GT, PRED, 4.0), match(GT, PRED, 8.0)
    assert b.tp.sum() == a.tp.sum() + 1 and b.fn.sum() == a.fn.sum() - 1 and b.fp.sum() == a.fp.sum() - 1


def test_threshold_override_and_sweep_trade_precision_for_recall():
    m = match(GT, PRED, 8.0, threshold=0.5)
    assert not m.set_index("frame").loc[4, "pred_visible"]  # score 0.3 falls below
    s = sweep(GT, PRED, 8.0, thresholds=[0.1, 0.5, 0.95])
    assert s.recall.is_monotonic_decreasing
    assert s.tp.iloc[-1] == 0


def test_prf_edge_cases():
    assert prf(0, 0, 0)["precision"] != prf(0, 0, 0)["precision"]  # NaN when undefined
    r = prf(3, 1, 2)
    assert r["precision"] == pytest.approx(0.75) and r["recall"] == pytest.approx(0.6)


def test_region_binning_uses_label_for_misses_and_prediction_for_false_alarms():
    m = match(GT, PRED, 4.0)
    gt_reg = np.array(["front_wall", "front_wall", "floor_back", None, "air", "door_strip"], dtype=object)
    pred_reg = np.array(["front_wall", "side_walls", None, "air", "air", "door_strip"], dtype=object)
    t = by_region(m, gt_reg, pred_reg, ("front_wall", "side_walls", "floor_back", "air", "door_strip")).set_index("region")
    assert t.loc["front_wall", ["tp", "fp", "fn"]].tolist() == [1, 0, 1]
    assert t.loc["side_walls", ["tp", "fp", "fn"]].tolist() == [0, 1, 0]
    assert t.loc["air", ["tp", "fp", "fn"]].tolist() == [0, 2, 1]
    assert t.loc["door_strip", "occ_recall"] == 1.0
    assert t.loc["all", ["tp", "fp", "fn"]].tolist() == [int(m.tp.sum()), int(m.fp.sum()), int(m.fn.sum())]


def test_agreement_counts_only_close_joint_detections():
    a = pd.DataFrame({"frame": [0, 1, 2, 3], "x": [0.0, 0.0, 0.0, np.nan], "y": [0.0] * 4, "visible": [True, True, True, False]})
    b = pd.DataFrame({"frame": [0, 1, 2, 3], "x": [5.0, 50.0, np.nan, 0.0], "y": [0.0] * 4, "visible": [True, True, False, True]})
    r = agreement(a, b, tol=20.0)
    assert (r["both_visible"], r["agree"], r["either_visible"]) == (2, 1, 4)


def test_copy_pixels_map_by_pixel_centres():
    xy = copy_to_original_px(np.array([[-0.5, -0.5], [1023.5, 575.5], [511.5, 287.5]]))
    assert np.allclose(xy, [[-0.5, -0.5], [1919.5, 1079.5], [959.5, 539.5]])


def test_frame_match_picks_the_candidate_with_the_same_content():
    rng = np.random.default_rng(0)
    base = rng.normal(100, 10, (144, 256)).astype(np.float32)
    frames = []
    for k in range(3):
        f = base.copy()
        f[60:80, 40 + 20 * k : 60 + 20 * k] = 250  # a moving blob
        frames.append(f)
    copy = frames[1] * 0.8 + 20 + rng.normal(0, 2, base.shape)  # tone-shifted and noisy
    assert best_match(copy.astype(np.float32), frames)[0] == 1


def test_tone_curve_is_monotonic_and_matches_distributions():
    rng = np.random.default_rng(1)
    ref = rng.integers(20, 230, (64, 64)).astype(np.uint8)
    src = (np.clip(ref.astype(int) * 0.5 + 120, 0, 255)).astype(np.uint8)  # brighter and flatter, like HLG decoded as SDR
    lut = match_curve(src, ref)
    assert np.all(np.diff(lut.astype(int)) >= 0)
    assert abs(float(np.median(lut[src])) - float(np.median(ref))) <= 2
    img = np.dstack([src] * 3)
    assert Tone(np.stack([lut] * 3, axis=1))(img).shape == img.shape


def test_candidate_ceiling_counts_any_blob_near_a_visible_label():
    from squashvid.ball.evaluate import candidate_recall

    cands = pd.DataFrame({"frame": [0, 0, 1, 4], "x": [300.0, 101.0, 900.0, 500.0], "y": [300.0, 100.0, 900.0, 103.0], "peak": [0.9] * 4, "mass": [1.0] * 4})
    r = candidate_recall(GT, cands, tol=8.0)
    assert (r["n_visible"], r["any_within"]) == (4, 2)  # frame 0 via its second blob, frame 4; frames 1-2 have none near

"""Fine-tuning's pure pieces: target placement, coordinate scaling, augmentation and the window splits."""

import numpy as np
import pytest

from squashvid.ball import finetune as F
from squashvid.ball.base import Geometry, resolve_geometry


def centroid(hm: np.ndarray) -> tuple[float, float]:
    ys, xs = np.mgrid[0 : hm.shape[0], 0 : hm.shape[1]]
    return float((xs * hm).sum() / hm.sum()), float((ys * hm).sum() / hm.sum())


def test_label_scales_to_the_model_pixel_the_detector_maps_back_from():
    geom = resolve_geometry(F.FRAME_HW, "576x1024", None)
    for x, y in [(0.0, 0.0), (500.3, 700.7), (1919.0, 1079.0)]:
        mx, my = F.to_model(x, y)
        bx, by = geom.to_frame(mx, my)
        assert (bx, by) == pytest.approx((x, y), abs=1e-9)


def test_heatmap_peaks_at_the_labelled_point_in_model_pixels():
    # A 1080p point that lands exactly on model pixel (300, 200).
    x, y = (300 + 0.5) * 1.875 - 0.5, (200 + 0.5) * 1.875 - 0.5
    mx, my = F.to_model(x, y)
    hm = F.heatmap(F.MODEL_HW, (mx, my))
    assert hm.shape == F.MODEL_HW
    assert hm.max() == 1.0
    assert centroid(hm) == pytest.approx((300, 200), abs=1e-4)
    assert hm[200, 300] == hm.max()
    assert hm[200, 300 + 6] == 0  # beyond the target's radius


def test_subpixel_label_moves_the_target_centroid():
    hm = F.heatmap((64, 64), (20.4, 30.0))
    assert centroid(hm)[0] == pytest.approx(20.4, abs=0.1)


def test_out_frames_and_off_image_points_give_empty_targets():
    assert not F.heatmap((32, 32), None).any()
    assert not F.heatmap((32, 32), (np.nan, np.nan)).any()
    assert not F.heatmap((32, 32), (500.0, 10.0)).any()


def test_flip_moves_the_target_to_the_mirrored_pixel():
    hw = (48, 80)
    stack = np.random.default_rng(0).integers(0, 255, (3, *hw, 3), dtype=np.uint8)
    pts = [(10.0, 20.0), None, (70.5, 5.0)]
    flipped, fpts = F.hflip(stack, pts)
    assert np.array_equal(flipped[0], stack[0][:, ::-1])
    assert fpts[1] is None
    for p, q in zip(pts, fpts):
        if p is not None:
            assert F.heatmap(hw, q) == pytest.approx(F.heatmap(hw, p)[:, ::-1])
    assert fpts[0] == (hw[1] - 1 - 10.0, 20.0)


def test_affine_moves_points_with_the_pixels():
    hw = (60, 100)
    m = F.affine(1.1, (3.0, -2.0), hw)
    img = np.zeros((*hw, 3), np.uint8)
    img[30, 40] = 255
    import cv2

    warped = cv2.warpAffine(img, m, (hw[1], hw[0]), flags=cv2.INTER_NEAREST)
    px, py = F.apply_affine(m, (40.0, 30.0))
    ys, xs = np.nonzero(warped[..., 0])
    assert abs(xs.mean() - px) <= 1.0 and abs(ys.mean() - py) <= 1.0
    assert F.apply_affine(m, None) is None


def test_crop_contains_a_visible_point_when_asked():
    rng = np.random.default_rng(1)
    for _ in range(50):
        pt = (float(rng.uniform(20, 1000)), float(rng.uniform(20, 550)))
        x0, y0 = F.pick_crop(rng, F.MODEL_HW, (288, 512), [None, pt, None], p_ball=1.0)
        assert 0 <= x0 <= F.MODEL_HW[1] - 512 and 0 <= y0 <= F.MODEL_HW[0] - 288
        assert x0 <= pt[0] < x0 + 512 and y0 <= pt[1] < y0 + 288


def test_photometric_keeps_shape_and_range():
    s = np.random.default_rng(2).integers(0, 255, (3, 8, 8, 3), dtype=np.uint8)
    out = F.photometric(s, 0.1, 1.2, 0.9)
    assert out.shape == s.shape and out.dtype == np.uint8
    assert np.array_equal(F.photometric(s, 0.0, 1.0, 1.0), s)


def test_validation_tail_is_contiguous_and_separated_from_training():
    frames = np.arange(1000, 1465)
    train, val = F.val_split(frames)
    assert val[-1] == frames[-1] and np.array_equal(val, np.arange(val[0], val[-1] + 1))
    assert train.max() + 2 < val.min()  # no 3-frame stack spans both
    assert len(val) / len(frames) == pytest.approx(F.VAL_FRAC, abs=0.01)


def test_folds_never_share_a_window_between_train_and_test():
    ids = ["IMG_4091-r12", "IMG_4091-r16", "IMG_4089-r11", "IMG_4089-r3", "IMG_4090-r18"]
    folds = F.make_folds(ids)
    held = [w for _, test in folds.values() for w in test]
    assert sorted(held) == sorted(ids)  # every window is tested exactly once
    for train, test in folds.values():
        assert not set(train) & set(test)
        assert set(train) | set(test) == set(ids)


def test_a_sparsely_labelled_window_takes_no_part():
    counts = {"IMG_4090-r18": 225, "IMG_4090-r9": 10, "IMG_4091-r12": 464}
    ids = F.eligible(counts)
    assert "IMG_4090-r9" not in ids
    for train, test in F.make_folds(ids).values():
        assert "IMG_4090-r9" not in train + test
    assert F.eligible(counts, min_labels=5)[0] == "IMG_4090-r18"  # the cut is the argument, not a constant buried elsewhere


def test_stack_items_skip_validation_frames_and_unlabelled_stacks():
    labels = {f: ("out", np.nan, np.nan) for f in range(10, 30)}
    wc = F.WindowCache("w", "IMG_4091", 8, 31, labels)
    train, val = F.val_split(np.array(sorted(labels)))
    items = F.stack_items({"w": wc}, {"w": train})
    allowed = set(train.tolist())
    for _, s in items:
        used = [f for f in range(s, s + 3) if f in labels]
        assert used and all(f in allowed for f in used)
        assert not any(f in set(val.tolist()) for f in range(s, s + 3))


def test_loss_ignores_masked_frames_and_rewards_the_right_peak():
    import torch

    t = torch.zeros(1, 3, 16, 16)
    t[0, 0, 8, 8] = 1.0
    good = torch.full((1, 3, 16, 16), -8.0)
    good[0, 0, 8, 8] = 8.0
    bad = torch.full((1, 3, 16, 16), -8.0)
    mask = torch.tensor([[1.0, 0.0, 0.0]])
    assert F.wbce(good, t, mask) < F.wbce(bad, t, mask)
    wrong_elsewhere = good.clone()
    wrong_elsewhere[0, 1:, 3, 3] = 8.0  # false peaks on masked frames do not count
    assert F.wbce(wrong_elsewhere, t, mask) == pytest.approx(float(F.wbce(good, t, mask)))


def test_geometry_scale_is_what_training_uses():
    g = Geometry((0, 0, 1920, 1080), (576, 1024))
    assert g.scale == pytest.approx(F.model_scale())

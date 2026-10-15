# Bodies: 3D skeletons and figures

Lifts each player's 2D COCO-17 keypoints to a metric 3D skeleton in the court frame (x across, y back wall to front wall, z up), writes it to `work/<stem>/bodies.parquet`, and the viewer draws a solid, lit figure from it. It is an estimate: the lifter guesses depth, nothing here is measured in 3D.

## Commands

```bash
source .venv/bin/activate
python -I -m squashvid.bodies IMG_4091 IMG_4090 IMG_4089 --validate   # ~4 min per 9k-frame clip on CPU
python -I -m squashvid.viewer export --all                            # adds <p>_body (int16 mm, n x 17 x 3) and <p>_bconf (u8) arrays
python -I -m squashvid.bodies --download                              # fetch the weights only
```

Outputs per clip: `bodies.parquet` (frame, t, player, state 1 seen / 2 bridged, conf 0..1, joints = 17 x 3 float32 court metres in H36M order), `bodies_summary.json` (per player fitted size, camera-axis choice, raw bone CV, frame counts), `bodies_validation.json` (the numbers below). Tests: `python -I -m pytest -q tests/test_bodies.py` (synthetic, never touches `work/`).

## Method

1. **Lifter**: MotionBERT (Apache-2.0), fine-tuned H36M 3D-pose checkpoint `FT_MB_release_MB_ft_h36m` from https://huggingface.co/walterzhu/MotionBERT (public, no login), kept in `third_party/weights/lifters/` (git-ignored). The DSTformer is vendored trimmed in `squashvid/bodies/dstformer.py` with identical parameter names (loads strictly). Chosen over VideoPose3D because it is the stronger model (about 39 mm vs 46 mm MPJPE on H36M) and takes confidence-weighted 2D input; only CPU is needed. Windows of 243 frames, 121-frame stride, triangular blending, horizontal-flip averaging.
2. **Input**: COCO-17 mapped to H36M-17 (`skeleton.py`: pelvis = hip midpoint, thorax = shoulder midpoint, spine = their mean, nose = neck/"head", head top extrapolated from the ears or nose), keypoints below score 0.3 treated as unseen, then MotionBERT's own normalisation (window bounding box to [-1, 1]).
3. **Gaps**: a frame needs both hips, both shoulders and the foot point. Gaps of at most 0.5 s are bridged by linear interpolation of the 2D joints before lifting (state 2, confidence halved); anything longer splits the track, and tracks under 0.3 s are dropped. No pose is produced across a long gap; the viewer falls back to the cylinder there.
4. **Bone directions only**: the lifter's output is smoothed (Savitzky-Golay 7/2) and only its bone *directions* are kept. Each player gets one set of bone lengths for the whole clip (median proportions, left/right averaged) and the skeleton is rebuilt from the pelvis down the kinematic tree. Bone length is therefore constant (CV 0) by construction.
5. **Court frame**: lifter axes (x right, y down, z away) are rotated by the calibrated camera. Whether the lifter's frame is the real camera frame or a level one is fitted per clip by reprojection (alpha 0, 0.5, 1 of the camera pitch); all six player-clips chose alpha = 1, the real camera axes.
6. **Size**: the bone-chain total (metres) is the single scale parameter, fitted per player and clip to minimise the median pixel distance between the reprojected joints and the detected keypoints. It is not assumed: the fitted statures (p90 of head-top height) are 1.36 m for A and 1.30 m for B in all three clips.
7. **Placement**: the ankle midpoint sits over the smoothed court foot point (the pelvis where the foot point itself came from the hips), and the lowest ankle is at the ankle-keypoint height `ANKLE_Z` = 0.08 m (`players/geometry.py`), so the sole is on the floor. The floor offset is smoothed (SG 9/2) and never allowed below the floor.
8. **Confidence**: mean score of the core joints (hips, knees, ankles, thorax, shoulders), halved in bridged frames, interpolated across the bridge. The viewer draws a figure above 0.25.

## Validation

Detector = the RTMPose COCO keypoints used as input. "Flat" = the viewer's previous lift (each keypoint ray meets the vertical plane through the floor point facing the camera), reimplemented in `validate.py`. Frames: seen frames only (bridged frames excluded). Per clip IMG_4091 / IMG_4090 / IMG_4089, players A and B; the ranges below span all six.

| Check | New (bodies) | Flat lift |
| --- | --- | --- |
| Reprojection error vs detected keypoints, 13 shared joints (px, 1080p) | median 8.3 to 12.1, p90 24 to 33; per-frame median 3.7 to 4.1 % of box height | 0 by construction (every point is on its keypoint ray), which is why it is flat |
| Bone-length CV, median over bones | 0 (fixed lengths); the raw lifter before fixing had median 0.085 to 0.098, max 0.14 to 0.16 | median 0.21 to 0.24, max 0.60 to 0.80 |
| Lowest ankle vs floor plane | median 0.000 m, p90 0.003 m, never below | lowest ankle more than 8 cm under the floor plane in 6.5 to 10 % of frames (then clamped) |
| Higher foot above the ankle plane | median 0.033 to 0.043 m, p90 0.11 to 0.12 m | not defined |
| Both feet over 5 cm up (float) | 0 % by construction (one foot is always grounded; a real jump would be flattened) | n/a |
| Ankle y (toward the front wall) vs the ankle keypoint's own ray meeting the ankle plane (m), median / p90 | 0.074 to 0.095 / 0.26 to 0.30 | 0.095 to 0.113 / 0.29 to 0.33 |
| Ankle x vs the same ray, median | 0.026 to 0.030 m | n/a |
| Front/back order of the two feet when the rays differ by more than 0.3 m (2,850 to 5,440 frames per player-clip) | right in 82.6 to 88.0 % | right in 63 to 70 % |
| Hit moments with a foot gap of at least 0.3 m (lunges or split steps; 18 to 38 per player-clip, 163 over all six): lead foot further toward the front wall than the hip | 128 of 163 (79 %) | 89 of 163 (55 %) |

Reading it: the figure follows the picture to about 4 % of the player's height, keeps believable proportions, stands on the floor, and gets the depth order of the feet right five times in six. The independent depth test uses the ankle keypoints through the camera, but the foot point already came from those ankles, so it checks the lifter's front/back separation of the feet, not the whole body. The hit-moment rows (`hit_moments_largest_foot_gap` in the JSON) show e.g. t = 226.16 s, a 1.1 m foot gap, lead foot 0.48 m and lead knee 0.28 m ahead of the hip.

## Rendering (`squashvid/viewer/static/view3d.js`)

A Mixamo-rigged humanoid (`vendor/models/Xbot.glb`, three.js example model, personal use) is loaded on demand with the vendored r128 `GLTFLoader`. Each frame the hips get the pelvis position and heading, three spine bones follow the hip and shoulder lines, the neck and head follow thorax to head top and the nose, and each limb bone is swung from its rest direction to the measured joint-to-joint direction (twist neutral). The rig is scaled from the measured leg and trunk lengths and lifted so its ankles stand at rest height. Vertex colours come from skin weights: shirt and upper sleeves in the player colour (`V.COLORS`), dark shorts, skin, white shoes. If the model cannot load, or until it has, a shaped capsule figure (tapered limbs, chest, belly and pelvis ellipsoids, neck, head, hands, shoes) with the same interface is used; the cylinder remains for frames without a 3D pose, and the old flat lift for clips without `bodies.parquet`. Lights: hemisphere, directional and a little ambient; the court and ball stay unlit. `?view=behind|side|top` selects the camera preset on load.

## Known limits

- Depth is the lifter's guess from a single camera behind the glass: a lunge toward the front wall is foreshortened, so lead-knee and hip depth are less reliable than the image-plane pose. 21 % of hit moments with a large foot gap still show the lead foot behind the hip.
- MotionBERT was trained on adult H36M proportions; bone proportions are the lifter's, only their overall size is fitted. A left/right swap in the 2D detector flips the figure's limbs.
- The lowest ankle is forced onto the floor every frame, so jumps and fully airborne frames are flattened.
- Both players are fitted with one constant size per clip; reprojection error is 3.7 to 4.1 % of height, largest for fast arm swings and when limbs are occluded (low-score keypoints were interpolated).
- The rig is a generic mannequin: the pelvis folds look lumpy at deep hip flexion, shoulders crumple at large arm swings, there is no racket, facial detail, hair or hand posing, and bone twist (forearm rotation, head yaw beyond the nose cue) is neutral.
- Segments are lifted per player independently; the two players are never constrained against each other (they can interpenetrate).
- Weights (170 MB) and the cloned MotionBERT repo live under `third_party/` (git-ignored); the code does not need the clone, only `motionbert_ft_h36m.bin`.

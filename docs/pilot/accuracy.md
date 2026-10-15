# Pilot accuracy sheet

Measured on court 5 from Deepti's iPhone originals (three clips, 1920x1080,
36,386 frames). The WhatsApp copies of the same clips (1024x576, 34,325
frames) are kept as the comparison: they show what compression costs. M3 Pro.
Each number replaces an estimate ([E]) in the research report. The "Status"
column is the pilot's go/no-go call.

| Clip | Original | WhatsApp copy |
|---|---|---|
| 1 | IMG_4091.MOV, 291.2 s | wa_3A16D3EC0BAFDCE1C46D.mp4, 291.2 s |
| 2 | IMG_4089.MOV, 601.7 s | wa_3A644526AAA5409BE3EF.mp4, 533.0 s (stops early) |
| 3 | IMG_4090.MOV, 320.9 s | wa_3AE908EFCE489E9F342F.mp4, 320.9 s |

The originals are HEVC 10-bit HLG at 8.5 Mb/s; the copies are H.264 Baseline
at 0.66 Mb/s. Both are about 30 fps. Audio starts at the same instant in both,
with no drift. WhatsApp resampled the iPhone's variable frame rate to a
constant 29.97 fps, so frame *k* of a copy is not frame *k* of the original:
the same picture is 0–33 ms apart and the gap drifts. Results move between the
two by time, with one frame of tolerance.

The originals are HDR (HLG with BT.2020 colours). Decoded naively they come
out bright, flat and half as saturated as the copies. `media.video` now maps
them to SDR by default. It undoes the HLG curve, converts the colours to
BT.709 and applies a fixed gain fitted to iOS's own export. Across the
three clips that matches the copies to about 4 grey levels on average, where
the naive decode is about 34 levels off. The player results below ran on the naive
decode, which predates the conversion. The ball results were re-run on it, and
it equals the ball stream's earlier histogram curve for the best detector. Court registration is unaffected: refitting
28 frames on converted frames moved the court by 0.07 px median.

Ground truth is thin except for the court and the ball. Hit labels were made
by Claude agents looking at the frames, not by a person; the ball labels (section
4) are the user's own hand labels. Nothing yet measures foot positions against a
known mark. Those come from the labelling
app and the Track B session.

## 1. Court calibration and camera registration

Pixel errors are in each video's own pixels: one original pixel is 0.53 of a
WhatsApp pixel. The target, 2 px at 1024 wide, is 3.75 px on the originals.
The centimetre table below needs no conversion.

| Measure | Originals (1080p) | WhatsApp copies | Target | Status |
|---|---|---|---|---|
| Court width (fitted) | **6.138 m** (6.135–6.145 across refits) | 6.147 m (6.146–6.156) | — | Found; the WSF standard is 6.40 m |
| Court length (fitted) | 9.91 m (9.89–9.92 across clip refits) | 9.94 m | — | Open: weakly observable, tape-measure in Track B |
| Service boxes | 1.61 x 1.61 m, lines 1.6 m from the walls | 1.62 x 1.62 m | — | Found |
| Lens | f = 892 px (476 px at 1024 wide; ≈94° field of view), k1 = 0.050, k2 = −0.022 | f = 478 px, k1 = 0.057, k2 = −0.038 | — | Found; shared by all clips |
| Fit residual | 0.70 px RMS, 19 frames | 0.43 px RMS, 17 frames (old line method) | ≤ 2 px at 1024 wide | Pass |
| **Held out: side-wall out lines** | 1.53 px RMS, median 0.59, p90 2.70 | 0.54 px RMS, median 0.27, p90 0.78 | ≤ 2 px at 1024 wide | Pass |
| **Held out: front-wall lines** | 0.65 px RMS, median 0.52, p90 0.80 | 0.40 px RMS, median 0.35, p90 0.51 | ≤ 2 px at 1024 wide | Pass |
| **Held out: wall/floor junctions** | 2.05 px RMS, median 1.23, p90 3.60 | 0.93 px RMS, median 0.44, p90 1.73 | ≤ 2 px at 1024 wide | Pass |
| **Held out: floor lines** | 2.17 px RMS, median 1.49, p90 3.51 | 1.20 px RMS, median 0.93, p90 1.88 | ≤ 2 px at 1024 wide | Pass (marginal: without the floor lines the length is unpinned and refits to 9.3 m, which inflates this test) |
| **Held out: a whole clip** (venue fitted on the other two) | 1.18–1.33 px RMS, median 0.39–0.42 | 0.62–0.71 px RMS, median 0.27–0.28 | ≤ 2 px at 1024 wide | Pass |
| Per-frame registration | 36,386 frames: median 0.70 px, p99 ≤ 0.81, worst 1.04 | 34,325 frames: median 0.43 px, p99 ≤ 0.49, worst 0.57 (old line method) | ≤ 2 px at 1024 wide | Pass |
| The two registrations against each other | Court points agree to median 0.50 px at 1080p, p95 0.73, worst 1.66 (registered independently) | | — | Consistent |
| Frame-to-frame tracking jitter | median 0.12 px, p99 0.38–0.45 | median 0.07 px, p99 0.30–0.37 | — | — |
| Camera moves found | Clip 2 at 298.9 s: 112 px in one frame, picked up again in one frame. Clip 2 at 393.8–395.8 s: a slow pan, tracked. No moves in the extra 69 s of clip 2. The phone's centre moves 5–8 cm per clip | Same events (61 px at 298.9 s) | Flagged | Pass |
| Registration speed | 4–5 fps per clip, 3 clips in parallel with three other jobs on the machine (load 50+); not measured idle | 37–43 fps per process | real time | Open: a static phone needs a pose only every few frames |
| Hand-clicked landmarks (13, clip 1, 145.6 s) | — | 3.3 px RMS; the clicks, not the model, are off the lines | — | Clicks too coarse to grade the fit |

Held-out WhatsApp figures use the current line method; the earlier sheet
(old method) had side-wall 0.70, front-wall 0.41, junctions 0.96, floor 1.27
and whole clip 0.75–0.85 px RMS.

**Floor and wall accuracy implied by calibration alone** (clip 1 at 145.6 s;
players and the ball add their own error on top):

| Location | Originals: cm per px, across x along | At median 0.40 px | At held-out RMS 1.26 px | WhatsApp: at median 0.28 px | WhatsApp: at RMS 0.66 px |
|---|---|---|---|---|---|
| Back floor (y = 1.5 m) | 0.40 x 0.99 | 0.2 x 0.4 cm | 0.5 x 1.2 cm | 0.2 x 0.5 cm | 0.5 x 1.2 cm |
| T | 0.71 x 3.19 | 0.3 x 1.3 cm | 0.9 x 4.0 cm | 0.4 x 1.7 cm | 0.9 x 4.0 cm |
| Front-left floor (1 m, 8 m) | 1.11 x 4.91 | 0.4 x 2.0 cm | 1.4 x 6.2 cm | 0.6 x 2.6 cm | 1.4 x 6.1 cm |
| Front-centre floor (y = 9 m) | 1.24 x 9.71 | 0.5 x 3.9 cm | 1.6 x 12.2 cm | 0.6 x 5.1 cm | 1.5 x 12.0 cm |
| Front wall | 1.34 x 1.34 | 0.5 cm | 1.7 cm | 0.7 cm | 1.7 cm |

Plus up to ±0.1 m along the court near the front wall until the court length is
measured.

**What the originals change.** For the court, little: the typical error is
about 25% smaller, and the RMS in centimetres is the same. The remaining error
(about 0.2 px median at 1024 wide, at both resolutions) comes from the lens
and court model, not from image noise, so doubling the resolution does not
halve it. The originals' value is in the players and the ball (sections 2 and
4): a slow ball is a crisp 2–3 px dot in the original and a smudge in the copy.
A fast ball is a faint streak in both, because the frame rate and shutter are
unchanged (`work/review/original_vs_whatsapp_ball.jpg`).

**Lessons for the method.**

- Matching moves model points only along their normals, so a fit started at
  the standard 6.40 m width never leaves it. The pose absorbs the error and the
  service-box lines look "off-spec". The width must be searched explicitly
  (inlier consensus), and the wall/floor junctions must be matched as
  brightness edges, because painted lines don't say where they end.
- Testing on synthetic frames needs ray-cast rendering. OpenCV's antialiased
  drawing shifts lines by up to 0.9 px, which is larger than the effect being
  tested.
- Pixel settings (kernel, search radii, thresholds) are stated for a frame
  1024 wide and scale with the frame. With fixed settings, near lines at 1080p
  are wider than the 9 px kernel and drop out (84% of samples matched instead
  of 96% on a synthetic frame).
- A line several pixels wide gives a flat-topped response, so its highest
  sample can sit anywhere across it. Lines are located at the centre of the
  response above half its peak. At 1080p this cut the residual p90 from 0.84
  to 0.63 px (1024-wide scale). It also improved the WhatsApp held-out
  side-wall lines from 0.70 to 0.54 px RMS.

Files:

- `work/venue_court5_1080p.json` (court and lens for the originals) and
  `work/venue_court5.json` (for the copies);
- `work/IMG_40xx/camera.parquet` and `work/wa_*/camera.parquet` (per-frame
  poses, segments, quality);
- `work/calibration_holdout_1080p.json`,
  `work/calibration_holdout_wa_centroid.json` and
  `work/calibration_holdout.json` (old method);
- `work/review/clip2_court_lock_*.mp4` (copies),
  `work/review/original_vs_whatsapp_ball.jpg`.

## 2. Players

YOLO26-m detection (input 1024 wide), RTMPose-m keypoints on 1080p crops,
tracklets joined into two identities by shirt colour and court-space motion,
then a robust Kalman smoother on the court floor. Ranges are across the three
clips. Clip 2's original is cut to the copy's 533 s for comparison.

| Measure | Originals (1080p) | WhatsApp copies | Estimate or threshold | Status |
|---|---|---|---|---|
| Both players detected | 85.9–92.4% of frames | 81.0–87.6% | — | Better on the originals |
| Neither detected | 0.5–4.4% | 1.1–5.1% | — | — |
| Detector confidence, median | 0.89 | 0.85 | — | — |
| Keypoint confidence, median of means | 0.74–0.75 | 0.68–0.71 | — | — |
| Foot point taken from the ankles | 82.8–92.3% (the rest: box bottom or hip height) | 79.8–88.1% | — | — |
| **Sustained identity switches** | **0** in 3 clips (every one of 222 crossings viewed) | 0 | about 1 across several rallies | Pass |
| Brief label errors (1–4 frames) | 2. One put B's label on a merged box for 4 frames and moved B's track up to 2.3 m | 3, each 1 frame | — | Open: merged boxes are labelled by colour alone |
| Same box under the same label, both runs | 99.92–99.99% of frames where both detect | | — | Consistent |
| Smoothed position, original against copy | median 3.7–4.5 cm, p90 12.6–23.7 cm | | — | Consistent |
| Foot jitter while standing still, median SD across x along | ankle 3.3–4.1 x 4.5–5.5 cm (148–284 windows per clip) | 3.3–4.3 x 4.5–5.7 cm | back half ±10–20 cm | Within the estimate. Mostly the player's own sway |
| Front court, ankle SD along the court | 8.6–12.8 cm (3–10 windows per clip) | 10.0–14.1 cm | ±0.5–1 m | Better than estimated, from few windows |
| Frame-to-frame noise along the court (second difference) | 1.31–1.53 cm | 1.64–2.03 cm | — | About 20% lower on the originals |
| Player lost (no box) | A 2.8–7.4%, B 4.8–9.6% of frames | | — | Mostly hidden behind the other player (32–75%), then the door strip (8–24%) |
| Rally distance, original against copy | within 1.3 m per player, except clip 3 B (53.3 against 50.1 m, the 4-frame error) | | T distance ±10–30 cm | Consistent |
| **Foot position against a known mark** | not measured | | per-zone estimates | Open: needs clicked feet or Track B |
| Speed | 8–15 fps (shared machine; decode on its own thread) | 15–16 fps | real time | Open |

**What the originals change.** More of each player is found and the keypoints
are surer, because pose runs on full-resolution crops. Detection itself runs at
1024 wide in both, so its gain comes from a cleaner picture. Running the detector
at 1920 wide cost 11x the time for 2% more boxes.

**Lessons for the method.**

- The naive HLG decode halved shirt saturation (median 12 against 23) and cut
  per-box colour accuracy from 99.3% to 97.4%. Before torso pixels were
  weighted away from the dark door hardware, it lost player A for 5 s. The
  shared SDR conversion now restores the saturation.
- Gating each point against the forward prediction let one bad foot point bend
  the velocity, reject the good points after it, and run the track away by
  1–3.5 m. The smoother now fits every point, drops the implausible ones
  against the smoothed track, and repeats. No smoothed jump over 0.5 m remains.
- The door-hardware mask sits about 6 px inside the plate's real edge, so a
  player cut off by the handle plate wasn't marked truncated, and the box
  bottom put them at the front wall. The mask edge now has 16 px of slack.

Files: `work/IMG_40xx/players.parquet`, `players_report.json`,
`topview_rally.png`, `overlay_rally.mp4`, `crossings/` and
`compare_vs_whatsapp.json`.

## 3. Audio hits

Onsets in the AAC track (the 4-channel spatial track can't be decoded by
ffmpeg, AVFoundation or afconvert), typed from rise time, level, spectrum
and channel balance. Ground truth: 134 events labelled frame by frame by
Claude agents blind to the audio (78 racket contacts, 54 of them high
confidence; 20 front-wall impacts, mostly low confidence). Clip 1's two
labellers, taken separately, give offsets 14 ms apart (+15.0 and +0.7 ms),
which bounds how well anything below can be measured.

| Measure | Originals | WhatsApp copies | Estimate or target | Status |
|---|---|---|---|---|
| **Audio–video offset** (audio minus video, after sound travel) | +1.5, +13.1, +12.1 ms (clips 1, 2, 3); clips 1+3 pooled +3.7 ms, 90% CI 1.6–11.8 | −21.4, −4.5, −1.0 ms | unknown; under one frame (33 ms) | Pass: under half a frame, known to ±10–15 ms |
| Copy audio against original audio | — | −0.4 to −0.5 ms, spread ~1 ms, no drift | — | Identical; it is the copy's video that moves (0–32 ms, varying) |
| Any onset at a racket label, ±1 / ±2 frames | 0.44–0.82 / 0.73–0.87 | 0.58–0.71 / 0.67–0.79 | chance 0.24 / 0.43 | Above chance, not selective (4.3 onsets/s) |
| **Racket contact typed**, P / R at ±1 frame | P 0.44–0.67, R 0.13–0.22 | P 0–0.80, R 0–0.11 | frame-level hit times | Fail: a racket contact is a 3.6–4.5 dB step |
| **Front-wall impact typed** (clip 1) | 15 of 19 typed onsets correct; 14 of 20 labels found at ±2 frames (10 at ±1) | 8 of 9 correct; 8 of 20 found | — | Usable: one per shot, the dependable audio event |
| Serve signature | 9 of 9 serves: a sharp 14–20 dB impact 119–183 ms after contact | | — | Found |
| Rally segmentation, clip 1 (visual count 24) | 14 matched, 10 missed, 4 spurious; overlap 0.51 | 9 matched, 15 missed, 5 spurious | — | Open: audio alone merges and misses rallies |
| Time between shots | labelled racket contacts: median 1.33 s, p10–p90 1.04–2.13 s (N=53); audio front wall to front wall: median 1.21–1.52 s | 1.18–1.65 s | elite 1.3–1.9 s | Consistent |
| Unmatched loud onsets in rallies (57 typed by eye) | half real wall or floor impacts (28), most of the rest footwork (25) | | — | Confounders as common as hits |

**What the originals change.** Front-wall recall nearly doubles (14 against 8 of
20) and rise times separate impacts better. The offset differs between the two
versions because WhatsApp moved the video, not the sound, so each version needs
its own offset.

**What this means for the design.** Racket contacts can't be timed from sound
alone on this footage. Time each shot from its front-wall impact (audio, or
the ball reversing at the wall), and place the racket contact 0.12–0.24 s
earlier using the ball track and the striker's pose. Rally boundaries need the
players and the ball as well as the audio.

Files: `work/<stem>/audio_onsets.parquet`, `audio_rallies.parquet`,
`audio_plot.png`, `audio_eval.json`, `hit_labels.csv`;
`work/audio_scratch/final_metrics.json`.

## 4. Ball

Labels: **hand labels from the user**, not model output. The user clicked the
ball in every frame of six 15-second rally windows (labelling app, one click
per frame, at the centre of the blur streak; 2,789 frames: 2,049 with the ball
visible, 740 with it out of view). Every record is a user click; none is an
accepted BlurBall pre-label, so the labels are independent of the detectors.
Windows: IMG_4091-r12 (464 frames), IMG_4091-r16, IMG_4089-r3, IMG_4089-r11,
IMG_4090-r18 and IMG_4090-r9 (465 each). Counts are from the final snapshot
(`work/ball_ft/labels_snapshot_v2.parquet`, taken when labelling stopped). The
first snapshot, taken mid-labelling, had IMG_4090-r18 at 225 of 465 and
IMG_4090-r9 at 10 of 465 (so r9 was mostly unlabelled and excluded); the four
other windows were identical in both. An earlier 166-frame sample labelled by a
Claude agent is no longer used. Tolerances are in 1080p pixels (4, 7.5, 15);
one point per frame, strict per-frame matching (a wrong point is a false
positive and a miss). All numbers below score exactly the same 2,789 frames.

**Zero-shot BlurBall at 576x1024 (Core ML), the baseline**

| Tolerance | Precision | Recall | F1 |
|---|---|---|---|
| 4 px | 0.29 | 0.21 | 0.24 |
| 7.5 px | 0.37 | 0.26 | 0.31 |
| 15 px | 0.40 | 0.28 | 0.33 |

The earlier sheet's 0.73 / 0.62 came from 158 frames that a Claude agent chose
to be easy. On every frame of a rally, BlurBall works on the front wall
(recall 0.63 at precision 0.84) and little elsewhere. Many false positives are
hundreds of pixels off: shoes and other ball-sized blobs.

**Fine-tuned on the user's labels, held out by whole window**

BlurBall's weights, fine-tuned at 576x1024 input on random 288x512 crops (flip,
brightness/contrast/gamma, small zoom and shift), BlurBall's own heatmap
targets (sigma 2.5) and weighted-BCE loss, AdamW 1e-4 cosine, batch 4, 10
epochs, batch-norm statistics frozen. About 4 minutes per epoch on the M3 Pro
GPU (MPS). Early stopping used the last 15% of each *training* window
(validation loss), and no test window was used for any choice except the
detection threshold below. Three folds hold out whole windows, so no
neighbouring frame of a test frame was ever seen:

| Fold | Held-out windows | Trained on |
|---|---|---|
| A | IMG_4091-r12, IMG_4089-r3 | the other four |
| B | IMG_4091-r16, IMG_4089-r11 | the other four |
| C | IMG_4090-r18, IMG_4090-r9 (a whole clip) | the other four, from clips 1 and 2 |

Each frame is scored once, by the model that never saw its window.

| Fold (7.5 px) | Zero-shot P / R / F1 | Fine-tuned P / R / F1 |
|---|---|---|
| A | 0.38 / 0.29 / 0.33 | 0.81 / 0.81 / 0.81 |
| B | 0.35 / 0.24 / 0.29 | 0.79 / 0.77 / 0.78 |
| C (a clip the model never saw) | 0.39 / 0.27 / 0.32 | 0.85 / 0.82 / 0.84 |
| **Pooled** | **0.37 / 0.26 / 0.31** | **0.82 / 0.80 / 0.81** |

| Pooled tolerance | Zero-shot P / R / F1 | Fine-tuned P / R / F1 |
|---|---|---|
| 4 px | 0.29 / 0.21 / 0.24 | 0.68 / 0.67 / 0.67 |
| 7.5 px | 0.37 / 0.26 / 0.31 | 0.82 / 0.80 / 0.81 |
| 15 px | 0.40 / 0.28 / 0.33 | 0.85 / 0.84 / 0.85 |

Median error on the hits is 1.9 px. By window (7.5 px F1, zero-shot to
fine-tuned): 4091-r12 0.42 to 0.87, 4091-r16 0.30 to 0.82, 4089-r11 0.26 to
0.72, 4089-r3 0.22 to 0.74, 4090-r18 0.37 to 0.86, 4090-r9 0.26 to 0.81.

By region at 7.5 px, pooled (precision / recall; fp counted in the region of the
predicted point):

| Region | Visible labels | Zero-shot | Fine-tuned | Fine-tuned F1 |
|---|---|---|---|---|
| Front wall | 353 | 0.84 / 0.63 | 0.84 / 0.92 | 0.88 |
| Side walls | 571 | 0.50 / 0.19 | 0.86 / 0.79 | 0.83 |
| Back floor | 238 | 0.00 / 0.00 | 0.83 / 0.68 | 0.75 |
| Front floor | 153 | 0.08 / 0.03 | 0.71 / 0.52 | 0.60 |
| Door strip | 733 | 0.29 / 0.29 | 0.78 / 0.85 | 0.81 |
| Air (above the front wall) | 1 | | | |

**Detection threshold.** The pretrained 0.7 is too strict for the fine-tuned
heatmaps. Pooled F1 at 7.5 px by threshold (the detector re-run at each value):
0.3: 0.76, 0.4: 0.79, **0.5: 0.81**, 0.6: 0.79, 0.7: 0.69, 0.8: 0.40. The post-hoc
`evaluate.sweep` on the 0.3 run peaks in the same place (0.45 to 0.5). The
table above uses 0.5, which was chosen on these pooled held-out frames, so it is
slightly optimistic; the 0.4 to 0.6 plateau is within 0.025 F1 of it.

**Shoe suppression.** A post-filter drops a detection inside a player's lower-leg
and foot zone (the bottom 15 to 40% of the box, widened) unless a detection
outside the zone within two frames links to it. On the pooled held-out
predictions it removes 5 to 37 detections and moves F1 from 0.809 to 0.808 to
0.810 across four settings. Fine-tuning already removed the shoe false
positives, so it is **not kept**.

| Measure | Result | Threshold | Status |
|---|---|---|---|
| **Ball recall in rally frames** (held-out windows, 7.5 px) | 0.80 at precision 0.82 (0.81 / 0.77 / 0.82 across folds; 0.67 at 4 px) | >= 0.5 for 3D ball physics | Pass, on a clip the model never saw too |
| Weakest regions | front floor 0.52 recall, back floor 0.68 | — | The floor is still the hard part |
| Speed, sustained (final model, Core ML on the GPU, full 1080p pipeline) | 7.5 fps over 3 minutes, load average about 3 (same as the zero-shot model's 7.7) | real time | Open: a 15-minute match takes about 30 minutes |

Limits: six rallies from one court and one phone; two clips supply the training
windows for fold C's clip-held-out test, so the clip-level result is one clip.
Players and lighting are the same across all windows. Windows are 15 s rallies
(mostly rally tempo), not whole matches, so idle-time false positives are not
measured here.

**Final model.** Trained on all six windows (10 epochs, no validation split) and
exported to Core ML (`blurball-ft:final@576x1024`, default threshold 0.5).
Files: `work/ball_ft/final/final.pt`.

Files: `squashvid/ball/finetune.py` (training, folds, scoring, shoe filter),
`work/ball_ft/` (snapshots, fold checkpoints and logs, `cv_report.json`,
`sweep_report.json`), `work/<stem>/ball/blurball-ft-fold*@576x1024-coreml.parquet`.
Fold checkpoints: `work/ball_ft/fold{A,B,C}/`.

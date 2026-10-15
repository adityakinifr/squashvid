# Ball track and court events

Stage after the ball detector: turns per-frame blob candidates into one linked ball track, finds the
moments the ball hits something, places them on the court in metres, and fits 3D flights.
Code: `squashvid/balltrack/`. Run: `python -I -m squashvid.balltrack <stem> [--variant blurball-ft-final]`
(IMG_4091 takes about 40 s including the 3D fit).

## Outputs (`work/<stem>/`)

| File | Columns |
|---|---|
| `ball_track.parquet` | frame, t, u, v, state (detected / interpolated / none), score, track_id, segment_id |
| `ball_events.parquet` | t, frame (float, sub-frame), kind (front_wall, side_wall_left, side_wall_right, floor, back_wall, racket), u, v, x, y, z, sigma_a, sigma_b, evidence (track / audio / both), confidence, player, player_dist_px |
| `ball_3d.parquet` | frame, t, x, y, z, segment_id, anchored, residual_px |

Event x, y, z are on the surface of the event kind (floor z = 0.02 m, the ball radius), found by
casting the camera ray. sigma_a and sigma_b are the principal axes of the position uncertainty (m).
Racket events carry the nearest player (A/B) and the pixel distance to that player's wrist; deciding
who really hit the ball is left to the analysis stage.

## How it works

1. **Linking** (`link.py`). Blobs within 4 px are merged; blobs that sit still (a spot that does not
   move for 25 frames) are masked. A dynamic programme finds the best chain of blobs (speed, speed
   change and gap penalties, reward for peak height), extracts it, removes it and repeats.
   Isolated high-peak blobs (>= 0.8) are kept as one-point tracks.
2. **Gap filling and kinks** (`link.py`, `kinks.py`). A gap is filled with a smooth curve only when the
   ball leaves and re-enters on the same flight. Otherwise the two sides are bridged through the
   intersection of their lines, which marks an impact and starts a new `segment_id`. Nothing is
   smoothed across an impact. Kinks are found by comparing a smooth local fit with a two-velocity fit.
3. **Events** (`events.py`, `world.py`). Each kink is tested against bounce physics for each plane (the
   normal speed reverses and shrinks, the tangential speed shrinks, image velocity must match), and
   as a racket hit (a player within reach and a hard velocity change). Audio onsets (corrected for
   the audio/video offset and sound travel time) confirm weak front-wall kinks and add audio-only
   front-wall events (position filled from the track within 3 frames, else NaN).
4. **3D** (`traj3d.py`). Each segment is fitted with gravity and quadratic drag (0.016 1/m) by
   reprojection error. `anchored = True` when a located wall/floor event sits at either end; those
   pass through the event point. Unanchored segments have unconstrained depth: they are estimates
   only. Fits outside the court box or worse than 8 px RMS are dropped.

## Measured numbers

Six labelled windows, held out by fold (A: IMG_4091-r12 + IMG_4089-r3, B: IMG_4091-r16 + IMG_4089-r11,
C: IMG_4090-r18 + IMG_4090-r9). Parameters were chosen on A+B; C (a clip never used for choosing)
confirms. Input: candidates with peak >= 0.5, rebuilt from the cross-validation files (see caveats).
Point correct = within 7.5 px of the user label on a labelled-visible frame.

### Track

| Set | Variant | Precision | Recall | F1 |
|---|---|---|---|---|
| A+B | raw detector, threshold 0.5 | 0.795 | 0.790 | 0.793 |
| A+B | linked (detected only) | 0.878 | 0.738 | 0.802 |
| A+B | linked + interpolated | 0.826 | 0.750 | 0.786 |
| C | raw detector, threshold 0.5 | 0.854 | 0.824 | 0.839 |
| C | linked (detected only) | 0.910 | 0.739 | 0.816 |
| C | linked + interpolated | 0.884 | 0.764 | 0.819 |

Linking raises precision by 5 to 8 points and costs 5 to 9 points of recall: F1 is level on A+B and
slightly lower on C. Its use is a cleaner track and segments, not a better point score.

Interpolated frames (reported apart): A+B 90 frames, of which 34 fall on labelled-visible frames and
16 of those are within 7.5 px; 56 fall on frames the user labelled out of view (the ball is hidden,
so the interpolation is a guess). C: 38 frames, 25 visible, 18 within 7.5 px, 13 on out frames.

### Events, against events derived from the user's labels

The reference is the same kink and physics machinery run on the label track (the user labelled
positions, not impacts), so these numbers measure losses from detection and linking, not whether the
rules are right (checked by eye, see below). Match tolerance 3 frames. Events where under 60% of the
surrounding frames are labelled visible are counted "unverifiable", not false positives.

| Kind | Set | Reference | TP | FP | FN | Precision | Recall | Median abs timing error (frames) |
|---|---|---|---|---|---|---|---|---|
| floor | A+B | 23 | 13 | 8 | 10 | 0.62 | 0.57 | 0.25 |
| floor | C | 11 | 3 | 1 | 8 | 0.75 | 0.27 | 0.25 |
| racket | A+B | 26 | 8 | 3 | 18 | 0.73 | 0.31 | 0.88 |
| racket | C | 9 | 4 | 1 | 5 | 0.80 | 0.44 | 0.25 |
| front_wall | A+B | 7 | 2 | 9 | 5 | 0.18 | 0.29 | 0.33 |
| front_wall | C | 4 | 3 | 5 | 1 | 0.38 | 0.75 | 0.75 |
| side_wall_left | A+B | 7 | 5 | 2 | 2 | 0.71 | 0.71 | 0.50 |
| side_wall_left | C | 6 | 5 | 0 | 1 | 1.00 | 0.83 | 0.00 |
| side_wall_right | A+B | 4 | 2 | 1 | 2 | 0.67 | 0.50 | 0.25 |
| side_wall_right | C | 2 | 0 | 2 | 2 | 0.00 | 0.00 | none |
| back_wall | A+B / C | 0 / 1 | 0 / 1 | 1 / 2 | 0 / 0 | | | |
| any kind | A+B | 67 | 34 | 20 | 33 | 0.63 | 0.51 | 0.25 |
| any kind | C | 33 | 19 | 8 | 14 | 0.70 | 0.58 | 0.25 |

Front-wall false positives in A+B (9) are all audio-only events with no track nearby; they are
audio-confirmed hits whose ball the camera could not see, not necessarily wrong.

Front-wall impact position, against label-derived positions: A+B n = 2, median error 0.11 m (0.06 m
sideways, 0.08 m height); C n = 3, median 0.07 m, p90 0.31 m. Samples are tiny.

### Against the hand-labelled hits (`hit_labels.csv`), IMG_4091, full clip

These labels lie almost entirely outside the six windows, so this uses the full-clip final-model
output. Recall within 3 frames only; precision is not measurable (labels cover a subset of hits).

| Label | Labels | Matched | Recall | Median abs timing error (frames) |
|---|---|---|---|---|
| front_wall | 20 | 18 | 0.90 | 0.9 |
| floor | 10 | 7 | 0.70 | 0.5 |
| racket | 40 | 12 | 0.30 | 1.0 |
| side_wall | 4 | 0 | 0.00 | none |

The front-wall figure leans on audio, which the labeller also used; it is not independent.

### 3D (IMG_4091)

2301 frames in 192 fitted segments: 75 anchored (908 frames), 117 unanchored (1393 frames). Median
reprojection error 0.6 px either way (a fit is well constrained by the image, not by depth, so low
residual does not validate the unanchored positions).

## Outputs per clip

| Clip | track | events | 3D |
|---|---|---|---|
| IMG_4091 | yes | yes (267 events: floor 90, front_wall 83, racket 60, side_wall_left 15, side_wall_right 14, back_wall 5) | yes |
| IMG_4090 | pending full-clip predictions | | |
| IMG_4089 | pending full-clip predictions | | |

## Caveats

- The cross-validation candidate files only keep blobs with peak >= 0.7. The tuning input was rebuilt
  by adding the top sub-0.7 blob per frame from the sweep cache; it approximates, not equals, the
  full-clip file (peak >= 0.5). Full-clip behaviour is checked only by the hit-label recall above.
- The camera sits behind the back glass, so the front wall is seen end-on and the ball often hides
  behind door glass near it. Front-wall impacts barely change image velocity; faint ones need audio.
  Audio-only events have no reliable position and carry NaN when no track exists within 3 frames.
- A back-wall hit and a front-wall hit look the same in image dynamics. `back_wall` is only emitted
  when no player is near; treat it as low trust (5 events on IMG_4091, mostly spurious in the labels).
- Racket recall is limited by gaps: a fast ball near the camera leaves no track around the hit, so no
  kink exists. The racket kind also over-claims where the ball vanishes out of view.
- Time base is the frame index through the clip's timestamps (variable frame rate); audio offsets
  were applied as stored in `audio_eval.json`.
- Uncertainties (sigma, confidence) are heuristics from pixel noise and plane geometry, not
  calibrated against ground truth.
- Reference events come from the same physics as the detector, so they share its blind spots.

## Failure modes seen (IMG_4091 frames)

- f7300.5: a ball crossing sideways was read as a front-wall kink in the label reference.
- f5085, f5175: smooth flight or fast deceleration, no impact; weak kinks near these were rejected.
- f5169, f7448: fast ball near the camera, no predicted track, impact missed.
- f2122.75: floor event with racket visible a frame later; floor vs racket is ambiguous by eye.
- f6822.25: racket event at the point where the ball leaves view, no racket visible.
- IMG_4089-r3 f2088, f2379: gaps in the candidates remove the events.

## Review images

`work/balltrack_review/` holds per-event contact sheets (`full_*` from the full clip, `gt_*` from the
label reference), and `evaluate.json` holds the full evaluation.

# Processing a new video (squashvid/process.py)

```bash
source .venv/bin/activate
python -I -m squashvid.process ~/Downloads/new_match.MOV [--stem NAME] [--from STAGE] [--only STAGE]
python -I -m squashvid.viewer serve        # then open http://127.0.0.1:8771/#NAME
```

One command takes a new phone video of **court 5** to everything the viewer reads. The video is recorded in `work/clips.json` (`stem -> absolute path`), which `ball.clips.clip_path` / `venue_path` / `is_original` consult after the built-in clips, so every stage module finds it by stem. The venue is chosen by frame size (`clips.venue_for_size`: `work/venue_court5.json` for 1024x576, `work/venue_court5_1080p.json` for 1920x1080). The stem defaults to the file name; a stem that is one of the built-in clips cannot be reused for a different file.

## Stages

Each stage skips itself when its outputs exist. `--from S` forces S and every later stage; `--only S` forces just S. A stage that raises stops the run, naming the stage.

| stage | what it runs | outputs in `work/<stem>/` |
| --- | --- | --- |
| register | `calib.register.register_video`: track every frame from the court-5 seed pose, split at jumps, smooth | `camera.parquet`, `camera_summary.json` (match_frac, rms, segments) |
| players | `players.__main__.run` | `detections.parquet`, `players.parquet`, `players_summary.json`, overlays |
| audio | `audio.__main__.main` | `audio_onsets.parquet`, `audio_rallies.parquet`, `audio_plot.png` |
| ball | `ball.predict.run_full`, `blurball-ft:final@576x1024`, Core ML, 60 s chunks | `ball/blurball-ft-final@576x1024-coreml.parquet`, `ball/cands/` |
| balltrack | `balltrack.__main__.run` | `ball_track.parquet`, `ball_events.parquet`, `ball_3d.parquet` |
| analysis | `analysis.run.run` | `analysis/` (rallies, hits, stats, insights, report) |
| bodies | `bodies.run.run` (MotionBERT lift on MPS) | `bodies.parquet` (3D joints for the viewer's player figures) |
| viewer | `viewer.proxy.make_proxy` + `viewer.export.export` | `proxy.mp4`, `viewer/` |

The ball output is saved per chunk, so an interrupted run leaves a partial file; the stage counts it as done only when it reaches the last frame, otherwise it reruns (finished chunks are merged by frame).

Without `hit_labels.csv` there is no `audio_eval.json`, so the balltrack audio offset is 0 (labelled clips measure 1-13 ms, under one frame) and a warning is logged.

## Limits

- **Court 5 by default.** The seed pose (`register.SEED_POSE`) and the venue files belong to the phone behind the back glass of court 5. Another court needs a venue first: `python -I -m squashvid.calib.new_venue <video> <clicks.json> --name NAME --out work/venue_NAME_1080p.json` (landmark clicks on one frame; the phone may pan and tilt), then `--venue` on this command, which pins it to the clip in `work/<stem>/venue.json`. A frame size with no venue file stops the run before any work. Same size on another court without `--venue` is not detected: check `camera_summary.json` (`match_frac_median` around 0.78 is normal; the command warns below 0.5).
- Players are not named; identity is fitted per clip.

## Timings

Measured on the 40 s excerpt (`work/test_inputs/excerpt4091.mov`, 1920x1080, 1201 frames) on this Mac. Players ran while another Core ML job was using the GPU, so it is a pessimistic number.

| stage | seconds | per minute of video |
| --- | --- | --- |
| register | 120 | 180 |
| players | 152 | 227 |
| audio | 2.4 | 4 |
| ball | 164 | 246 |
| balltrack | 0.6 | 1 |
| analysis | 3.5 | 5 |
| viewer | 40 | 59 |
| total | 482 | about 720 (12 min per minute) |

A 5-minute clip therefore takes about an hour.

## Check against the existing clip

The excerpt is IMG_4091 from 160 s, stream-copied, so it can be compared with `work/IMG_4091` (excerpt frame k is original frame k + 4785; the copy starts on a keyframe 0.4 s early, so the first 12 frames have negative time):

- camera: median probe-point difference 0.048 px (p95 0.11, max 0.17) at 1080p;
- ball track: 785 frames located in both, every one identical (0 px); state agrees on 97.9 % of frames;
- ball events: 25 vs 24; 24/24 of the original's events have the same kind within one frame.

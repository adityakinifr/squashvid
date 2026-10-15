# Match viewer

To get a new video into the viewer in one step, see [process.md](process.md).

A local web app (never served publicly) that plays a match video beside a 3D court, a floor plan and a front-wall view, synced, with tracking, events and statistics.

## Run it

```bash
source .venv/bin/activate
python -I -m squashvid.viewer proxy --all     # work/<stem>/proxy.mp4 (H.264 SDR, ~4 min per clip)
python -I -m squashvid.viewer export --all    # work/<stem>/viewer/ (meta.json + .bin arrays, ~2 MB)
python -I -m squashvid.viewer serve           # http://127.0.0.1:8771/#IMG_4091  (?t=60 seeks)
```

Re-run `export` whenever the ball or analysis files change; every upstream file is optional and the app says what is missing in its header. The proxy encoder is `h264_videotoolbox`; pass `--encoder libx264` elsewhere. `proxy <stem> --check-only` re-runs the PTS check.

Proxy: frames come from `Video(sdr=True)` (the project's HLG to SDR conversion), keyframe every 15 frames, `+faststart`, first audio stream copied. Presentation timestamps are written from the originals, so proxy time equals original time. The iPhone files also hold trailing packets that never decode (32 in IMG_4091), so the proxy has fewer frames than `Video.timestamps`; the check requires the shared PTS to match within one 1/600 s tick and every missing PTS to lie after the last proxy frame.

## Panels

- **Video** with a canvas overlay: court lines projected through that frame's camera (a calibration check), player boxes and skeletons (dashed box = interpolated, no keypoints), the ball with a 12-frame trail (yellow ring = detected, orange = linked, dashed white = interpolated).
- **3D court** (three.js r128, orbit controls, presets behind glass / side / top): walls to the out lines, tin, lines, players as solid lit figures posed from the 3D joints in `bodies.parquet` (a rigged humanoid, or a shaped capsule figure if the model fails to load; see [bodies.md](bodies.md)), a cylinder where there is no 3D pose, ball and recent impacts. The "skeletons" box switches the figures on and off. Without `bodies.parquet` the pane draws the older flat keypoint lift as thin lines.
- **Top view**: floor plan, T, players with 1.5 s trails, optional occupancy heatmap (whole clip or current rally, per player), floor bounces with uncertainty ellipses.
- **Front wall**: tin, service and out lines, front-wall impacts with ellipses, coloured by player or rally, for the current rally or the whole clip.
- **Timeline**: rally bands, event ticks (analysis hits in player colours, audio onsets faint), ball coverage and player presence strips; click or drag to seek. Clicking an impact or bounce in the top or front view seeks to it.
- **Side panel**: Stats renders `analysis/stats.json` (per-player objects become comparison tables). Insights shows the coaching as cards per player: what to try next time and what to keep doing, each with why, the moments to watch (clickable chips), a drill, and how sure folded away. Events lists every event. Report shows the report, rendered by the server on request so it carries the current names. Insights and Report have a **This clip / All clips** switch: All clips shows the coaching and report pooled over every clip (`work/analysis/`, from `python -I -m squashvid.analysis --all --pooled`), where more findings hold up; a moment in another clip opens that clip and seeks to it.
- **Timeline readout**: under the timeline, the current rally, who served it and from which side, and who won it (inferred from who served next; see [analysis.md](analysis.md)).
- **Links**: `?view=behind|side|top` picks the 3D camera preset on load (for screenshots; the buttons do the same), `?t=<seconds>` seeks and `?tab=stats|insights|events|report` opens a side tab, e.g. `http://127.0.0.1:8771/?t=166&tab=report#IMG_4091`.
- **Names**: the Names button saves A/B names to `work/viewer_settings.json` through the server; the form says which shirt is which (A light blue, B dark), and a blank box keeps the default name. The viewer, its Report tab and the report files on disk use them at once, with no re-run.

Keys: space play/pause, left/right one frame (pauses first), shift+arrow 5 s, `[` / `]` previous/next rally. The `<video>` element is the clock (`requestVideoFrameCallback`, falling back to `requestAnimationFrame`); media time maps to a frame by binary search on the exported per-frame `t`.

## Measured vs estimated

| Shown | Status |
| --- | --- |
| Player court positions, boxes, keypoints | measured by the player tracker; interpolated frames are drawn dashed and have no keypoints |
| Court lines and camera | fitted venue plus smoothed per-frame pose; any offset is calibration error |
| Ball pixel position | detector output (`detected`), or tracker output when `ball_track.parquet` exists (`linked`, `interpolated`) |
| Ball in 3D | measured only from `ball_3d.parquet`. Without it the 3D view shows the camera ray and its floor intersection, labelled ESTIMATED: the height is unknown, and a ray that lands outside the court is not drawn |
| Figure in 3D | an estimate, not a measurement: a temporal 2D to 3D lifter (MotionBERT) gives the limb directions, the size is fitted per player through the court camera, and the feet are placed on the court foot point and the floor. Depth (toward or away from the camera) is the lifter's guess; see the validation numbers in `bodies.md`. The mesh's body shape, face and clothing are cosmetic, and the figure has no racket |
| Figure without `bodies.parquet` | the older flat lift: each keypoint ray is intersected with a vertical plane through the floor point facing the camera |
| Cylinder | drawn where no 3D pose exists (long gaps); fixed stylised 1.7 m (the tracker's hip height is not stature) |
| Impacts and ellipses | from `ball_events.parquet`; ellipses use `sigma_a` / `sigma_b` as axis-aligned semi-axes (x and y on the floor, x and z on the front wall), as the file gives no angle |
| Tracking summary stats | computed in the viewer from tracked positions during rallies (distance, speed, T proximity); the Analysis section is the analysis stage's own |
| Rallies | `analysis/rallies.parquet` if present, else the audio rallies |

## Files read (all optional)

`camera.parquet`, `players.parquet`, `players_summary.json`, `bodies.parquet`, `audio_rallies.parquet`, `audio_onsets.parquet`, `ball/blurball-ft-final@576x1024-coreml.parquet` (else the next best raw output), `ball_track.parquet`, `ball_events.parquet`, `ball_3d.parquet`, `analysis/{rallies,hits}.parquet`, `analysis/{stats,insights}.json`, `analysis/report.md`. The export format is `schema_version` 1: see `squashvid/viewer/export.py`.

Vendored libraries (three.js MIT, d3 ISC) live in `squashvid/viewer/static/vendor/` with licences.

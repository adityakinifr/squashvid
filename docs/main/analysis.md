# Match analysis (squashvid/analysis)

Run: `python -I -m squashvid.analysis <stem>|--all [--no-ball] [--no-review] [--pooled]`.
Evaluate: `python -I -m squashvid.analysis.evaluate`.
Outputs per clip in `work/<stem>/analysis/`: `rallies.parquet`, `hits.parquet`, `stats.json`, `insights.json`, `report.md`. Pooled outputs in `work/analysis/`. Review plots in `work/analysis_review/`.

## What it computes
- **Hits**: candidates are per-player wrist-speed peaks, audio front-wall onsets with no pose peak, and ball racket events. A Viterbi pass picks a sequence that favours alternating players, front-wall support after the hit, a rival swing, and a still hitter. With ball events present (`work/<stem>/ball_events.parquet`) they are used automatically and hit times snap to the ball's racket event.
- **Rallies**: a per-second play score (logistic on player speeds, swing rate, impact rate, inter-player distance, optional ball presence) with hysteresis, then chained and filtered (at least 2 shots, both players hitting). A gap of more than 2.8 s between shots ends a rally only if the play score also drops below 0.7 inside the gap (gaps over 6 s always do): missed shots in continuous play leave such gaps. Trailing shots where both players move at walking pace (faster under 1.5 m/s and slower under 0.8 m/s, averaged over ±1 s) are dropped first: after a rally the players walk, pick up the ball and bounce it before the serve, and those touches passed as shots. Start is first hit minus 0.6 s, end last hit plus 0.9 s.
- **Per-hit context**: T distance of both players, available time before the shot, distance the returner moved, 3x3 court cells, recovery time to within 1.5 m of the T, and a rule-of-thumb situation label (approximates the six Murray situations; no centroids available).
- **Stats**: per player, inside rallies only. Every headline number carries n and a 90% bootstrap interval over rallies.
- **Insights**: candidates compared player against opponent, early against late, recovery after short against long available time, and two situations: getting back up to the T after a shot from the back of the court, and how often the next shot is rushed when a player is more than 2 m from the T as the opponent hits. A candidate is suppressed when its 90% interval crosses zero, when it rests on fewer than 8 rallies, or when the gap is below its metric's `min_effect` (the smallest difference worth coaching, e.g. 0.3 m of distance from the T or 8 percentage points of time on it).
- **Coaching** (`coach.py`), told twice. For the player: at most three things to try next time and two to keep doing, each one line of fixed coaching text with no numbers. For a coach or parent: why it matters (the numbers, and for position items what being away from the T cost), the clip moments to watch and what to look at in them, one drill, and how sure (clear when the whole interval clears `min_effect`, else likely). Findings are grouped into themes (getting back to the T, length, taking time away, keeping it up to the end) and each theme gives at most one item per kind. A theme never appears as both a thing to try and a thing to keep, except a narrow thing to try beside a broad thing to keep (holding the T well overall but not after shots from the back). Every sentence with a number cites its candidates and passes the verifier, which rejects any number, timestamp or clip name not in them and bans technique claims, won/lost claims and elite-norm comparisons. Winners are never inferred.
- **Names**: generated text names the players `{A}` and `{B}`. The report and the viewer fill in the names saved in `work/viewer_settings.json` (the viewer's Names button; A is the light-blue shirt, B the dark shirt in every clip so far), so renaming needs no re-run.

## How far to trust it (IMG_4089, 4090, 4091)
Ground truth is thin; treat all numbers as indicative.
- Hit labels: 78 racket labels from Claude agents in labelled spans only. No ball events, pooled at +-0.2 s: precision 0.68, recall 0.68, hitter accuracy 0.93. Pose-only: P 0.62, R 0.58. Timing MAD about 0.05-0.07 s.
- With ball events (IMG_4091 only, preliminary): +-0.2 s precision 0.79, recall 0.85, hitter accuracy 1.0, timing MAD 0.02 s (no ball: 0.74 / 0.70 / 0.93).
- Hand-labelled ball frames gave only 34 contacts: recall 0.59, hitter accuracy 0.75. Too few to conclude.
- Rallies against a by-eye census (clips 1 and 3; partly used to fit the play score, so optimistic), with ball events: IMG_4091 P 0.94, R 0.67, IoU 0.68, end error median +1.2 s (audio-chain baseline P 0.78, R 0.58, IoU 0.49); IMG_4090 P 0.93, R 0.93, IoU 0.71, end error -0.1 s (baseline 0.43 / 0.60 / 0.48). The walking-pace thresholds were read off these same two clips, so these are in-sample. IMG_4089 has no census.
  The census is rough: it was written from 1 s contact sheets of the WhatsApp copies and several of its rallies start 0.4-1.4 s after the previous one ends, which a real rally cannot (the server has to fetch the ball). Merging those gives 16 and 12 rallies, and the scores become IMG_4091 P 0.88, R 0.94 and IMG_4090 P 0.73, R 0.92. Checked by eye on the originals: the census is right that IMG_4090 293-298 s is one rally (the detector used to split it) and that IMG_4091 rally 1 ends near 15 s (walking and pre-serve ball bouncing carried it to 19.8 s; with the tail trim it ends at 17.8 s, one touch at 16.9 s being just above walking pace).
- Rally boundaries are typically within about 2-3 s.

## Caveats
- Foot positions are unverified against tape marks; front-court depth is coarse.
- Winner and point information is absent, so distance covered and speed are reported but not interpreted.
- Situation rules are approximate.
- Ball metrics (height bands, thirds, near-tin, bounce length) exist only where ball events exist (IMG_4091 so far).
- The pooled report assumes player A/B identity is consistent across clips.
- The coaching thresholds (`min_effect`, 2 m from the T, the rushed-shot situations) were chosen by judgement, not fitted, and the drills are standard junior drills, not prescribed from the data.

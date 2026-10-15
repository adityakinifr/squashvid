# Match analysis (squashvid/analysis)

Run: `python -I -m squashvid.analysis <stem>|--all [--no-ball] [--no-review] [--pooled]`.
Evaluate: `python -I -m squashvid.analysis.evaluate`.
Outputs per clip in `work/<stem>/analysis/`: `rallies.parquet`, `hits.parquet`, `stats.json`, `insights.json`, `report.md`. Pooled outputs in `work/analysis/`. Review plots in `work/analysis_review/`.

## What it computes
- **Hits**: candidates are per-player wrist-speed peaks, audio front-wall onsets with no pose peak, and ball racket events. A Viterbi pass picks a sequence that favours alternating players, front-wall support after the hit, a rival swing, and a still hitter. With ball events present (`work/<stem>/ball_events.parquet`) they are used automatically and hit times snap to the ball's racket event.
- **Rallies**: a per-second play score (logistic on player speeds, swing rate, impact rate, inter-player distance, optional ball presence) with hysteresis, then chained and filtered (at least 2 shots, both players hitting). Start is first hit minus 0.6 s, end last hit plus 0.9 s.
- **Per-hit context**: T distance of both players, available time before the shot, distance the returner moved, 3x3 court cells, recovery time to within 1.5 m of the T, and a rule-of-thumb situation label (approximates the six Murray situations; no centroids available).
- **Stats**: per player, inside rallies only. Every headline number carries n and a 90% bootstrap interval over rallies.
- **Insights**: candidates compared player against opponent, early against late, and recovery after short against long available time. They are suppressed when the interval crosses zero or there are fewer than 8 rallies. A templated editor (optional LLM editor only if `ANTHROPIC_API_KEY` is set) writes cues and drills. A verifier rejects any number, timestamp or clip name not in the cited candidates, and bans technique claims, won/lost claims and elite-norm comparisons. Winners are never inferred.

## How far to trust it (IMG_4089, 4090, 4091)
Ground truth is thin; treat all numbers as indicative.
- Hit labels: 78 racket labels from Claude agents in labelled spans only. No ball events, pooled at +-0.2 s: precision 0.68, recall 0.68, hitter accuracy 0.93. Pose-only: P 0.62, R 0.58. Timing MAD about 0.05-0.07 s.
- With ball events (IMG_4091 only, preliminary): +-0.2 s precision 0.79, recall 0.85, hitter accuracy 1.0, timing MAD 0.02 s (no ball: 0.74 / 0.70 / 0.93).
- Hand-labelled ball frames gave only 34 contacts: recall 0.59, hitter accuracy 0.75. Too few to conclude.
- Rallies against a by-eye census (clips 1 and 3; partly used to fit the play score, so optimistic): IMG_4091 P 0.94, R 0.71, IoU 0.69 (audio-chain baseline P 0.78, R 0.58, IoU 0.49); IMG_4090 P 0.54, R 0.87, IoU 0.66 (baseline 0.43 / 0.60 / 0.48). IMG_4089 has no census.
- Rally boundaries are typically within about 2-3 s.

## Caveats
- Foot positions are unverified against tape marks; front-court depth is coarse.
- Winner and point information is absent, so distance covered and speed are reported but not interpreted.
- Situation rules are approximate.
- Ball metrics (height bands, thirds, near-tin, bounce length) exist only where ball events exist (IMG_4091 so far).
- The pooled report assumes player A/B identity is consistent across clips.
- The LLM editor is untested with a real key.

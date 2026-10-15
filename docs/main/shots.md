# Shots and rally endings

Two fact tables for the coaching layer, built from hits, rallies, ball events and `ball_3d`.
No coaching text. Every field says how much is known; the value `unknown` means the camera
did not show enough, never a guess.

Code: `squashvid/analysis/shots.py` (per shot), `squashvid/analysis/endings.py` (per rally).
Run: the analysis stage (`python -I -m squashvid.analysis`, and so `squashvid.process`) writes
`work/<stem>/analysis/shots.parquet` and `endings.parquet` straight after the hits and rallies;
`python -I -m squashvid.analysis.shots <stem>... | --all` rebuilds them alone and prints coverage and mixes.

`placement()` sums each player's known facts by where the shot was hit from (back, mid, front cell),
leaving out each rally's first shot; the reports show it as "Where the shots went" (counts of known
shots only, and nothing coached from it: how a choice of shot worked out is not measured). The
viewer's timeline readout names the current shot's known facts.

## Shot table (one row per hit)

A shot's interval runs from the hit to the next hit in the rally. For the last shot it runs to
rally end + 1.5 s, at most 3.5 s after the hit. Front-wall events within 0.25 s are one impact,
the best-placed one is used (audio-only impacts have no x or z). A front-wall event after a floor
bounce more than 1.5 m from the front wall is dropped (`fw_after_floor`): it is a later shot.

Shot type, first match wins (constants are named in `shots.py`, each with its reason):

1. boast: a side wall is hit before the front wall, at least 0.5 m in front of the hitter. With no
   front-wall event it counts only within 0.5 s of the hit and 4 m ahead of the hitter.
2. lob: front-wall z >= 3.2 m.
3. unknown (`front_wall_below_tin_top`): placed z under the tin top. Never called a kill.
4. front-wall z < 0.9: kill if foot-to-wall speed >= 13 m/s; drop if short and < 11 m/s; else unknown.
5. drop: bounce short, speed < 11 m/s, z unknown or < 1.5.
6. drive: z known, under 3.2, not taken above. A fast ball that lands short is still a drive (`length` says short).
7. no z: drive if deep and >= 13 m/s; lob if deep and < 8 m/s; else unknown.

Speed is mean foot-to-front-wall; NaN if the flight is under 0.2 s or over 40 m/s (timing error).

Other fields: `direction` (straight/cross; unknown within 0.4 m of the middle line), `length`
(deep/mid/short from the spec lines), `tight` (bounce within 0.6 m of a side wall or a side-wall
event after the front wall), `volley` (serve / bounce / volley / unknown: a floor event between
hits means bounce; else ball height at the hit, >= 1.0 m volley, <= 0.4 m bounce). Each has a
`*_source` column. Serves are often lobs; filter `volley == "serve"` when studying shot types.

## Ending table (one row per rally)

`ending` is `tin`, `out`, `not_returned` or `unclear`; `inferred_winner` is the last hitter for
not_returned, the opponent for tin and out, `none` for unclear. `confidence` is an ordinal 0-1
score, not a probability; unclear rows carry 0.

- out: front-wall z over the out line, side-wall z over its sloped line, or back-wall z over its line.
- tin: placed front-wall z under the tin top, unless the ball visibly came back (floor bounce more than
  1.5 m out, or >= 3 `ball_3d` samples back in the court above 0.3 m): then unclear.
- not_returned: needs a double bounce. One bounce or none: unclear. Audio-only front wall plus a double bounce: lower confidence.
- unclear: no front-wall event after the last hit; or an opponent racket event >= 0.3 s after it.

## Limits that matter

- Lets and strokes are not detected. A let or stroke rally ends as whatever the ball did next; the winner is the player the rules would credit, not the referee's.
- Missed final hit: hit recall is about 0.87, so the "last hit" is sometimes an earlier hit and the rally continued. Then the ending and the winner are shifted by one shot. This is the main source of wrong not_returned rows.
- Hidden impacts: the front wall is far and small, a door frame hides the centre, and a ball flying along the line of sight barely moves in the image.
- Only about 43% of shots have a placed front-wall impact. Front-wall x clusters near the walls, so direction and tight from front-wall x are weaker than from the bounce. Front-wall z has a median near 2 m and may be biased.
- Drop vs kill is barely separable; volley height evidence is weak; side-wall-first without a front-wall event is the weakest boast rule.
- No `out` ending occurs in the three clips; that rule is tested only on synthetic data.

## Coverage per clip (share of shots with the field known)

| Clip | Shots | Rallies | fw seen | fw placed | bounce | side_first | volley | shot_type | direction | length | tight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IMG_4089 | 178 | 29 | 52.8% | 42.7% | 35.4% | 54.5% | 51.7% | 41.6% | 50.0% | 35.4% | 43.3% |
| IMG_4090 | 125 | 15 | 51.2% | 43.2% | 43.2% | 55.2% | 64.0% | 41.6% | 56.0% | 43.2% | 48.8% |
| IMG_4091 | 114 | 17 | 50.0% | 43.0% | 43.9% | 51.8% | 62.3% | 39.5% | 54.4% | 43.9% | 46.5% |

## Mixes (pooled, 417 shots)

Shot type: A drive 74, boast 6, drop 2, kill 3, lob 2, unknown 120. B drive 62, boast 10, kill 4, lob 8, unknown 126.
Endings: IMG_4089 unclear 27, tin 1, not_returned 1; IMG_4090 unclear 14, not_returned 1; IMG_4091 unclear 14, not_returned 3.
Of 61 unclear, 39 had no front-wall event after the last hit.

## Validation

By eye on contact sheets, 23 shots and 17 endings; see `shots_validation.md`. Thresholds were fitted on
the same three clips, so the numbers are in-sample.

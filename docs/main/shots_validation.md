# Shots and endings: validation by eye

Method: contact sheets (frames around the hit and the rally end) for IMG_4089, 4090, 4091, viewed by
one person. Limits: the front wall is far and small, the ball is a tiny circle, a dark door frame hides
the centre, a ball flying along the line of sight barely moves. Side-wall-first cannot be checked reliably
by eye. Labels are the final code's output. **The thresholds were tuned after looking at these same
clips, so everything below is in-sample.** The fixes listed at the end were made from clear errors seen here.

## Endings (17 rallies)

| Clip rally | Label | Verdict | Why |
|---|---|---|---|
| 4091 r10 | unclear | right | was tin; ball visibly kept flying (fixed) |
| 4089 r12 | unclear | right | was tin; ball kept flying (fixed) |
| 4089 r8 | unclear | right | was not_returned; rally already over, no real last hit |
| 4089 r15 | unclear | right | was not_returned; rally continued after a missed hit at 314.6 s |
| 4091 r3 | not_returned | right | ball died |
| 4090 r12 | not_returned | right | ball died |
| 4091 r2, 4090 r5, 4089 r19, 4091 r14, 4091 r1, 4090 r7 | unclear | right | nothing decisive visible |
| 4089 r22 | tin (z 0.41) | uncertain | cannot see the impact |
| 4091 r6, r11 | not_returned | uncertain | ball rolled to the back-right corner; plausible |
| 4089 r26 | not_returned | uncertain, leaning wrong | dark player kept playing at the left; a hit near 551.5 s may be missed |
| 4089 r3 | unclear | uncertain | |

Tally: 12 right, 5 uncertain, 0 clearly wrong in the final labels (4 wrong before the fixes).
Unclear is a cheap label; the informative classes are thin: tin 1 (uncertain), not_returned 5
(2 right, 3 uncertain), out 0.

## Shots (23)

| Shot | Label | Verdict |
|---|---|---|
| 4090 r4 s16; 4089 r3 s7 | boast | right |
| 4091 r2 s3; 4090 r4 s12; 4089 r3 s12 | lob | right |
| 4091 r7 s4; 4089 r3 s4 | kill | right |
| 4091 r16 s2; 4090 r3 s2; 4090 r2 s2; 4089 r24 s2 | drive | right |
| 4089 r2 s7; 4091 r1 s6; 4089 r19 s2; 4091 r14 s3 | unknown | right to be unknown (r14 s3 was a 43.6 m/s kill, now capped) |
| 4089 r18 s13 | drop | wrong: no real shot visible, rally over |
| 4091 r6 s2; 4089 r4 s12; 4090 r15 s18 | boast | uncertain (side-wall-first not checkable) |
| 4089 r15 s4 | drop | uncertain, leaning right |
| 4089 r6 s3 | unknown (was kill) | uncertain |
| 4089 r3 s5; 4091 r9 s3 | drive | uncertain |

Tally: about 14 right (counting appropriate unknowns), 1 wrong, 8 uncertain.

## Rule changes made from clear errors

Tin-height impacts no longer called kills (11 shots); boast without a front-wall event limited to 0.5 s
and 4 m ahead; drive no longer needs a non-short bounce; speed invalid under 0.2 s flight or over 40 m/s;
front-wall event after a floor bounce discarded; tin becomes unclear if the ball is seen returning;
not_returned requires a double bounce.

## Weakest

Boast without a front-wall event; front-wall x and z bias; spurious or missed last hit (lets and strokes
are not seen either); drop vs kill; volley height; 42% of shots unplaced.

# Live twin runs re-scored with the current rules

The recorded live runs (`tests/api/fixtures/live`, 1920x1080, and `tests/api/fixtures/live_phone`, 1280x720 at a phone-like bitrate) re-scored offline with `python scripts/rescore_live.py --set grain|phone` after the harmonic fix and the artifact rule. The recordings are not modified.

## 1920x1080

| fault | Spark phrases | skipped as not clear | SAM missed | IoU washing machine | IoU door | rpm range (true 1000) | rule verdict (SAM masks) | Spark fault | Spark agrees | evidence v/c/d | Spark confidence | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| healthy | washing machine, circular front door, front control panel | - | front control panel | 0.981 | 0.955 (circular front door) | 126 to 1600 | none (ok) | none | yes | 0/4/0 | medium | $0.0590 |
| imbalance | washing machine, circular front door, front control panel | - | front control panel | 0.982 | 0.962 (circular front door) | 126 to 1600 | imbalance (ok) | imbalance | yes | 2/2/0 | high | $0.0626 |
| looseness | washing machine, round front door, top control panel | - | - | 0.981 | 0.959 (round front door) | 400 to 1600 | looseness (ok) | looseness | yes | 4/1/0 | high | $0.0698 |
| misalignment | washing machine, front control panel, round glass door | - | front control panel | 0.981 | 0.961 (round glass door) | 126 to 1600 | misalignment (ok) | misalignment | yes | 2/3/0 | high | $0.0617 |

| fault | plan (Spark) | segment (SAM) | diagnose (Spark) | total |
|---|---|---|---|---|
| healthy | 0.0110 | 0.0360 | 0.0120 | 0.0590 |
| imbalance | 0.0141 | 0.0360 | 0.0125 | 0.0626 |
| looseness | 0.0097 | 0.0360 | 0.0240 | 0.0698 |
| misalignment | 0.0097 | 0.0360 | 0.0159 | 0.0617 |

Re-scored with the current rules; changed: misalignment: looseness -> misalignment.

## 1280x720, phone-like

| fault | Spark phrases | skipped as not clear | SAM missed | IoU washing machine | IoU door | rpm range (true 1000) | rule verdict (SAM masks) | Spark fault | Spark agrees | evidence v/c/d | Spark confidence | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| healthy | washing machine, washer door | - | - | 0.982 | 0.973 (washer door) | 400 to 1600 | none (ok) | none | yes | 0/4/0 | medium | $0.0584 |
| imbalance | washing machine, washer front door | - | - | 0.984 | 0.975 (washer front door) | 400 to 1600 | imbalance (ok) | imbalance | yes | 3/2/0 | high | $0.0502 |
| looseness | washing machine, washer front door | - | - | 0.984 | 0.975 (washer front door) | 400 to 1600 | looseness (ok) | looseness | yes | 6/2/0 | high | $0.0518 |
| misalignment | washing machine, washer front door | - | - | 0.984 | 0.974 (washer front door) | 126 to 1600 | misalignment (ok) | misalignment | yes | 2/2/0 | high | $0.0460 |

| fault | plan (Spark) | segment (SAM) | diagnose (Spark) | total |
|---|---|---|---|---|
| healthy | 0.0152 | 0.0240 | 0.0192 | 0.0584 |
| imbalance | 0.0128 | 0.0240 | 0.0134 | 0.0502 |
| looseness | 0.0111 | 0.0240 | 0.0167 | 0.0518 |
| misalignment | 0.0102 | 0.0240 | 0.0118 | 0.0460 |

Re-scored with the current rules; no verdict changed.

## What moved

- 1920x1080 misalignment: looseness to misalignment (the harmonic fix; a 4x at 2% of 2x no longer counts as a harmonic). Spark now agrees, and its confidence is no longer capped.
- 1280x720 healthy: the reading falls from 6.10 to 2.50 mm/s (alert 7). Nine artifact peaks are left out, among them 83.4 Hz and 116.7 Hz, which were labelled 5x and 7x: they are 100 Hz flicker minus and plus running speed. The verdict stays none.
- Other runs: verdicts unchanged; velocities move by under 3%.
- The replayed misalignment diagnosis was written when the rules said looseness. The page now notes when a replayed diagnosis was recorded against a different rule verdict.

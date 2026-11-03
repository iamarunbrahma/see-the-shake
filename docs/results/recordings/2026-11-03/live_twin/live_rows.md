# Live twin runs: SAM 3.1 and Muse Spark

One full run per twin fault (6 s, 1920x1080, 240 fps). Spark plans the parts, SAM 3.1 masks them on a 60-frame sub-clip, measurement and rules run on the SAM masks, Spark diagnoses and code verifies it.

| fault | Spark phrases | skipped as not clear | SAM missed | IoU washing machine | IoU door | rpm range (true 1000) | rule verdict (SAM masks) | Spark fault | Spark agrees | evidence v/c/d | Spark confidence | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| healthy | washing machine, washing machine front door | - | - | 0.981 | 0.979 (washing machine front door) | 126 to 1600 | none (ok) | none | yes | 0/2/0 | medium | $0.0539 |
| imbalance | washing machine, washing machine front door | - | - | 0.981 | 0.979 (washing machine front door) | 126 to 1600 | imbalance (ok) | imbalance | yes | 2/2/0 | high | $0.0520 |
| looseness | washing machine, washing machine front door | - | - | 0.982 | 0.980 (washing machine front door) | 126 to 1600 | looseness (ok) | looseness | yes | 6/2/0 | high | $0.0541 |
| misalignment | washing machine, washing machine front door | - | - | 0.979 | 0.981 (washing machine front door) | 400 to 1600 | misalignment (ok) | misalignment | yes | 2/3/0 | high | $0.0483 |

Total API cost: $0.2083. Evidence v/c/d: verified, context (real peak, not what the fault rests on), dropped.

Cost per step (USD):

| fault | plan (Spark) | segment (SAM) | diagnose (Spark) | total |
|---|---|---|---|---|
| healthy | 0.0175 | 0.0240 | 0.0124 | 0.0539 |
| imbalance | 0.0130 | 0.0240 | 0.0150 | 0.0520 |
| looseness | 0.0144 | 0.0240 | 0.0157 | 0.0541 |
| misalignment | 0.0107 | 0.0240 | 0.0136 | 0.0483 |

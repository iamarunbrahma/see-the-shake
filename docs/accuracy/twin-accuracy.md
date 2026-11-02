# Twin accuracy: measurement and rules, ground-truth masks

Offline, no API calls. Masks are the renderer's; the running-speed range is the true rpm +/- 10% (standing in for Spark's range). 6 s clips at 240 fps, FFT bin 0.167 Hz.

**Scoring.** A truth component is scored when its amplitude is at least 10x the measured noise floor. Pass: frequency within 1 bin and amplitude within 20%.

**Truth.** Each part's expected translation is the cabinet translation plus its rocking. The rocking is taken as the structure-weighted least-squares translation of the rotation field over the part's rest image (`twin.rocking_lever`), which is what any translation estimate of a rotating region aims at. The plain centroid displacement, and the error against it, are in `twin-accuracy-rows.csv` (`true_amp_centroid_px`, `amp_err_centroid`).

## Main table: x264 crf 16, -tune grain, 1920x1080

-tune grain keeps the tiny inter-frame changes that sub-pixel vibration makes, as a high-bitrate hardware encoder does. It is kinder than real phone footage; see the phone-like section.

| Fault | Seed | Rule verdict | Correct | Components passed | Max freq err (bins) | Max amp err | Missed | Noise floor (px) |
|---|---|---|---|---|---|---|---|---|
| healthy | 1 | none (medium) | yes | 4/4 | 0.15 | 14.0% | 0 | 0.00014 |
| healthy | 2 | none (medium) | yes | 4/4 | 0.15 | 15.1% | 0 | 0.00014 |
| healthy | 3 | none (medium) | yes | 4/4 | 0.14 | 15.4% | 0 | 0.00014 |
| imbalance | 1 | imbalance (high) | yes | 8/8 | 0.36 | 16.0% | 0 | 0.00020 |
| imbalance | 2 | imbalance (high) | yes | 8/8 | 0.25 | 13.8% | 0 | 0.00018 |
| imbalance | 3 | imbalance (high) | yes | 8/8 | 0.31 | 9.9% | 0 | 0.00021 |
| looseness | 1 | looseness (high) | yes | 16/16 | 0.41 | 14.2% | 0 | 0.00022 |
| looseness | 2 | looseness (high) | yes | 16/16 | 0.38 | 13.8% | 0 | 0.00021 |
| looseness | 3 | looseness (high) | yes | 16/16 | 0.41 | 13.1% | 0 | 0.00021 |
| misalignment | 1 | misalignment (high) | yes | 12/12 | 0.43 | 7.5% | 0 | 0.00018 |
| misalignment | 2 | misalignment (high) | yes | 12/12 | 0.40 | 13.2% | 0 | 0.00019 |
| misalignment | 3 | misalignment (high) | yes | 12/12 | 0.40 | 8.8% | 0 | 0.00020 |

**Verdicts correct: 12/12. Components within tolerance: 120/120.**

## Phone-like encoding (labelled; not the headline): 1280x720, default x264 tuning, 24Mbit/s

Closer to what a phone saves: 720p and an encoder that discards small changes. The bitrate is an assumption until it is read from the real Redmi clips.

| Fault | Seed | Rule verdict | Correct | Components passed | Max freq err (bins) | Max amp err | Missed | Noise floor (px) |
|---|---|---|---|---|---|---|---|---|
| healthy | 1 | none (medium) | yes | 0/4 | 0.20 | 45.2% | 0 | 0.00023 |
| imbalance | 1 | imbalance (high) | yes | 6/8 | 0.36 | 25.1% | 0 | 0.00032 |
| looseness | 1 | looseness (high) | yes | 14/16 | 0.43 | 24.9% | 0 | 0.00031 |
| misalignment | 1 | misalignment (high) | yes | 11/12 | 0.41 | 21.3% | 0 | 0.00031 |

**Verdicts correct: 4/4. Components within tolerance: 31/40.**

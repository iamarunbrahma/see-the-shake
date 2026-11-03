# Twin accuracy: measurement and rules, ground-truth masks

Offline, no API calls. Masks are the renderer's; the running-speed range is the true rpm +/- 10% (standing in for Spark's range). 6 s clips at 240 fps, FFT bin 0.167 Hz.

**Scoring.** A truth component is scored when its amplitude is at least 10x the measured noise floor. Pass: frequency within 1 bin and amplitude within 20%.

**Truth.** Each part's expected translation is the cabinet translation plus its rocking. The rocking is taken as the structure-weighted least-squares translation of the rotation field over the part's rest image (`twin.rocking_lever`), which is what any translation estimate of a rotating region aims at. The plain centroid displacement, and the error against it, are in `twin-accuracy-rows.csv` (`true_amp_centroid_px`, `amp_err_centroid`). The drum's speed drifts 0.4% at 0.2 Hz, which spreads the higher orders over neighbouring bins in any spectrum, so each scored amplitude is the applied amplitude times what a pure line at that order of the drifting speed reads in the pipeline's spectrum over the same window (`evaluate.drift_factor`: 0.99 at 1x, 0.89 at 3x, 0.73 at 5x for 6 s). The applied amplitude is in the CSV too (`true_amp_applied_px`); which components are scored still follows it.

## Main table: x264 crf 16, -tune grain, 1920x1080

-tune grain keeps the tiny inter-frame changes that sub-pixel vibration makes, as a high-bitrate hardware encoder does. It is kinder than real phone footage; see the phone-like section.

| Fault | Seed | Rule verdict | Correct | Components passed | Max freq err (bins) | Max amp err | Missed | Noise floor (px) |
|---|---|---|---|---|---|---|---|---|
| healthy | 1 | none (medium) | yes | 4/4 | 0.15 | 13.1% | 0 | 0.00014 |
| healthy | 2 | none (medium) | yes | 4/4 | 0.15 | 14.2% | 0 | 0.00014 |
| healthy | 3 | none (medium) | yes | 4/4 | 0.14 | 14.5% | 0 | 0.00014 |
| imbalance | 1 | imbalance (high) | yes | 7/8 | 0.36 | 21.6% | 0 | 0.00020 |
| imbalance | 2 | imbalance (high) | yes | 8/8 | 0.25 | 9.6% | 0 | 0.00018 |
| imbalance | 3 | imbalance (high) | yes | 8/8 | 0.31 | 11.0% | 0 | 0.00021 |
| looseness | 1 | looseness (high) | yes | 16/16 | 0.41 | 7.8% | 0 | 0.00022 |
| looseness | 2 | looseness (high) | yes | 16/16 | 0.38 | 9.3% | 0 | 0.00021 |
| looseness | 3 | looseness (high) | yes | 16/16 | 0.41 | 9.6% | 0 | 0.00021 |
| misalignment | 1 | misalignment (high) | yes | 12/12 | 0.43 | 12.7% | 0 | 0.00018 |
| misalignment | 2 | misalignment (high) | yes | 12/12 | 0.40 | 7.1% | 0 | 0.00019 |
| misalignment | 3 | misalignment (high) | yes | 12/12 | 0.40 | 8.0% | 0 | 0.00020 |

**Verdicts correct: 12/12. Components within tolerance: 119/120.**

## Phone-like encoding (labelled; not the headline): 1280x720, default x264 tuning, 24Mbit/s

Closer to what a phone saves: 720p and an encoder that discards small changes. The bitrate is an assumption until it is read from the real Redmi clips.

| Fault | Seed | Rule verdict | Correct | Components passed | Max freq err (bins) | Max amp err | Missed | Noise floor (px) |
|---|---|---|---|---|---|---|---|---|
| healthy | 1 | none (high) | yes | 0/4 | 0.14 | 44.9% | 0 | 0.00023 |
| imbalance | 1 | imbalance (high) | yes | 7/8 | 0.33 | 25.9% | 0 | 0.00033 |
| looseness | 1 | looseness (high) | yes | 16/16 | 0.43 | 18.9% | 0 | 0.00034 |
| misalignment | 1 | misalignment (high) | yes | 12/12 | 0.43 | 10.4% | 0 | 0.00031 |

**Verdicts correct: 4/4. Components within tolerance: 35/40.**

**Regenerated after the artifact rule (Nov 2, evening; superseded below).** The rules began to leave out peaks that are compression or
flicker artifacts: a peak the background moves with too (at 30% of its amplitude or more), or, when flicker
shows, a peak on the flicker pattern (multiples of 20 Hz at 240 fps) or its sidebands with running speed; never
1x, and orders below 5x never count as sidebands. Every component and every verdict in that day's table was
unchanged. Only the velocities moved: phone-like healthy 5.41 to 2.36 mm/s (confidence medium to high, the alert is 7 mm/s);
the other phone-like runs by 0.1 to 0.9%; the 1080p runs by at most 1%.

**Regenerated on Nov 3 after the QA fixes.** Two changes moved the numbers:
- **Scored truth follows the drum's speed drift** (see Truth above). The old truth was the applied amplitude, which
  no spectrum can read once the 0.4% drift spreads a line, so higher orders read low: 3x by 9.0% on average at
  1080p. Against the drift-aware truth the mean amplitude error at 1080p goes from -3.9% to +0.1% (3x: -9.0% to
  +1.7%), and phone-like from -12.6% to -8.9%.
- **The running speed and the energy share** (an rpm range with no real peak is set aside; orders above 6x count in
  the energy). No verdict moved.

Phone-like components go from 31 of 40 to 35 of 40. At 1080p one component now fails, giving 119 of 120: imbalance
seed 1, the door's horizontal 2x. It reads 0.0232 px against 0.0200 applied, 16% high even against the old truth,
and 21.6% high against the drift-aware truth of 0.0191. That is a real over-read, possibly 2x distortion from
the door's 0.24 px 1x, and the old truth had partly hidden it. Every verdict is still right: 12 of 12 and 4 of 4.

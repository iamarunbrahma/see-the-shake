# Real clips: a washer at 120 fps, and a rotor rig against eddy-current probes

Run on Nov 2 2026 with the live API (`python -m shake.pipeline <clip> --api record`) on the two openly licensed
clips in `samples/real/` (credits in `samples/real/README.md`). Recordings are in `tests/api/fixtures/real/`,
replayed offline by `tests/api/test_replay_real.py`. The rotor table comes from `scripts/compare_rotor_probes.py`.
Total API cost: $1.17, including $0.46 on runs that exposed the two plan bugs fixed below.

## A. AEG washer, 1400 rpm spin, handheld phone at 120 fps

`python -m shake.pipeline samples/real/aeg-washer-120fps/aeg-washer-1400rpm-120fps.webm --api record --ref "washing machine=600"`.
No rpm was given: Spark's range (400 to 1600 rpm) chose the running speed.

| | result |
|---|---|
| capture rate | 120 fps, from the `com.android.capture.fps` tag |
| Spark's parts | washing machine, washer front door (both found by SAM 3.1) |
| running speed | 22.75 Hz (1365 rpm), with 2x at 45.5 Hz; 22.79 and 22.72 Hz in the two halves of the clip |
| hand shake | background moves 0.1 to 0.4 px at 2 to 12 Hz; after subtraction the parts keep 0.03 to 0.09 px there, and the artifact rule flags all of it as whole-frame |
| vibration | door region 1x: 0.19 px horizontal, 0.41 px vertical (0.27 mm at 0.657 mm/px); cabinet 1x: 0.06 px |
| rule verdict | imbalance (high): 1x holds 95% of the vibration energy, on the door region (33 mm/s RMS) |
| Spark | imbalance, high, agrees: "a spin imbalance from uneven laundry"; 2 verified, 2 context, 0 dropped |
| visual check | door, vertical: verified |
| time, cost | 170 s, $0.055 |

- **The verdict makes sense for a real machine.** The door region moves 6 times more than the cabinet, likely
  because the tub on its springs shows through the glass. That tub motion is pure 1x, the signature of an uneven
  load in spin.
- **Running speed:** 22.75 Hz rather than the 24.0 Hz from the quick check. The clip shows no peak at 24 Hz, and
  both halves agree within 0.07 Hz. 22.75 Hz is 2.5% under the nominal 1400 rpm, which is usual under load. The
  untrimmed original (0 to 8 s and 18.5 to 26.9 s) is not in the repo, so a different speed there cannot be
  ruled out.
- **Hand shake is removed for the verdict, but not fully from the motion.** Subtraction cuts it about 5 times.
  The rest is likely camera rotation and zoom, which a translation reference cannot cancel. The whole-frame
  artifact rule then leaves out every one of those peaks, so none of it reaches the features or the verdict.

## B. XJTU rotor rig, 500 fps camera against eddy-current probes

`python -m shake.pipeline samples/real/xjtu-rotor-500fps/<run>.avi --api record --capture-fps 500 --rpm <rpm>`,
with `--parts "large metal disc, long metal shaft"` for runs 2 to 9 (the phrases Spark chose on run 1), so the
runs compare like for like.

- **Probe units:** the CSV values are millimetres. The smallest step in every file is 3.157e-5, which is a 16-bit
  converter over +/-10 V (20.69 V / 65536) divided by the 10 V/mm sensitivity. Probes #2 and #3 clip at 2000 rpm
  (flat tops on 8 to 46% of samples), so they are left out there.
- **Scale:** in the STEP model, the disc ("150 disc") has radius 75 mm and the shaft 11.5 mm in its main span.
  The disc's SAM mask is 244 to 248 px tall, which gives 0.605 to 0.615 mm/px.
- **Axis:** the camera looks at the rig side-on and the probes sit under the shaft, so the video's vertical axis
  is compared with the probes.

| run | 1x video / probes (Hz) | disc 1x (um) | shaft 1x (um) | probe #1 1x (um) | probes 1x range (um) | 2x/1x video / probes | vs Unb1: disc, shaft, probe #1 | rules | Spark | visual check | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000rpm-Unb1 | 16.66 / 16.67 | 22.6 | 20.8 | 25.1 | 19 to 46 | 1.84 / 0.96 to 1.37 |  | looseness | misalignment (capped at low) | not verified | $0.092 |
| 1000rpm-Unb2 | 16.66 / 16.67 | 30.1 | 29.7 | 26.0 | 21 to 47 | 1.13 / 1.11 to 1.48 | 1.33, 1.43, 1.03 | looseness | misalignment (capped at low) | partly verified | $0.089 |
| 1000rpm-Unb3 | 16.66 / 16.67 | 15.8 | 20.8 | 26.1 | 20 to 49 | 2.64 / 1.04 to 1.34 | 0.70, 1.00, 1.04 | looseness | misalignment (capped at low) | not verified | $0.073 |
| 2000rpm-Unb1 | 33.35 / 33.34 | 319.9 | 280.2 | 284.8 | 184 to 285 | 0.04 / 0.02 to 0.03 |  | imbalance | imbalance | verified | $0.060 |
| 2000rpm-Unb2 | 33.35 / 33.33 | 435.3 | 402.0 | 407.4 | 237 to 407 | 0.04 / 0.01 to 0.02 | 1.36, 1.43, 1.43 | imbalance | imbalance | verified | $0.065 |
| 2000rpm-Unb3 | 33.35 / 33.34 | 390.8 | 359.2 | 356.1 | 224 to 356 | 0.05 / 0.01 to 0.03 | 1.22, 1.28, 1.25 | imbalance | imbalance | partly verified | $0.074 |
| 3000rpm-Unb1 | 50.01 / 50.00 | 38.4 | 33.4 | 38.5 | 33 to 49 | 0.28 / 0.01 to 0.05 |  | looseness | imbalance (capped at low) | verified | $0.072 |
| 3000rpm-Unb2 | 50.00 / 50.00 | 45.2 | 28.4 | 46.9 | 25 to 64 | 0.10 / 0.01 to 0.05 | 1.18, 0.85, 1.22 | looseness | imbalance (capped at low) | verified | $0.064 |
| 3000rpm-Unb3 | 50.00 / 50.00 | 54.6 | 47.0 | 55.8 | 55 to 71 | 0.23 / 0.01 to 0.05 | 1.42, 1.41, 1.45 | looseness | imbalance (capped at low) | verified | $0.060 |


The "vs Unb1" column gives 1x amplitude relative to the same speed's Unb1 run. Each rotor run took 86 to 146 s.

- **1x frequency matches the probes within 0.02 Hz on all nine runs.** That is from 1 s of video, with a
  1 Hz bin, thanks to peak interpolation.
- **Absolute amplitude matches where the signal is strong.** At 2000 rpm the shaft reads 280, 402 and 359 µm
  against probe #1's 285, 407 and 356 µm, within 2%. At 3000 rpm the disc reads 38.4, 45.2 and 54.6 µm against
  probe #1's 38.5, 46.9 and 55.8 µm, within 4%. At 1000 rpm the readings fall inside the probes' range, but the
  motion is about 0.03 px and noisier. Probe #1 seems to sit nearest the disc. The probes' positions along the
  shaft are not in the dataset notes.
- **Unbalance ratios match at 2000 and 3000 rpm.** At 2000 rpm, Unb2 and Unb3 against Unb1 are 1.36 and 1.22
  (disc), 1.43 and 1.28 (shaft), and 1.43 and 1.25 (probe #1). At 3000 rpm, the disc's 1.18 and 1.42 match probe
  #1's 1.22 and 1.45. At 1000 rpm the unbalance masses barely change the probes' 1x (within 10%), and the video
  does not resolve that.
- **Verdicts:**
  - **2000 rpm:** imbalance on all three, and Spark agrees.
  - **1000 rpm:** the probes show 2x as large as 1x (2x/1x 0.96 to 1.48), likely 2x running speed exciting the
    first critical speed. Spark said misalignment from that 2x every time. The rules said looseness from 3x to
    6x harmonics that the probes do not show.
  - **3000 rpm:** the probes show a clean 1x (2x/1x under 0.05) and Spark said imbalance. The rules said
    looseness, because 2x (100 Hz) and 4x (200 Hz) sit on the light flicker and 3x and 5x on its sidebands.
  - Spark's verdict matches what the probes support on all nine runs; the rule verdict on three. On the other
    six, the card did what the design asks: the rule verdict stayed the headline, Spark was capped at low, and
    both were shown.
- **Visual check:** verified on 5 runs, partly verified on 2, not verified on 2. Spark reads the vertical
  motion well and sometimes calls the disc "both".

## What broke

Fixed in this branch:

1. **The plan dropped the parts that vibrate.** Spark saw a "balancing machine" with a "shaft" and a "disk", and
   the plan dropped both as rotating parts. Spark now marks each part's rotation, and a smooth part whose outline
   stays the same as it turns (a shaft, a solid disc) is kept and measured directly, as the probes do (c03a57a).
   On the next run Spark named "long metal shaft" and "large metal disc", and SAM 3.1 found both (confidence
   0.85 and 0.95).
2. **Parts typed by the user were filtered too.** `--parts "large metal disc, long metal shaft"` lost the shaft
   to the rotating-name filter. A user's parts are now never dropped (afa3e0f).

Needed in files this session does not own:

1. **The running speed picks a harmonic when the range is wide** (`measure.running_speed`). With Spark's 300 to
   3600 rpm range and no rpm given, the strongest peak was the 2x (33.4 Hz). Every order was then mislabelled and
   the verdict was looseness (0.5x at 0.76 of 1x). With `--rpm 1000` it was right. Suggestion: when the range
   covers both f and f/2 and both peaks are strong, warn and ask for the rpm on the page.
2. **A moving part left out of the masks corrupts the camera-shake reference** (`measure` / `pipeline`). With
   only the disc masked, the shaft, which crosses the whole frame and moves up to 0.4 mm, counted as
   "background" and was subtracted. The disc then read 141 µm at 2000 rpm Unb1. With the shaft masked it read
   320 µm (probe #1: 285 µm). Suggestion: leave every SAM mask, including the whole machine's, out of the
   background, and warn when the background moves in step with the parts at 1x.
3. **Orders that coincide with light flicker count as harmonics** (`fingerprint`). At 3000 rpm under 50 Hz mains,
   2x is 100 Hz, 4x is 200 Hz, and 3x and 5x sit on the flicker sidebands, so every rotor run at 3000 rpm read
   looseness. Two-pole motors near 3000 rpm are common in small plants, so this will happen on real pumps and
   motors. Suggestion: an order within a bin of a flicker frequency or sideband should not count toward the
   harmonics, and the page should say the running speed coincides with mains (film in daylight or under DC
   light). This extends the 6x-on-flicker label already queued.
4. **Surface texture turning with a part adds harmonics** (`measure`). At 1000 rpm the video shows 3x to 5x at
   20 to 60% of 1x, where the probes show under 6%. The horizontal axis of a horizontal shaft has almost no edges
   to track, and marks on the turning surface repeat every revolution. Suggestion: report per-axis tracking
   confidence (the structure tensor's conditioning) and down-weight a poorly conditioned axis, or require a
   harmonic on the well-conditioned axis.
5. **The scale reference uses mask width only** (`pipeline`, CLI and page). The disc's 150 mm diameter shows as
   its height. `measure.Scale.from_reference` already takes `axis="height"`. The pipeline needs a way to ask for
   it, for example `--ref "large metal disc=150:height"`. This analysis set the scale outside the pipeline.
6. **The synthesised-frame check fires on both real clips** (`ingest`). The washer (VP9, whiteness 0.23) and the
   high-speed camera (MPEG-4) are both flagged "synthesised", and the page then says the spectrum would be
   meaningless. The probes show the spectrum is right. Compression removes the frame-to-frame sensor noise the
   check relies on. Suggestion: report "inconclusive" for heavily compressed clips, or calibrate the threshold
   on real footage.
7. **The codec warnings mislead** (`ingest`). "codec vp9 is not H.264 or H.265; SAM 3.1 needs one of those"
   appears although the pipeline re-encodes the SAM sub-clip to H.264.
8. **Short clips work.** A 1 s window gives a 1 Hz bin, and the page warns that 4 s or more is better. The 1x
   still lands within 0.02 Hz.

## Cost and time

| run | API cost | wall time |
|---|---|---|
| AEG washer | $0.055 | 170 s |
| rotor, 1000 rpm Unb1, first try (no rpm; plan dropped shaft and disc) | $0.058 | 103 s |
| rotor, 1000 rpm Unb1 (Spark's smooth parts, rpm given) | $0.092 | 123 s |
| rotor, 8 runs with the disc only (user-parts bug) | $0.406 | 87 to 113 s each |
| rotor, 8 runs with disc and shaft | $0.557 | 86 to 146 s each |
| total | $1.168 | |

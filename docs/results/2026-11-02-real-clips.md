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

## Before and after the fixes, on the same recordings

Replayed on Nov 2 at no cost with `python scripts/replay_real.py --rotor-ref-height`: Spark's plan and diagnosis and
the SAM 3.1 masks come from the recordings, and ingest, measurement, the rules and magnification are the current
code. The table comes from `python scripts/compare_rotor_probes.py --runs out/real_replay`. The rotor runs now take
their mm scale from the disc's 150 mm diameter as its mask height (`--ref "large metal disc=150:height"`), so the
7 mm/s alert applies. The recordings had no scale.

| run | disc 1x (um) | probe #1 1x (um) | 2x/1x disc / probes | 3x to 6x over 1x: shaft, disc / probes | velocity (mm/s): video / probes | rules before | rules now | pattern without the alert |
|---|---|---|---|---|---|---|---|---|
| 1000rpm-Unb1 | 18.5 | 25.1 | 1.71 / 0.96 to 1.37 | 0.10, 0.38 / 0.06 or less | 7.2 / 3.7 to 7.6 | looseness | looseness | looseness |
| 1000rpm-Unb2 | 30.2 | 26.0 | 1.10 / 1.11 to 1.48 | 0.04, 0.09 / 0.05 or less | 6.7 / 4.0 to 8.5 | looseness | none | misalignment |
| 1000rpm-Unb3 | 16.1 | 26.1 | 1.98 / 1.04 to 1.34 | 0.09, 0.68 / 0.06 or less | 6.7 / 3.8 to 8.5 | looseness | none | misalignment |
| 2000rpm-Unb1 | 299.4 | 284.8 | 0.01 / 0.02 to 0.03 | 0.06, 0.03 / 0.08 or less | 47.5 / 29.3 to 42.7 | imbalance | imbalance | imbalance |
| 2000rpm-Unb2 | 437.4 | 407.4 | 0.02 / 0.01 to 0.02 | 0.10, 0.02 / 0.11 or less | 67.9 / 40.1 to 61.2 | imbalance | imbalance | imbalance |
| 2000rpm-Unb3 | 358.5 | 356.1 | 0.03 / 0.01 to 0.03 | 0.07, 0.06 / 0.08 or less | 56.7 / 35.8 to 53.4 | imbalance | imbalance | imbalance |
| 3000rpm-Unb1 | 37.9 | 38.5 | 0.12 / 0.01 to 0.05 | 0.04, 0.08 / 0.08 or less | 8.5 / 8.4 to 11.0 | looseness | imbalance | imbalance |
| 3000rpm-Unb2 | 39.1 | 46.9 | 0.07 / 0.01 to 0.05 | 0.08, 0.07 / 0.06 or less | 9.1 / 5.8 to 14.3 | looseness | imbalance | imbalance |
| 3000rpm-Unb3 | 63.8 | 55.8 | 0.03 / 0.01 to 0.05 | 0.06, 0.08 / 0.05 or less | 14.3 / 12.3 to 15.8 | looseness | imbalance | imbalance |

Column notes:
- "3x to 6x over 1x" is the largest of those orders over 1x on the vertical axis, the one the probes read. It
  includes orders on the light flicker, which the rules leave out. The disc's 0.38 and 0.68 at 1000 rpm are mostly
  5x and 6x, which sit on 100 Hz flicker and its sideband.
- The video's velocity covers both axes of the lead part; each probe's covers one vertical axis.
- "Pattern without the alert" is the rules' verdict with the alert level set to zero.

Verdicts:
- **The washer** still reads imbalance (1x holds 95% of the vibration energy).
- **Before:** the rules matched the probes on 3 of 9 rotor runs.
- **Now:** they match on 6 of 9, and on the pattern at 1000 rpm for 2 of the other 3.

**What each fix did on these clips:**

1. **Synthesised frames.** The washer (VP9) and the rotor clips (MPEG-4) now read "inconclusive", with the reason
   that compression removes the sensor noise the check relies on. Before, they read "synthesised", and the page said
   the spectrum would be meaningless. Twin clips still read "ok", both H.264 at 1080p and the phone-like 720p.
   These still read "synthesised":
   - a frame-blended or motion-interpolated copy;
   - a phone clip tagged 480 fps or more.
2. **Background.** Every SAM mask is left out of the camera-shake reference. The whole machine is also segmented
   when it is not one of the parts; a replay does this only when its recording holds that call, so these runs don't
   use it. A warning appears when the background moves at 1x with 30% or more of the part's 1x.
3. **Running speed.** Without `--rpm` and with Spark's wide range, 1000rpm-Unb1 and Unb2 now take 16.67 Hz, where
   Unb1 took 33.4 Hz. The page says the speed is ambiguous (2000 rpm also fits) and asks for the rpm. The 2000 rpm runs
   are not flagged.
4. **Flicker.** An order on the flicker pattern or its sidebands counts only at 50% of 1x or more, so a real 2x
   still counts. All three 3000 rpm runs move from looseness to imbalance, and the page says the running speed
   coincides with the mains.
5. **Weak axis and turning texture.**
   - **Weak axis.** The shaft's horizontal axis gets 1 to 2% of its vertical axis's edge information, so it is left
     out, and the page says why.
   - **Turning texture.** Each part now also leaves out pixels whose change its own shift does not explain, such
     as marks turning with the surface. The part is tracked again when these cover 10% or more of it. On the rotor
     that is 19,000 to 23,000 px of the shaft and 5,000 to 12,000 of the disc; it never happens on the washer or the
     twin clips.
   - **Effect on the shaft.** At 1000 rpm its vertical 3x to 6x fall from 21 to 25% of 1x to 4 to 10%. The probes
     show 6% or less.
6. **Codec warning.** Gone. The pipeline re-encodes SAM's sub-clip to H.264 itself.
7. **Scale by height.** `--ref "part=150:height"` on the command line, and a width or height choice on the page.

**Still off:**

- **1000 rpm severity.** The probes read 3.7 to 8.5 mm/s on one axis, with 5.4 to 6.0 at probe #1 near the disc.
  That is right at the 7 mm/s alert, and the video's 6.7 to 7.2 mm/s on two axes lands on both sides of it. Unb2
  and Unb3 read none, with the probes' misalignment pattern underneath (2x at 1.1 times 1x).
- **1000rpm-Unb1** still reads looseness. The disc's 3x to 5x are 16 to 25% of its 2x, and the probes show under
  6%. Texture on the disc's face and rim survives the 3x-noise threshold. Clearing it would take a rule change,
  for example counting harmonics only well above 10% of the largest order, and this was not made.
- **1x amplitude moves with the pixels tracked.**
  - At 2000 rpm the disc came closer to probe #1: within 1 to 7%, where it was within 7 to 12%. The shaft stayed
    within 3%.
  - At 3000 rpm the disc reads 38, 39 and 64 um against 38.5, 46.9 and 55.8 um, where it was within 4%. The shaft
    reads lower, 19 to 38 um, where it read 28 to 47.
  - The shaft's 1x is an average along a shaft whose motion changes along its length, so it depends on which pixels
    remain.
- **Twin accuracy is unchanged** row for row: 12/12 and 120/120 at 1080p, 4/4 and 31/40 phone-like. No twin part
  reaches the 10% gate, so none is tracked again. On the two twin clips checked, texture covered at most 4.4% of a
  part.

## Cost and time

| run | API cost | wall time |
|---|---|---|
| AEG washer | $0.055 | 170 s |
| rotor, 1000 rpm Unb1, first try (no rpm; plan dropped shaft and disc) | $0.058 | 103 s |
| rotor, 1000 rpm Unb1 (Spark's smooth parts, rpm given) | $0.092 | 123 s |
| rotor, 8 runs with the disc only (user-parts bug) | $0.406 | 87 to 113 s each |
| rotor, 8 runs with disc and shaft | $0.557 | 86 to 146 s each |
| total | $1.168 | |

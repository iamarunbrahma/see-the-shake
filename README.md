# See the Shake

Every common machine fault leaves a vibration fingerprint: an unbalanced rotor shakes once per revolution (1x), a
misaligned shaft shows a strong 2x, a loose mount rattles at 0.5x and a row of harmonics. Factories read these with
accelerometers and trained analysts; a small workshop or a home has neither, but it has a phone that films 240 fps
slow motion. See the Shake turns that clip into a diagnosis: Muse Spark plans which parts to watch, SAM 3.1 finds
them, our code measures each part's sub-pixel vibration and plays the clip back with the shake magnified so you can
see it, fixed rules name the fault from the measured peaks, and Muse Spark explains it and suggests a fix, with every
claim it makes checked against the measurement.

Built for the Meta Global AI Developer Hackathon (Indie track). Every model call goes through the Meta Model API.

## Quickstart

You need Python 3.10 or later and ffmpeg (`apt install ffmpeg` or `brew install ffmpeg`).

```bash
pip install -r requirements.txt
```

```bash
export MODEL_API_KEY=your-meta-model-api-key
```

```bash
uvicorn shake.web.app:app --port 8000
```

Open http://localhost:8000. The key is only for live runs. With no key at all, open the **Digital twin** tab and pick
**Recorded live run, no cost**: it replays a real Muse Spark and SAM 3.1 run on a rendered washing machine (the first
time renders and measures the clip, a few minutes). From the command line, the same zero-cost replay is
`python -m shake.pipeline --twin imbalance --api replay --seconds 6`. To analyse your own clip, use the **Your clip**
tab with Muse Spark and SAM 3.1 on.

## How it works

1. **Ingest.** Reads the clip on its capture time base (phones often save 240 fps as a slowed 30 fps file), refuses
   clips below 120 fps, flags synthesised in-between frames (or says the check is inconclusive when compression
   has removed the sensor noise it needs), and picks a steady window of up to 10 s.
2. **Plan (Muse Spark).** Names the machine, a running-speed range and up to four parts to measure.
3. **Find the parts (SAM 3.1).** One stable mask per part phrase.
4. **Measure.** Sub-pixel tracking of each part minus the background (camera shake; every masked part stays out
   of it), with light flicker, compression artifacts, spinning content (drums, blades) and marks turning on a part's
   surface left out. An axis a part's edges cannot pin down (a horizontal shaft's horizontal) is left out of the
   rules. A spectrum per part, peaks labelled as orders of running speed, in mm when you give one reference length.
5. **Fingerprint.** Fixed rules give the official verdict: none, imbalance, misalignment, looseness or unclear (and,
   for a fan with a known blade count, high vibration at blade pass), citing the peaks it rests on.
6. **Magnify.** Each part is moved by alpha times its motion around 1x over an inpainted background; original and
   magnified play side by side.
7. **Diagnose (Muse Spark).** A plain-words explanation, fix steps and what to check next, verified by code.

## How the Meta models are used

**Muse Spark (`muse-spark-1.3`)** has three jobs:

- **Plan.** It watches a 4 s, 480p real-time preview and returns JSON: machine type, a running-speed range (which
  only gates which peak counts as 1x) and part phrases. Code normalises the phrases and drops rotating parts such as
  blades and drums, which are measured through the housing that holds them.
- **Diagnose.** It gets the peak table, the features, the rule verdict, the spectrum image and the side-by-side
  magnified clip, and returns a fault, evidence as `[part, frequency, order]`, an explanation and fix steps. Code
  checks each evidence item against the measured peaks within one FFT bin: verified, context (a real peak that is
  not what the fault rests on) or dropped. When Spark disagrees with the rules or cites a peak that is not there,
  its confidence is capped at low and both are shown; the rule verdict stays the headline.
- **Check the video.** Spark also says which part moves most in the magnified half and in which direction. Code
  checks that against the measured motion in the magnification band (rocking is marked unverifiable, since tracking
  measures translation).

**SAM 3.1 (`sam-3.1`)** gets one streaming request per part phrase on a 60-frame sub-clip at source resolution
(uploaded once with the Files API). Vibration is sub-pixel, so one stable mask per part (a per-pixel majority over
8 decoded frames) serves the whole window. A part SAM cannot find is reported; if none are found, masks or boxes you
supplied are the fallback.

All calls use the OpenAI SDK on the Meta Model API with retries and a cost ledger. Every live run can be recorded
and replayed, so the test suite runs offline. A full run costs about 5 to 8 cents.

## Accuracy

- **Digital twin, renderer's masks, offline** ([docs/accuracy/twin-accuracy.md](docs/accuracy/twin-accuracy.md)):
  at 1920x1080, 6 s, 4 faults x 3 seeds, the rule verdict is right 12 of 12 times and 120 of 120 vibration
  components are within one FFT bin and 20% amplitude. A phone-like section (720p, an encoder that discards small
  motion) gets 4 of 4 verdicts and 31 of 40 components; small components read low.
- **Live twin runs with SAM 3.1 and Muse Spark** ([docs/results/2026-11-02-live-twin.md](docs/results/2026-11-02-live-twin.md)):
  SAM 3.1's masks match the renderer's at IoU 0.95 to 0.98; in the latest round the rules are right on 5 of 5 runs and
  Spark agrees on 4 of 5, at 108 to 197 s and $0.05 to $0.08 per run. Re-scored with the current rules:
  [docs/accuracy/live-rescore.md](docs/accuracy/live-rescore.md).
- **Real clips** ([docs/results/2026-11-02-real-clips.md](docs/results/2026-11-02-real-clips.md)): a phone-filmed
  washing machine at 120 fps and a laboratory rotor rig at 500 fps with displacement probes for reference. The 1x
  is within 0.02 Hz of the probes on all 9 rotor runs. The rule verdict matches what the probes support on 6 of 9
  (imbalance at 2000 and 3000 rpm); at 1000 rpm, where the rig sits right at the alert level, 2 of 3 show the
  probes' 2x pattern. The washer reads imbalance.
- **A second machine:** a pedestal fan twin (healthy, imbalance, looseness) gets 3 of 3 verdicts, measured through its
  housing, guard rim and stand while the blades spin.

## Limits

- **A steady view.** Prop the phone up; it must not touch the machine or be handheld. Camera shake is subtracted using
  the background, so the background must be static and visible.
- **True slow motion.** At least 120 fps captured (240 fps is best). Interpolated slow-motion modes (such as 960 fps
  on many phones) invent frames; ingest flags them, and the spectrum would be meaningless.
- **Translation only.** Each part is tracked as a rigid shift in the image plane. Rocking, twisting and motion toward
  the camera are not measured, and amplitudes are fractions of a pixel, so the smallest components read low on
  compressed phone video.
- **Rules-based verdicts.** Four fault patterns plus blade pass, with thresholds tuned on the twin. Muse Spark
  explains; it does not decide. This is a hackathon prototype, not a certified condition-monitoring or safety tool,
  and it makes no ISO claims.

## Filming tips

- Film 10 s of steady running in 240 fps slow motion (720p is fine), with the phone propped on something that does not
  touch the machine, and a still background in view.
- Keep the original file: transfer by USB or a cloud drive, not a messaging app. If the phone saved the clip as a 30 fps
  file, enter the capture rate (240) when you upload.
- Give one reference length to get millimetres: the machine's width, or a height (a disc's diameter, say).
- **Fans:** the spinning blades cannot be tracked, so film from a 3/4 angle with the motor housing, guard rim and stand
  in view (for a ceiling fan, the housing and downrod) and measure those, never the blade disc. Keep curtains and paper
  out of the frame, since the fan moves them. Enter the blade count so blade pass (3x for three blades) is labelled
  and cannot push the verdict toward a fault pattern.

## Credits

- `samples/real/aeg-washer-120fps/`: "Washing machine 120fps" by Golha, Wikimedia Commons,
  https://commons.wikimedia.org/wiki/File:Washing_machine_120fps.webm, CC BY-SA 4.0, trimmed. Anything made from
  it keeps the credit and the license.
- `samples/real/xjtu-rotor-500fps/`: Kang, Zheng, Liu, Pei, "Vibration Displacement Dataset of Rotors under Various
  Operational Conditions: High-speed Camera Video Data and Eddy current displacement sensor data", Mendeley Data, V1,
  2024, doi:10.17632/47y7rwc2yg.1, CC BY 4.0.
- Details and changes: [samples/real/README.md](samples/real/README.md).

## License

The code is MIT licensed; see [LICENSE](LICENSE). The clips in `samples/real/` keep their own licenses, listed above.

## Reference

**Environment variables:** `MODEL_API_KEY` (live runs only), `MODEL_API_BASE` (default `https://api.meta.ai/v1`),
`SHAKE_RUNS` (where the web app keeps runs, default `./runs`), `SHAKE_RECORDINGS` (recorded live twin runs, default
`tests/api/fixtures/live`). Check the key with one small call:

```bash
python -m shake.api_check
```

**Command line.** `python -m shake.pipeline clip.mp4 --api live --ref "washing machine=600"` runs your clip with Muse
Spark and SAM 3.1 (`--ref "large metal disc=150:height"` takes the length as the mask's height);
`--api off --mask "part=mask.png"` uses your own masks. Other options: `--capture-fps 240`, `--rpm`
or `--rpm-range`, `--blades`, `--parts "phrase, phrase"`. Twins: `--twin imbalance` (washing machine faults: healthy,
imbalance, looseness, misalignment) or `--twin fan-imbalance` (fan: healthy, imbalance, looseness), with `--seconds`
and `--scale`. Results go to `runs/<run id>/results.json` with the videos and plots beside it. Scripts that call the
pipeline need an `if __name__ == "__main__":` guard, or SAM mask decoding falls back to serial.

**Tests** run offline (recorded API responses) in about 2 minutes after a first run that renders small twin clips:

```bash
python -m pytest -q
```

Live API tests cost money and run only with `python -m pytest -m live`. Regenerate the twin accuracy table with
`python scripts/eval_twin.py --seeds 1 2 3 --seconds 6 --phone`, and the live twin runs (about 25 cents) with
`python scripts/live_twin.py --max-usd 1.0`. Replay the real clips through the current code at no cost with
`python scripts/replay_real.py --rotor-ref-height`, then compare with the probes with
`python scripts/compare_rotor_probes.py --runs out/real_replay`.

**Layout.** `shake/` holds the pipeline: `ingest`, `twin` and `twin_fan` (renderers with complete truth), `measure`,
`fingerprint` (rules), `magnify`, `meta_api` (client, cost, record and replay), `plan_parts`, `segment`, `diagnose`,
`pipeline` and `web/` (FastAPI and one page). `scripts/` builds the accuracy tables, `docs/` holds the design, plan
and results, and `reference/spike/` the Oct 27 spike this was ported from.

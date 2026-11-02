# See the Shake

Film a running machine with a phone in slow motion (240 fps). See the Shake finds the machine's parts, measures how
each part vibrates, plays the clip back with that vibration magnified so you can see it, and explains the likely
fault and the fix.

Every common machine fault leaves a vibration fingerprint. An unbalanced rotor shakes once per revolution (1x).
A misaligned shaft shows a strong 2x. A loose mount rattles at 0.5x and a row of harmonics. Factories read these
with accelerometers and trained analysts; a small workshop has neither, but it has a phone.

Built for the Meta Global AI Developer Hackathon (Indie track). Every model call goes through the Meta Model API:
Muse Spark plans the inspection and explains the result, SAM 3.1 finds the parts.

## How it works

1. **Ingest.** Reads the clip on its capture time base (a 240 fps clip saved as slowed 30 fps is handled),
   refuses clips below 120 fps, flags synthesised in-between frames, and picks a steady 4 to 10 s window.
2. **Plan the inspection (Muse Spark).** Spark watches a downscaled copy and names the machine, a plausible
   running-speed range and up to four parts to measure, preferring parts that do not rotate.
3. **Find the parts (SAM 3.1).** One request per part phrase on a 60-frame sub-clip; one stable mask per part.
4. **Measure.** Sub-pixel tracking of each part, minus the background's motion (camera shake), with light flicker
   and rotating content (drum, blades) rejected. Spectrum per part, peaks labelled as orders of running speed,
   in mm when you give one reference length.
5. **Fingerprint.** Fixed rules turn the features into the official verdict: none, imbalance, looseness,
   misalignment or unclear, citing the peaks it rests on.
6. **Magnify.** Each part's layer is moved by alpha times its band-passed motion (around 1x) over an inpainted
   background. Original and magnified play side by side; pixels outside the parts are untouched.
7. **Diagnose (Muse Spark).** Spark explains the result and suggests a fix. Code checks every peak it cites
   against the measurement: each item is marked verified, context or dropped, and Spark's confidence is capped at
   low when it disagrees with the rules or cites something that is not there. The rule verdict stays the headline.

## Setup

Python 3.10 or later and ffmpeg (with ffprobe).

```bash
sudo apt install ffmpeg            # macOS: brew install ffmpeg
```

```bash
pip install -r requirements.txt
```

Environment variables:

| Variable | Needed for | Default |
|---|---|---|
| `MODEL_API_KEY` | Muse Spark and SAM 3.1 (live runs only) | none |
| `MODEL_API_BASE` | Meta Model API base URL | `https://api.meta.ai/v1` |
| `SHAKE_RUNS` | where the web app keeps runs | `./runs` |
| `SHAKE_RECORDINGS` | recorded live twin runs, replayed at no cost | `tests/api/fixtures/live` |

Check the key with one small call (exit code 0 on success; on HTTP 402 the credit is used up):

```bash
python -m shake.api_check
```

Without a key everything except live model calls works: the digital twin supplies its own footage and exact part
masks, and recorded live runs replay offline.

## Run it

### Web app

```bash
uvicorn shake.web.app:app --port 8000
```

Open http://localhost:8000. Two tabs:

- **Digital twin.** Pick a fault (healthy, imbalance, looseness, misalignment), render it and analyse it. Muse
  Spark and SAM 3.1 can be off (the renderer's masks), a recorded live run (no cost), or live. Results are compared
  with the twin's truth.
- **Your clip.** Upload an MP4 or MOV. With Muse Spark and SAM 3.1 on, Spark names the parts and SAM 3.1 masks
  them; masks or boxes you add are the fallback. Optional: capture fps, rpm or rpm range, and one reference length
  (for example "washing machine, 600 mm") for results in mm.

The page shows live progress through the seven steps, the verdict with its peaks, Spark's plan and its checked
diagnosis, the side-by-side video, a spectrum per part, a space-time slice, the API cost, and a JSON report.
Every run is cached under `runs/`, so a repeat run replays instantly.

### Command line

```bash
python -m shake.pipeline --twin imbalance
```

```bash
python -m shake.pipeline --twin imbalance --api replay --seconds 6
```

```bash
python -m shake.pipeline clip.mp4 --api live --ref "washing machine=600"
```

```bash
python -m shake.pipeline clip.mp4 --mask "washing machine=machine.png" --mask "door=door.png" --rpm-range 600 1400
```

`--api` is `off` (default; give `--mask`), `live`, `record` (live, and the responses are saved so the run can be
replayed) or `replay` (with `--cassette`). Other options: `--capture-fps 240` when the phone saved slow motion at a
lower playback rate, `--rpm`, `--parts "phrase, phrase"` to replace Spark's part list, `--alpha 40` for a fixed
magnification. Results go to `runs/<run id>/results.json` with the videos and plots beside it.

Render a twin clip on its own:

```bash
python -m shake.twin looseness out/twin/looseness --seconds 6
```

## Filming tips

- Slow motion at 240 fps (720p is fine). Avoid 960 fps modes: they invent frames, and ingest will flag them.
- Prop the phone up; it must not touch the machine. Film 10 s of steady running.
- Keep the original file: transfer by USB or a cloud drive, not a messaging app.
- If the phone saved slow motion as a 30 fps file, enter the capture rate (240).

## Tests

```bash
python -m pytest -q
```

Offline: the Meta Model API tests replay recorded live responses, so the suite makes no network calls and costs
nothing. The first run renders small twin clips into `out/test_cache/` (a few minutes); later runs take about
1.5 to 2 minutes. Every test has a timeout that dumps all stacks on a hang.

Live API tests cost money and run only when asked:

```bash
python -m pytest -m live
```

## Accuracy

On the digital twin, with the renderer's exact masks (offline, repeatable):
[docs/accuracy/twin-accuracy.md](docs/accuracy/twin-accuracy.md). At 1920x1080, 6 s, 4 faults x 3 seeds: the rule
verdict is right 12 of 12 times, and 120 of 120 vibration components are within one FFT bin and 20% amplitude.
A labelled phone-like section (720p, an encoder that discards small motion) gets 4 of 4 verdicts and 31 of 40
components, with small components reading low.

Live runs with SAM 3.1 and Muse Spark on the twin: [docs/results/2026-11-02-live-twin.md](docs/results/2026-11-02-live-twin.md).
SAM 3.1's masks overlap the renderer's at IoU 0.955 to 0.982, and a full run costs about 6 to 7 cents.
Re-scored with the current rules, including the compression and flicker artifact rule: [docs/accuracy/live-rescore.md](docs/accuracy/live-rescore.md).

Regenerate:

```bash
python scripts/eval_twin.py --seeds 1 2 3 --seconds 6 --phone
```

```bash
python scripts/live_twin.py --max-usd 1.0
```

The second one calls the API (about 25 cents for all four faults) and rewrites the recordings the tests replay.

## Repository layout

| Path | What |
|---|---|
| `shake/ingest.py` | capture fps, synthesised-frame check, steady window |
| `shake/twin.py` | digital twin renderer with injected faults and complete truth |
| `shake/measure.py` | per-part sub-pixel tracking, spectra, orders, mm scale |
| `shake/fingerprint.py` | features and the fixed fault rules |
| `shake/magnify.py` | layer-based magnification and the side-by-side video |
| `shake/meta_api.py` | Meta Model API client: retries, cost ledger, record and replay |
| `shake/plan_parts.py`, `shake/segment.py`, `shake/diagnose.py` | Muse Spark plan, SAM 3.1 masks, Spark diagnosis with code checks |
| `shake/pipeline.py` | the seven steps, cached per run |
| `shake/web/` | FastAPI app and the single page |
| `scripts/` | twin accuracy table and live twin runs |
| `docs/` | design, plan, accuracy and live results |
| `reference/spike/` | the Oct 27 spike this was ported from |

## Limits

See the Shake is a hackathon prototype, not a certified condition-monitoring or safety tool, and makes no ISO
claims. Its thresholds were tuned on the digital twin. Real phone footage carries more noise than the twin, and the
smallest vibration components will read low.

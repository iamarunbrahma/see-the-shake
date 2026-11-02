# Live recordings on real footage

Recorded Meta Model API responses (Muse Spark and SAM 3.1) and the pipeline results for the clips in
`samples/real/`, replayed offline by `tests/api/test_replay_real.py`. See `docs/results/2026-11-02-real-clips.md`.

- `aeg-washer/` is derived from "Washing machine 120fps" by Golha, Wikimedia Commons,
  https://commons.wikimedia.org/wiki/File:Washing_machine_120fps.webm, licensed CC BY-SA 4.0
  (https://creativecommons.org/licenses/by-sa/4.0/). These derived recordings are shared under the same license.
- `rotor-*/` are derived from Kang, Zheng, Liu, Pei, "Vibration Displacement Dataset of Rotors under Various
  Operational Conditions", Mendeley Data, V1, 2024, doi:10.17632/47y7rwc2yg.1, licensed CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/).

`aeg-washer-54492b8/` and `rotor-2000rpm-Unb2-54492b8/` were recorded on main 54492b8 (after the measurement fixes),
for the demo: the real washer (imbalance, frames check inconclusive for VP9) and the rotor at 2000 rpm with the
scale from the disc's 150 mm height (`--ref "large metal disc=150:height"`), against the probes.

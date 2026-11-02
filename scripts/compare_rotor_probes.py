"""Compare the video pipeline on the XJTU rotor clips with the rig's eddy-current probes. Offline: reads the
recorded live runs (tests/api/fixtures/real/rotor-*) and the probe CSVs (samples/real/xjtu-rotor-500fps).

- Probe CSVs are displacement in mm: the smallest step in every file is 3.157e-5, which is a 16-bit converter over
  +/-10 V (20.69 V / 65536) divided by the 10 V/mm sensitivity. Probes #2 and #3 clip at 2000 rpm (flat tops on
  8 to 46% of samples), as the dataset README warns, so they are left out there.
- Video: the vertical axis, since the probes sit under the shaft and read vertical motion; the camera looks at the
  rig side-on. Scale from the disc's 150 mm diameter (STEP model part "150 disc", radius 75 mm) over the height of
  its SAM 3.1 mask's bounding box.
- Both use the pipeline's own spectrum estimator (detrend, Hann, refined peak).

usage: python scripts/compare_rotor_probes.py [--out docs/results/rotor_table.md]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shake import measure  # noqa: E402

FIX = ROOT / "tests" / "api" / "fixtures" / "real"
CSV = ROOT / "samples" / "real" / "xjtu-rotor-500fps"
DISC_MM = 150.0
PROBE_FS = 1000.0
CLIPPED = {2000: (2, 3)}


def probes(name, rpm):
    a = np.loadtxt(CSV / f"{name}.csv", delimiter=",")
    out = {}
    for c in range(a.shape[1]):
        if c + 1 in CLIPPED.get(rpm, ()):
            continue
        f, Z = measure.spectrum(a[:, c], PROBE_FS)
        A = np.abs(Z)
        a1, h1, _ = measure.amplitude_at(f, A, rpm / 60.0, search_bins=3)
        a2, _, _ = measure.amplitude_at(f, A, 2 * h1, search_bins=3)
        out[c + 1] = {"f1_hz": h1, "a1_um": a1 * 1000, "a2_um": a2 * 1000}
    return out


def fixture(name):
    """The recording for a run: rotor-<name>, or rotor-<name>-<note> (the 1000 rpm Unb1 run kept Spark's own parts)."""
    return next(iter(sorted(FIX.glob(f"rotor-{name}*"))), None)


def video(name):
    r = json.loads((fixture(name) / "results.json").read_text())
    seg = r["segment"]["parts"]
    disc = next(p for p in seg if "disc" in p)
    x0, y0, x1, y1 = seg[disc]["bbox"]
    mm_px = DISC_MM / (y1 - y0)
    parts = {}
    for p, e in r["measurement"]["parts"].items():
        o = e["axes"]["y"]["orders"]
        parts["disc" if "disc" in p else "shaft" if "shaft" in p else p] = {
            "f1_hz": o["1x"]["hz"], "a1_um": o["1x"]["amp_px"] * mm_px * 1000, "a2_um": o["2x"]["amp_px"] * mm_px * 1000}
    sp = r["diagnosis"]["spark"]
    return {"parts": parts, "mm_per_px": mm_px, "rules": r["fingerprint"]["verdict"], "reason": r["fingerprint"]["reason"],
            "spark": sp.get("fault"), "spark_conf": sp.get("confidence"), "agrees": sp.get("agrees"),
            "visual": (sp.get("visual") or {}).get("status"), "usd": r["api"]["usd"]["total"],
            "seconds": round(sum(s["seconds"] or 0 for s in r["steps"]), 0)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    a = p.parse_args(argv)
    rows, base = [], {}
    lines = ["| run | 1x video / probes (Hz) | disc 1x (um) | shaft 1x (um) | probe #1 1x (um) | probes 1x range (um) | "
             "2x/1x video / probes | vs Unb1: disc, shaft, probe #1 | rules | Spark | visual check | cost |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rpm in (1000, 2000, 3000):
        for u in (1, 2, 3):
            name = f"{rpm}rpm-Unb{u}"
            if fixture(name) is None:
                continue
            v, pr = video(name), probes(name, rpm)
            d, s = v["parts"].get("disc", {}), v["parts"].get("shaft", {})
            if u == 1:
                base[rpm] = (d.get("a1_um"), s.get("a1_um"), pr[1]["a1_um"])
            b = base.get(rpm)
            ratio = "" if u == 1 or not b else (f"{d['a1_um'] / b[0]:.2f}, {s['a1_um'] / b[1]:.2f}, "
                                                f"{pr[1]['a1_um'] / b[2]:.2f}")
            r21v = d["a2_um"] / d["a1_um"]
            r21p = [x["a2_um"] / x["a1_um"] for x in pr.values()]
            f1p = np.mean([x["f1_hz"] for x in pr.values()])
            a1p = [x["a1_um"] for x in pr.values()]
            lines.append(
                f"| {name} | {d['f1_hz']:.2f} / {f1p:.2f} | {d['a1_um']:.1f} | {s['a1_um']:.1f} | {pr[1]['a1_um']:.1f} | "
                f"{min(a1p):.0f} to {max(a1p):.0f} | {r21v:.2f} / {min(r21p):.2f} to {max(r21p):.2f} | {ratio} | "
                f"{v['rules']} | {v['spark']}{'' if v['agrees'] else ' (capped at ' + str(v['spark_conf']) + ')'} | "
                f"{v['visual']} | ${v['usd']:.3f} |")
            rows.append({"run": name, "video": v, "probes": pr})
    md = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

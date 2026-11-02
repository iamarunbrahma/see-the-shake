"""Offline accuracy table: measurement and rules on twin clips, with the renderer's ground-truth masks.

No API calls, so it repeats exactly and costs nothing.

usage: python scripts/eval_twin.py [--seeds 1 2 3] [--seconds 6] [--scale 1.0] [--out out/eval]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shake import evaluate, fingerprint, measure, twin  # noqa: E402

RPM_RANGE_FRAC = 0.1  # stands in for Spark's plausible running-speed range: true rpm +/- 10%


def run_one(fault, seed, seconds, scale, root, rerender=False):
    d = root / f"{fault}_seed{seed}"
    truth_path = d / "truth.json"
    cfg = twin.TwinConfig(fault, seconds=seconds, seed=seed, out_scale=scale)
    t0 = time.time()
    if rerender or not truth_path.exists() or json.loads(truth_path.read_text()).get("seconds") != seconds \
            or json.loads(truth_path.read_text()).get("out_scale") != scale:
        twin.render(cfg, d)
    t_render = time.time() - t0
    truth = json.loads(truth_path.read_text())
    masks = {n: cv2.imread(str(d / truth["parts"][n]["mask"]), 0) for n in truth["parts"]}
    sc = measure.Scale.from_reference(masks, truth["reference"]["part"], truth["reference"]["width_mm"])
    rpm = truth["rpm"]
    t0 = time.time()
    res, arrays = measure.measure(str(d / "clip.mp4"), masks, truth["fps"],
                                  rpm_range=(rpm * (1 - RPM_RANGE_FRAC), rpm * (1 + RPM_RANGE_FRAC)), scale=sc)
    fp = fingerprint.fingerprint(res)
    t_measure = time.time() - t0
    cmp = evaluate.compare(truth, res, fp["verdict"])
    (d / "measured.json").write_text(json.dumps(res, indent=2))
    (d / "fingerprint.json").write_text(json.dumps(fp, indent=2))
    measure.plot_spectra(res, arrays, d / "spectrum.png", f"twin {fault}, seed {seed}")
    return truth, res, fp, cmp, t_render, t_measure


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--faults", nargs="+", default=list(twin.FAULTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--out", default="out/eval")
    p.add_argument("--rerender", action="store_true")
    a = p.parse_args(argv)
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)

    rows, summary = [], []
    for fault in a.faults:
        for seed in a.seeds:
            truth, res, fp, cmp, tr, tm = run_one(fault, seed, a.seconds, a.scale, root, a.rerender)
            for r in cmp["rows"]:
                rows.append({"fault": fault, "seed": seed, **r})
            scored = [r for r in cmp["rows"] if r["scored"]]
            summary.append({
                "fault": fault, "seed": seed, "verdict": fp["verdict"], "expected": truth["expected_verdict"],
                "verdict_ok": cmp["verdict_ok"], "confidence": fp["confidence"],
                "scored": cmp["scored"], "passed": cmp["passed"],
                "max_freq_err_bins": max((r["freq_err_bins"] or 0) for r in scored) if scored else None,
                "max_abs_amp_err": max(abs(r["amp_err"]) for r in scored if r["amp_err"] is not None) if scored else None,
                "missed": sum(not r["found"] for r in scored),
                "velocity": fp["parts"][fp["lead_part"]]["features"]["velocity_rms"],
                "render_s": round(tr, 1), "measure_s": round(tm, 1)})
            s = summary[-1]
            print(f"{fault:13s} seed {seed}: verdict {s['verdict']:12s} ({'ok' if s['verdict_ok'] else 'WRONG'}), "
                  f"components {s['passed']}/{s['scored']} within tolerance, max amp err {s['max_abs_amp_err']}, "
                  f"render {s['render_s']} s, measure {s['measure_s']} s", flush=True)

    with open(root / "accuracy_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (root / "accuracy.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    bin_hz = 1 / a.seconds
    lines = [
        "# Twin accuracy: measurement and rules, ground-truth masks",
        "",
        f"Offline, no API calls. {len(a.faults)} faults x {len(a.seeds)} seeds, {a.seconds:g} s clips at 240 fps, "
        f"output scale {a.scale:g} ({int(1920 * a.scale)}x{int(1080 * a.scale)}), FFT bin {bin_hz:.3f} Hz. "
        f"Masks are the renderer's; running-speed range is true rpm +/- {RPM_RANGE_FRAC:.0%}.",
        "",
        f"A component is scored when its true amplitude is at least {evaluate.EVAL_SNR:g}x the measured noise floor. "
        f"Pass: frequency within {evaluate.FREQ_TOL_BINS:g} bin and amplitude within {evaluate.AMP_TOL:.0%}.",
        "",
        "| Fault | Seed | Rule verdict | Correct | Components passed | Max freq err (bins) | Max amp err | Missed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        amp = f"{s['max_abs_amp_err']:.1%}" if s["max_abs_amp_err"] is not None else "n/a"
        fe = f"{s['max_freq_err_bins']:.2f}" if s["max_freq_err_bins"] is not None else "n/a"
        lines.append(f"| {s['fault']} | {s['seed']} | {s['verdict']} ({s['confidence']}) | "
                     f"{'yes' if s['verdict_ok'] else 'no'} | {s['passed']}/{s['scored']} | {fe} | {amp} | {s['missed']} |")
    n_ok = sum(s["verdict_ok"] for s in summary)
    n_pass = sum(s["passed"] for s in summary)
    n_scored = sum(s["scored"] for s in summary)
    lines += ["", f"**Verdicts correct: {n_ok}/{len(summary)}. Components within tolerance: {n_pass}/{n_scored}.**", ""]
    (root / "accuracy.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0 if n_ok == len(summary) and n_pass == n_scored else 1


if __name__ == "__main__":
    sys.exit(main())

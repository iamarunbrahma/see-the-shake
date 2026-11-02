"""Offline accuracy table: measurement and rules on twin clips, with the renderer's ground-truth masks.

No API calls, so it repeats exactly and costs nothing.

Rendered clips are cached under <out>/clips/ by fault, seed, duration, scale, encoding and a hash of the renderer.

usage: python scripts/eval_twin.py [--seeds 1 2 3] [--seconds 6] [--scale 1.0] [--phone] [--out out/eval]
  --phone adds a labelled section with phone-like encoding (720p, default x264 tuning, fixed bitrate).
"""
import argparse
import csv
import faulthandler
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shake import evaluate, fingerprint, measure, twin  # noqa: E402

RPM_RANGE_FRAC = 0.1  # stands in for Spark's plausible running-speed range: true rpm +/- 10%


TWIN_HASH = hashlib.sha1(Path(twin.__file__).read_bytes()).hexdigest()[:10]
MEASURE_HASH = hashlib.sha1(Path(measure.__file__).read_bytes()).hexdigest()[:10]


def run_one(fault, seed, seconds, scale, root, rerender=False, encoding="grain"):
    d = root / "clips" / f"{fault}_seed{seed}_{seconds:g}s_{scale:.3g}x_{encoding}_{TWIN_HASH}"
    truth_path = d / "truth.json"
    cfg = twin.TwinConfig(fault, seconds=seconds, seed=seed, out_scale=scale, encoding=encoding)
    t0 = time.time()
    if rerender or not truth_path.exists():
        twin.render(cfg, d)
    t_render = time.time() - t0
    truth = json.loads(truth_path.read_text())
    masks = {n: cv2.imread(str(d / truth["parts"][n]["mask"]), 0) for n in truth["parts"]}
    sc = measure.Scale.from_reference(masks, truth["reference"]["part"], truth["reference"]["width_mm"])
    rpm = truth["rpm"]
    t0 = time.time()
    mpath = d / "measured.json"
    cached = json.loads(mpath.read_text()) if mpath.exists() and not rerender else None
    if cached and cached.get("measure_hash") == MEASURE_HASH:  # same clip, same measurement code
        res = cached
    else:
        res, arrays = measure.measure(str(d / "clip.mp4"), masks, truth["fps"],
                                      rpm_range=(rpm * (1 - RPM_RANGE_FRAC), rpm * (1 + RPM_RANGE_FRAC)), scale=sc)
        res["measure_hash"] = MEASURE_HASH
        mpath.write_text(json.dumps(res, indent=2))
        measure.plot_spectra(res, arrays, d / "spectrum.png", f"twin {fault}, seed {seed}")
    fp = fingerprint.fingerprint(res)
    t_measure = time.time() - t0
    cmp = evaluate.compare(truth, res, fp["verdict"])
    (d / "fingerprint.json").write_text(json.dumps(fp, indent=2))
    return truth, res, fp, cmp, t_render, t_measure


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--faults", nargs="+", default=list(twin.FAULTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--out", default="out/eval")
    p.add_argument("--rerender", action="store_true")
    p.add_argument("--phone", action="store_true", help="add the phone-like encoding section (seed 1 per fault)")
    p.add_argument("--hang-timeout", type=float, default=900,
                   help="seconds per clip (render + measure) before dumping all stacks and exiting")
    a = p.parse_args(argv)
    faulthandler.enable()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)

    jobs = [("grain", fault, seed, a.scale) for fault in a.faults for seed in a.seeds]
    if a.phone:
        jobs += [("phone", fault, a.seeds[0], twin.PHONE_SCALE) for fault in a.faults]
    rows, summary = [], []
    for enc, fault, seed, scale in jobs:
            faulthandler.dump_traceback_later(a.hang_timeout, exit=True)  # a hang fails with a traceback
            truth, res, fp, cmp, tr, tm = run_one(fault, seed, a.seconds, scale, root, a.rerender, enc)
            faulthandler.cancel_dump_traceback_later()
            for r in cmp["rows"]:
                rows.append({"encoding": enc, "fault": fault, "seed": seed, **r})
            scored = [r for r in cmp["rows"] if r["scored"]]
            summary.append({
                "encoding": enc, "size": truth["size"], "noise_floor_px": min(
                    res["parts"][n]["axes"][ax]["noise_floor_px"] for n in res["parts"] for ax in "xy"),
                "fault": fault, "seed": seed, "verdict": fp["verdict"], "expected": truth["expected_verdict"],
                "verdict_ok": cmp["verdict_ok"], "confidence": fp["confidence"],
                "scored": cmp["scored"], "passed": cmp["passed"],
                "max_freq_err_bins": max((r["freq_err_bins"] or 0) for r in scored) if scored else None,
                "max_abs_amp_err": max(abs(r["amp_err"]) for r in scored if r["amp_err"] is not None) if scored else None,
                "missed": sum(not r["found"] for r in scored),
                "velocity": fp["parts"][fp["lead_part"]]["features"]["velocity_rms"],
                "render_s": round(tr, 1), "measure_s": round(tm, 1)})
            s = summary[-1]
            print(f"[{enc}] {fault:13s} seed {seed}: verdict {s['verdict']:12s} ({'ok' if s['verdict_ok'] else 'WRONG'}), "
                  f"components {s['passed']}/{s['scored']} within tolerance, max amp err {s['max_abs_amp_err']}, "
                  f"render {s['render_s']} s, measure {s['measure_s']} s", flush=True)

    with open(root / "accuracy_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    (root / "accuracy.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    bin_hz = 1 / a.seconds

    def table(items):
        out = ["| Fault | Seed | Rule verdict | Correct | Components passed | Max freq err (bins) | Max amp err | "
               "Missed | Noise floor (px) |", "|---|---|---|---|---|---|---|---|---|"]
        for s_ in items:
            amp = f"{s_['max_abs_amp_err']:.1%}" if s_["max_abs_amp_err"] is not None else "n/a"
            fe = f"{s_['max_freq_err_bins']:.2f}" if s_["max_freq_err_bins"] is not None else "n/a"
            out.append(f"| {s_['fault']} | {s_['seed']} | {s_['verdict']} ({s_['confidence']}) | "
                       f"{'yes' if s_['verdict_ok'] else 'no'} | {s_['passed']}/{s_['scored']} | {fe} | {amp} | "
                       f"{s_['missed']} | {s_['noise_floor_px']:.5f} |")
        return out

    def totals(items):
        return (f"**Verdicts correct: {sum(x['verdict_ok'] for x in items)}/{len(items)}. Components within "
                f"tolerance: {sum(x['passed'] for x in items)}/{sum(x['scored'] for x in items)}.**")

    main_ = [x for x in summary if x["encoding"] == "grain"]
    phone = [x for x in summary if x["encoding"] == "phone"]
    lines = [
        "# Twin accuracy: measurement and rules, ground-truth masks",
        "",
        f"Offline, no API calls. Masks are the renderer's; the running-speed range is the true rpm +/- "
        f"{RPM_RANGE_FRAC:.0%} (standing in for Spark's range). {a.seconds:g} s clips at 240 fps, FFT bin {bin_hz:.3f} Hz.",
        "",
        "**Scoring.** A truth component is scored when its amplitude is at least "
        f"{evaluate.EVAL_SNR:g}x the measured noise floor. Pass: frequency within {evaluate.FREQ_TOL_BINS:g} bin and "
        f"amplitude within {evaluate.AMP_TOL:.0%}.",
        "",
        "**Truth.** Each part's expected translation is the cabinet translation plus its rocking. The rocking is taken "
        "as the structure-weighted least-squares translation of the rotation field over the part's rest image "
        "(`twin.rocking_lever`), which is what any translation estimate of a rotating region aims at. The plain "
        "centroid displacement, and the error against it, are in `accuracy_rows.csv` "
        "(`true_amp_centroid_px`, `amp_err_centroid`).",
        "",
        f"## Main table: x264 crf {twin.CRF}, -tune grain, {int(1920 * a.scale)}x{int(1080 * a.scale)}",
        "",
        "-tune grain keeps the tiny inter-frame changes that sub-pixel vibration makes, as a high-bitrate hardware "
        "encoder does. It is kinder than real phone footage; see the phone-like section.",
        "",
        *table(main_),
        "",
        totals(main_),
        "",
    ]
    if phone:
        w_, h_ = phone[0]["size"]
        lines += [
            f"## Phone-like encoding (labelled; not the headline): {w_}x{h_}, default x264 tuning, "
            f"{twin.PHONE_BITRATE}bit/s",
            "",
            "Closer to what a phone saves: 720p and an encoder that discards small changes. The bitrate is an "
            "assumption until it is read from the real Redmi clips.",
            "",
            *table(phone),
            "",
            totals(phone),
            "",
        ]
    (root / "accuracy.md").write_text("\n".join(lines))
    print("\n".join(lines))
    ok = all(x["verdict_ok"] for x in main_) and all(x["passed"] == x["scored"] for x in main_)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

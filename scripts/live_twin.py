"""Live evaluation on the twin: one full run per fault with Muse Spark and SAM 3.1. Costs money.

Per fault: ingest, Spark plans the parts, SAM 3.1 masks each phrase on a short sub-clip, measurement and rules
run on the SAM masks, Spark diagnoses and code verifies it. It reports the two live rows from the design:
- SAM mask IoU against the renderer's exact mask (best-matching phrase per renderer part; if Spark's phrases
  miss a renderer part, its canonical phrase is segmented once more so the row is measured, and marked);
- whether Spark agrees with the rule verdict;
plus the rule verdict on SAM masks against the injected fault, and the API cost of each step.

Every API response is recorded to <fixtures>/<fault>/cassette.json.gz with the measurement, so the default test
suite replays the run offline.

usage: python scripts/live_twin.py [--faults imbalance ...] [--seed 1] [--seconds 6] [--out out/live]
                                    [--fixtures tests/api/fixtures/live] [--max-usd 2.0]
"""
import argparse
import faulthandler
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shake import diagnose, fingerprint, ingest, measure, plan_parts, segment, twin  # noqa: E402
from shake.meta_api import CreditExhausted, Ledger, MetaAPI  # noqa: E402

IOU_MATCH = 0.5  # below this, no Spark phrase is taken to cover the renderer part


def clip_dir(out, fault, seed, seconds):
    return out / "clips" / f"{fault}_seed{seed}_{seconds:g}s"


def run_fault(fault, seed, seconds, out, fixtures, ledger_path):
    d = clip_dir(out, fault, seed, seconds)
    if not (d / "truth.json").exists():
        twin.render(twin.TwinConfig(fault, seconds=seconds, seed=seed), d)
    truth = json.loads((d / "truth.json").read_text())
    gt = {n: cv2.imread(str(d / truth["parts"][n]["mask"]), 0) for n in truth["parts"]}
    clip = d / "clip.mp4"
    shape = gt["washing machine"].shape

    fx = fixtures / fault
    fx.mkdir(parents=True, exist_ok=True)
    cassette = fx / "cassette.json.gz"
    if cassette.exists():
        cassette.unlink()
    api = MetaAPI("record", cassette=cassette, ledger=Ledger(ledger_path))
    t0 = time.time()
    ing = ingest.ingest(str(clip))
    start, count = ing.window
    fps = ing.fps

    plan = plan_parts.plan_parts(api, clip, fps, start_frame=start, label=f"{fault}:plan")
    phrases = plan["parts"]
    sub_start = start + max(0, count // 2 - segment.SUB_FRAMES // 2)
    masks, seg = segment.segment_parts(api, clip, phrases, shape, start_frame=sub_start,
                                       fallback_phrase=plan["fallback_phrase"], label=f"{fault}:segment")
    # IoU per renderer part against the best-matching SAM mask
    added = []
    iou = {}
    for part, m in gt.items():
        best = max(((segment.iou(sm, m), p) for p, sm in masks.items()), default=(0.0, None))
        if best[0] < IOU_MATCH and part not in seg["parts"]:
            sm, info = segment.segment_phrase(api, seg["file_id"], part, shape, seg["subclip"]["frames"])
            info["added_for_iou"] = True
            seg["parts"][part] = info
            added.append(part)
            if sm is not None:
                masks.setdefault(part, sm)
                best = max(best, (segment.iou(sm, m), part))
        iou[part] = {"iou": round(best[0], 4), "phrase": best[1], "added_for_iou": part in added}
    seg["usd"] = round(sum(i.get("usd", 0.0) for i in seg["parts"].values()), 6)
    for p, m in masks.items():
        cv2.imwrite(str(d / f"sam_{twin.slug(p)}.png"), m)

    # measure and rules on the masks of the planned phrases (the pipeline's parts)
    planned = {p: masks[p] for p in phrases if p in masks} or dict(masks)
    ref_part = plan["fallback_phrase"] if plan["fallback_phrase"] in planned else next(iter(planned))
    sc = measure.Scale.from_reference(planned, ref_part, truth["reference"]["width_mm"])
    res, arrays = measure.measure(str(clip), planned, fps, start=start, count=count,
                                  rpm_range=tuple(plan["rpm_range"]) if plan["rpm_range"] else None, scale=sc)
    fp = fingerprint.fingerprint(res)
    png = d / "spectrum_sam.png"
    measure.plot_spectra(res, arrays, png, f"twin {fault} (SAM masks)")
    card = diagnose.diagnose(api, res, fp, spectrum_png=png, plan=plan, label=f"{fault}:diagnose")

    costs = {"plan": plan["usd"], "segment": seg["usd"], "diagnose": card.get("usd", 0.0)}
    costs["total"] = round(sum(costs.values()), 6)
    sp = card["spark"]
    report = {
        "fault": fault, "seed": seed, "seconds": seconds, "expected_verdict": truth["expected_verdict"],
        "window": {"start_frame": start, "frames": count}, "fps": fps,
        "plan": {k: plan[k] for k in ("machine_type", "rpm_range", "rpm_reason", "parts", "rotating_parts_seen",
                                      "warnings")},
        "true_rpm": truth["rpm"],
        "rpm_in_plan_range": bool(plan["rpm_range"]) and plan["rpm_range"][0] <= truth["rpm"] <= plan["rpm_range"][1],
        "segment": {"subclip": seg["subclip"], "missing": seg["missing"], "added_for_iou": added,
                    "parts": {p: {k: v for k, v in i.items() if k != "per_object"} for p, i in seg["parts"].items()}},
        "iou": iou,
        "measured_parts": list(planned), "scale_basis": sc.basis,
        "running_speed": res["running_speed"],
        "rule_verdict": fp["verdict"], "rule_confidence": fp["confidence"], "rule_reason": fp["reason"],
        "rule_ok": fp["verdict"] == truth["expected_verdict"],
        "spark": {k: sp.get(k) for k in ("fault", "confidence_reported", "confidence", "agrees", "supported",
                                         "cap_reasons", "counts", "error")},
        "card": card,
        "costs_usd": costs, "seconds_elapsed": round(time.time() - t0, 1),
    }
    (fx / "measured.json").write_text(json.dumps(res, indent=1))
    (fx / "fingerprint.json").write_text(json.dumps(fp, indent=1))
    (fx / "report.json").write_text(json.dumps(report, indent=1))
    (d / "report.json").write_text(json.dumps(report, indent=1))
    return report


def rng(r):
    pr = r["plan"]["rpm_range"]
    return f"{pr[0]:.0f} to {pr[1]:.0f}" if pr else "none usable"


def table(reports):
    lines = ["# Live twin runs: SAM 3.1 and Muse Spark", "",
             "One full run per twin fault (6 s, 1920x1080, 240 fps). Spark plans the parts, SAM 3.1 masks them on a "
             "60-frame sub-clip, measurement and rules run on the SAM masks, Spark diagnoses and code verifies it.",
             "", "| fault | Spark phrases | IoU washing machine | IoU door | rpm range (true 1000) | rule verdict "
             "(SAM masks) | Spark fault | Spark agrees | evidence v/c/d | Spark confidence | cost |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        io = r["iou"]

        def cell(p):
            i = io.get(p, {})
            s = f"{i.get('iou', 0):.3f}"
            if i.get("phrase") and i["phrase"] != p:
                s += f" ({i['phrase']})"
            if i.get("added_for_iou"):
                s += " (added)"
            return s
        sp = r["spark"]
        c = sp.get("counts") or {}
        conf = f"{sp.get('confidence_reported')} -> {sp.get('confidence')}" if sp.get("confidence") != sp.get(
            "confidence_reported") else str(sp.get("confidence"))
        lines.append(f"| {r['fault']} | {', '.join(r['plan']['parts'])} | {cell('washing machine')} | {cell('door')} | "
                     f"{rng(r)} | {r['rule_verdict']} "
                     f"({'ok' if r['rule_ok'] else 'expected ' + r['expected_verdict']}) | {sp.get('fault')} | "
                     f"{'yes' if sp.get('agrees') else 'no'} | {c.get('verified', 0)}/{c.get('context', 0)}/"
                     f"{c.get('dropped', 0)} | {conf} | ${r['costs_usd']['total']:.4f} |")
    total = sum(r["costs_usd"]["total"] for r in reports)
    lines += ["", f"Total API cost: ${total:.4f}. Evidence v/c/d: verified, context (real peak, not what the fault "
              "rests on), dropped.", "", "Cost per step (USD):", "",
              "| fault | plan (Spark) | segment (SAM) | diagnose (Spark) | total |", "|---|---|---|---|---|"]
    for r in reports:
        c = r["costs_usd"]
        lines.append(f"| {r['fault']} | {c['plan']:.4f} | {c['segment']:.4f} | {c['diagnose']:.4f} | {c['total']:.4f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--faults", nargs="+", default=list(twin.FAULTS))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--out", default="out/live")
    p.add_argument("--fixtures", default=str(ROOT / "tests" / "api" / "fixtures" / "live"))
    p.add_argument("--max-usd", type=float, default=2.0, help="stop before a fault once the run has spent this much")
    a = p.parse_args(argv)
    faulthandler.enable()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reports, spent = [], 0.0
    for fault in a.faults:
        if spent >= a.max_usd:
            print(f"stopping: spent ${spent:.4f} >= --max-usd {a.max_usd}")
            break
        try:
            r = run_fault(fault, a.seed, a.seconds, out, Path(a.fixtures), out / "ledger.jsonl")
        except CreditExhausted as e:
            print(f"STOP: {e}")
            return 3
        spent += r["costs_usd"]["total"]
        reports.append(r)
        print(f"{fault}: iou {json.dumps({k: v['iou'] for k, v in r['iou'].items()})}, rules {r['rule_verdict']}, "
              f"spark {r['spark']['fault']} ({'agrees' if r['spark']['agrees'] else 'disagrees'}), "
              f"${r['costs_usd']['total']:.4f}, {r['seconds_elapsed']} s", flush=True)
    md = table(reports)
    (out / "live_rows.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

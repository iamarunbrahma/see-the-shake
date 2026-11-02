"""Re-score recorded live twin runs with the current rules, offline and free.

Each recording keeps the measurement made on SAM masks (measured.json) and Spark's diagnosis (spark_raw in
report.json). This re-runs the fixed rules on that measurement and re-verifies Spark's claim against the new rule
verdict, so the live table follows rule changes without new API calls. The recordings are not modified.

usage: python scripts/rescore_live.py [--set grain|phone] [--out out/live/live_rows_rescored.md]
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from shake import diagnose, fingerprint, twin  # noqa: E402
import live_twin  # noqa: E402

SETS = {"grain": ROOT / "tests" / "api" / "fixtures" / "live", "phone": ROOT / "tests" / "api" / "fixtures" / "live_phone"}


def rescore(d):
    rep = json.loads((d / "report.json").read_text())
    res = json.loads((d / "measured.json").read_text())
    fp = fingerprint.fingerprint(res)
    r = copy.deepcopy(rep)
    r["rule_verdict"], r["rule_confidence"], r["rule_reason"] = fp["verdict"], fp["confidence"], fp["reason"]
    r["rule_ok"] = fp["verdict"] == rep["expected_verdict"]
    raw = rep["card"].get("spark_raw")
    if raw:
        card = diagnose.verify(raw, res, fp)
        sp = card["spark"]
        r["card"] = {**card, "spark_raw": raw}
        r["spark"] = {k: sp.get(k) for k in ("fault", "confidence_reported", "confidence", "agrees", "supported",
                                             "cap_reasons", "counts", "error")}
    r["recorded_rule_verdict"] = rep["rule_verdict"]
    return r


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=sorted(SETS), default="grain")
    p.add_argument("--out")
    a = p.parse_args(argv)
    reps = [rescore(SETS[a.set] / f) for f in twin.FAULTS if (SETS[a.set] / f / "report.json").exists()]
    md = live_twin.table(reps)
    changed = [f"{r['fault']}: {r['recorded_rule_verdict']} -> {r['rule_verdict']}" for r in reps
               if r["recorded_rule_verdict"] != r["rule_verdict"]]
    md += "\nRe-scored with the current rules" + (f"; changed: {'; '.join(changed)}." if changed else "; no verdict changed.") + "\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

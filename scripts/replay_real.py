"""Replay the live runs on real footage (tests/api/fixtures/real/) through the current pipeline, offline and free.

Spark's plan, the SAM 3.1 masks and Spark's diagnosis come from the recordings; ingest, measurement, the rules and
the magnified video are today's code. Prints each run's rule verdict next to what the reference supports (the
eddy-current probes for the rotor rig), the pattern the rules see with the alert level set aside (the 1000 rpm runs
sit near it), and writes the runs under --runs for scripts/compare_rotor_probes.py.

usage: python scripts/replay_real.py [--runs out/real_replay] [--only rotor-2000] [--rotor-ref-height]
  --rotor-ref-height  scale the rotor runs from the disc's 150 mm diameter as its mask height (the pipeline's
                      reference by height), so amplitudes come out in mm.
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "api" / "fixtures" / "real"
SAMPLES = ROOT / "samples" / "real"
# what the reference supports, in the rules' terms: the probes show a clean 1x at 2000 and 3000 rpm, and at
# 1000 rpm a 2x as large as 1x (the rules' misalignment pattern); the washer's spin is a 1x tub motion
EXPECTED = {"1000rpm": "misalignment", "2000rpm": "imbalance", "3000rpm": "imbalance", "aeg-washer": "imbalance"}
OPTION_KEYS = ("capture_fps", "rpm", "rpm_range", "reference", "alpha", "band", "video_seconds", "min_seconds",
               "max_seconds", "mains_hz", "spark_video", "user_parts", "blades")


def expected(name):
    return next((v for k, v in EXPECTED.items() if k in name), None)


def pattern_verdict(out):
    """The rules' verdict with the alert level at zero: the fault pattern alone, whatever the severity."""
    from shake import fingerprint
    fp = fingerprint.fingerprint(out["measurement"], alert=0.0, blades=out["options"].get("blades"))
    return f"{fp['verdict']} ({fp['reason']})"


def clip_for(r):
    hits = sorted(SAMPLES.rglob(r["clip"]["name"]))
    if not hits:
        raise FileNotFoundError(r["clip"]["name"])
    return hits[0]


def replay(name, runs, rotor_ref_height=False):
    from shake import pipeline
    d = FIX / name
    r = json.loads((d / "results.json").read_text())
    cas = json.loads(gzip.decompress((d / "cassette.json.gz").read_bytes()))
    label = cas[0]["key"].rsplit(":plan:preview", 1)[0]
    o = {k: v for k, v in r["options"].items() if k in OPTION_KEYS and v is not None}
    for k in ("rpm_range", "reference", "band"):
        if k in o:
            o[k] = tuple(o[k])
    if rotor_ref_height and name.startswith("rotor"):
        disc = next(p for p in r["plan"]["parts"] if "disc" in p)
        o["reference"] = (disc, 150.0, "height")
    opts = pipeline.Options(api="replay", cassette=str(d / "cassette.json.gz"), api_label=label, **o)
    t = time.time()
    out = pipeline.run(clip_for(r), None, runs, opts, mask_source="supplied")
    out["_seconds"] = round(time.time() - t, 1)
    return r, out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(ROOT / "out" / "real_replay"))
    p.add_argument("--only", default="")
    p.add_argument("--rotor-ref-height", action="store_true")
    a = p.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    names = sorted(x.name for x in FIX.iterdir() if (x / "results.json").exists() and a.only in x.name)
    rows = ["| run | expected | recorded rules | rules now | reason now | pattern without the alert | "
            "interpolation check now |", "|---|---|---|---|---|---|---|"]
    index = {}
    for name in names:
        rec, now = replay(name, a.runs, a.rotor_ref_height)
        index[name] = now["run_id"]
        fp = now["fingerprint"]
        pattern = pattern_verdict(now)
        exp = expected(name)
        ok = "right" if fp["verdict"] == exp else "WRONG"
        print(f"{name:32s} expected {exp:12s} recorded {rec['fingerprint']['verdict']:12s} now {fp['verdict']:12s} "
              f"({ok}) {fp['reason'][:80]} | pattern {pattern} | {now['_seconds']} s", flush=True)
        for part, e in now["measurement"]["parts"].items():
            print(f"    {part:20s} axis confidence {e.get('axis_confidence')}, surface texture left out "
                  f"{e.get('excluded_texture_px')} px", flush=True)
        for w in now.get("warnings", []):
            print(f"    warning: {w}", flush=True)
        rows.append(f"| {name} | {exp} | {rec['fingerprint']['verdict']} | {fp['verdict']} | {fp['reason']} | "
                    f"{pattern} | {now['ingest']['interpolation']['status']} |")
    Path(a.runs).mkdir(parents=True, exist_ok=True)
    (Path(a.runs) / "index.json").write_text(json.dumps(index, indent=1))
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())

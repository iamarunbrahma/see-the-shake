"""Build the runs the hosted demo shows, at no cost, from the committed recordings.

Every run is a replay: Muse Spark's plan and diagnosis and the SAM 3.1 masks come from a recording, and ingest,
measurement, the rules and magnification are today's code. The twins render their clips first (seed 1, 1920x1080,
6 s, as recorded). Each finished run folder is copied into --out with prepared.json, which
scripts/export_static.py turns into the static site; the rendered twin clips stay behind in --work, since the page
shows each run's own side-by-side video. Local paths in results.json are cut down to the file name.

usage: python scripts/prepare_hosted.py [--out hosted_runs] [--work out/hosted_work]
"""
import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# key, label, then ("twin", machine, fault) or ("real", recording in tests/api/fixtures/real)
PREPARED = [
    ("washer-healthy", "Washing machine twin: healthy", ("twin", "washer", "healthy")),
    ("washer-imbalance", "Washing machine twin: imbalance", ("twin", "washer", "imbalance")),
    ("washer-looseness", "Washing machine twin: looseness", ("twin", "washer", "looseness")),
    ("washer-misalignment", "Washing machine twin: misalignment", ("twin", "washer", "misalignment")),
    ("fan-healthy", "Pedestal fan twin: healthy", ("twin", "fan", "healthy")),
    ("fan-imbalance", "Pedestal fan twin: imbalance", ("twin", "fan", "imbalance")),
    ("fan-looseness", "Pedestal fan twin: looseness", ("twin", "fan", "looseness")),
    ("real-washer", "Real washing machine, phone clip at 120 fps", ("real", "aeg-washer-54492b8")),
    ("real-rotor", "Lab rotor rig at 2000 rpm, 500 fps camera", ("real", "rotor-2000rpm-Unb2-54492b8")),
]


def _replay_real():
    spec = importlib.util.spec_from_file_location("replay_real", ROOT / "scripts" / "replay_real.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(key, source, work):
    from shake import pipeline
    if source[0] == "twin":
        _, machine, fault = source
        r = pipeline.run_twin(fault, work, seed=1, seconds=6.0, scale=1.0, api="replay", machine=machine)
        return r, machine, fault
    name = source[1]
    _, r = _replay_real().replay(name, work, rotor_ref_height=name.startswith("rotor"))
    return r, "real", None


def copy_run(work, out, run_id):
    dst = out / run_id
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(work / run_id, dst)
    res = json.loads((dst / "results.json").read_text())
    clip = res.get("ingest", {}).get("clip", {})
    if clip.get("path"):
        clip["path"] = Path(clip["path"]).name  # no local paths in the published runs
    (dst / "results.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "hosted_runs"))
    p.add_argument("--work", default=str(ROOT / "out" / "hosted_work"))
    a = p.parse_args(argv)
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    t0, index = time.time(), []
    for key, label, source in PREPARED:
        t = time.time()
        r, machine, fault = build(key, source, work)
        if r["state"] != "done":
            raise SystemExit(f"{key}: the run did not finish ({r.get('error')})")
        res = copy_run(work, out, r["run_id"])
        fp = res["fingerprint"]
        index.append({"key": key, "label": label, "machine": machine, "fault": fault, "run_id": r["run_id"],
                      "verdict": fp["verdict"]})
        print(f"{key:22s} run {r['run_id']}  {fp['verdict']:12s} {time.time() - t:6.0f} s", flush=True)
    (out / "prepared.json").write_text(json.dumps({"runs": index}, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{len(index)} runs in {out}, {size / 1e6:.1f} MB, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

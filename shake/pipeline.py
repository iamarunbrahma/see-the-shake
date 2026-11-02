"""Pipeline: one clip in, results JSON plus magnified video out. No model calls happen here.

Steps follow the design. Our code runs ingest, measure, fingerprint and magnify. The Muse Spark parts plan,
SAM 3.1 segmentation and the Spark diagnosis are slots: pass callables (plan=, segment=, diagnose=) once those
modules exist; until then masks are an input (the renderer's masks for twin clips, uploaded masks otherwise).

Every run is cached under <runs>/<run_id>/, where run_id hashes the clip bytes, the masks, the options and the code,
so a repeat run returns at once.

usage: python -m shake.pipeline clip.mp4 --mask "washing machine=mask.png" [--mask door=door.png]
           [--rpm-range 900 1100] [--ref "washing machine=600"] [--capture-fps 240] [--runs runs]
       python -m shake.pipeline --twin imbalance [--seed 1] [--seconds 4] [--scale 0.5]
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import evaluate, fingerprint, ingest, magnify, measure, twin

STEPS = [
    ("ingest", "Ingest the clip"),
    ("plan", "Plan the inspection (Muse Spark)"),
    ("segment", "Find the parts (SAM 3.1)"),
    ("measure", "Measure the vibration"),
    ("fingerprint", "Match fault fingerprints"),
    ("magnify", "Magnify the shake"),
    ("diagnose", "Explain and suggest a fix (Muse Spark)"),
]
CODE_FILES = ("pipeline.py", "ingest.py", "measure.py", "fingerprint.py", "magnify.py")
RPM_RANGE_TWIN = 0.1  # twin runs stand in for Spark's running-speed range with the true rpm +/- 10%


@dataclass
class Options:
    capture_fps: float = None
    rpm: float = None
    rpm_range: tuple = None
    reference: tuple = None  # (part, length_mm): the part's mask width is this long
    alpha: object = "auto"
    band: tuple = magnify.DEFAULT_BAND
    video_seconds: float = magnify.VIDEO_SECONDS
    min_seconds: float = 4.0
    max_seconds: float = 10.0
    verify: bool = False  # track the magnified clip and report its gain (slow; a display check)
    mains_hz: float = measure.MAINS_HZ
    extra: dict = field(default_factory=dict)

    def key(self):
        d = asdict(self)
        d["rpm_range"] = list(self.rpm_range) if self.rpm_range else None
        d["reference"] = list(self.reference) if self.reference else None
        d["band"] = list(self.band)
        return d


def _sha_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_masks(masks):
    """{name: array or path} to {name: uint8 0/255}."""
    out = {}
    for name, m in masks.items():
        if isinstance(m, (str, Path)):
            img = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"cannot read mask {m}")
            m = img
        out[name] = ((np.asarray(m) > 0) * 255).astype(np.uint8)
    return out


def code_version():
    h = hashlib.sha1()
    for f in CODE_FILES:
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:10]


def run_id(clip, masks, options):
    h = hashlib.sha1()
    h.update(_sha_file(clip).encode())
    for name in sorted(masks):
        h.update(name.encode())
        h.update(hashlib.sha1(np.ascontiguousarray(masks[name]).tobytes()).digest())
        h.update(str(masks[name].shape).encode())
    h.update(json.dumps(options.key(), sort_keys=True, default=str).encode())
    h.update(code_version().encode())
    return h.hexdigest()[:12]


class _Status:
    """Step statuses, written to status.json after every change so a page can poll them."""

    def __init__(self, out, rid, progress=None):
        self.path = out / "status.json"
        self.progress = progress
        self.data = {"run_id": rid, "state": "running", "error": None,
                     "steps": [{"key": k, "label": label, "status": "pending", "detail": None, "seconds": None}
                               for k, label in STEPS]}
        self._t = {}
        self.write()

    def write(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    def set(self, key, status, detail=None):
        step = next(s for s in self.data["steps"] if s["key"] == key)
        if status == "running":
            self._t[key] = time.time()
        elif key in self._t:
            step["seconds"] = round(time.time() - self._t[key], 2)
        step["status"], step["detail"] = status, detail
        self.write()
        if self.progress:
            self.progress(key, status, detail)

    def finish(self, state, error=None):
        self.data["state"], self.data["error"] = state, error
        self.write()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _spectrum_pngs(result, arrays, out):
    files = {}
    for part in result["parts"]:
        sub = {**result, "parts": {part: result["parts"][part]}}
        name = f"spectrum_{twin.slug(part)}.png"
        measure.plot_spectra(sub, arrays, out / name, part)
        files[part] = name
    return files


def run(clip, masks, runs_dir="runs", options=None, plan=None, segment=None, diagnose=None, truth=None,
        mask_source="supplied", progress=None, use_cache=True):
    """Run one clip. masks: {part name: HxW array or PNG path}. Returns the results dict (also results.json).

    plan(clip, ingested) -> {"machine", "rpm_range", "parts"}; segment(clip, phrases) -> {part: mask};
    diagnose(results) -> dict. Each is optional; without it the step is reported as not connected yet.
    truth: a twin truth dict, to compare against.
    """
    clip = Path(clip)
    options = options or Options()
    masks = load_masks(masks)
    rid = run_id(clip, masks, options)
    out = Path(runs_dir) / rid
    res_path = out / "results.json"
    if use_cache and res_path.exists():
        cached = json.loads(res_path.read_text())
        if cached.get("state") == "done":
            if progress:
                for s in cached["steps"]:
                    progress(s["key"], s["status"], s["detail"])
            return cached
    out.mkdir(parents=True, exist_ok=True)
    st = _Status(out, rid, progress)
    results = {"run_id": rid, "state": "running", "code_version": code_version(), "clip": {"name": clip.name},
               "options": options.key(), "mask_source": mask_source, "files": {}}
    (out / "masks").mkdir(exist_ok=True)
    for name, m in masks.items():
        cv2.imwrite(str(out / "masks" / f"{twin.slug(name)}.png"), m)
    results["files"]["masks"] = {n: f"masks/{twin.slug(n)}.png" for n in masks}
    step = "ingest"
    try:
        st.set(step, "running")
        ing = ingest.ingest(clip, capture_fps=options.capture_fps, min_s=options.min_seconds, max_s=options.max_seconds)
        results["ingest"] = ing.to_dict()
        start, count = ing.window
        fps = ing.fps
        st.set(step, "done", f"{fps:g} fps ({ing.info.fps_source}), window {count / fps:.1f} s, "
                             f"frames check: {ing.interpolation['status']}")
        if ing.interpolation["flagged"]:
            results["warning"] = "In-between frames look synthesised; treat the spectrum with suspicion."

        step = "plan"
        rpm_range = options.rpm_range
        if plan:
            st.set(step, "running")
            results["plan"] = plan(clip, ing)
            rpm_range = rpm_range or tuple(results["plan"].get("rpm_range") or ()) or None
            st.set(step, "done", ", ".join(results["plan"].get("parts", [])))
        else:
            results["plan"] = {"status": "not_connected", "parts": list(masks)}
            st.set(step, "skipped", "Muse Spark parts plan not connected yet; parts come with the masks")

        step = "segment"
        if segment and plan:
            st.set(step, "running")
            masks = load_masks(segment(clip, results["plan"]["parts"]))
            mask_source = "sam"
            st.set(step, "done", f"{len(masks)} parts")
        else:
            st.set(step, "skipped", f"masks supplied ({mask_source})")
        results["mask_source"] = mask_source

        step = "measure"
        st.set(step, "running")
        scale = None
        if options.reference:
            part, mm = options.reference
            scale = measure.Scale.from_reference(masks, part, float(mm))
        res, arrays = measure.measure(clip, masks, fps, start=start, count=count, rpm=options.rpm,
                                      rpm_range=rpm_range, scale=scale, mains_hz=options.mains_hz)
        results["measurement"] = res
        results["files"]["spectra"] = _spectrum_pngs(res, arrays, out)
        rs = res["running_speed"]
        st.set(step, "done", f"running speed {rs['hz']:.2f} Hz ({rs['rpm']:.0f} rpm, from {rs['source']})")

        step = "fingerprint"
        st.set(step, "running")
        fp = fingerprint.fingerprint(res)
        results["fingerprint"] = fp
        st.set(step, "done", f"{fp['verdict']} ({fp['confidence']})")

        step = "magnify"
        st.set(step, "running")
        info = magnify.magnify(clip, masks, arrays["rel"], fps, rs["hz"], out, start=start, alpha=options.alpha,
                               band=options.band, seconds=options.video_seconds, write_native=options.verify)
        results["magnify"] = info
        results["files"]["side_by_side"] = info["files"]["side_by_side"]
        results["files"]["slice"] = info["files"]["slice"]
        if options.verify:
            rot = magnify.rotating_mask(clip, start, info["frames"])
            results["magnify"]["verify"] = magnify.verify(out / info["files"]["native"], masks, arrays["rel"], fps,
                                                          rs["hz"], info, rot)
        st.set(step, "done", f"{info['alpha']:.0f}x, band {info['band_hz'][0]:.1f} to {info['band_hz'][1]:.1f} Hz")

        if truth is not None:
            results["truth_comparison"] = evaluate.compare(truth, res, fp["verdict"])
            results["truth"] = {"fault": truth["fault"], "expected_verdict": truth["expected_verdict"],
                                "rpm": truth["rpm"], "seed": truth["seed"]}

        step = "diagnose"
        if diagnose:
            st.set(step, "running")
            results["diagnosis"] = diagnose(results)
            st.set(step, "done")
        else:
            results["diagnosis"] = {"status": "not_connected",
                                    "headline": fp["verdict"], "note": "The rule verdict is the official result."}
            st.set(step, "skipped", "Muse Spark diagnosis not connected yet; the rule verdict stands")
        results["state"] = "done"
        results["steps"] = st.data["steps"]
        res_path.write_text(json.dumps(_jsonable(results), indent=2))
        st.finish("done")
        return json.loads(res_path.read_text())
    except Exception as e:  # report which step failed, keep what was done
        st.set(step, "failed", f"{type(e).__name__}: {e}")
        results["state"] = "failed"
        results["error"] = {"step": step, "message": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
        results["steps"] = st.data["steps"]
        res_path.write_text(json.dumps(_jsonable(results), indent=2))
        st.finish("failed", results["error"]["message"])
        raise


def twin_clip(fault, runs_dir="runs", seed=1, seconds=4.0, scale=0.5):
    """Render (or reuse) a twin clip for the app. Returns (clip path, masks, truth)."""
    h = hashlib.sha1(Path(twin.__file__).read_bytes()).hexdigest()[:10]
    d = Path(runs_dir) / "twin_clips" / f"{fault}_seed{seed}_{seconds:g}s_{scale:.3g}x_{h}"
    if not (d / "truth.json").exists():
        tmp = d.with_name(d.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        twin.render(twin.TwinConfig(fault, seconds=seconds, seed=seed, out_scale=scale), tmp)
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)
    truth = json.loads((d / "truth.json").read_text())
    masks = {n: d / truth["parts"][n]["mask"] for n in truth["parts"]}
    return d / "clip.mp4", masks, truth


def twin_options(truth, **kw):
    rpm = truth["rpm"]
    return Options(rpm_range=(rpm * (1 - RPM_RANGE_TWIN), rpm * (1 + RPM_RANGE_TWIN)),
                   reference=(truth["reference"]["part"], truth["reference"]["width_mm"]),
                   min_seconds=min(4.0, truth["seconds"]), **kw)


def run_twin(fault, runs_dir="runs", seed=1, seconds=4.0, scale=0.5, progress=None, **kw):
    clip, masks, truth = twin_clip(fault, runs_dir, seed, seconds, scale)
    return run(clip, masks, runs_dir, twin_options(truth, **kw), truth=truth, mask_source="twin renderer",
               progress=progress)


def main(argv=None):
    p = argparse.ArgumentParser(description="Run See the Shake on one clip")
    p.add_argument("clip", nargs="?")
    p.add_argument("--mask", action="append", default=[], help='"part name=mask.png" (repeat per part)')
    p.add_argument("--twin", choices=sorted(twin.FAULTS))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--rpm", type=float)
    p.add_argument("--rpm-range", type=float, nargs=2)
    p.add_argument("--capture-fps", type=float)
    p.add_argument("--ref", help='"part name=length in mm" across the part\'s mask width')
    p.add_argument("--alpha", default="auto")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--runs", default="runs")
    a = p.parse_args(argv)
    alpha = a.alpha if a.alpha == "auto" else float(a.alpha)
    show = lambda k, s, d: print(f"[{s:8s}] {k:12s} {d or ''}", flush=True)  # noqa: E731
    if a.twin:
        r = run_twin(a.twin, a.runs, a.seed, a.seconds, a.scale, progress=show, alpha=alpha, verify=a.verify)
    else:
        if not a.clip or not a.mask:
            p.error("give a clip and at least one --mask (or --twin)")
        masks = dict(m.split("=", 1) for m in a.mask)
        ref = None
        if a.ref:
            part, mm = a.ref.split("=", 1)
            ref = (part, float(mm))
        opts = Options(capture_fps=a.capture_fps, rpm=a.rpm, rpm_range=tuple(a.rpm_range) if a.rpm_range else None,
                       reference=ref, alpha=alpha, verify=a.verify)
        r = run(a.clip, masks, a.runs, opts, progress=show)
    fp = r["fingerprint"]
    print(f"verdict: {fp['verdict']} ({fp['confidence']}): {fp['reason']}")
    print(f"results: {Path(a.runs) / r['run_id'] / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

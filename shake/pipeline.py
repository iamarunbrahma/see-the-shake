"""Pipeline: one clip in, results JSON plus magnified video out.

The seven steps of the design:
1. ingest (our code);
2. plan the inspection (Muse Spark, shake.plan_parts): machine type, running-speed range, part phrases;
3. segment (SAM 3.1, shake.segment): one stable mask per phrase;
4. measure, 5. fingerprint, 6. magnify (our code);
7. diagnose (Muse Spark, shake.diagnose), with every evidence item checked by code; the rule verdict stays the headline.

Model calls go through shake.meta_api.MetaAPI, chosen by Options.api:
- "off": no model calls; parts come from the supplied masks (the renderer's for twin clips);
- "live": real calls; "record": real calls, saved to <run>/cassette.json.gz so the run can be replayed;
- "replay": recorded responses only, no network (tests and the twin demo use these).
Supplied masks are the fallback when the plan or SAM fails or finds nothing. A 402 stops the run: the credit is gone.

Every run is cached under <runs>/<run_id>/, where run_id hashes the clip bytes, the masks, the options, the code and
the run's context, so a repeat run returns at once.

usage: python -m shake.pipeline clip.mp4 [--mask "washing machine=mask.png"] [--api live] [--rpm-range 900 1100]
           [--ref "washing machine=600"] [--capture-fps 240] [--runs runs]
       python -m shake.pipeline --twin imbalance [--api replay] [--seed 1] [--seconds 4] [--scale 0.5]
       python -m shake.pipeline --twin fan-looseness [--seconds 6] [--scale 1]
"""
import argparse
import contextlib
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import evaluate, fingerprint, ingest, magnify, measure, twin, twin_fan

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("ingest", "Ingest the clip"),
    ("plan", "Plan the inspection (Muse Spark)"),
    ("segment", "Find the parts (SAM 3.1)"),
    ("measure", "Measure the vibration"),
    ("fingerprint", "Match fault fingerprints"),
    ("magnify", "Magnify the shake"),
    ("diagnose", "Explain and suggest a fix (Muse Spark)"),
]
CODE_FILES = ("pipeline.py", "ingest.py", "measure.py", "fingerprint.py", "magnify.py", "meta_api.py",
              "plan_parts.py", "segment.py", "diagnose.py")
BG_STEP_WARN = 0.3  # background 1x at this share of the part's 1x or more: warn
RPM_RANGE_TWIN = 0.1  # twin runs without Spark stand in for its running-speed range with the true rpm +/- 10%
API_MODES = ("off", "live", "record", "replay")
# Recorded live twin runs (scripts/live_twin.py): seed 1, 6 s, 1920x1080, labels prefixed with the fault name.
RECORDINGS = Path(os.environ.get("SHAKE_RECORDINGS", ROOT / "tests" / "api" / "fixtures" / "live"))
RECORDED_TWIN = {"seed": 1, "seconds": 6.0, "scale": 1.0}
IOU_MATCH = 0.5  # a SAM mask stands for a renderer part when their IoU reaches this
CREDIT_MESSAGE = ("HTTP 402 from the Meta Model API: the credit is used up. API work has stopped; tell Arun. "
                  "Do not add a card.")


@dataclass
class Options:
    capture_fps: float = None
    rpm: float = None
    rpm_range: tuple = None
    reference: tuple = None  # (part, length_mm[, axis]): the part's mask width (or "height") is this long
    alpha: object = "auto"
    band: tuple = magnify.DEFAULT_BAND
    video_seconds: float = magnify.VIDEO_SECONDS
    min_seconds: float = 4.0
    max_seconds: float = 10.0
    verify: bool = False  # track the magnified clip and report its gain (slow; a display check)
    mains_hz: float = measure.MAINS_HZ
    api: str = "off"  # off, live, record, replay
    cassette: str = None  # replay: the recording to serve; record: where to save (default <run>/cassette.json.gz)
    api_label: str = "run"  # label prefix of the calls; a replay must use the recording's (the fault for twin runs)
    spark_video: bool = True  # send the side-by-side clip to Spark with the diagnosis request
    user_parts: list = None  # replace Spark's part phrases (still normalised by plan_parts)
    blades: int = None  # a fan's blade count: blades x running speed is then treated as blade pass
    extra: dict = field(default_factory=dict)

    def key(self):
        d = asdict(self)
        d["rpm_range"] = list(self.rpm_range) if self.rpm_range else None
        d["reference"] = list(self.reference) if self.reference else None
        d["band"] = list(self.band)
        if self.api == "replay" and self.cassette:
            d["cassette"] = _sha_file(self.cassette)[:16]  # the recording's content, not where it lives
        elif self.api != "replay":
            d["cassette"] = None
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
    for name, m in (masks or {}).items():
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


def run_id(clip, masks, options, mask_source="supplied", truth=None, ignore=None, magnify_masks=None):
    """Hash of everything a run depends on, including its context (where the masks came from, which truth)."""
    h = hashlib.sha1()
    h.update(_sha_file(clip).encode())
    for name in sorted(masks):
        h.update(name.encode())
        h.update(hashlib.sha1(np.ascontiguousarray(masks[name]).tobytes()).digest())
        h.update(str(masks[name].shape).encode())
    h.update(json.dumps(options.key(), sort_keys=True, default=str).encode())
    h.update(code_version().encode())
    h.update(mask_source.encode())
    if truth is not None:
        h.update(json.dumps([truth["fault"], truth["seed"], truth["rpm"], truth["seconds"]]
                            + ([truth["machine"]] if "machine" in truth else [])).encode())
    if ignore is not None:
        h.update(b"ignore" + hashlib.sha1(np.ascontiguousarray(ignore).tobytes()).digest())
    for name in sorted(magnify_masks or {}):
        h.update(b"magnify" + name.encode() + hashlib.sha1(np.ascontiguousarray(magnify_masks[name]).tobytes()).digest())
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


def _spectrum_pngs(result, arrays, out, fp=None):
    skip = {}
    for part, e in ((fp or {}).get("parts") or {}).items():
        skip[part] = {(a["axis"], a["hz"]) for a in e["features"].get("artifacts", [])}
    files = {}
    for part in result["parts"]:
        sub = {**result, "parts": {part: result["parts"][part]}}
        name = f"spectrum_{twin.slug(part)}.png"
        measure.plot_spectra(sub, arrays, out / name, part, skip=skip)
        files[part] = name
    measure.plot_spectra(result, arrays, out / "spectrum.png", skip=skip)  # all parts, for Spark
    return files


def _frame_shape(clip):
    cap = cv2.VideoCapture(str(clip))
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise ingest.IngestError(f"cannot decode a frame of {clip}")
    return f.shape[:2]  # as decoded (display orientation), which is also what ffmpeg hands SAM


def make_api(options, out):
    """The MetaAPI for this run, or None when model calls are off."""
    if options.api not in API_MODES:
        raise ValueError(f"api must be one of {API_MODES}, not {options.api!r}")
    if options.api == "off":
        return None
    from .meta_api import Ledger, MetaAPI
    ledger_path = out / "ledger.jsonl"
    ledger_path.unlink(missing_ok=True)
    ledger = Ledger(ledger_path)
    if options.api == "replay":
        if not options.cassette or not Path(options.cassette).exists():
            raise ValueError(f"replay needs a recording; {options.cassette} does not exist")
        return MetaAPI("replay", cassette=Path(options.cassette), ledger=ledger)
    if options.api == "record":
        return MetaAPI("record", cassette=Path(options.cassette or out / "cassette.json.gz"), ledger=ledger)
    return MetaAPI("live", ledger=ledger)


@contextlib.contextmanager
def _workers_can_start():
    """Let segment's process pool start when __main__ cannot be re-imported (code run from stdin or -c).

    Its forkserver workers re-import the parent's __main__ from its file; '<stdin>' is no file, so every worker
    died, the pool broke and the masks were decoded serially (about 65 s per part instead of a few seconds),
    with a traceback from each child. While SAM runs, such a __main__ is not offered for re-import. Under
    python -m shake.pipeline, uvicorn and pytest __main__ is a real module and nothing changes.
    """
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    hide = main is not None and getattr(main, "__spec__", None) is None and path is not None and not os.path.isfile(path)
    if hide:
        del main.__file__
    try:
        yield
    finally:
        if hide:
            main.__file__ = path


def _recorded_rule_verdict(api, label):
    """In replay: the rule verdict Spark was shown when the diagnosis was recorded (from the recorded prompt)."""
    import re
    for it in getattr(api.cassette, "items", []):
        if it.get("op") == "spark" and it.get("key") == label:
            m = re.search(r"Official rule verdict \(.*?\): ([a-z]+) \(", json.dumps(it.get("request")))
            return m.group(1) if m else None
    return None


def _iou_table(sam_masks, ref_masks):
    """For each reference (renderer) part, the SAM phrase with the best IoU."""
    from .segment import iou
    out = {}
    for part, ref in ref_masks.items():
        best = max(((iou(m, ref), p) for p, m in sam_masks.items()), default=(0.0, None))
        out[part] = {"iou": round(best[0], 4), "phrase": best[1]}
    return out


def run(clip, masks=None, runs_dir="runs", options=None, plan=None, segment=None, diagnose=None, truth=None,
        mask_source="supplied", progress=None, use_cache=True, on_start=None, ignore=None, magnify_masks=None):
    """Run one clip. Returns the results dict (also written to results.json).

    masks: {part name: HxW array or PNG path}. Required when options.api is "off"; otherwise the fallback for
    when Spark or SAM fails or finds nothing.
    plan, segment, diagnose: optional callables used only when options.api is "off" (stand-ins for tests):
    plan(clip, ingested) -> {"machine_type", "rpm_range", "parts"}; segment(clip, phrases) -> {part: mask};
    diagnose(results) -> dict.
    truth: a twin truth dict, to compare against. on_start(run_id) is called once the run id is known.
    ignore: a region the background leaves out although no part covers it (a fan's blade disc).
    magnify_masks: masks for the magnified video when they should differ from the measured parts (a fan's whole
    guard, blades included, wobbles as one); same part names.
    """
    clip = Path(clip)
    options = options or Options()
    masks = load_masks(masks)
    if not masks and options.api == "off":
        raise ValueError("give part masks, or turn on Muse Spark and SAM 3.1 (options.api)")
    ignore = None if ignore is None else (load_masks({"i": ignore})["i"])
    magnify_masks = load_masks(magnify_masks) if magnify_masks else None
    rid = run_id(clip, masks, options, mask_source, truth, ignore, magnify_masks)
    if on_start:
        on_start(rid)
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
               "options": options.key(), "mask_source": mask_source, "files": {}, "warnings": []}
    supplied = dict(masks)
    step = "ingest"
    api = None
    usd = {}
    try:
        api = make_api(options, out)
        st.set(step, "running")
        ing = ingest.ingest(clip, capture_fps=options.capture_fps, min_s=options.min_seconds, max_s=options.max_seconds)
        results["ingest"] = ing.to_dict()
        start, count = ing.window
        fps = ing.fps
        st.set(step, "done", f"{fps:g} fps ({ing.info.fps_source}), window {count / fps:.1f} s, "
                             f"frames check: {ing.interpolation['status']}")
        if ing.interpolation["flagged"]:
            results["warnings"].append("In-between frames look synthesised; treat the spectrum with suspicion.")

        # 2. plan the inspection
        step = "plan"
        rpm_range = options.rpm_range
        plan_info = None
        if api:
            from . import plan_parts
            from .meta_api import CreditExhausted
            st.set(step, "running")
            try:
                plan_info = plan_parts.plan_parts(api, clip, fps, start_frame=start, user_parts=options.user_parts,
                                                  workdir=out, label=f"{options.api_label}:plan")
            except CreditExhausted:
                raise
            except Exception as e:
                if not supplied:
                    raise
                results["warnings"].append(f"The Muse Spark plan failed ({type(e).__name__}: {e}); "
                                           "the supplied parts are used.")
                results["plan"] = {"status": "failed", "error": f"{type(e).__name__}: {e}", "parts": list(supplied)}
                st.set(step, "fallback", "Muse Spark plan failed; using the supplied parts")
            if plan_info:
                usd["plan"] = plan_info.get("usd", 0.0)
                results["plan"] = {"status": "done", **{k: v for k, v in plan_info.items() if k != "spark_raw"}}
                if not rpm_range and plan_info.get("rpm_range"):
                    rpm_range = tuple(plan_info["rpm_range"])
                st.set(step, "done", f"{plan_info['machine_type']}: {', '.join(plan_info['parts'])}")
        elif plan:
            st.set(step, "running")
            results["plan"] = {"status": "done", **plan(clip, ing)}
            rpm_range = rpm_range or tuple(results["plan"].get("rpm_range") or ()) or None
            st.set(step, "done", ", ".join(results["plan"].get("parts", [])))
        else:
            results["plan"] = {"status": "not_connected", "parts": list(masks)}
            st.set(step, "skipped", "Muse Spark is off; parts come with the masks")

        # 3. segment
        step = "segment"
        if api and plan_info:
            from . import segment as seg_mod
            from .meta_api import CreditExhausted
            st.set(step, "running")
            shape = _frame_shape(clip)
            sub_start = start + max(0, count // 2 - seg_mod.SUB_FRAMES // 2)
            # The whole machine is segmented too when it is not one of the parts, so that every moving thing SAM
            # can see stays out of the camera-shake reference (an unmasked shaft read as "background" halved the
            # disc's motion on the rotor rig). A replay asks for it only if its recording holds that call.
            machine = plan_info.get("fallback_phrase")
            context = []
            if machine and machine not in plan_info["parts"]:
                recorded = [it for it in getattr(getattr(api, "cassette", None), "items", []) or []
                            if it.get("op") == "sam" and it.get("key") == machine]
                if options.api != "replay" or recorded:
                    context = [machine]
            try:
                with _workers_can_start():
                    sam_masks, seg = seg_mod.segment_parts(api, clip, plan_info["parts"] + context, shape,
                                                           start_frame=sub_start,
                                                           fallback_phrase=plan_info["fallback_phrase"], workdir=out,
                                                           label=f"{options.api_label}:segment")
            except CreditExhausted:
                raise
            except Exception as e:
                sam_masks, seg = {}, {"error": f"{type(e).__name__}: {e}", "missing": plan_info["parts"]}
            usd["segment"] = seg.get("usd", 0.0)
            results["segment"] = {k: v for k, v in seg.items() if k != "parts"}
            results["segment"]["parts"] = {p: {k: v for k, v in i.items() if k != "per_object"}
                                           for p, i in (seg.get("parts") or {}).items()}
            parts_found = {p_: m_ for p_, m_ in sam_masks.items() if p_ not in context}
            ctx_found = {p_: m_ for p_, m_ in sam_masks.items() if p_ in context}
            if ctx_found and not parts_found:  # only the whole machine was found: measure it
                parts_found, ctx_found = ctx_found, {}
            if ctx_found:
                ctx = np.max(np.stack(list(load_masks(ctx_found).values())), axis=0)
                ignore = ctx if ignore is None else np.maximum(ignore, ctx)
                results["segment"]["background_excludes"] = sorted(ctx_found)
            sam_masks = parts_found
            if sam_masks:
                if supplied and truth is not None:
                    results["sam_iou"] = _iou_table(sam_masks, supplied)
                masks, mask_source = load_masks(sam_masks), "SAM 3.1"
                missing = [p_ for p_ in plan_info["parts"] if p_ not in sam_masks]
                st.set(step, "done", f"{len(masks)} parts" + (f"; not found: {', '.join(missing)}" if missing else ""))
            elif supplied:
                why = seg.get("error") or "SAM 3.1 found none of the parts"
                results["warnings"].append(f"{why}; the supplied masks are used.")
                st.set(step, "fallback", f"{why}; using the supplied masks")
            else:
                raise RuntimeError("SAM 3.1 found none of the parts and no masks were supplied")
        elif segment and plan and not api:
            st.set(step, "running")
            masks = load_masks(segment(clip, results["plan"]["parts"]))
            mask_source = "segment hook"
            st.set(step, "done", f"{len(masks)} parts")
        elif api:
            st.set(step, "fallback", f"no plan to segment; using the supplied masks ({mask_source})")
        else:
            st.set(step, "skipped", f"SAM 3.1 is off; masks supplied ({mask_source})")
        results["mask_source"] = mask_source
        (out / "masks").mkdir(exist_ok=True)
        for name, m in masks.items():
            cv2.imwrite(str(out / "masks" / f"{twin.slug(name)}.png"), m)
        results["files"]["masks"] = {n: f"masks/{twin.slug(n)}.png" for n in masks}

        # 4. measure
        step = "measure"
        st.set(step, "running")
        scale = None
        if options.reference:
            part, mm, *axis = options.reference
            axis = axis[0] if axis else "width"
            if part not in masks:
                alt = plan_info["fallback_phrase"] if plan_info and plan_info["fallback_phrase"] in masks else None
                if alt:
                    results["warnings"].append(f"The reference part '{part}' is measured as '{alt}'.")
                    part = alt
                else:
                    results["warnings"].append(f"The reference part '{part}' is not among the parts; "
                                               "amplitudes stay in px.")
                    part = None
            if part:
                scale = measure.Scale.from_reference(masks, part, float(mm), axis=axis)
        res, arrays = measure.measure(clip, masks, fps, start=start, count=count, rpm=options.rpm,
                                      rpm_range=rpm_range, scale=scale, mains_hz=options.mains_hz, ignore=ignore)
        results["measurement"] = res
        rs = res["running_speed"]
        if rs.get("ambiguous"):
            alts = ", ".join(f"{h * 60:.0f}" for h in rs["alternatives_hz"])
            results["warnings"].append(f"The running speed is ambiguous: {rs['rpm']:.0f} rpm was taken, but {alts} rpm "
                                       "also fits the peaks. Enter the rpm and run again to be sure.")
        lead_1x = max((float(np.hypot(e["axes"]["x"]["orders"]["1x"]["amp_px"], e["axes"]["y"]["orders"]["1x"]["amp_px"]))
                       for e in res["parts"].values()), default=0.0)
        bg = res["background"]["axes"]
        bg_1x = float(np.hypot(bg["x"]["orders"]["1x"]["amp_px"], bg["y"]["orders"]["1x"]["amp_px"]))
        if lead_1x > 0 and bg_1x >= BG_STEP_WARN * lead_1x:
            results["warnings"].append(f"The background moves with the machine at running speed ({bg_1x / lead_1x:.0%} "
                                       "of the part's 1x): the camera or its support is shaking, or a moving part is "
                                       "not masked. Readings may be low.")
        st.set(step, "done", f"running speed {rs['hz']:.2f} Hz ({rs['rpm']:.0f} rpm, from the "
                             f"{rs['source'].replace('_', ' ')})")

        # 5. fingerprint
        step = "fingerprint"
        st.set(step, "running")
        blades, blades_from = options.blades, "given"
        if not blades and plan_info and plan_info.get("blades"):
            blades, blades_from = plan_info["blades"], "plan"
        fp = fingerprint.fingerprint(res, blades=blades)
        if blades:
            results["blades"] = {"count": int(blades), "source": blades_from}
        # spectra after the rules, so peaks they left out as artifacts are not labelled on the plots
        results["files"]["spectra"] = _spectrum_pngs(res, arrays, out, fp)
        results["files"]["spectrum"] = "spectrum.png"
        results["fingerprint"] = fp
        results["warnings"].extend(fp.get("notes", []))
        st.set(step, "done", f"{fp['verdict']} ({fp['confidence']})")

        # 6. magnify
        step = "magnify"
        st.set(step, "running")
        shown = {k: magnify_masks.get(k, m) for k, m in masks.items()} if magnify_masks else masks
        info = magnify.magnify(clip, shown, arrays["rel"], fps, rs["hz"], out, start=start, alpha=options.alpha,
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
            cmp_res, mapping = res, None
            if mask_source == "SAM 3.1" and results.get("sam_iou"):
                mapping = {p: v["phrase"] for p, v in results["sam_iou"].items() if v["iou"] >= IOU_MATCH}
                cmp_res = {**res, "parts": {p: res["parts"][ph] for p, ph in mapping.items() if ph in res["parts"]}}
            results["truth_comparison"] = evaluate.compare(truth, cmp_res, fp["verdict"])
            results["truth_comparison"]["parts_matched_by_iou"] = mapping
            results["truth"] = {"machine": truth.get("machine", "washer"), "fault": truth["fault"],
                                "expected_verdict": truth["expected_verdict"],
                                "rpm": truth["rpm"], "seed": truth["seed"]}

        # 7. diagnose
        step = "diagnose"
        if api:
            from . import diagnose as diag_mod
            st.set(step, "running")
            video = out / info["files"]["side_by_side"] if options.spark_video else None
            card = diag_mod.diagnose(api, res, fp, spectrum_png=out / "spectrum.png", magnified_clip=video,
                                     plan=plan_info, label=f"{options.api_label}:diagnose",
                                     band_orders=tuple(options.band))
            usd["diagnose"] = card.get("usd", 0.0)
            results["diagnosis"] = {"status": "done" if "error" not in card["spark"] else "failed", **card}
            if options.api == "replay":
                then = _recorded_rule_verdict(api, f"{options.api_label}:diagnose")
                if then and then != fp["verdict"]:
                    results["diagnosis"]["recorded_against"] = {"rule_verdict": then, "rule_verdict_now": fp["verdict"]}
            sp = card["spark"]
            if "error" in sp:
                st.set(step, "fallback", "Muse Spark diagnosis failed; the rule verdict stands")
            else:
                c = sp["counts"]
                st.set(step, "done", f"Spark: {sp['fault']} ({sp['confidence']}); evidence {c['verified']} verified, "
                                     f"{c['context']} context, {c['dropped']} dropped")
        elif diagnose:
            st.set(step, "running")
            results["diagnosis"] = {"status": "done", **diagnose(results)}
            st.set(step, "done")
        else:
            results["diagnosis"] = {"status": "not_connected",
                                    "headline": fp["verdict"], "note": "The rule verdict is the official result."}
            st.set(step, "skipped", "Muse Spark is off; the rule verdict stands")

        if api:
            results["api"] = {"mode": options.api, "usd": {**{k: round(v, 6) for k, v in usd.items()},
                                                           "total": api.ledger.total()},
                              "calls": api.ledger.summary()}
        results["state"] = "done"
        results["steps"] = st.data["steps"]
        res_path.write_text(json.dumps(_jsonable(results), indent=2))
        st.finish("done")
        return json.loads(res_path.read_text())
    except Exception as e:  # report which step failed, keep what was done
        msg = f"{type(e).__name__}: {e}"
        if type(e).__name__ == "CreditExhausted":
            msg = CREDIT_MESSAGE
        st.set(step, "failed", msg)
        results["state"] = "failed"
        results["error"] = {"step": step, "message": msg, "trace": traceback.format_exc()}
        if api:
            results["api"] = {"mode": options.api, "usd": {"total": api.ledger.total()}, "calls": api.ledger.summary()}
        results["steps"] = st.data["steps"]
        res_path.write_text(json.dumps(_jsonable(results), indent=2))
        st.finish("failed", msg)
        raise


MACHINES = {"washer": "Washing machine", "fan": "Pedestal fan"}


def twin_faults(machine="washer"):
    return list((twin_fan if machine == "fan" else twin).FAULTS)


def twin_clip(fault, runs_dir="runs", seed=1, seconds=4.0, scale=0.5, machine="washer"):
    """Render (or reuse) a twin clip for the app. Returns (clip path, masks, truth)."""
    if machine not in MACHINES:
        raise ValueError(f"machine must be one of {sorted(MACHINES)}, not {machine!r}")
    if machine == "fan":
        h = hashlib.sha1(Path(twin_fan.__file__).read_bytes() + Path(twin.__file__).read_bytes()).hexdigest()[:10]
        name = f"fan_{fault}_seed{seed}_{seconds:g}s_{scale:.3g}x_{h}"
    else:
        h = hashlib.sha1(Path(twin.__file__).read_bytes()).hexdigest()[:10]
        name = f"{fault}_seed{seed}_{seconds:g}s_{scale:.3g}x_{h}"
    d = Path(runs_dir) / "twin_clips" / name
    if not (d / "truth.json").exists():
        tmp = d.with_name(d.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        if machine == "fan":
            twin_fan.render(twin_fan.FanConfig(fault, seconds=seconds, seed=seed, out_scale=scale), tmp)
        else:
            twin.render(twin.TwinConfig(fault, seconds=seconds, seed=seed, out_scale=scale), tmp)
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)
    truth = json.loads((d / "truth.json").read_text())
    masks = {n: d / truth["parts"][n]["mask"] for n in truth["parts"]}
    return d / "clip.mp4", masks, truth


def twin_options(truth, **kw):
    """Options for a twin clip: mm scale from the reference length; without Spark, the true rpm +/- 10% as the range."""
    rpm = truth["rpm"]
    api = kw.get("api", "off")
    rng = None if api != "off" else (rpm * (1 - RPM_RANGE_TWIN), rpm * (1 + RPM_RANGE_TWIN))
    ref = truth["reference"]
    reference = ((ref["part"], ref["height_mm"], "height") if ref.get("axis") == "height"
                 else (ref["part"], ref["width_mm"]))
    if truth.get("blades") and "blades" not in kw:
        kw["blades"] = truth["blades"]  # a fan twin: the renderer's blade count
    return Options(rpm_range=rng, reference=reference, min_seconds=min(4.0, truth["seconds"]), **kw)


def recording(fault):
    return RECORDINGS / fault / "cassette.json.gz"


def run_twin(fault, runs_dir="runs", seed=1, seconds=4.0, scale=0.5, api="off", progress=None, on_start=None,
             machine="washer", **kw):
    """Render (or reuse) a twin clip and run it.

    api "replay" serves the recorded live run for this fault (seed 1, 1920x1080; any duration works, 6 s matches the
    recording exactly; the fan's are under fan-<fault>), so seed and scale are fixed to the recording's. api "live" records the calls.
    For the fan, the background leaves out the blade disc (its grill and hub move with the head), and the magnified
    video moves the whole guard, blades included, with the guard rim's measured motion.
    """
    api_kw = {}
    if api == "replay":
        name = f"fan-{fault}" if machine == "fan" else fault
        seed, scale = RECORDED_TWIN["seed"], RECORDED_TWIN["scale"]
        # the fan's recordings are pipeline runs that sent the magnified clip with the diagnosis; the washer's are not
        api_kw = dict(api="replay", cassette=str(recording(name)), api_label=name, spark_video=machine == "fan")
    elif api in ("live", "record"):
        api_kw = dict(api="record", api_label=f"fan-{fault}" if machine == "fan" else fault)
    clip, masks, truth = twin_clip(fault, runs_dir, seed, seconds, scale, machine)
    extra = {}
    if machine == "fan":
        disc = load_masks({"d": clip.parent / truth["rotating_mask"]})["d"]
        rim = load_masks({"r": masks["fan guard rim"]})["r"]
        extra = dict(ignore=disc, magnify_masks={"fan guard rim": np.maximum(rim, disc)})
    return run(clip, masks, runs_dir, twin_options(truth, **api_kw, **kw), truth=truth, mask_source="twin renderer",
               progress=progress, on_start=on_start, **extra)


def parse_twin(name):
    """'imbalance' (washing machine) or 'fan-imbalance' -> (machine, fault)."""
    machine, fault = ("fan", name[4:]) if name.startswith("fan-") else ("washer", name)
    if fault not in twin_faults(machine):
        raise ValueError(f"unknown twin {name!r}")
    return machine, fault


def parse_reference(text):
    """'part=600' (the part's mask width is 600 mm) or 'part=150:height' (its height) -> (part, mm[, axis])."""
    part, rest = text.split("=", 1)
    mm, _, axis = rest.partition(":")
    axis = axis.strip().lower() or "width"
    if axis not in ("width", "height"):
        raise ValueError(f"reference axis must be width or height, not {axis!r}")
    return (part.strip(), float(mm)) if axis == "width" else (part.strip(), float(mm), "height")


def main(argv=None):
    p = argparse.ArgumentParser(description="Run See the Shake on one clip")
    p.add_argument("clip", nargs="?")
    p.add_argument("--mask", action="append", default=[], help='"part name=mask.png" (repeat per part)')
    p.add_argument("--api", choices=API_MODES, default="off",
                   help="Muse Spark and SAM 3.1: off, live, record (live and saved), replay (needs --cassette)")
    p.add_argument("--cassette", help="recording to replay, or where to save one")
    p.add_argument("--parts", help="comma-separated part phrases to use instead of Spark's")
    p.add_argument("--no-spark-video", action="store_true", help="do not send the magnified clip to Spark")
    p.add_argument("--twin", choices=sorted(twin.FAULTS) + [f"fan-{f}" for f in twin_fan.FAULTS],
                   help="a twin clip: washing machine faults, or fan-healthy, fan-imbalance, fan-looseness")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--rpm", type=float)
    p.add_argument("--blades", type=int, help="a fan's blade count: blades x running speed is blade pass")
    p.add_argument("--rpm-range", type=float, nargs=2)
    p.add_argument("--capture-fps", type=float)
    p.add_argument("--ref", help='"part name=length in mm" across the part\'s mask width, or "part=150:height"')
    p.add_argument("--alpha", default="auto")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--runs", default="runs")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    alpha = a.alpha if a.alpha == "auto" else float(a.alpha)
    show = lambda k, s, d: print(f"[{s:8s}] {k:12s} {d or ''}", flush=True)  # noqa: E731
    if a.twin:
        machine, fault = parse_twin(a.twin)
        r = run_twin(fault, a.runs, a.seed, a.seconds, a.scale, api=a.api, progress=show, alpha=alpha, machine=machine,
                     verify=a.verify)
    else:
        if not a.clip or (not a.mask and a.api == "off"):
            p.error("give a clip and at least one --mask, or --api live (or --twin)")
        masks = dict(m.split("=", 1) for m in a.mask)
        ref = None
        if a.ref:
            ref = parse_reference(a.ref)
        opts = Options(capture_fps=a.capture_fps, rpm=a.rpm, rpm_range=tuple(a.rpm_range) if a.rpm_range else None,
                       reference=ref, alpha=alpha, verify=a.verify, api=a.api, cassette=a.cassette, blades=a.blades,
                       spark_video=not a.no_spark_video,
                       user_parts=[s.strip() for s in a.parts.split(",") if s.strip()] if a.parts else None)
        r = run(a.clip, masks, a.runs, opts, progress=show)
    fp = r["fingerprint"]
    print(f"verdict: {fp['verdict']} ({fp['confidence']}): {fp['reason']}")
    if r.get("truth_comparison"):
        tc = r["truth_comparison"]
        print(f"truth: {r['truth']['machine']} {r['truth']['fault']}, expected {tc['expected_verdict']} "
              f"({'right' if tc['verdict_ok'] else 'WRONG'}); components within tolerance {tc['passed']}/{tc['scored']}")
    if r.get("api"):
        print(f"API cost: ${r['api']['usd']['total']:.4f}")
    print(f"results: {Path(a.runs) / r['run_id'] / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

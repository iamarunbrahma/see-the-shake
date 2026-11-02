"""See the Shake web app: upload a clip or pick a twin clip, follow the seven steps, see the results.

usage: uvicorn shake.web.app:app --port 8000      (runs are stored under $SHAKE_RUNS, default ./runs)
Muse Spark and SAM 3.1 run only when the request asks for them: "live" calls the Meta Model API (and records
the run), "replay" serves a recorded live twin run at no cost, "off" uses supplied or renderer masks.
"""
import hashlib
import os
import re
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import ingest, pipeline, twin

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 600 * 1024 * 1024
BOX = re.compile(r"^\s*(?P<name>[^:]+?)\s*:\s*(?P<x>\d+)\s*,\s*(?P<y>\d+)\s*,\s*(?P<w>\d+)\s*,\s*(?P<h>\d+)\s*$")
FAULT_LABELS = {"healthy": "Healthy", "imbalance": "Imbalance", "looseness": "Looseness", "misalignment": "Misalignment"}
FAN_FAULT_LABELS = {"healthy": "Healthy", "imbalance": "Imbalance (head wobbles once a turn)",
                    "looseness": "Looseness (loose head mount)"}


class TwinRequest(BaseModel):
    fault: str
    seed: int = 1
    seconds: float = 4.0
    scale: float = 0.5
    alpha: str = "auto"
    api: str = "off"  # off, replay (recorded live run, no cost), live (costs money)
    machine: str = "washer"  # washer or fan


def _alpha(v):
    v = (v or "auto").strip()
    if v == "auto":
        return "auto"
    try:
        return float(v)
    except ValueError:
        raise HTTPException(400, "Magnification must be a number or auto")


def _blades(v):
    if v is None or str(v).strip() == "":
        return None
    try:
        n = int(str(v).strip())
    except ValueError:
        raise HTTPException(400, "Fan blades must be a whole number")
    if not 2 <= n <= 12:
        raise HTTPException(400, "Fan blades must be between 2 and 12")
    return n


def _opt_float(v, name):
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        raise HTTPException(400, f"{name} must be a number")


def create_app(runs_dir=None):
    runs = Path(runs_dir or os.environ.get("SHAKE_RUNS", "runs")).resolve()
    (runs / "uploads").mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="See the Shake")
    pool = ThreadPoolExecutor(max_workers=1)  # measurement is CPU heavy: one run at a time
    jobs, lock = {}, threading.Lock()

    def update(job_id, **kw):
        with lock:
            jobs[job_id].update(kw)

    def submit(job_id, fn):
        def work():
            update(job_id, state="running")
            try:
                r = fn()
                update(job_id, state=r["state"], run_id=r["run_id"])
            except Exception as e:
                update(job_id, state="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())
        with lock:
            jobs[job_id] = {"job_id": job_id, "state": "queued", "run_id": None, "phase": None, "error": None}
        pool.submit(work)

    def progress_for(job_id, rid_box):
        def cb(key, status, detail):
            update(job_id, phase=f"{key}: {status}")
        return cb

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/api/samples")
    def samples():
        return {"twin_faults": [{"key": k, "label": FAULT_LABELS[k],
                                 "recorded": pipeline.recording(k).exists()} for k in twin.FAULTS],
                "machines": [{"key": "washer", "label": pipeline.MACHINES["washer"],
                              "faults": [{"key": k, "label": FAULT_LABELS[k], "recorded": pipeline.recording(k).exists()}
                                         for k in twin.FAULTS]},
                             {"key": "fan", "label": pipeline.MACHINES["fan"],
                              "faults": [{"key": k, "label": FAN_FAULT_LABELS[k], "recorded": False}
                                         for k in pipeline.twin_faults("fan")]}],
                "recorded_twin": pipeline.RECORDED_TWIN,
                "twin_defaults": TwinRequest(fault="imbalance").model_dump(),
                "steps": [{"key": k, "label": label} for k, label in pipeline.STEPS]}

    @app.post("/api/twin")
    def start_twin(req: TwinRequest):
        if req.machine not in pipeline.MACHINES:
            raise HTTPException(400, f"Unknown machine {req.machine}")
        if req.fault not in pipeline.twin_faults(req.machine):
            raise HTTPException(400, f"Unknown fault {req.fault} for the {pipeline.MACHINES[req.machine].lower()}")
        if req.api == "replay" and req.machine != "washer":
            raise HTTPException(400, "Recorded live runs exist for the washing machine only")
        if not (0.5 <= req.seconds <= 10 and 0.2 <= req.scale <= 1.0):
            raise HTTPException(400, "Twin clips are 0.5 to 10 s long at a scale of 0.2 to 1")
        if req.api not in ("off", "replay", "live"):
            raise HTTPException(400, "Muse Spark and SAM 3.1 must be off, replay or live")
        if req.api == "replay" and not pipeline.recording(req.fault).exists():
            raise HTTPException(400, f"There is no recorded live run for {req.fault}")
        if req.api == "replay" and req.seconds > pipeline.RECORDED_TWIN["seconds"]:
            raise HTTPException(400, "A recorded run covers at most 6 s")
        alpha = _alpha(req.alpha)
        job_id = uuid.uuid4().hex[:12]

        def fn():
            update(job_id, phase="rendering the twin clip")
            return pipeline.run_twin(req.fault, runs, req.seed, req.seconds, req.scale, api=req.api, alpha=alpha,
                                     machine=req.machine,
                                     progress=progress_for(job_id, None),
                                     on_start=lambda rid: update(job_id, run_id=rid))
        submit(job_id, fn)
        return {"job_id": job_id}

    @app.post("/api/runs")
    async def start_upload(clip: UploadFile = File(...), masks: list[UploadFile] = File(default=[]),
                           mask_names: str = Form(""), boxes: str = Form(""), capture_fps: str = Form(""),
                           rpm: str = Form(""), rpm_min: str = Form(""), rpm_max: str = Form(""),
                           ref_part: str = Form(""), ref_mm: str = Form(""), ref_axis: str = Form("width"),
                           alpha: str = Form("auto"),
                           min_seconds: str = Form("4"), verify: bool = Form(False), api: str = Form("off"),
                           parts: str = Form(""), blades: str = Form("")):
        if api not in ("off", "live"):
            raise HTTPException(400, "Muse Spark and SAM 3.1 must be off or live for an uploaded clip")
        # store the clip under its hash, so the same file always maps to the same run
        tmp = runs / "uploads" / f"incoming_{uuid.uuid4().hex}"
        h, size = hashlib.sha256(), 0
        with open(tmp, "wb") as fh:
            while chunk := await clip.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    fh.close()
                    tmp.unlink(missing_ok=True)
                    raise HTTPException(413, "The clip is larger than 600 MB")
                h.update(chunk)
                fh.write(chunk)
        ext = Path(clip.filename or "clip.mp4").suffix.lower() or ".mp4"
        dest = runs / "uploads" / f"{h.hexdigest()[:16]}{ext}"
        if dest.exists():
            tmp.unlink()
        else:
            shutil.move(tmp, dest)
        try:
            # the real check, before any job starts: a slowed slow-motion file needs its capture rate
            info = ingest.clip_info(dest, capture_fps=_opt_float(capture_fps, "Capture fps"))
        except ingest.IngestError as e:
            if e.field:
                raise HTTPException(400, {"message": str(e), "field": e.field})
            raise HTTPException(400, str(e))
        H, W = info.height, info.width
        names = [n.strip() for n in mask_names.split(",") if n.strip()]
        part_masks = {}
        for i, up in enumerate(masks):
            if not up.filename:
                continue
            img = cv2.imdecode(np.frombuffer(await up.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise HTTPException(400, f"Mask {up.filename} is not an image")
            if img.shape != (H, W):
                raise HTTPException(400, f"Mask {up.filename} is {img.shape[1]}x{img.shape[0]}, the clip is {W}x{H}")
            name = names[i] if i < len(names) else Path(up.filename).stem.replace("_", " ")
            part_masks[name] = img
        for line in boxes.splitlines():
            if not line.strip():
                continue
            m = BOX.match(line)
            if not m:
                raise HTTPException(400, f"Cannot read the box '{line.strip()}'. Use: part name: x, y, width, height")
            x, y, w, hh = (int(m[k]) for k in ("x", "y", "w", "h"))
            box = np.zeros((H, W), np.uint8)
            box[y:y + hh, x:x + w] = 255
            part_masks[m["name"]] = box
        if not part_masks and api == "off":
            raise HTTPException(400, "Add at least one part: a mask image or a box, "
                                     "or turn on Muse Spark and SAM 3.1 to find the parts.")
        rmin, rmax = _opt_float(rpm_min, "Lowest rpm"), _opt_float(rpm_max, "Highest rpm")
        ref = None
        if ref_part.strip() and ref_mm.strip():
            if api == "off" and ref_part.strip() not in part_masks:
                raise HTTPException(400, f"The reference part '{ref_part}' is not one of the parts")
            axis = (ref_axis or "width").strip().lower()
            if axis not in ("width", "height"):
                raise HTTPException(400, "The reference length is the part's width or height")
            ref = (ref_part.strip(), _opt_float(ref_mm, "Reference length")) + (("height",) if axis == "height" else ())
        opts = pipeline.Options(capture_fps=_opt_float(capture_fps, "Capture fps"), rpm=_opt_float(rpm, "Rpm"),
                                rpm_range=(rmin, rmax) if rmin and rmax else None, reference=ref,
                                alpha=_alpha(alpha), min_seconds=_opt_float(min_seconds, "Shortest window") or 4.0,
                                verify=verify, api="record" if api == "live" else "off",
                                user_parts=[s.strip() for s in parts.split(",") if s.strip()] or None,
                                blades=_blades(blades))
        job_id = uuid.uuid4().hex[:12]

        def fn():
            return pipeline.run(dest, part_masks, runs, opts, mask_source="uploaded",
                                progress=progress_for(job_id, None), on_start=lambda rid: update(job_id, run_id=rid))
        submit(job_id, fn)
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        with lock:
            j = dict(jobs.get(job_id) or {})
        if not j:
            raise HTTPException(404, "No such job")
        rid = j.get("run_id")
        if rid:
            d = runs / rid
            if (d / "status.json").exists():
                j["status"] = _read_json(d / "status.json")
            if j["state"] == "done" and (d / "results.json").exists():
                j["results"] = _read_json(d / "results.json")
                j["base"] = f"/runs/{rid}/"
        return j

    @app.get("/api/runs/{run_id}")
    def run_results(run_id: str):
        if not re.fullmatch(r"[0-9a-f]{12}", run_id) or not (runs / run_id / "results.json").exists():
            raise HTTPException(404, "No such run")
        return {"results": _read_json(runs / run_id / "results.json"), "base": f"/runs/{run_id}/"}

    app.mount("/runs", StaticFiles(directory=runs), name="runs")
    app.state.runs = runs
    app.state.jobs = jobs
    return app


def _read_json(p):
    import json
    for _ in range(3):
        try:
            return json.loads(Path(p).read_text())
        except json.JSONDecodeError:  # being replaced right now
            continue
    return None


app = create_app()

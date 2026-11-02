"""See the Shake web app: upload a clip or pick a twin clip, follow the seven steps, see the results.

usage: uvicorn shake.web.app:app --port 8000      (runs are stored under $SHAKE_RUNS, default ./runs)
No model calls happen here; the Muse Spark and SAM 3.1 steps show as slots until they are connected.
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


class TwinRequest(BaseModel):
    fault: str
    seed: int = 1
    seconds: float = 4.0
    scale: float = 0.5
    alpha: str = "auto"


def _alpha(v):
    v = (v or "auto").strip()
    if v == "auto":
        return "auto"
    try:
        return float(v)
    except ValueError:
        raise HTTPException(400, "Magnification must be a number or auto")


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
        return {"twin_faults": [{"key": k, "label": FAULT_LABELS[k]} for k in twin.FAULTS],
                "twin_defaults": TwinRequest(fault="imbalance").model_dump(),
                "steps": [{"key": k, "label": label} for k, label in pipeline.STEPS]}

    @app.post("/api/twin")
    def start_twin(req: TwinRequest):
        if req.fault not in twin.FAULTS:
            raise HTTPException(400, f"Unknown fault {req.fault}")
        if not (0.5 <= req.seconds <= 10 and 0.2 <= req.scale <= 1.0):
            raise HTTPException(400, "Twin clips are 0.5 to 10 s long at a scale of 0.2 to 1")
        alpha = _alpha(req.alpha)
        job_id = uuid.uuid4().hex[:12]

        def fn():
            update(job_id, phase="rendering the twin clip")
            clip, masks, truth = pipeline.twin_clip(req.fault, runs, req.seed, req.seconds, req.scale)
            opts = pipeline.twin_options(truth, alpha=alpha)
            rid = pipeline.run_id(clip, pipeline.load_masks(masks), opts, "twin renderer", truth)
            update(job_id, run_id=rid)
            return pipeline.run(clip, masks, runs, opts, truth=truth, mask_source="twin renderer",
                                progress=progress_for(job_id, None))
        submit(job_id, fn)
        return {"job_id": job_id}

    @app.post("/api/runs")
    async def start_upload(clip: UploadFile = File(...), masks: list[UploadFile] = File(default=[]),
                           mask_names: str = Form(""), boxes: str = Form(""), capture_fps: str = Form(""),
                           rpm: str = Form(""), rpm_min: str = Form(""), rpm_max: str = Form(""),
                           ref_part: str = Form(""), ref_mm: str = Form(""), alpha: str = Form("auto"),
                           min_seconds: str = Form("4"), verify: bool = Form(False)):
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
            info = ingest.clip_info(dest, capture_fps=_opt_float(capture_fps, "Capture fps"), min_fps=1)
        except ingest.IngestError as e:
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
        if not part_masks:
            raise HTTPException(400, "Add at least one part: a mask image or a box. "
                                     "SAM 3.1 will find parts automatically once it is connected.")
        rmin, rmax = _opt_float(rpm_min, "Lowest rpm"), _opt_float(rpm_max, "Highest rpm")
        ref = None
        if ref_part.strip() and ref_mm.strip():
            if ref_part.strip() not in part_masks:
                raise HTTPException(400, f"The reference part '{ref_part}' is not one of the parts")
            ref = (ref_part.strip(), _opt_float(ref_mm, "Reference length"))
        opts = pipeline.Options(capture_fps=_opt_float(capture_fps, "Capture fps"), rpm=_opt_float(rpm, "Rpm"),
                                rpm_range=(rmin, rmax) if rmin and rmax else None, reference=ref,
                                alpha=_alpha(alpha), min_seconds=_opt_float(min_seconds, "Shortest window") or 4.0,
                                verify=verify)
        job_id = uuid.uuid4().hex[:12]

        def fn():
            update(job_id, run_id=pipeline.run_id(dest, pipeline.load_masks(part_masks), opts, "uploaded"))
            return pipeline.run(dest, part_masks, runs, opts, mask_source="uploaded",
                                progress=progress_for(job_id, None))
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

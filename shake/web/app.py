"""See the Shake web app: upload a clip or pick a twin clip, follow the seven steps, see the results.

usage: uvicorn shake.web.app:app --port 8000      (runs are stored under $SHAKE_RUNS, default ./runs)
Muse Spark and SAM 3.1 run only when the request asks for them: "live" calls the Meta Model API (and records
the run), "replay" serves a recorded live twin run at no cost, "off" uses supplied or renderer masks.

The app is meant for your own machine (uvicorn listens on 127.0.0.1 by default). It still refuses requests that
other web sites make on your behalf (a cross-site POST could otherwise start a paid live run), and it stops reading
an upload as soon as it passes the size limit.
"""
import hashlib
import json
import logging
import math
import os
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import ingest, pipeline, twin
from ..meta_api import MissingKey, require_key

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 600 * 1024 * 1024
BODY_SLACK = 32 * 1024 * 1024  # masks and form fields on top of the clip
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv")
BOX = re.compile(r"^\s*(?P<name>[^:]+?)\s*:\s*(?P<x>\d+)\s*,\s*(?P<y>\d+)\s*,\s*(?P<w>\d+)\s*,\s*(?P<h>\d+)\s*$")
FAULT_LABELS = {"healthy": "Healthy", "imbalance": "Imbalance", "looseness": "Looseness", "misalignment": "Misalignment"}
FAN_FAULT_LABELS = {"healthy": "Healthy", "imbalance": "Imbalance (head wobbles once a turn)",
                    "looseness": "Looseness (loose head mount)"}
log = logging.getLogger("shake.web")


class TwinRequest(BaseModel):
    fault: str
    seed: int = 1
    seconds: float = 4.0
    scale: float = 0.5
    alpha: str = "auto"
    api: str = "off"  # off, replay (recorded live run, no cost), live (costs money)
    machine: str = "washer"  # washer or fan


def _size(n):
    return f"{n / (1024 * 1024):.0f} MB" if n >= 1024 * 1024 else f"{n} bytes"


def _bad(message, field=None, status=400):
    """An error the page shows; with field, the page also points at that input."""
    return HTTPException(status, {"message": message, "field": field} if field else message)


def _alpha(v):
    v = (v or "auto").strip()
    if v == "auto":
        return "auto"
    try:
        x = float(v)
    except ValueError:
        x = float("nan")
    lo, hi = pipeline.magnify.ALPHA_RANGE
    if not (math.isfinite(x) and lo <= x <= hi):
        raise _bad(f"Magnification must be auto or a number from {lo:g} to {hi:g}", "alpha")
    return x


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


def _opt_float(v, name, field=None):
    """A typed number, or None when blank. Must be finite; whether it is in range is checked with the options."""
    if v is None or str(v).strip() == "":
        return None
    try:
        x = float(v)
    except ValueError:
        raise _bad(f"{name} must be a number", field)
    if not math.isfinite(x):
        raise _bad(f"{name} must be a finite number", field)
    return x


def _clean(o):
    """JSON from disk made safe to send: NaN and infinity (which JSON cannot carry) become null."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


async def _send_json(send, status, detail):
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                            (b"connection", b"close")]})
    await send({"type": "http.response.body", "body": body})


class SameOriginOnly:
    """Refuse state-changing requests that another site's page makes through the user's browser (CSRF). Browsers
    mark those with Sec-Fetch-Site and Origin; requests from curl or scripts carry neither and pass."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] not in ("GET", "HEAD", "OPTIONS"):
            h = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
            site, origin = h.get("sec-fetch-site"), h.get("origin")
            cross = site is not None and site not in ("same-origin", "none")
            if origin is not None and not cross:
                cross = origin == "null" or urlsplit(origin).netloc.lower() != h.get("host", "").lower()
            if cross:
                return await _send_json(send, 403, "Requests from other web sites are not accepted. Open the app "
                                                   "itself and start the run there.")
        return await self.app(scope, receive, send)


class BodyLimit:
    """Stop reading an upload once it passes the limit, instead of receiving it all first (the form parser would
    otherwise spool the whole body to disk). A declared Content-Length over the limit is refused before reading."""

    def __init__(self, app, limit, paths):
        self.app, self.limit, self.paths = app, limit, set(paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in self.paths:
            return await self.app(scope, receive, send)
        message = f"The upload is larger than {_size(self.limit)}."
        length = dict(scope["headers"]).get(b"content-length")
        try:
            if length is not None and int(length) > self.limit:
                return await _send_json(send, 413, message)
        except ValueError:
            return await _send_json(send, 400, "Bad Content-Length header.")
        seen, over, started = 0, False, False

        async def counted():
            nonlocal seen, over
            if over:
                return {"type": "http.disconnect"}
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.limit:
                    over = True
                    return {"type": "http.disconnect"}
            return msg

        async def guarded(msg):
            nonlocal started
            if over:  # the app's reply to a body cut short is replaced by the 413 below
                return
            started = True
            await send(msg)

        try:
            await self.app(scope, counted, guarded)
        except Exception:
            if not over:
                raise
        if over and not started:
            await _send_json(send, 413, message)


def create_app(runs_dir=None, max_upload=MAX_UPLOAD, max_body=None):
    """max_upload limits the clip; max_body (default max_upload plus room for masks) limits the whole request."""
    runs = Path(runs_dir or os.environ.get("SHAKE_RUNS", "runs")).resolve()
    (runs / "uploads").mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="See the Shake")
    pool = ThreadPoolExecutor(max_workers=1)  # measurement is CPU heavy: one run at a time
    jobs, lock = {}, threading.Lock()
    in_use = {}  # upload path -> number of queued or running jobs that read it

    def update(job_id, **kw):
        with lock:
            jobs[job_id].update(kw)

    def hold(path):
        with lock:
            in_use[path] = in_use.get(path, 0) + 1

    def release(path, delete):
        """A job is done with an upload; a failed job's upload is deleted once no other job needs it."""
        with lock:
            in_use[path] = in_use.get(path, 1) - 1
            if in_use[path] <= 0:
                in_use.pop(path, None)
                if delete:
                    Path(path).unlink(missing_ok=True)

    def submit(job_id, fn, upload=None):
        def work():
            update(job_id, state="running")
            failed = True
            try:
                r = fn()
                failed = r["state"] != "done"
                update(job_id, state=r["state"], run_id=r["run_id"], from_cache=bool(r.get("from_cache")))
            except Exception as e:  # the page gets one plain line; the traceback goes to the server log
                log.error("job %s failed", job_id, exc_info=True)
                update(job_id, state="failed", error=pipeline.user_message(e))
            finally:
                if upload is not None:
                    release(upload, failed)
        with lock:
            jobs[job_id] = {"job_id": job_id, "state": "queued", "run_id": None, "phase": None, "error": None}
        if upload is not None:
            hold(upload)
        pool.submit(work)

    def progress_for(job_id):
        def cb(key, status, detail):
            update(job_id, phase=f"{key}: {status}")
        return cb

    def need_key(field):
        try:
            require_key()
        except MissingKey as e:
            raise _bad(str(e), field)

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
                              "faults": [{"key": k, "label": FAN_FAULT_LABELS[k],
                                          "recorded": pipeline.recording(f"fan-{k}").exists()}
                                         for k in pipeline.twin_faults("fan")]}],
                "recorded_twin": pipeline.RECORDED_TWIN,
                "twin_defaults": TwinRequest(fault="imbalance").model_dump(),
                "max_upload_bytes": max_upload,
                "steps": [{"key": k, "label": label} for k, label in pipeline.STEPS]}

    @app.post("/api/twin")
    def start_twin(req: TwinRequest):
        if req.machine not in pipeline.MACHINES:
            raise HTTPException(400, f"Unknown machine {req.machine}")
        if req.fault not in pipeline.twin_faults(req.machine):
            raise HTTPException(400, f"Unknown fault {req.fault} for the {pipeline.MACHINES[req.machine].lower()}")
        if not (0.5 <= req.seconds <= 10 and 0.2 <= req.scale <= 1.0):
            raise HTTPException(400, "Twin clips are 0.5 to 10 s long at a scale of 0.2 to 1")
        if not 0 <= req.seed < 2 ** 32:
            raise _bad("The seed must be a whole number from 0 to 4294967295", "seed")
        if req.api not in ("off", "replay", "live"):
            raise HTTPException(400, "Muse Spark and SAM 3.1 must be off, replay or live")
        if req.api == "live":
            need_key("twin_api")
        if req.api == "replay" and not pipeline.recording(
                f"fan-{req.fault}" if req.machine == "fan" else req.fault).exists():
            raise HTTPException(400, f"There is no recorded live run for {req.fault}")
        if req.api == "replay" and req.seconds > pipeline.RECORDED_TWIN["seconds"]:
            raise HTTPException(400, "A recorded run covers at most 6 s")
        alpha = _alpha(req.alpha)
        job_id = uuid.uuid4().hex[:12]

        def fn():
            update(job_id, phase="rendering the twin clip")
            return pipeline.run_twin(req.fault, runs, req.seed, req.seconds, req.scale, api=req.api, alpha=alpha,
                                     machine=req.machine,
                                     progress=progress_for(job_id),
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
        if api == "live":
            need_key("clip_api")
        ext = Path(clip.filename or "clip.mp4").suffix.lower() or ".mp4"
        if ext not in VIDEO_EXTS:
            shown = ext if len(ext) <= 12 else ext[:12] + "..."
            raise _bad(f"Choose a video file (MP4, MOV, WebM, AVI or MKV); this file ends in '{shown}'.", "clip")
        # store the clip under its hash, so the same file always maps to the same run
        tmp = runs / "uploads" / f"incoming_{uuid.uuid4().hex}"
        h, size = hashlib.sha256(), 0
        try:
            with open(tmp, "wb") as fh:
                while chunk := await clip.read(1 << 20):
                    size += len(chunk)
                    if size > max_upload:
                        raise _bad(f"The clip is larger than {_size(max_upload)}", "clip", 413)
                    h.update(chunk)
                    fh.write(chunk)
            dest = runs / "uploads" / f"{h.hexdigest()[:16]}{ext}"
            created = not dest.exists()
            if created:
                shutil.move(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        try:
            opts, part_masks = await _check_upload(dest, masks, mask_names, boxes, capture_fps, rpm, rpm_min, rpm_max,
                                                   ref_part, ref_mm, ref_axis, alpha, min_seconds, verify, api, parts,
                                                   blades)
        except Exception:
            with lock:
                busy = dest in in_use
            if created and not busy:
                dest.unlink(missing_ok=True)  # a rejected upload is not kept
            raise
        job_id = uuid.uuid4().hex[:12]

        def fn():
            return pipeline.run(dest, part_masks, runs, opts, mask_source="uploaded",
                                progress=progress_for(job_id), on_start=lambda rid: update(job_id, run_id=rid))
        submit(job_id, fn, upload=dest)
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        with lock:
            j = dict(jobs.get(job_id) or {})
        if not j:
            raise HTTPException(404, "No such job: the server may have been restarted. Start the run again.")
        rid = j.get("run_id")
        if rid:
            d = runs / rid
            if (d / "status.json").exists():
                j["status"] = _read_json(d / "status.json")
            if j["state"] == "done" and (d / "results.json").exists():
                j["results"] = _read_json(d / "results.json")
                j["base"] = f"/runs/{rid}/"
        return _clean(j)

    @app.get("/api/runs/{run_id}")
    def run_results(run_id: str):
        if not re.fullmatch(r"[0-9a-f]{12}", run_id) or not (runs / run_id / "results.json").exists():
            raise HTTPException(404, "No such run")
        return _clean({"results": _read_json(runs / run_id / "results.json"), "base": f"/runs/{run_id}/"})

    app.mount("/runs", StaticFiles(directory=runs), name="runs")
    app.add_middleware(BodyLimit, limit=max_body or max_upload + BODY_SLACK, paths=["/api/runs"])
    app.add_middleware(SameOriginOnly)  # outermost: a cross-site request is refused before its body is read
    app.state.runs = runs
    app.state.jobs = jobs
    return app


async def _check_upload(dest, masks, mask_names, boxes, capture_fps, rpm, rpm_min, rpm_max, ref_part, ref_mm,
                        ref_axis, alpha, min_seconds, verify, api, parts, blades):
    """Everything about an upload that can be checked before a job starts. Returns (options, part masks)."""
    fps_in = _opt_float(capture_fps, "Capture fps", "capture_fps")
    if fps_in is not None and fps_in <= 0:
        raise _bad("Capture fps must be greater than 0", "capture_fps")
    try:
        # the real check, before any job starts: a slowed slow-motion file needs its capture rate
        info = ingest.clip_info(dest, capture_fps=fps_in)
        H, W = pipeline._frame_shape(dest)  # as decoded: a portrait phone clip is taller than ffprobe's stored size
    except ingest.IngestError as e:
        raise _bad(str(e), e.field or "clip")
    names = [n.strip() for n in mask_names.split(",") if n.strip()]
    part_masks = {}
    try:
        for i, up in enumerate(masks):
            if not up.filename:
                continue
            img = cv2.imdecode(np.frombuffer(await up.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise HTTPException(400, f"Mask {Path(up.filename).name} is not an image")
            if img.shape != (H, W):
                raise HTTPException(400, f"Mask {Path(up.filename).name} is {img.shape[1]}x{img.shape[0]}, the clip "
                                         f"is {W}x{H}")
            name = names[i] if i < len(names) else Path(up.filename).stem.replace("_", " ")
            part_masks[_part_name(name, "mask_names")] = img
        for line in boxes.splitlines():
            if not line.strip():
                continue
            m = BOX.match(line)
            if not m:
                raise HTTPException(400, f"Cannot read the box '{line.strip()}'. Use: part name: x, y, width, height")
            x, y, w, hh = (int(m[k]) for k in ("x", "y", "w", "h"))
            name = _part_name(m["name"], "boxes")
            if w == 0 or hh == 0:
                raise _bad(f"The box '{name}' has no width or height.", "boxes")
            if x >= W or y >= H:
                raise _bad(f"The box '{name}' starts at {x}, {y}, outside the {W}x{H} frame.", "boxes")
            box = np.zeros((H, W), np.uint8)
            box[y:y + hh, x:x + w] = 255
            part_masks[name] = box
        if not part_masks and api == "off":
            raise HTTPException(400, "Add at least one part: a mask image or a box, "
                                     "or turn on Muse Spark and SAM 3.1 to find the parts.")
        if part_masks:
            pipeline.check_masks(part_masks, (H, W))
    except pipeline.OptionsError as e:
        raise _bad(str(e), e.field or "boxes")
    rmin, rmax = _opt_float(rpm_min, "Lowest rpm", "rpm_min"), _opt_float(rpm_max, "Highest rpm", "rpm_max")
    if (rmin is None) != (rmax is None):
        raise _bad("Give both the lowest and the highest rpm, or neither", "rpm_min" if rmin is None else "rpm_max")
    ref = None
    if ref_part.strip() or ref_mm.strip():
        if not (ref_part.strip() and ref_mm.strip()):
            raise _bad("Give the reference part and its size, or neither",
                       "ref_part" if not ref_part.strip() else "ref_mm")
        if api == "off" and ref_part.strip() not in part_masks:
            raise HTTPException(400, f"The reference part '{ref_part}' is not one of the parts")
        axis = (ref_axis or "width").strip().lower()
        if axis not in ("width", "height"):
            raise HTTPException(400, "The reference length is the part's width or height")
        ref = ((ref_part.strip(), _opt_float(ref_mm, "Reference length", "ref_mm"))
               + (("height",) if axis == "height" else ()))
    shortest = _opt_float(min_seconds, "Shortest window", "min_seconds")
    opts = pipeline.Options(capture_fps=fps_in, rpm=_opt_float(rpm, "Rpm", "rpm"),
                            rpm_range=(rmin, rmax) if rmin is not None else None, reference=ref,
                            alpha=_alpha(alpha), min_seconds=4.0 if shortest is None else shortest,
                            verify=verify, api="record" if api == "live" else "off",
                            user_parts=[s.strip() for s in parts.split(",") if s.strip()] or None,
                            blades=_blades(blades))
    try:
        pipeline.validate_options(opts, fps=info.capture_fps)
    except pipeline.OptionsError as e:
        raise _bad(str(e), e.field)
    return opts, part_masks


def _part_name(name, field):
    try:
        return pipeline.check_part_name(name)
    except pipeline.OptionsError as e:
        raise _bad(str(e), field)


def _read_json(p):
    for _ in range(3):
        try:
            return json.loads(Path(p).read_text())
        except json.JSONDecodeError:  # being replaced right now
            continue
    return None


app = create_app()

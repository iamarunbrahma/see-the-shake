"""Plan the inspection (Muse Spark): what machine this is, its plausible running speed, and which parts to measure.

Spark watches a downscaled, real-time copy of the steady window (480p, 30 fps, a few seconds) uploaded with the
Files API and passed as an input_file. It returns strict JSON; code validates and clamps it:
- the running-speed range is sorted, widened if implausibly narrow, and clamped to what the clip can measure
  (above F_MIN, below Nyquist); a range entirely outside that band is dropped with a warning, and running speed
  then falls back to the strongest peak;
- Spark marks each part's visibility, and only parts it can clearly see as separate objects are kept (each SAM
  request costs money, and faint parts such as a strip of control panel come back empty);
- part phrases are normalised to short noun phrases, de-duplicated and capped at four, with the whole machine
  first (SAM's fallback phrase);
- parts whose outline changes as they turn (blades, spokes, a drum's contents) are dropped: they are measured
  through their housing, downrod or cabinet instead. Spark marks each part's rotation; a smooth round part whose
  outline stays the same as it turns (a plain shaft, a solid disc) is kept and measured directly, as an
  eddy-current probe would. For phrases without that mark (earlier recordings), names that sound like rotating
  parts are dropped while a static part remains.
The range only gates which spectral peak counts as 1x, so Spark is asked for a deliberately wide range from the
machine type and visible clues, not from apparent rotation (the real-time preview aliases it). An RPM the user
enters overrides the range downstream. The user can also replace the part list (user_parts); it goes through the
same normalisation.
"""
import re
import subprocess
import tempfile
from pathlib import Path

from .meta_api import MetaAPI

MAX_PARTS = 4
MAX_WORDS = 5
PREVIEW_HEIGHT = 480
PREVIEW_FPS = 30
PREVIEW_SECONDS = 4.0
F_MIN_HZ = 2.0  # measure.F_MIN: lowest frequency the peak picker reports
MIN_SPAN = 0.20  # an rpm range narrower than +/-20% of its centre is widened to that
DEFAULT_RPM = (300.0, 3000.0)  # when Spark gives no usable range

ROTATING = {"blade", "blades", "rotor", "rotors", "impeller", "propeller", "drum", "pulley", "belt", "shaft",
            "spindle", "flywheel", "gear", "gears", "agitator", "fan blade", "fan blades", "spinner", "turbine"}
STATIC_HINTS = {"housing", "cover", "guard", "mount", "mounting", "bracket", "casing", "case", "downrod", "frame",
                "cabinet", "base", "panel", "door", "canopy", "stand", "body", "shroud"}
FILLER = {"the", "a", "an", "its", "this", "that", "visible", "front-loading"}
VISIBILITY = ("clear", "partial", "unsure")  # only "clear" parts are segmented (each SAM request costs $0.012)
ROTATION = ("none", "smooth", "changing")  # "changing" parts are dropped; "smooth" ones (a plain shaft) are kept

SCHEMA = {
    "type": "object",
    "properties": {
        "machine_type": {"type": "string"},
        "rpm_min": {"type": "number"},
        "rpm_max": {"type": "number"},
        "rpm_reason": {"type": "string"},
        "parts": {"type": "array", "items": {
            "type": "object",
            "properties": {"phrase": {"type": "string"}, "visibility": {"type": "string", "enum": list(VISIBILITY)},
                           "rotation": {"type": "string", "enum": list(ROTATION)}},
            "required": ["phrase", "visibility", "rotation"], "additionalProperties": False}},
        "rotating_parts_seen": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["machine_type", "rpm_min", "rpm_max", "rpm_reason", "parts", "rotating_parts_seen"],
    "additionalProperties": False,
}

PROMPT = """You are planning a vibration inspection of a running machine from a phone video.
The original clip was filmed in slow motion at {fps:g} fps. This copy is downscaled and plays at real time ({seconds:.1f} s).
A later step measures sub-pixel vibration of each part you name, using a segmentation model that takes one short noun phrase per part.

Return JSON with:
- machine_type: the machine's common name in 1 to 3 words (for example "washing machine", "ceiling fan", "water pump").
- rpm_min, rpm_max: a deliberately wide range of running speeds, in revolutions per minute, for this type of machine running as shown. It only tells the measurement which spectral peak is the running speed, so wide is safe and too narrow is not. Base it on what the machine is and visible clues (model, size, motor, mode), for example roughly 400 to 1600 for a washing machine on spin, or 150 to 400 for a ceiling fan.
  This copy is too slow to judge rotation from: the machine turns faster than its frame rate can show, so how fast or which way a part appears to turn is meaningless. Do not estimate speed from apparent rotation.
- rpm_reason: one short sentence on how you chose the range.
- parts: 1 to 4 parts to measure, each with:
  - phrase: a short noun phrase (2 to 4 words, common nouns) that a segmentation model could match to one object.
  - visibility: "clear" only if the part is a distinct object with its own visible outline that stands out from what surrounds it; "partial" if it is small, faint, blends into a larger surface or is partly hidden; "unsure" otherwise.
  The first must be the whole machine (its machine_type), marked "clear". Add another part only if it is clearly visible as a separate object, such as a door with a visible rim, a motor housing or a downrod. A flat area or strip of a larger surface is not a separate part. Fewer parts are better than a doubtful one: the whole machine alone is fine. Do not name hidden parts or sides that are out of view.
  - rotation: "none" if the part does not turn; "smooth" if it turns but its outline stays the same as it turns (a plain shaft, a solid disc, a coupling sleeve), so its wobble can be measured directly; "changing" if its outline changes as it turns (fan blades, spokes, a drum's contents, a belt). Do not name "changing" parts: they are measured through the static part that holds them.
- rotating_parts_seen: rotating parts you can see, so they can be excluded.
Do not use em dashes."""


class PlanError(ValueError):
    pass


def preview_clip(clip, out, start_frame, capture_fps, seconds=PREVIEW_SECONDS, height=PREVIEW_HEIGHT,
                 fps=PREVIEW_FPS):
    """Real-time, downscaled copy of the window starting at start_frame: every (capture_fps / fps)-th frame."""
    step = max(1, int(round(capture_fps / fps)))
    n_src = int(round(seconds * capture_fps))
    vf = (f"select='between(n\\,{start_frame}\\,{start_frame + n_src - 1})*not(mod(n-{start_frame}\\,{step}))',"
          f"setpts=N/{fps}/TB,scale=-2:{height}")
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(clip), "-vf", vf, "-r", str(fps),
           "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise PlanError(f"ffmpeg could not make the preview: {r.stderr.strip()[:300]}")
    return {"path": str(out), "seconds": round(n_src / capture_fps, 3), "fps": fps, "height": height, "step": step}


def is_rotating(phrase):
    words = set(phrase.split())
    if words & STATIC_HINTS:
        return False
    return bool(words & ROTATING) or any(r in phrase for r in ROTATING if " " in r)


def normalise_phrase(p):
    s = re.sub(r"[^a-z0-9 \-]", " ", str(p).lower().replace("_", " "))
    words = [w for w in s.split() if w not in FILLER]
    return " ".join(words[:MAX_WORDS])


def clamp_rpm(rpm_min, rpm_max, capture_fps, warnings):
    """Sorted, widened and clamped running-speed range in rpm, or None when none of it is measurable."""
    lo_ok = 60 * F_MIN_HZ * 1.05
    hi_ok = 60 * (capture_fps / 2 - 1)
    try:
        lo, hi = float(rpm_min), float(rpm_max)
        if not (lo > 0 and hi > 0):
            raise ValueError
    except (TypeError, ValueError):
        warnings.append(f"no usable rpm range from Spark ({rpm_min!r}, {rpm_max!r}); using {DEFAULT_RPM}")
        lo, hi = DEFAULT_RPM
    if lo > hi:
        lo, hi = hi, lo
        warnings.append("rpm range was reversed")
    c = (lo + hi) / 2
    if (hi - lo) < 2 * MIN_SPAN * c:
        lo, hi = c * (1 - MIN_SPAN), c * (1 + MIN_SPAN)
        warnings.append(f"rpm range widened to +/-{MIN_SPAN:.0%} of {c:.0f}")
    if lo < lo_ok or hi > hi_ok:
        nlo, nhi = max(lo, lo_ok), min(hi, hi_ok)
        if nlo >= nhi:
            warnings.append(f"rpm range {lo:.0f} to {hi:.0f} is outside what {capture_fps:g} fps can measure "
                            f"({lo_ok:.0f} to {hi_ok:.0f} rpm); running speed falls back to the strongest peak")
            return None
        warnings.append(f"rpm range clamped from {lo:.0f}-{hi:.0f} to {nlo:.0f}-{nhi:.0f} (measurable band)")
        lo, hi = nlo, nhi
    return round(lo, 1), round(hi, 1)


def clean_parts(parts, machine_type, warnings, trusted=()):
    """Normalised, de-duplicated phrases, the machine first. Phrases that sound like rotating parts are dropped
    while a static one remains, unless Spark said they do not turn or turn smoothly (trusted)."""
    out = []
    for p in parts or []:
        n = normalise_phrase(p)
        if n and n not in out:
            out.append(n)
    machine = normalise_phrase(machine_type) or "machine"
    rotating = [p for p in out if is_rotating(p) and p not in trusted]
    static = [p for p in out if p not in rotating]
    if rotating and static:
        warnings.append(f"dropped rotating parts {rotating}; they are measured through static parts")
        out = static
    if machine in out:
        out.remove(machine)
    out = [machine] + out
    if len(out) > MAX_PARTS:
        warnings.append(f"kept the first {MAX_PARTS} of {len(out)} parts")
        out = out[:MAX_PARTS]
    return out


def clear_phrases(parts, warnings):
    """(phrases Spark marked as clearly visible, the unclear ones, the kept phrases Spark said do not turn or turn
    smoothly). Plain strings (earlier recordings) count as clear, with no rotation given. Parts whose outline
    changes as they turn are dropped here."""
    keep, unclear, smooth, changing = [], [], set(), []
    for p in parts or []:
        if isinstance(p, dict):
            phrase, vis = p.get("phrase", ""), str(p.get("visibility", "unsure")).strip().lower()
            rot = str(p.get("rotation", "")).strip().lower()
        else:
            phrase, vis, rot = p, "clear", ""
        if rot == "changing":
            changing.append(normalise_phrase(phrase))
            continue
        (keep if vis == "clear" else unclear).append(phrase)
        if vis == "clear" and rot in ("none", "smooth"):
            smooth.add(normalise_phrase(phrase))
    if unclear:
        warnings.append(f"skipped parts Spark could not clearly see: {[normalise_phrase(u) for u in unclear]}")
    if changing:
        warnings.append(f"dropped parts whose outline changes as they turn: {changing}")
    return keep, [normalise_phrase(u) for u in unclear], smooth


def validate(raw, capture_fps, user_parts=None):
    """Validated plan from Spark's JSON (raw dict)."""
    if not isinstance(raw, dict):
        raise PlanError(f"plan is not a JSON object: {raw!r}")
    warnings = []
    machine = normalise_phrase(raw.get("machine_type", "")) or "machine"
    rpm = clamp_rpm(raw.get("rpm_min"), raw.get("rpm_max"), capture_fps, warnings)
    clear, unclear, smooth = clear_phrases(raw.get("parts"), warnings)
    parts = clean_parts(clear, machine, warnings, trusted=smooth)
    source = "spark"
    if user_parts:
        # the user named these parts: none is dropped for sounding like a rotating part
        parts = clean_parts(user_parts, user_parts[0], warnings, trusted={normalise_phrase(u) for u in user_parts})
        source = "user"
    return {"machine_type": machine, "rpm_range": list(rpm) if rpm else None, "rpm_reason": str(raw.get("rpm_reason", ""))[:300],
            "parts": parts, "parts_source": source, "fallback_phrase": machine, "parts_not_clear": unclear,
            "rotating_parts_seen": [normalise_phrase(p) for p in raw.get("rotating_parts_seen") or []][:8],
            "warnings": warnings, "spark_raw": raw}


def plan_from_file(api: MetaAPI, file_id, capture_fps, seconds, user_parts=None, label="plan_parts"):
    """Ask Spark for the plan on an already uploaded preview."""
    content = [{"type": "input_text", "text": PROMPT.format(fps=capture_fps, seconds=seconds)},
               {"type": "input_file", "file_id": file_id}]
    reply = api.spark(content, label=label, schema=SCHEMA, schema_name="inspection_plan")
    plan = validate(reply.json(), capture_fps, user_parts)
    plan["usd"] = round(reply.usd, 6)
    plan["usage"] = {k: reply.usage.get(k) for k in ("input_tokens", "output_tokens")}
    return plan


def plan_parts(api: MetaAPI, clip, capture_fps, start_frame=0, user_parts=None, workdir=None, label="plan_parts"):
    """Downscale the window, upload it, and ask Spark for the inspection plan."""
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        prev = preview_clip(clip, Path(td) / "preview.mp4", start_frame, capture_fps)
        fid = api.upload(prev["path"], label=f"{label}:preview")
        plan = plan_from_file(api, fid, capture_fps, prev["seconds"], user_parts, label)
    plan["preview"] = {k: prev[k] for k in ("seconds", "fps", "height", "step")}
    plan["preview_file_id"] = fid
    return plan

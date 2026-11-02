"""Ingest: read a slow-motion clip on its capture time base and check it is fit to measure.

- Capture fps comes from, in order: the user's override, capture-fps metadata, the stream rate.
  A phone may save 240 fps capture as a 30 fps file slowed 8x; every frame is still there, so the
  time base is 1/capture_fps per frame.
- Clips captured below 120 fps are rejected.
- Synthesised in-between frames (interpolated "slow motion") are flagged: they lack the frame-to-frame
  independent sensor noise of real frames.
- A steady 4 to 10 s window is picked for measurement.

Ported from the spike's frame reader (reference/spike/measure.py) and hardened.
"""
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction

import cv2
import numpy as np

MIN_CAPTURE_FPS = 120.0
SAM_CODECS = {"h264", "hevc"}
SAM_CONTAINERS = {"mov", "mp4"}
CAPTURE_TAG = re.compile(r"capture[._-]?fps|capture[._-]?frame[._-]?rate", re.I)


class IngestError(ValueError):
    """A clip that cannot be measured. field names the input that would fix it (for example "capture_fps")."""

    def __init__(self, message, field=None):
        super().__init__(message)
        self.field = field


@dataclass
class ClipInfo:
    path: str
    width: int
    height: int
    codec: str
    container: str
    stream_fps: float
    capture_fps: float
    fps_source: str  # "override", "metadata:<tag>" or "stream"
    frames: int
    duration_s: float  # on the capture time base
    time_scale: float  # capture_fps / stream_fps (8 for a 240 fps capture saved at 30 fps)
    rotation: int = 0
    warnings: list = field(default_factory=list)
    variable_rate: bool = False  # container timestamps are uneven; frames are still timed by capture_fps

    def to_dict(self):
        return asdict(self)


def probe(path):
    """ffprobe JSON for the file: format and streams."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                            str(path)], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    except subprocess.TimeoutExpired:
        raise IngestError(f"ffprobe timed out on {path}")
    if r.returncode != 0:
        raise IngestError(f"ffprobe could not read {path}: {r.stderr.strip()}")
    return json.loads(r.stdout)


def _rate(s):
    try:
        f = Fraction(s)
        return float(f) if f > 0 else 0.0
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.0


def _capture_tag(meta, video):
    """(fps, tag name) from any capture-fps style tag in the format or video stream tags."""
    for tags in (meta.get("format", {}).get("tags", {}), video.get("tags", {})):
        for k, v in tags.items():
            if CAPTURE_TAG.search(k):
                fps = _rate(str(v).strip())
                if fps > 0:
                    return fps, k
    return None, None


def _rotation(video):
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(sd["rotation"])
    try:
        return int(video.get("tags", {}).get("rotate", 0))
    except ValueError:
        return 0


def clip_info(path, capture_fps=None, min_fps=MIN_CAPTURE_FPS):
    """Read metadata and settle the capture frame rate. Raises IngestError if the clip is unusable."""
    meta = probe(path)
    videos = [s for s in meta.get("streams", []) if s.get("codec_type") == "video"]
    if not videos:
        raise IngestError(f"{path} has no video stream")
    v = videos[0]
    # avg_frame_rate (frames over duration), not r_frame_rate: a phone's variable-rate file can report r_frame_rate
    # 240 while most of it plays at 30
    avg_fps, r_fps = _rate(v.get("avg_frame_rate")), _rate(v.get("r_frame_rate"))
    stream_fps = avg_fps or r_fps
    duration = float(v.get("duration") or meta.get("format", {}).get("duration") or 0)
    frames = int(v.get("nb_frames") or 0) or count_frames(path)
    if not frames and duration and stream_fps:
        frames = int(round(duration * stream_fps))
    if not stream_fps and frames and duration:
        stream_fps = frames / duration
    warnings = []
    variable = bool(avg_fps and r_fps and abs(r_fps - avg_fps) > 0.05 * avg_fps)

    tag_fps, tag = _capture_tag(meta, v)
    if capture_fps:
        fps, source = float(capture_fps), "override"
        if tag_fps and abs(tag_fps - fps) > 1:
            warnings.append(f"override {fps:g} fps differs from metadata {tag} = {tag_fps:g} fps")
        elif not tag_fps and stream_fps >= min_fps and abs(stream_fps - fps) > 1:
            warnings.append(f"you entered {fps:g} fps but the file itself plays at {stream_fps:.4g} fps; "
                            f"every frame is taken as 1/{fps:g} s")
    elif tag_fps:
        fps, source = tag_fps, f"metadata:{tag}"
        if tag_fps < stream_fps - 1:
            warnings.append(f"capture fps {tag_fps:g} is below the stream rate {stream_fps:g}")
    else:
        fps, source = stream_fps, "stream"
        if stream_fps <= 60:
            warnings.append(f"stream plays at {stream_fps:g} fps with no capture-fps metadata; "
                            "if this is slow motion, enter the capture rate")
    if fps < min_fps:
        if source == "override":
            msg = f"You entered a capture rate of {fps:g} fps; measuring needs at least {min_fps:g} fps."
        elif source.startswith("metadata:"):
            msg = (f"The file's capture-rate tag ({tag}) says {fps:g} fps; measuring needs at least {min_fps:g} fps. "
                   "If the tag is wrong, enter the capture rate.")
        else:
            msg = (f"This file plays at {stream_fps:.4g} fps and has no capture-rate tag. Phones often save "
                   f"slow motion as a slowed {round(stream_fps)} fps file that still holds every frame; enter the "
                   f"rate it was captured at (for example 240). Measuring needs at least {min_fps:g} fps.")
        raise IngestError(msg, field="capture_fps")
    if variable:
        warnings.append(f"variable frame rate (timestamps between {avg_fps:.4g} and {r_fps:.4g} fps): frames are "
                        f"timed by the capture rate, one every 1/{fps:g} s, not by the file's timestamps")

    codec = v.get("codec_name", "")
    container = meta.get("format", {}).get("format_name", "")
    if codec not in SAM_CODECS:
        warnings.append(f"codec {codec} is not H.264 or H.265; SAM 3.1 needs one of those")
    if not SAM_CONTAINERS & set(container.split(",")):
        warnings.append(f"container {container} is not MP4 or MOV; SAM 3.1 needs one of those")
    return ClipInfo(path=str(path), width=int(v.get("width", 0)), height=int(v.get("height", 0)), codec=codec,
                    container=container, stream_fps=round(stream_fps, 4), capture_fps=round(fps, 4), fps_source=source,
                    frames=frames, duration_s=round(frames / fps, 4) if fps else 0.0,
                    time_scale=round(fps / stream_fps, 4) if stream_fps else 1.0, rotation=_rotation(v),
                    warnings=warnings, variable_rate=variable)


def count_frames(path):
    """Video packets in the file (one per frame), for containers that do not state a frame count. 0 if unknown."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                            "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
        return int(r.stdout.strip().split(",")[0] or 0) if r.returncode == 0 else 0
    except (subprocess.TimeoutExpired, ValueError):
        return 0


def read_gray(path, start=0, count=None, scale=1.0, blur=None):
    """Yield float32 grayscale frames [start, start + count), optionally downscaled and blurred."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IngestError(f"cannot open {path}")
    try:
        i = 0
        while count is None or i < start + count:
            ok, f = cap.read()
            if not ok:
                break
            if i >= start:
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                if scale != 1.0:
                    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                g = g.astype(np.float32)
                if blur:
                    g = cv2.GaussianBlur(g, (0, 0), blur)
                yield g
            i += 1
    finally:
        cap.release()


# ---------------------------------------------------------------- synthesised-frame check

def _tile_means(img, grid):
    h, w = img.shape
    th, tw = h // grid, w // grid
    return img[:th * grid, :tw * grid].reshape(grid, th, grid, tw).mean(axis=(1, 3))


def _expand(t, shape):
    """Tile values back to a full image (nearest), padded to shape."""
    grid = t.shape[0]
    th, tw = shape[0] // grid, shape[1] // grid
    out = np.empty(shape, np.float32)
    out[:] = t.mean()
    out[:th * grid, :tw * grid] = np.kron(t, np.ones((th, tw), np.float32))
    return out


def frame_energies(path, max_frames=1440, scale=0.5, grid=6):
    """Per-tile temporal statistics over the first max_frames frames.

    Each tile is gain-normalised first (divided by its mean brightness), as the tracker does, so LED flicker
    does not count as change.
    Returns dict with e1 [tiles] (mean first-difference energy), e2 [tiles] (mean second-difference energy,
    f[i] - (f[i-1] + f[i+1]) / 2), brightness [tiles], global_diff [n-1] (per-frame mean first-difference energy).
    """
    win, gdiff = [], []
    e1 = e2 = bright = None
    n1 = n2 = 0
    for g in read_gray(path, count=max_frames, scale=scale):
        m = _tile_means(g, grid)
        bright = m if bright is None else bright + m
        g = g / np.maximum(_expand(m, g.shape), 1.0)
        win.append(g)
        if len(win) >= 2:
            d2 = (win[-1] - win[-2]) ** 2
            gdiff.append(float(d2.mean()))
            t = _tile_means(d2, grid).ravel()
            e1 = t if e1 is None else e1 + t
            n1 += 1
        if len(win) == 3:
            r = win[1] - 0.5 * (win[0] + win[2])
            t = _tile_means(r * r, grid).ravel()
            e2 = t if e2 is None else e2 + t
            n2 += 1
            win.pop(0)
    if n2 < 16:
        raise IngestError(f"{path}: only {n2 + 2} frames decoded")
    return {"e1": e1 / n1, "e2": e2 / n2, "brightness": (bright / (n1 + 1)).ravel(), "global_diff": np.array(gdiff)}


def interpolation_check(path, threshold=0.70, min_noise=0.3, max_frames=720):
    """Flag clips whose in-between frames look synthesised.

    Real frames carry independent sensor noise from frame to frame. For white temporal noise the ratio of
    second-difference to first-difference energy is 0.75. An in-between frame made by blending or by
    motion-compensated interpolation lies close to the average of its neighbours, so its second difference
    collapses and the ratio drops (about 0.6 for 60 to 240 fps on the twin, versus about 0.8 for real frames).
    The median over image tiles is used, so a few tiles with real smooth motion (a drum, fan blades) do not
    decide it. If the clip carries almost no temporal noise (an encoder can freeze it in static areas),
    the test cannot tell and says so instead of flagging.

    Known limit: a phone's temporal noise reduction also lowers the ratio, so a flag means "check", and the
    fallback is to film at a mode that is certainly native (1080p 120 fps on the Redmi).
    """
    st = frame_energies(path, max_frames=max_frames)
    ratio = st["e2"] / np.maximum(st["e1"], 1e-12)
    r = float(np.median(ratio))
    noise = float(np.median(np.sqrt(st["e1"] / 2) * st["brightness"]))  # temporal noise, grey levels
    conclusive = noise >= min_noise
    flagged = bool(conclusive and r < threshold)
    status = "synthesised" if flagged else ("ok" if conclusive else "inconclusive")
    return {"flagged": flagged, "status": status, "whiteness": round(r, 3), "threshold": threshold,
            "temporal_noise_grey": round(noise, 3), "min_noise_grey": min_noise,
            "tile_whiteness_p10_p90": [round(float(v), 3) for v in np.percentile(ratio, [10, 90])],
            "frames_checked": min(max_frames, len(st["global_diff"]) + 1)}


# ---------------------------------------------------------------- steady window

def steady_window(global_diff, fps, min_s=4.0, max_s=10.0, step_s=0.25):
    """(start, count) of the steadiest window: least variation in global difference energy.

    Uses the whole clip when it is shorter than max_s; short clips get a warning from ingest().
    """
    n = len(global_diff) + 1
    L = int(round(min(max_s, n / fps) * fps))
    if L >= n:
        return 0, n
    e = np.asarray(global_diff, np.float64)
    step = max(1, int(step_s * fps))
    best, best_s = None, 0
    for s in range(0, n - L + 1, step):
        seg = e[s:s + L - 1]
        cost = seg.std() / (seg.mean() + 1e-9) + 0.5 * np.percentile(seg, 99) / (np.median(seg) + 1e-9)
        if best is None or cost < best:
            best, best_s = cost, s
    return best_s, L


@dataclass
class Ingested:
    info: ClipInfo
    interpolation: dict
    window: tuple  # (start frame, frame count)

    @property
    def fps(self):
        return self.info.capture_fps

    def to_dict(self):
        s, c = self.window
        return {"clip": self.info.to_dict(), "interpolation": self.interpolation,
                "window": {"start_frame": s, "frames": c, "start_s": round(s / self.fps, 4),
                           "seconds": round(c / self.fps, 4)}}


def ingest(path, capture_fps=None, min_fps=MIN_CAPTURE_FPS, min_s=4.0, max_s=10.0, check_frames=720):
    """Metadata, capture fps, synthesised-frame check and steady window for one clip."""
    info = clip_info(path, capture_fps=capture_fps, min_fps=min_fps)
    check = interpolation_check(path, max_frames=check_frames)
    if check["flagged"]:
        info.warnings.append(f"in-between frames look synthesised (noise whiteness {check['whiteness']} below "
                             f"{check['threshold']}); the spectrum would be meaningless. Re-film at a native capture "
                             "rate (e.g. 1080p 120 fps).")
    elif check["status"] == "inconclusive":
        info.warnings.append("too little sensor noise to check for synthesised frames")
    if info.frames and info.frames <= int(round(max_s * info.capture_fps)):
        window = (0, info.frames)  # the whole clip fits the window: steady_window would return it unchanged
    else:
        gdiff = frame_energies(path, max_frames=None, scale=0.25, grid=1)["global_diff"]
        window = steady_window(gdiff, info.capture_fps, min_s=min_s, max_s=max_s)
    if window[1] / info.capture_fps < min_s:
        info.warnings.append(f"only {window[1] / info.capture_fps:.1f} s of video; {min_s:g} s or more is better")
    return Ingested(info, check, window)

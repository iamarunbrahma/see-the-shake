"""Magnify: layer-based motion magnification of each part, the side-by-side video, and a check of the result.

Ported from reference/spike/magnify.py (Oct 27). Kept from the spike: each part is cut out with its feathered
mask and moved by (alpha - 1) times its band-passed measured motion, over an inpainted plate; a side-by-side
video at 30 fps; a space-time slice image.
Changed:
- several parts, larger ones first; a part inside another (the door in the machine) is pasted over the outer
  layer instead of over the plate, so it never opens a hole to the wall behind;
- no full-frame stabilising warp: every pixel outside a part's zone (its mask grown by the largest shift plus
  the feather) is copied unchanged from the source frame. A strict "only inside the mask" rule would freeze the
  silhouette, and the moving outline is what makes the shake visible;
- band-pass around 1x by default (zero-phase Butterworth), alpha chosen so the largest shift is about 8 px;
- verify(): track the magnified clip (track_large here: coarse to fine, motion-compensated template; the official
  tracker in measure.py is untouched) and compare its gain with the expected 1 + (alpha - 1) |H(f)|^2. It is a
  display check: about 30% agreement is the bar, not the measurement's 20%.
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfreqz

from . import measure

DEFAULT_BAND = (0.8, 1.25)  # x running speed; geometric centre exactly 1x, where the gain is 1
FILTER_ORDER = 2  # butter order per edge (a 4th-order band-pass), applied forward and back
TARGET_SHIFT_PX = 8.0  # auto alpha: the largest shift (99th percentile) is about this
ALPHA_RANGE = (2.0, 200.0)
FEATHER_SIGMA = 1.0
MASK_GROW = 5  # px, as the tracker's part weights
PLAYBACK_FPS = 30
VIDEO_SECONDS = 3.0  # of capture time; 3 s at 240 fps plays for 24 s at 30 fps
ENCODE_WAIT_S = 300


def band_filter(fps, f1, band=DEFAULT_BAND):
    lo, hi = band[0] * f1, band[1] * f1
    hi = min(hi, 0.45 * fps)
    return butter(FILTER_ORDER, [lo, hi], btype="bandpass", fs=fps, output="sos"), (lo, hi)


def expected_gain(sos, fps, alpha, hz):
    """Gain of the magnified motion at hz: 1 + (alpha - 1) |H(hz)|^2 (the filter runs forward and back)."""
    _, h = sosfreqz(sos, worN=[hz], fs=fps)
    return float(1 + (alpha - 1) * np.abs(h[0]) ** 2)


def shifts(rel, fps, f1, alpha="auto", band=DEFAULT_BAND):
    """Band-passed motion per part and the alpha to use. Returns (alpha, {part: (alpha - 1) * bp, (n, 2)}, sos, band_hz)."""
    sos, band_hz = band_filter(fps, f1, band)
    bp = {k: sosfiltfilt(sos, v - v.mean(axis=0), axis=0) for k, v in rel.items()}
    if alpha == "auto":
        peak = max(float(np.percentile(np.hypot(v[:, 0], v[:, 1]), 99)) for v in bp.values())
        alpha = 1 + TARGET_SHIFT_PX / max(peak, 1e-6)
    alpha = float(np.clip(float(alpha), *ALPHA_RANGE))
    return alpha, {k: (alpha - 1) * v for k, v in bp.items()}, sos, band_hz


@dataclass
class Layer:
    name: str
    outer: bool
    shift: np.ndarray  # (n, 2) px applied per frame
    box: tuple  # y0, y1, x0, x1 of the zone
    zone: np.ndarray  # bool, crop: pixels this layer may change
    fm: np.ndarray  # float32, crop: feathered mask at rest


def build_layers(masks, shift_by_part, shape):
    """Layers ordered outer and larger first, each with its zone (mask grown by its largest shift and feather)."""
    H, W = shape
    order = sorted(masks, key=lambda k: -int((masks[k] > 0).sum()))
    layers = []
    for i, name in enumerate(order):
        m = masks[name] > 0
        inside_other = any(((m & (masks[o] > 0)).sum() > 0.9 * m.sum()) for o in order[:i])
        s = shift_by_part[name]
        pad = int(np.ceil(np.abs(s).max())) + 3 * int(np.ceil(FEATHER_SIGMA)) + 2
        grown = cv2.dilate(m.astype(np.uint8), np.ones((MASK_GROW, MASK_GROW), np.uint8))
        zone = cv2.dilate(grown, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0
        ys, xs = np.nonzero(zone)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        fm = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), FEATHER_SIGMA)
        layers.append(Layer(name, not inside_other, s, (y0, y1, x0, x1), zone[y0:y1, x0:x1], fm[y0:y1, x0:x1]))
    return layers


def read_bgr(path, start, count):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        i = 0
        while i < start + count:
            ok, f = cap.read()
            if not ok:
                break
            if i >= start:
                yield f
            i += 1
    finally:
        cap.release()


def make_plate(path, start, count, layers, shape, samples=60):
    """Background behind the outer parts: mean frame with each outer part's footprint inpainted."""
    step = max(1, count // samples)
    acc, k = None, 0
    for j, f in enumerate(read_bgr(path, start, count)):
        if j % step == 0:
            acc = f.astype(np.float64) if acc is None else acc + f
            k += 1
    plate = (acc / k).astype(np.uint8)
    hole = np.zeros(shape, np.uint8)
    for L in layers:
        if L.outer:
            y0, y1, x0, x1 = L.box
            hole[y0:y1, x0:x1] |= (L.fm > 0.01).astype(np.uint8)
    hole = cv2.dilate(hole, np.ones((9, 9), np.uint8))
    return cv2.inpaint(plate, hole, 9, cv2.INPAINT_TELEA).astype(np.float32)


def magnify_frame(frame, i, layers, plate):
    """Magnified copy of one BGR frame. Pixels outside every layer's zone are the input's, bit for bit."""
    out = frame.copy()
    for L in layers:
        y0, y1, x0, x1 = L.box
        cur = out[y0:y1, x0:x1].astype(np.float32)
        src = frame[y0:y1, x0:x1].astype(np.float32)
        fm = L.fm[..., None]
        base = cur * (1 - fm) + plate[y0:y1, x0:x1] * fm if L.outer else cur
        sx, sy = L.shift[i]
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        size = (x1 - x0, y1 - y0)
        lay = cv2.warpAffine(src, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        wm = cv2.warpAffine(L.fm, M, size, flags=cv2.INTER_LINEAR)[..., None]
        comp = np.clip(np.rint(base * (1 - wm) + lay * wm), 0, 255).astype(np.uint8)
        region = out[y0:y1, x0:x1]
        region[L.zone] = comp[L.zone]
    return out


class _Writer:
    """ffmpeg H.264 writer that never leaves the encoder waiting on a pipe."""

    def __init__(self, path, w, h, fps, args):
        self.path = Path(path)
        self.ff = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", *args, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(self.path)], stdin=subprocess.PIPE)

    def write(self, img):
        self.ff.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self, ok=True):
        try:
            self.ff.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.ff.wait(timeout=ENCODE_WAIT_S if ok else 5)
        except subprocess.TimeoutExpired:
            self.ff.kill()
            self.ff.wait()
            raise RuntimeError(f"ffmpeg did not finish {self.path} within {ENCODE_WAIT_S} s")
        if ok and self.ff.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {self.ff.returncode} writing {self.path}")


def _even(v):
    return int(v) - int(v) % 2


def _outline(img, masks, scale, color=(40, 200, 255)):
    for m in masks.values():
        small = cv2.resize((m > 0).astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, color, 1, cv2.LINE_AA)


def _label(img, text):
    cv2.putText(img, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)


def magnify(path, masks, rel, fps, f1, out_dir, start=0, alpha="auto", band=DEFAULT_BAND, seconds=VIDEO_SECONDS,
            write_native=True, outline=True):
    """Write side_by_side.mp4 (30 fps), magnified_native.mp4 (capture fps, for verify) and slice.png.

    rel: {part: (n, 2)} measured part-minus-background motion for frames start .. start + n.
    Returns a JSON-ready dict with alpha, band, applied shifts and file names.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_all = min(len(v) for v in rel.values())
    m = min(n_all, int(round(seconds * fps))) if seconds else n_all
    alpha, sh, sos, band_hz = shifts({k: v[:n_all] for k, v in rel.items()}, fps, f1, alpha, band)
    sh = {k: v[:m] for k, v in sh.items()}
    shape = next(iter(masks.values())).shape
    H, W = shape
    s = min(1.0, 960.0 / W)
    hw, hh = _even(W * s), _even(H * s)
    slow = fps / PLAYBACK_FPS
    if not write_native:
        return _magnify_display(path, masks, sh, alpha, band, band_hz, fps, start, m, out, s, hw, hh, slow, outline)
    layers = build_layers(masks, sh, shape)
    plate = make_plate(path, start, n_all, layers, shape)

    sbs = _Writer(out / "side_by_side.mp4", 2 * hw, hh, PLAYBACK_FPS, ["-preset", "veryfast", "-crf", "20"])
    nat = _Writer(out / "magnified_native.mp4", W, H, int(round(fps)),
                  ["-preset", "veryfast", "-tune", "grain", "-crf", "16"]) if write_native else None
    lead = layers[0]
    lm = masks[lead.name] > 0
    row = int(np.nonzero(lm.any(1))[0].mean())
    xs = np.nonzero(lm[row])[0]
    xl = int(xs.min())
    c0, c1 = max(0, xl - 60), min(W, xl + 60)
    slices = {"original": [], "magnified": []}
    ok = False
    try:
        for i, f in enumerate(read_bgr(path, start, m)):
            g = magnify_frame(f, i, layers, plate)
            if nat:
                nat.write(g)
            if i < int(0.5 * fps):
                slices["original"].append(f[row, c0:c1].copy())
                slices["magnified"].append(g[row, c0:c1].copy())
            a = cv2.resize(f, (hw, hh), interpolation=cv2.INTER_AREA)
            b = cv2.resize(g, (hw, hh), interpolation=cv2.INTER_AREA)
            if outline:
                _outline(a, masks, s)
                _outline(b, masks, s)
            _label(a, f"Original, {slow:g}x slower")
            _label(b, f"Shake magnified {alpha:.0f}x")
            sbs.write(np.hstack([a, b]))
        ok = True
    finally:
        sbs.close(ok)
        if nat:
            nat.close(ok)
    tiles = []
    for k in ("original", "magnified"):
        arr = np.array(slices[k], np.uint8)
        tiles.append(cv2.resize(arr, (arr.shape[1] * 4, arr.shape[0] * 2), interpolation=cv2.INTER_NEAREST))
    gap = np.full((tiles[0].shape[0], 16, 3), 255, np.uint8)
    cv2.imwrite(str(out / "slice.png"), np.hstack([tiles[0], gap, tiles[1]]))

    ref_index = int(np.argmin(np.max([np.hypot(v[:, 0], v[:, 1]) for v in sh.values()], axis=0)))
    return {
        "alpha": round(alpha, 3), "band_hz": [round(band_hz[0], 3), round(band_hz[1], 3)],
        "band_orders": list(band), "frames": m, "seconds": round(m / fps, 3), "playback_fps": PLAYBACK_FPS,
        "slow_factor": slow, "start_frame": start, "ref_index": ref_index,
        "zones_px": {L.name: int(L.zone.sum()) for L in layers},
        "layers": [{"part": L.name, "outer": L.outer, "max_shift_px": round(float(np.abs(L.shift).max()), 3)}
                   for L in layers],
        "slice": {"row": row, "cols": [c0, c1], "seconds": 0.5},
        "files": {"side_by_side": "side_by_side.mp4", "native": "magnified_native.mp4" if write_native else None,
                  "slice": "slice.png"},
    }


def _magnify_display(path, masks, sh, alpha, band, band_hz, fps, start, m, out, s, hw, hh, slow, outline,
                     workers=4):
    """The side-by-side video composited at display resolution: what the page shows, for a quarter of the work.

    The rendered frames are decoded once, downscaled, and reused for the plate; shifts are scaled to display px,
    so the visible wobble is the same as compositing at full resolution and shrinking afterwards. Frames are
    composited in parallel threads and written in order.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    frames = [cv2.resize(f, (hw, hh), interpolation=cv2.INTER_AREA) for f in read_bgr(path, start, m)]
    m = len(frames)
    dmasks = {k: ((cv2.resize(v, (hw, hh), interpolation=cv2.INTER_AREA) > 127) * 255).astype(np.uint8)
              for k, v in masks.items()}
    dsh = {k: v[:m] * s for k, v in sh.items()}
    layers = build_layers(dmasks, dsh, (hh, hw))
    step = max(1, m // 60)
    plate = np.mean(np.stack(frames[::step]).astype(np.float64), axis=0).astype(np.uint8)
    hole = np.zeros((hh, hw), np.uint8)
    for L in layers:
        if L.outer:
            y0, y1, x0, x1 = L.box
            hole[y0:y1, x0:x1] |= (L.fm > 0.01).astype(np.uint8)
    hole = cv2.dilate(hole, np.ones((5, 5), np.uint8))
    plate = cv2.inpaint(plate, hole, 5, cv2.INPAINT_TELEA).astype(np.float32)

    lead = layers[0]
    lm = dmasks[lead.name] > 0
    row = int(np.nonzero(lm.any(1))[0].mean())
    xl = int(np.nonzero(lm[row])[0].min())
    c0, c1 = max(0, xl - 30), min(hw, xl + 30)
    n_slice = min(m, int(0.5 * fps))

    def render(i):
        a = frames[i]
        b = magnify_frame(a, i, layers, plate)
        sl = (a[row, c0:c1].copy(), b[row, c0:c1].copy()) if i < n_slice else None
        a = a.copy()
        if outline:
            _outline(a, masks, s)
            _outline(b, masks, s)
        _label(a, f"Original, {slow:g}x slower")
        _label(b, f"Shake magnified {alpha:.0f}x")
        return np.hstack([a, b]), sl

    sbs = _Writer(out / "side_by_side.mp4", 2 * hw, hh, PLAYBACK_FPS, ["-preset", "veryfast", "-crf", "20"])
    slices = {"original": [], "magnified": []}
    ok = False
    try:
        with ThreadPoolExecutor(workers, thread_name_prefix="magnify") as ex:
            pending = deque()

            def emit(fut):
                img, sl = fut.result()
                sbs.write(img)
                if sl:
                    slices["original"].append(sl[0])
                    slices["magnified"].append(sl[1])
            for i in range(m):  # a sliding window keeps frame order and bounds the frames held in memory
                pending.append(ex.submit(render, i))
                while len(pending) > 2 * workers:
                    emit(pending.popleft())
            while pending:
                emit(pending.popleft())
        ok = True
    finally:
        sbs.close(ok)
    tiles = []
    for k in ("original", "magnified"):
        arr = np.array(slices[k], np.uint8)
        tiles.append(cv2.resize(arr, (arr.shape[1] * 8, arr.shape[0] * 2), interpolation=cv2.INTER_NEAREST))
    gap = np.full((tiles[0].shape[0], 16, 3), 255, np.uint8)
    cv2.imwrite(str(out / "slice.png"), np.hstack([tiles[0], gap, tiles[1]]))
    ref_index = int(np.argmin(np.max([np.hypot(v[:, 0], v[:, 1]) for v in sh.values()], axis=0)))
    return {
        "alpha": round(alpha, 3), "band_hz": [round(band_hz[0], 3), round(band_hz[1], 3)],
        "band_orders": list(band), "frames": m, "seconds": round(m / fps, 3), "playback_fps": PLAYBACK_FPS,
        "slow_factor": slow, "start_frame": start, "ref_index": ref_index, "render_scale": round(s, 4),
        "zones_px": {L.name: int(L.zone.sum()) for L in layers},
        "layers": [{"part": L.name, "outer": L.outer, "max_shift_px": round(float(np.abs(L.shift).max()) / s, 3)}
                   for L in layers],
        "slice": {"row": row, "cols": [c0, c1], "seconds": 0.5, "scale": round(s, 4)},
        "files": {"side_by_side": "side_by_side.mp4", "native": None, "slice": "slice.png"},
    }


# ---------------------------------------------------------------- verification tracker (display check only)

class _Region(measure.Region):
    """measure.Region with a starting estimate (the official tracker in measure.py is unchanged)."""

    def track(self, img, d0=None, iters=measure.LK_ITERS + 2):
        cur = self.crop(img)
        d = np.zeros(2) if d0 is None else np.array(d0, np.float64)
        for _ in range(iters):
            M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
            al = cv2.warpAffine(cur, M, (cur.shape[1], cur.shape[0]), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
            al *= self.ref_mean / (float(np.sum(al * self.w)) / self.wsum)
            e = self.ref - al
            d += self.Hinv @ np.array([np.sum(self.gx * e), np.sum(self.gy * e)])
        return d


class _Pyramid:
    """Coarse-to-fine tracking for motions of several px."""

    def __init__(self, weight, levels):
        self.regs = []
        w = weight.astype(np.float32)
        for _ in range(levels):
            self.regs.append(_Region(w))
            w = cv2.pyrDown(w)

    def set_ref(self, ref):
        for reg in self.regs:
            reg.set_ref(ref)
            ref = cv2.pyrDown(ref)

    def track(self, pyr):
        d = np.zeros(2)
        for lvl in reversed(range(len(self.regs))):
            d = self.regs[lvl].track(pyr[lvl], d0=d)
            if lvl:
                d = d * 2.0
        return d


def _pyr(img, levels):
    out = [img]
    for _ in range(levels - 1):
        out.append(cv2.pyrDown(out[-1]))
    return out


def track_large(path, masks, start, count, rotating, levels=4, passes=2):
    """Track a clip whose parts move several px (the magnified clip), for verify() only.

    Pass 1 tracks coarse to fine against the mean frame, which the large motion smears. The next pass rebuilds each
    region's template as the mean of the frames shifted back by pass 1's estimates (sharp, and averaged over codec
    noise like an ordinary mean) and tracks again. It uses nothing but the clip: a wrong template position only adds
    a constant offset to the series, not amplitude. rotating comes from the original clip, where magnified edges
    cannot be mistaken for rotating content.
    """
    ref, _, n, _ = measure.mean_and_motion_mask(path, start, count)
    weights, bg = measure.part_weights(masks, ref.shape, rotating)
    W = {**weights, "background": bg}
    regions = {k: _Pyramid(w, levels) for k, w in W.items()}
    for r in regions.values():
        r.set_ref(ref)
    series = {k: np.empty((n, 2)) for k in W}
    for p in range(passes):
        if p:
            acc = {}
            for i, g in enumerate(measure.read_gray(path, start=start, count=n, blur=measure.BLUR)):
                for k, r in regions.items():
                    crop = r.regs[0].crop(g)
                    M = np.float32([[1, 0, series[k][i][0]], [0, 1, series[k][i][1]]])
                    al = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                                        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                    acc[k] = al.astype(np.float64) if k not in acc else acc[k] + al
            for k, r in regions.items():
                r0 = r.regs[0]
                full = ref.copy()
                full[r0.y0:r0.y1, r0.x0:r0.x1] = (acc[k] / n).astype(np.float32)
                r.set_ref(full)
        for i, g in enumerate(measure.read_gray(path, start=start, count=n, blur=measure.BLUR)):
            pyr = _pyr(g, levels)
            for k, r in regions.items():
                series[k][i] = r.track(pyr)
    return series


def rotating_mask(path, start, count):
    """The original clip's rotating-content mask, as measure computes it."""
    return measure.mean_and_motion_mask(path, start, count)[1]


def verify(native_path, masks, rel, fps, f1, info, rotating, snr=10.0):
    """Track the magnified clip and compare its motion with the original's, order by order.

    For each part and axis, measured gain = amplitude in the magnified clip / amplitude in the original
    (same frames), at 1x and 2x, against the expected 1 + (alpha - 1) |H(f)|^2. Only components at least
    snr x the original's noise floor are judged.
    """
    m = info["frames"]
    sos, _ = band_filter(fps, f1, tuple(info["band_orders"]))
    series = track_large(native_path, masks, 0, m, rotating)
    rows = []
    for part in masks:
        mag = series[part] - series["background"]
        for ai, axis in enumerate("xy"):
            fo, Zo = measure.spectrum(rel[part][:m, ai], fps)
            fm_, Zm = measure.spectrum(mag[:, ai], fps)
            Ao, Am = np.abs(Zo), np.abs(Zm)
            band = (fo > measure.F_MIN) & (fo < fo[-1] - 1)
            floor = float(np.median(Ao[band]))
            for order in (1, 2):
                a_o, hz, _ = measure.amplitude_at(fo, Ao, order * f1)
                a_m, _, _ = measure.amplitude_at(fm_, Am, order * f1)
                exp = expected_gain(sos, fps, info["alpha"], order * f1)
                judged = a_o >= snr * floor
                gain = a_m / a_o if a_o > 0 else None
                rows.append({"part": part, "axis": axis, "order": order, "hz": round(hz, 3),
                             "orig_amp_px": round(a_o, 5), "mag_amp_px": round(a_m, 5),
                             "gain": round(gain, 3) if gain else None, "expected_gain": round(exp, 3),
                             "ratio": round(gain / exp, 3) if gain else None, "judged": bool(judged)})
    judged = [r for r in rows if r["judged"]]
    worst = max((abs(r["ratio"] - 1) for r in judged), default=None)
    return {"rows": rows, "judged": len(judged), "worst_ratio_error": round(worst, 3) if worst is not None else None}

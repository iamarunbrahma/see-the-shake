"""Measure: per-part sub-pixel vibration, its spectrum, and peaks labelled as orders of running speed.

Ported from reference/spike/measure.py (Oct 27). Kept from the spike: region Lucas-Kanade translation, 3 iterations
against the mean frame, gain normalisation (rejects LED flicker), background motion subtracted (camera shake),
quadratic detrend, Hann window, peaks above 8x the median floor.
Added: any number of parts with the background outside their union; rotating content (drum glass, fan blades)
dropped from each part's weight; texture turning on a part's surface (marks on a shaft or disc), whose change no
shift explains, dropped too; peak frequency and amplitude refined by log-parabolic interpolation; running
speed from an RPM or an RPM range; order labels; amplitudes read at every order; mm scale from one reference length.
"""
import os
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import find_peaks

from .ingest import read_gray

BLUR = 1.5
LK_ITERS = 3
PART_DILATE = 5  # px, as in the spike's body mask
BG_DILATE = 61  # px; background lies outside this dilation of the part union
BG_BORDER = 40  # px removed at the frame edges
PEAK_FACTOR = 8.0  # peaks must exceed this multiple of the median floor
F_MIN = 2.0  # Hz
MAX_VIB_PX = 1.0  # temporal change beyond what this much motion explains is treated as rotating content
TEXTURE_K = 3.0  # a part pixel whose change after the part's own shift exceeds this many times the pixel noise
TEXTURE_MIN = 0.10  # a part is re-tracked without such pixels when they cover at least this share of it
RESID_Q = 64.0  # fixed-point steps per grey level for the residual sums (integer sums: the same for any thread order)
ORDERS = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6]
WORKERS = min(4, os.cpu_count() or 1)  # frames tracked in parallel; results are identical for any count
MAINS_HZ = 50.0  # India; lights flicker at twice this


def flicker_hz(fps, mains_hz=MAINS_HZ):
    """Where light flicker (2x mains) lands in the spectrum, folded below Nyquist."""
    F = 2 * mains_hz
    return abs(F - fps * round(F / fps))


def _dilate(m, k):
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


class Region:
    """Sub-pixel translation of one weighted image region against a reference frame (spike's tracker)."""

    def __init__(self, weight, margin=12):
        ys, xs = np.nonzero(weight > 0)
        if len(ys) < 50:
            raise ValueError("region has too few pixels to track")
        self.y0, self.y1 = max(ys.min() - margin, 0), ys.max() + margin
        self.x0, self.x1 = max(xs.min() - margin, 0), xs.max() + margin
        self.w = weight[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        self.wsum = float(self.w.sum())
        self.stats = None

    def collect(self):
        """Also sum each pixel's residual after tracking (the change the region's shift does not explain)."""
        self.stats = [np.zeros(self.w.shape, np.int64), np.zeros(self.w.shape, np.int64), 0, threading.Lock()]

    def residual_std(self):
        """Per-pixel std (grey levels) over the tracked frames of the residual, on the region's crop."""
        s1, s2, n, _ = self.stats
        m1, m2 = s1 / (n * RESID_Q), s2 / (n * RESID_Q * RESID_Q)
        return np.sqrt(np.maximum(m2 - m1 * m1, 0))

    def crop(self, img):
        return img[self.y0:self.y1, self.x0:self.x1]

    def set_ref(self, ref):
        self.ref = self.crop(ref)
        gy, gx = np.gradient(self.ref)
        self.gx, self.gy = gx * self.w, gy * self.w
        Hm = np.array([[np.sum(self.gx * gx), np.sum(self.gx * gy)], [np.sum(self.gy * gx), np.sum(self.gy * gy)]])
        self.Hinv = np.linalg.inv(Hm)
        self.ref_mean = float(np.sum(self.ref * self.w)) / self.wsum

    def track(self, img):
        cur = self.crop(img)
        d = np.zeros(2)
        for _ in range(LK_ITERS):
            M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
            al = cv2.warpAffine(cur, M, (cur.shape[1], cur.shape[0]), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
            al *= self.ref_mean / (float(np.sum(al * self.w)) / self.wsum)  # cancel light flicker
            e = self.ref - al
            d += self.Hinv @ np.array([np.sum(self.gx * e), np.sum(self.gy * e)])
        if self.stats is not None:  # the last step's residual; its update is a small fraction of a pixel
            q = np.rint(e * RESID_Q).astype(np.int64)
            with self.stats[3]:
                self.stats[0] += q
                self.stats[1] += q * q
                self.stats[2] += 1
        return d


def _blur(gray_u8, blur):
    g = gray_u8.astype(np.float32)  # exactly read_gray's steps after decoding: grey, float32, Gaussian blur
    return cv2.GaussianBlur(g, (0, 0), blur) if blur else g


def frames_ahead(path, start, count, blur=BLUR, depth=16, blur_threads=3):
    """The frames read_gray(path, start, count, blur=blur) yields, in the same order, but decoded in a background
    thread and blurred in a small pool, so both overlap the work on earlier frames."""
    q, stop, done, err = queue.Queue(maxsize=depth), threading.Event(), object(), []
    pool = ThreadPoolExecutor(blur_threads, thread_name_prefix="blur")

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise OSError(f"cannot open {path}")
            i = 0
            while count is None or i < start + count:
                ok, f = cap.read()
                if not ok:
                    break
                if i >= start and not put(pool.submit(_blur, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), blur)):
                    return
                i += 1
        except BaseException as e:  # surfaced in the consumer
            err.append(e)
        finally:
            cap.release()
            put(done)

    t = threading.Thread(target=produce, name="decode", daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is done:
                break
            yield item.result()
    finally:
        stop.set()
        t.join(timeout=5)
        pool.shutdown(wait=False, cancel_futures=True)
    if err:
        raise err[0]


def _strips(shape, k):
    edges = np.linspace(0, shape[0], k + 1).astype(int)
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def mean_and_motion_mask(path, start, count):
    """Mean frame (blurred grey) and a mask of pixels whose temporal change no sub-pixel motion explains.

    Frames are gain-normalised for the variance, so flicker does not count. A pixel is 'rotating content' when its
    temporal std exceeds 4x the noise plus MAX_VIB_PX times the mean frame's gradient there.
    """
    acc = acc2 = None
    n = 0

    def add(sl, g, inv):  # one horizontal strip; per pixel, the same operations in the same frame order
        gn = g[sl] * inv
        acc[sl] += g[sl]
        acc2[sl] += gn.astype(np.float64) ** 2
        gnsum[sl] += gn

    with ThreadPoolExecutor(WORKERS, thread_name_prefix="mean") as ex:
        strips = None
        for g in frames_ahead(path, start, count):
            gm = float(g.mean())
            inv = 1.0 / gm
            if acc is None:
                gn = g * inv
                acc, acc2, gsum = g.astype(np.float64), (gn.astype(np.float64)) ** 2, gm
                gnsum = gn.astype(np.float64)
                strips = _strips(g.shape, WORKERS)
            else:
                list(ex.map(add, strips, [g] * len(strips), [inv] * len(strips)))
                gsum += gm
            n += 1
    if n < 32:
        raise ValueError(f"only {n} frames in the window")
    ref = (acc / n).astype(np.float32)
    scale = gsum / n
    std = np.sqrt(np.maximum(acc2 / n - (gnsum / n) ** 2, 0)).astype(np.float32) * scale
    gy, gx = np.gradient(ref)
    grad = np.sqrt(gx * gx + gy * gy)
    noise = float(np.median(std))
    rotating = std > 4 * noise + MAX_VIB_PX * grad
    rotating = _dilate(cv2.morphologyEx(rotating.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0, 7)
    return ref, rotating, n, noise


def part_weights(masks, shape, rotating=None, ignore=None):
    """Tracking weights: each part's mask dilated 5x5 minus rotating content, and the background.
    ignore: a region the background must leave out although no part covers it (a fan's blade disc, whose grill and
    hub move with the head); None leaves everything as it was."""
    union = np.zeros(shape, bool)
    weights = {}
    for name, m in masks.items():
        m = m > 0
        union |= m
        w = _dilate(m, PART_DILATE)
        if rotating is not None:
            kept = w & ~rotating
            if kept.sum() > 0.2 * w.sum():  # never let the exclusion swallow a part
                w = kept
        weights[name] = w
    if ignore is not None:
        union |= np.asarray(ignore) > 0
    bg = ~_dilate(union, BG_DILATE)
    bg[:BG_BORDER], bg[-BG_BORDER:], bg[:, :BG_BORDER], bg[:, -BG_BORDER:] = False, False, False, False
    if rotating is not None:
        bg &= ~rotating
    return weights, bg


def spectrum(x, fps):
    """Quadratic detrend, Hann window, single-sided amplitude spectrum. Returns f, complex Z with |Z| = amplitude."""
    x = np.asarray(x, np.float64)
    i = np.arange(len(x))
    x = x - np.polyval(np.polyfit(i, x, 2), i)
    w = np.hanning(len(x))
    Z = 2 * np.fft.rfft(x * w) / w.sum()
    f = np.fft.rfftfreq(len(x), 1 / fps)
    return f, Z


def _hann_gain(delta):
    """Hann window response at an offset of delta bins (1 at 0)."""
    delta = np.clip(abs(delta), 1e-9, 0.999)
    return float(np.sinc(delta) / (1 - delta ** 2))


def refine(f, A, k):
    """Log-parabolic interpolation around bin k: (frequency, amplitude) corrected for Hann scalloping."""
    if k <= 0 or k >= len(A) - 1 or min(A[k - 1], A[k], A[k + 1]) <= 0:
        return float(f[k]), float(A[k])
    a, b, c = np.log(A[k - 1]), np.log(A[k]), np.log(A[k + 1])
    den = a - 2 * b + c
    delta = 0.5 * (a - c) / den if den < 0 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    df = f[1] - f[0]
    return float(f[k] + delta * df), float(A[k] / _hann_gain(delta))


def pick_peaks(f, A, factor=PEAK_FACTOR, fmin=F_MIN, n=10):
    """Peaks above factor x the median floor in [fmin, Nyquist - 1 Hz]. Returns (peaks [(hz, amp, bin)], floor)."""
    band = (f > fmin) & (f < f[-1] - 1)
    floor = float(np.median(A[band]))
    df = f[1] - f[0]
    idx, _ = find_peaks(np.where(band, A, 0), height=factor * floor, distance=max(1, int(0.8 / df)))
    idx = idx[np.argsort(A[idx])[::-1][:n]]
    out = []
    for k in idx:
        hz, amp = refine(f, A, k)
        out.append((hz, amp, int(k)))
    return sorted(out), floor


def amplitude_at(f, A, hz, search_bins=1.5):
    """Refined amplitude of the largest bin within search_bins of hz (for ratios, even below the peak threshold)."""
    df = f[1] - f[0]
    lo, hi = max(1, int(np.floor((hz / df) - search_bins))), min(len(A) - 2, int(np.ceil((hz / df) + search_bins)))
    if hi < lo:
        return 0.0, hz, None
    k = lo + int(np.argmax(A[lo:hi + 1]))
    rf, ra = refine(f, A, k)
    return ra, rf, k


HALF_SWITCH = 0.8  # a peak at half the strongest frequency this strong is taken as 1x (a big 2x over its 1x)
AMBIGUOUS = 0.5  # half or double the chosen speed this strong in the range: say the running speed is ambiguous
RPM_TOL = 0.05  # an entered rpm is a nameplate value: induction motors run 2 to 4% under it
OUTSIDE_RATIO = 4.0  # a peak outside the rpm range this much stronger, and no harmonic of it, sets the range aside
HARMONIC_N = range(2, 7)  # multiples of the in-range peak that a stronger peak outside it may be (2x, blade pass)


def _is_peak(A, k, floor):
    """A real peak: above the peak threshold and a local maximum, not the edge of a slope."""
    return bool(0 < k < len(A) - 1 and A[k] > PEAK_FACTOR * floor and A[k] >= A[k - 1] and A[k] >= A[k + 1])


def _pick(f, A, sel, floor, info):
    """The strongest bin in sel, taken at half its frequency when that is nearly as strong (HALF_SWITCH), and
    marked ambiguous when half or double the result is merely strong (AMBIGUOUS). Returns (hz, amp, half_taken)."""
    df = f[1] - f[0]
    idx = np.flatnonzero(sel)
    k = int(idx[np.argmax(A[sel])])
    hz, amp = refine(f, A, k)
    lo, hi = f[idx[0]] - df, f[idx[-1]] + df
    inside = lambda h: lo <= h <= hi and F_MIN < h < f[-1] - 1  # noqa: E731
    half_taken = False
    if inside(hz / 2):
        a_h, f_h, k_h = amplitude_at(f, A, hz / 2)
        if a_h >= HALF_SWITCH * amp and _is_peak(A, k_h, floor):
            info["alternatives_hz"].append(round(hz, 3))
            hz, amp, half_taken = f_h, a_h, True
    for h in (hz / 2, hz * 2):
        if inside(h):
            a_h, f_h, k_h = amplitude_at(f, A, h)
            if a_h >= AMBIGUOUS * amp and _is_peak(A, k_h, floor) and round(f_h, 3) not in info["alternatives_hz"]:
                info["alternatives_hz"].append(round(f_h, 3))
    info["ambiguous"] = bool(info["alternatives_hz"])
    return hz, amp, half_taken


def _bin(f, A, hz):
    """The largest bin next to the bin nearest hz (a refined peak frequency sits within half a bin of its bin)."""
    k = int(np.argmin(np.abs(f - hz)))
    return max(range(max(1, k - 1), min(len(A) - 1, k + 2)), key=lambda j: A[j])


def _harmonic_of(h, hz, df):
    return any(abs(h - n * hz) <= max(1.5 * df, 0.02 * h) for n in HARMONIC_N)


def running_speed(f, power, rpm=None, rpm_range=None, detail=False, avoid=()):
    """Running speed in Hz from summed spectral power.

    - rpm entered: the strongest peak within RPM_TOL of it (a nameplate rpm is a few percent high under load).
    - rpm range (Spark's, or the user's): the strongest peak inside it. A range that holds no real peak, or that a
      much stronger peak outside it shows to be wrong (OUTSIDE_RATIO, and not a harmonic of the in-range peak), is
      set aside with a note, and the strongest peak overall is taken.
    - neither: the strongest peak overall. Frequencies in avoid (the light flicker) are never taken from outside a
      range.
    A wide range or the whole spectrum can hold both f and 2f. If the peak at half the strongest frequency is nearly
    as strong (HALF_SWITCH), that half is taken as 1x: a 2x that large over a matching 1x is far more common than a
    0.5x that large. If half or double the chosen speed is merely strong (AMBIGUOUS), the result is marked ambiguous
    so the user can be asked for the rpm. info["clear"] is False when the speed taken is no real peak (1x at the
    noise floor); info["notes"] says what was set aside and what the speed settled on. With detail=True returns
    (hz, source, info)."""
    df = f[1] - f[0]
    A = np.sqrt(power)
    band = (f > F_MIN) & (f < f[-1] - 1)
    floor = float(np.median(A[band]))
    keep = band.copy()
    for h in avoid:
        if h:
            keep &= np.abs(f - h) > 1.5 * df
    info = {"ambiguous": False, "alternatives_hz": [], "clear": True, "notes": []}
    out = None
    if rpm:
        hz0 = rpm / 60.0
        sel = band & (f >= hz0 * (1 - RPM_TOL) - df / 2) & (f <= hz0 * (1 + RPM_TOL) + df / 2)
        k = int(np.flatnonzero(sel)[np.argmax(A[sel])]) if sel.any() else None
        info["entered_rpm"] = float(rpm)
        if k is not None and _is_peak(A, k, floor):
            hz = refine(f, A, k)[0]
            out = (hz, "entered rpm, strongest peak within 5%")
            off = hz * 60 / rpm - 1
            if abs(off) >= 0.01:
                info["notes"].append(f"The running speed settled on {hz * 60:.0f} rpm, {abs(off):.1%} "
                                     f"{'under' if off < 0 else 'over'} the {rpm:.0f} rpm entered: it is the "
                                     "strongest peak within 5% (motors run a few percent under their nameplate).")
        else:
            out = (amplitude_at(f, A, hz0, search_bins=1.5)[1], "entered rpm")
    elif rpm_range:
        lo, hi = rpm_range[0] / 60.0, rpm_range[1] / 60.0
        sel = band & (f >= lo - df) & (f <= hi + df)
        if sel.any():
            trial = {"ambiguous": False, "alternatives_hz": []}
            hz, amp, half = _pick(f, A, sel, floor, trial)
            k = _bin(f, A, hz)
            others = keep & ~sel
            k_g = int(np.flatnonzero(others)[np.argmax(A[others])]) if others.any() else None
            stronger = (k_g is not None and _is_peak(A, k_g, floor) and A[k_g] >= OUTSIDE_RATIO * amp
                        and not _harmonic_of(f[k_g], hz, df))
            rng = f"{rpm_range[0]:.0f} to {rpm_range[1]:.0f} rpm"
            if not _is_peak(A, k, floor) and not half:
                info["notes"].append(f"The rpm range {rng} holds no clear vibration peak, so it was set aside and "
                                     "the strongest peak was taken as the running speed. Enter the rpm if you know it.")
            elif stronger:
                info["notes"].append(f"A peak at {f[k_g] * 60:.0f} rpm, {A[k_g] / amp:.0f} times stronger than "
                                     f"anything in the rpm range {rng}, shows the range is wrong, so it was set "
                                     "aside. Enter the rpm if you know it.")
            else:
                info.update(trial)
                out = (hz, "rpm_range, half of the strongest peak" if half else "rpm_range")
            if out is None:
                info["range_set_aside"] = list(rpm_range)
    if out is None:
        hz, amp, half = _pick(f, A, keep if keep.any() else band, floor, info)
        source = "strongest_peak, half of it" if half else "strongest_peak"
        out = (hz, source + (" (the rpm range was set aside)" if info.get("range_set_aside") else ""))
    info["clear"] = _is_peak(A, _bin(f, A, out[0]), floor)
    if not info["clear"]:
        info["notes"].append(f"No clear peak at the running speed ({out[0] * 60:.0f} rpm): 1x is at the noise "
                             "floor, so the fault pattern cannot be read. Enter the rpm, or film closer.")
    return (*out, info) if detail else out


def axis_confidence(ref, weight):
    """How well a region's edges pin down each axis: the structure tensor's information about x once y is free
    (and the other way round), relative to the better axis. A horizontal shaft has edges that fix y well and x
    hardly at all; its x then reads mostly texture and noise."""
    gy, gx = np.gradient(ref.astype(np.float64))
    w = weight.astype(np.float64)
    sxx, syy, sxy = float(np.sum(w * gx * gx)), float(np.sum(w * gy * gy)), float(np.sum(w * gx * gy))
    if sxx <= 0 or syy <= 0:
        return {"x": 0.0 if sxx <= 0 else 1.0, "y": 0.0 if syy <= 0 else 1.0}
    ix, iy = sxx - sxy * sxy / syy, syy - sxy * sxy / sxx
    top = max(ix, iy)
    return {"x": round(ix / top, 4), "y": round(iy / top, 4)}


def label_order(hz, f1, df, flicker=None):
    """'0.5x', '1x', '2x' ... when within tolerance of a half-integer order, else 'non-synchronous'.
    A peak at the light-flicker frequency is labelled 'flicker' and given no order, even if it sits on one, except
    1x and 0.5x: running speed itself is never discounted (the rules judge higher orders on the flicker by their
    strength, see fingerprint.on_flicker)."""
    o = round(2 * hz / f1) / 2 if f1 > 0 else 0.0
    on_order = o >= 0.5 and abs(hz - o * f1) <= max(1.5 * df, 0.02 * hz)
    if flicker is not None and abs(hz - flicker) <= 1.5 * df and not (on_order and o <= 1):
        return None, "flicker"
    if f1 <= 0:
        return None, "unknown"
    if on_order:
        return float(o), f"{o:g}x"
    return None, "non-synchronous"


@dataclass
class Scale:
    """Pixel to mm conversion from one reference length."""
    mm_per_px: float = None
    basis: str = "px"

    @classmethod
    def from_reference(cls, masks, part, length_mm, axis="width"):
        m = masks[part] > 0
        span = np.nonzero(m.any(0 if axis == "width" else 1))[0]
        px = span.max() - span.min() + 1
        return cls(length_mm / px, f"{part} {axis} = {length_mm:g} mm over {px} px")


def _track_frames(path, start, n, regions, series, workers):
    def one(i, g):
        for k, r in regions.items():
            series[k][i] = r.track(g)

    if workers <= 1:
        for i, g in enumerate(read_gray(path, start=start, count=n, blur=BLUR)):
            one(i, g)
    else:
        with ThreadPoolExecutor(workers, thread_name_prefix="track") as ex:
            pending = deque()
            for i, g in enumerate(frames_ahead(path, start, n)):
                pending.append(ex.submit(one, i, g))
                while len(pending) > 2 * workers:  # bound the frames held in memory
                    pending.popleft().result()
            for f in pending:
                f.result()


def surface_texture(region, noise):
    """Pixels of a part whose change over the clip its own shift does not explain: marks on a turning shaft or disc
    pass by once per revolution and read as false harmonics. Returns a mask on the region's crop."""
    return (region.residual_std() > TEXTURE_K * noise) & (region.w > 0)


def track(path, masks, start, count, exclude_rotating=True, workers=None, ignore=None):
    """Per-frame translation (px) of every part and of the background, over frames [start, start + count).

    Each frame is tracked against the fixed mean frame from zero, independently of every other frame, so frames
    run in parallel threads (workers) with results identical to the sequential loop (workers=1). A part whose
    surface texture changes as it turns (TEXTURE_MIN of it or more) is tracked again without those pixels."""
    workers = WORKERS if workers is None else workers
    ref, rotating, n, noise = mean_and_motion_mask(path, start, count)
    weights, bg = part_weights(masks, ref.shape, rotating if exclude_rotating else None, ignore)
    regions = {name: Region(w) for name, w in weights.items()}
    regions["background"] = Region(bg)
    for k, r in regions.items():
        r.set_ref(ref)
        if k in masks:
            r.collect()
    series = {k: np.empty((n, 2)) for k in regions}
    _track_frames(path, start, n, regions, series, workers)
    texture, again = {}, {}
    for name in masks:
        r = regions[name]
        bad = surface_texture(r, noise)
        kept = weights[name].copy()
        kept[r.y0:r.y1, r.x0:r.x1] &= ~bad
        texture[name] = 0
        if bad.sum() >= TEXTURE_MIN * weights[name].sum() and kept.sum() > 0.2 * weights[name].sum():
            texture[name] = int(bad.sum())
            weights[name] = kept
            again[name] = Region(kept)
            again[name].set_ref(ref)
    if again:
        redo = {k: np.empty((n, 2)) for k in again}
        _track_frames(path, start, n, again, redo, workers)
        series.update(redo)
    excluded = {name: int((_dilate(masks[name] > 0, PART_DILATE) & ~weights[name]).sum()) - texture[name]
                for name in masks}
    return series, {"ref": ref, "rotating": rotating, "weights": weights, "background": bg,
                    "pixel_noise": noise, "excluded_px": excluded, "texture_px": texture}


def analyse_axis(x, fps, f1, scale, mains_hz=MAINS_HZ):
    f, Z = spectrum(x, fps)
    A = np.abs(Z)
    df = f[1] - f[0]
    pk, floor = pick_peaks(f, A)
    k_mm = scale.mm_per_px or 1.0
    peaks = []
    for hz, amp, k in pk:
        order, label = label_order(hz, f1, df, flicker_hz(fps, mains_hz))
        peaks.append({"hz": round(hz, 3), "amp_px": round(amp, 5), "amp_mm": round(amp * k_mm, 5) if scale.mm_per_px else None,
                      "order": order, "label": label, "phase": round(float(np.angle(Z[k])), 4),
                      "snr": round(amp / floor, 1) if floor > 0 else None})
    orders = {}
    for o in ORDERS:
        amp, hz, k = amplitude_at(f, A, o * f1)
        orders[f"{o:g}x"] = {"hz": round(hz, 3), "amp_px": round(amp, 5),
                             "above_floor": bool(amp > PEAK_FACTOR * floor),
                             "phase": round(float(np.angle(Z[k])), 4) if k is not None else None}
    return {"noise_floor_px": round(floor, 6), "peaks": peaks, "orders": orders}, (f, A)


def velocity_rms(peaks_x, peaks_y, mm_per_px):
    """RMS vibration velocity from the peak tables of both axes (mm/s when scaled, else px/s)."""
    k = mm_per_px or 1.0
    s = sum((2 * np.pi * p["hz"] * p["amp_px"] * k) ** 2 / 2 for p in peaks_x + peaks_y)
    return float(np.sqrt(s))


def measure(path, masks, fps, start=0, count=None, rpm=None, rpm_range=None, scale=None, exclude_rotating=True,
            mains_hz=MAINS_HZ, workers=None, ignore=None):
    """Full measurement for one clip window. masks: {part name: HxW array, nonzero inside}.

    Returns (result dict, arrays dict). The result is JSON-ready; arrays holds series and spectra for plots.
    """
    series, aux = track(path, masks, start, count, exclude_rotating, workers, ignore)
    return analyse(series, list(masks), fps, start=start, rpm=rpm, rpm_range=rpm_range, scale=scale,
                   mains_hz=mains_hz, aux=aux)


def analyse(series, parts, fps, start=0, rpm=None, rpm_range=None, scale=None, mains_hz=MAINS_HZ, aux=None):
    """Spectra, running speed, order tables and peaks from tracked series ({part: (n, 2) px, 'background': ...}).
    aux: track()'s second return (pixel noise, excluded pixels, weights); None for series made up in tests."""
    scale = scale or Scale()
    n = len(series["background"])
    rel = {k: series[k] - series["background"] for k in parts}
    # running speed from the summed power of every part, both axes
    f = None
    power = 0
    for k in parts:
        for ai in range(2):
            f, Z = spectrum(rel[k][:, ai], fps)
            power = power + np.abs(Z) ** 2
    f1, f1_source, f1_info = running_speed(f, power, rpm=rpm, rpm_range=rpm_range, detail=True,
                                           avoid=(flicker_hz(fps, mains_hz),))
    df = f[1] - f[0]
    result = {"fps": fps, "frames": n, "start_frame": start, "seconds": round(n / fps, 4), "bin_hz": round(df, 5),
              "nyquist_hz": fps / 2, "running_speed": {"hz": round(f1, 4), "rpm": round(f1 * 60, 1), "source": f1_source,
                                                       **f1_info},
              "scale": {"mm_per_px": scale.mm_per_px, "basis": scale.basis},
              "flicker_hz": flicker_hz(fps, mains_hz),
              "pixel_noise_grey": round(aux["pixel_noise"], 4) if aux else None, "parts": {}}
    arrays = {"series": series, "rel": rel, "spectra": {}}
    for k in list(parts) + ["background"]:
        x = rel[k] if k in rel else series[k]
        axes = {}
        for ai, axis in enumerate("xy"):
            axes[axis], arrays["spectra"][(k, axis)] = analyse_axis(x[:, ai], fps, f1, scale, mains_hz)
        # every peak of both axes, artifacts and weak axes included; the rules' velocity, which leaves those out,
        # is the fingerprint's features["velocity_rms"]
        v = velocity_rms(axes["x"]["peaks"], axes["y"]["peaks"], scale.mm_per_px)
        entry = {"axes": axes, "velocity_rms_all_peaks": round(v, 3),
                 "velocity_unit": "mm/s" if scale.mm_per_px else "px/s"}
        if k in rel:
            if aux:
                entry["excluded_rotating_px"] = aux["excluded_px"][k]
                entry["excluded_texture_px"] = aux["texture_px"][k]
                entry["axis_confidence"] = axis_confidence(aux["ref"], aux["weights"][k])
            result["parts"][k] = entry
        else:
            result["background"] = entry
    # The background's amplitude at each part peak: whole-frame artifacts (the encoder's, a flicker pattern) show
    # in the background's motion too, real part motion does not. New fields only; nothing above changes.
    for k in parts:
        for axis in "xy":
            fb, Ab = arrays["spectra"][("background", axis)]
            for p in result["parts"][k]["axes"][axis]["peaks"]:
                p["bg_amp_px"] = round(amplitude_at(fb, Ab, p["hz"], search_bins=1.0)[0], 5)
    result["mains_hz"] = mains_hz
    return result, arrays


def plot_spectra(result, arrays, out_png, title="", skip=None):
    """One panel per part (x and y overlaid), peaks labelled with their orders.
    skip: {part: {(axis, hz)}} peaks not to label, such as the fingerprint's artifacts (a flicker sideband that
    happens to sit near 5x must not be drawn as a 5x on the plot Spark is sent)."""
    skip = skip or {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts = list(result["parts"])
    fig, axs = plt.subplots(len(parts), 1, figsize=(10, 3.2 * len(parts)), sharex=True, squeeze=False)
    k_mm = result["scale"]["mm_per_px"]
    unit = "mm" if k_mm else "px"
    for ax, name in zip(axs[:, 0], parts):
        for axis, col in (("x", "C0"), ("y", "C1")):
            f, A = arrays["spectra"][(name, axis)]
            ax.semilogy(f, A * (k_mm or 1), lw=0.8, color=col, label=f"{axis} (horizontal)" if axis == "x" else f"{axis} (vertical)")
            for p in result["parts"][name]["axes"][axis]["peaks"]:
                if (axis, p["hz"]) in skip.get(name, ()):
                    continue
                if p["order"] is not None or p["label"] == "flicker":
                    ax.annotate(p["label"], (p["hz"], p["amp_px"] * (k_mm or 1)), textcoords="offset points",
                                xytext=(0, 5), ha="center", fontsize=8, color=col)
        ax.set_ylabel(f"{name}\namplitude ({unit})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3, which="both")
    axs[-1, 0].set_xlabel("frequency (Hz)")
    rs = result["running_speed"]
    axs[0, 0].set_title(f"{title}  running speed {rs['hz']:.2f} Hz ({rs['rpm']:.0f} rpm)".strip())
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

"""Measure: per-part sub-pixel vibration, its spectrum, and peaks labelled as orders of running speed.

Ported from reference/spike/measure.py (Oct 27). Kept from the spike: region Lucas-Kanade translation, 3 iterations
against the mean frame, gain normalisation (rejects LED flicker), background motion subtracted (camera shake),
quadratic detrend, Hann window, peaks above 8x the median floor.
Added: any number of parts with the background outside their union; rotating content (drum glass, fan blades)
dropped from each part's weight; peak frequency and amplitude refined by log-parabolic interpolation; running
speed from an RPM or an RPM range; order labels; amplitudes read at every order; mm scale from one reference length.
"""
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
ORDERS = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6]
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
        return d


def mean_and_motion_mask(path, start, count):
    """Mean frame (blurred grey) and a mask of pixels whose temporal change no sub-pixel motion explains.

    Frames are gain-normalised for the variance, so flicker does not count. A pixel is 'rotating content' when its
    temporal std exceeds 4x the noise plus MAX_VIB_PX times the mean frame's gradient there.
    """
    acc = acc2 = None
    n = 0
    for g in read_gray(path, start=start, count=count, blur=BLUR):
        gm = float(g.mean())
        gn = g * (1.0 / gm)
        if acc is None:
            acc, acc2, gsum = g.astype(np.float64), (gn.astype(np.float64)) ** 2, gm
            gnsum = gn.astype(np.float64)
        else:
            acc += g
            acc2 += gn.astype(np.float64) ** 2
            gnsum += gn
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


def part_weights(masks, shape, rotating=None):
    """Tracking weights: each part's mask dilated 5x5 minus rotating content, and the background."""
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


def running_speed(f, power, rpm=None, rpm_range=None):
    """Running speed in Hz from summed spectral power: the user's RPM (snapped to the nearest strong bin),
    else the strongest peak inside the RPM range, else the strongest peak overall."""
    df = f[1] - f[0]
    A = np.sqrt(power)
    band = (f > F_MIN) & (f < f[-1] - 1)
    if rpm:
        hz = rpm / 60.0
        _, rf, _ = amplitude_at(f, A, hz, search_bins=1.5)
        return rf, "rpm"
    if rpm_range:
        lo, hi = rpm_range[0] / 60.0, rpm_range[1] / 60.0
        sel = band & (f >= lo - df) & (f <= hi + df)
        if sel.any():
            k = int(np.flatnonzero(sel)[np.argmax(A[sel])])
            return refine(f, A, k)[0], "rpm_range"
    k = int(np.flatnonzero(band)[np.argmax(A[band])])
    return refine(f, A, k)[0], "strongest_peak"


def label_order(hz, f1, df, flicker=None):
    """'0.5x', '1x', '2x' ... when within tolerance of a half-integer order, else 'non-synchronous'.
    A peak at the light-flicker frequency is labelled 'flicker' and given no order, even if it sits on one."""
    if flicker is not None and abs(hz - flicker) <= 1.5 * df:
        return None, "flicker"
    if f1 <= 0:
        return None, "unknown"
    o = round(2 * hz / f1) / 2
    if o >= 0.5 and abs(hz - o * f1) <= max(1.5 * df, 0.02 * hz):
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


def track(path, masks, start, count, exclude_rotating=True):
    """Per-frame translation (px) of every part and of the background, over frames [start, start + count)."""
    ref, rotating, n, noise = mean_and_motion_mask(path, start, count)
    weights, bg = part_weights(masks, ref.shape, rotating if exclude_rotating else None)
    regions = {name: Region(w) for name, w in weights.items()}
    regions["background"] = Region(bg)
    for r in regions.values():
        r.set_ref(ref)
    series = {k: np.empty((n, 2)) for k in regions}
    for i, g in enumerate(read_gray(path, start=start, count=n, blur=BLUR)):
        for k, r in regions.items():
            series[k][i] = r.track(g)
    excluded = {name: int((_dilate(masks[name] > 0, PART_DILATE) & ~weights[name]).sum()) for name in masks}
    return series, {"ref": ref, "rotating": rotating, "weights": weights, "background": bg,
                    "pixel_noise": noise, "excluded_px": excluded}


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
            mains_hz=MAINS_HZ):
    """Full measurement for one clip window. masks: {part name: HxW array, nonzero inside}.

    Returns (result dict, arrays dict). The result is JSON-ready; arrays holds series and spectra for plots.
    """
    scale = scale or Scale()
    series, aux = track(path, masks, start, count, exclude_rotating)
    n = len(series["background"])
    rel = {k: series[k] - series["background"] for k in masks}
    # running speed from the summed power of every part, both axes
    f = None
    power = 0
    for k in masks:
        for ai in range(2):
            f, Z = spectrum(rel[k][:, ai], fps)
            power = power + np.abs(Z) ** 2
    f1, f1_source = running_speed(f, power, rpm=rpm, rpm_range=rpm_range)
    df = f[1] - f[0]
    result = {"fps": fps, "frames": n, "start_frame": start, "seconds": round(n / fps, 4), "bin_hz": round(df, 5),
              "nyquist_hz": fps / 2, "running_speed": {"hz": round(f1, 4), "rpm": round(f1 * 60, 1), "source": f1_source},
              "scale": {"mm_per_px": scale.mm_per_px, "basis": scale.basis},
              "flicker_hz": flicker_hz(fps, mains_hz), "pixel_noise_grey": round(aux["pixel_noise"], 4), "parts": {}}
    arrays = {"series": series, "rel": rel, "spectra": {}}
    for k in list(masks) + ["background"]:
        x = rel[k] if k in rel else series[k]
        axes = {}
        for ai, axis in enumerate("xy"):
            axes[axis], arrays["spectra"][(k, axis)] = analyse_axis(x[:, ai], fps, f1, scale, mains_hz)
        v = velocity_rms(axes["x"]["peaks"], axes["y"]["peaks"], scale.mm_per_px)
        entry = {"axes": axes, "velocity_rms": round(v, 3), "velocity_unit": "mm/s" if scale.mm_per_px else "px/s"}
        if k in masks:
            entry["excluded_rotating_px"] = aux["excluded_px"][k]
            result["parts"][k] = entry
        else:
            result["background"] = entry
    return result, arrays


def plot_spectra(result, arrays, out_png, title=""):
    """One panel per part (x and y overlaid), peaks labelled with their orders."""
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

"""Throwaway spike: measure per-part vibration from a slow-mo video and a part mask.

usage: python measure.py <video> <body_mask.png> <exclude_mask.png|-> <out_prefix> [width_mm]
Region Lucas-Kanade translation per frame (sub-pixel), minus the background's motion (camera shake),
then an amplitude spectrum and its peaks.
"""
import json, sys
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

BLUR = 1.5


def frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield fps, cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), BLUR)
    cap.release()


class Region:
    def __init__(self, weight):
        ys, xs = np.nonzero(weight > 0)
        m = 12
        self.y0, self.y1 = max(ys.min() - m, 0), ys.max() + m
        self.x0, self.x1 = max(xs.min() - m, 0), xs.max() + m
        self.w = weight[self.y0:self.y1, self.x0:self.x1].astype(np.float32)

    def crop(self, img):
        return img[self.y0:self.y1, self.x0:self.x1]

    def set_ref(self, ref):
        self.ref = self.crop(ref)
        gy, gx = np.gradient(self.ref)
        self.gx, self.gy = gx * self.w, gy * self.w
        H = np.array([[np.sum(self.gx * gx), np.sum(self.gx * gy)], [np.sum(self.gy * gx), np.sum(self.gy * gy)]])
        self.Hinv = np.linalg.inv(H)
        self.ref_mean = np.sum(self.ref * self.w) / np.sum(self.w)

    def track(self, img):
        cur = self.crop(img)
        d = np.zeros(2)
        for _ in range(3):
            M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
            al = cv2.warpAffine(cur, M, (cur.shape[1], cur.shape[0]), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            al *= self.ref_mean / (np.sum(al * self.w) / np.sum(self.w))  # cancel light flicker
            e = self.ref - al
            d += self.Hinv @ np.array([np.sum(self.gx * e), np.sum(self.gy * e)])
        return d


def spectrum(x, fps):
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 2), np.arange(len(x)))
    w = np.hanning(len(x))
    A = 2 * np.abs(np.fft.rfft(x * w)) / w.sum()
    f = np.fft.rfftfreq(len(x), 1 / fps)
    return f, A


def peaks(f, A, n=6):
    band = (f > 2) & (f < f[-1] - 1)
    floor = np.median(A[band])
    idx, _ = find_peaks(np.where(band, A, 0), height=8 * floor, distance=max(1, int(0.8 / (f[1] - f[0]))))
    idx = idx[np.argsort(A[idx])[::-1][:n]]
    return sorted([(float(f[i]), float(A[i])) for i in idx]), float(floor)


def main(video, body_png, excl_png, prefix, width_mm=600.0):
    body = cv2.imread(body_png, 0) > 0
    excl = cv2.imread(excl_png, 0) > 0 if excl_png != "-" else np.zeros_like(body)
    excl = cv2.dilate(excl.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    body_w = (cv2.dilate(body.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~excl
    bg_w = ~(cv2.dilate(body.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0)
    bg_w[:40], bg_w[-40:], bg_w[:, :40], bg_w[:, -40:] = False, False, False, False
    xs = np.nonzero(body.any(0))[0]
    mm_per_px = width_mm / (xs.max() - xs.min() + 1)

    acc, n, fps = None, 0, 240.0
    for fps, g in frames(video):
        acc = g.astype(np.float64) if acc is None else acc + g
        n += 1
    ref = (acc / n).astype(np.float32)
    regions = {"body": Region(body_w), "background": Region(bg_w)}
    for r in regions.values():
        r.set_ref(ref)
    series = {k: [] for k in regions}
    for _, g in frames(video):
        for k, r in regions.items():
            series[k].append(r.track(g))
    series = {k: np.array(v) for k, v in series.items()}
    rel = series["body"] - series["background"]

    result = {"video": str(video), "fps": fps, "frames": n, "mm_per_px": mm_per_px, "axes": {}}
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ai, axis in enumerate("xy"):
        f, A = spectrum(rel[:, ai], fps)
        pk, floor = peaks(f, A)
        v_rms = float(np.sqrt(np.sum([(2 * np.pi * fq * a * mm_per_px) ** 2 / 2 for fq, a in pk])))
        result["axes"][axis] = {"noise_floor_px": floor, "velocity_rms_mm_s": round(v_rms, 2),
                                "peaks": [{"hz": round(fq, 2), "amp_px": round(a, 4), "amp_mm": round(a * mm_per_px, 4)} for fq, a in pk]}
        ax = axs[ai]
        ax.semilogy(f, A * mm_per_px, lw=0.8)
        for fq, a in pk:
            ax.annotate(f"{fq:.1f} Hz", (fq, a * mm_per_px), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_ylabel(f"{axis} amplitude (mm)")
        ax.set_ylim(1e-4, 2)
    axs[-1].set_xlabel("frequency (Hz)")
    axs[0].set_title(Path(video).name)
    fig.tight_layout()
    fig.savefig(f"{prefix}_spectrum.png", dpi=110)
    np.savez(f"{prefix}_series.npz", body=series["body"], background=series["background"], rel=rel, fps=fps, ref=ref)
    Path(f"{prefix}_measured.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:5], *(float(a) for a in sys.argv[5:6]))

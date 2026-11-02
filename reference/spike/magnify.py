"""Throwaway spike: layer-based motion magnification using a part mask and the measured motion.

usage: python magnify.py <video> <body_mask.png> <series.npz> <out_prefix> [alpha] [seconds]
The part is cut out with its mask, moved by alpha x its measured vibration, and pasted over a clean plate.
Writes a side-by-side MP4 (stabilised original | magnified) at 30 fps and a space-time slice PNG.
"""
import subprocess, sys
import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt


def main(video, body_png, series_npz, prefix, alpha=40.0, seconds=2.0):
    S = np.load(series_npz)
    fps = float(S["fps"])
    rel, cam = S["rel"], S["background"]
    d = sosfiltfilt(butter(4, 3.0, "highpass", fs=fps, output="sos"), rel, axis=0)
    body = cv2.imread(body_png, 0)
    layer_mask = cv2.GaussianBlur(cv2.dilate(body, np.ones((5, 5), np.uint8)), (0, 0), 1.0).astype(np.float32)[..., None] / 255

    cap = cv2.VideoCapture(video)
    acc, k = None, 0
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % 8 == 0:
            acc = f.astype(np.float64) if acc is None else acc + f
            k += 1
        i += 1
    plate = (acc / k).astype(np.uint8)
    hole = cv2.dilate(body, np.ones((45, 45), np.uint8))
    plate = cv2.inpaint(plate, hole, 9, cv2.INPAINT_TELEA).astype(np.float32)
    H, W = plate.shape[:2]

    n = int(seconds * fps)
    cap = cv2.VideoCapture(video)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H // 2}", "-r", "30", "-i", "-",
                           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", f"{prefix}_magnified.mp4"], stdin=subprocess.PIPE)
    row = int(np.nonzero(body.any(1))[0].mean())
    xs = np.nonzero(body[row])[0]
    x_left = xs.min()
    slices = {"original": [], "magnified": []}
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        Mst = np.float32([[1, 0, cam[i, 0]], [0, 1, cam[i, 1]]])
        st = cv2.warpAffine(f, Mst, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        sh = (alpha - 1) * d[i]
        M = np.float32([[1, 0, sh[0]], [0, 1, sh[1]]])
        lay = cv2.warpAffine(st, M, (W, H), flags=cv2.INTER_CUBIC)
        m = cv2.warpAffine(layer_mask, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
        mag = plate * (1 - m) + lay * m
        slices["original"].append(st[row, x_left - 60:x_left + 60].copy())
        slices["magnified"].append(mag[row, x_left - 60:x_left + 60].copy())
        a = cv2.resize(st, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        b = cv2.resize(mag, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        for img, txt in ((a, "Real, 8x slow motion"), (b, f"Shake magnified {alpha:.0f}x")):
            cv2.putText(img, txt, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3, cv2.LINE_AA)
        ff.stdin.write(np.clip(np.hstack([a, b]), 0, 255).astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    # space-time slice: one image row across the cabinet's left edge, stacked over the first 0.5 s
    m = int(0.5 * fps)
    tiles = [cv2.resize(np.array(slices[k][:m]).astype(np.uint8), (480, m * 2), interpolation=cv2.INTER_NEAREST) for k in ("original", "magnified")]
    gap = np.full((m * 2, 20, 3), 255, np.uint8)
    cv2.imwrite(f"{prefix}_slice.png", np.hstack([tiles[0], gap, tiles[1]]))
    print("wrote", f"{prefix}_magnified.mp4", f"{prefix}_slice.png")


if __name__ == "__main__":
    main(*sys.argv[1:5], *(float(a) for a in sys.argv[5:7]))

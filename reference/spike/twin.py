"""Throwaway spike: render a 2D 'digital twin' washing machine at 240fps with an injected vibration fault.

usage: python twin.py <fault> <out_dir>   fault in {healthy, imbalance, loose}
Writes <fault>.mp4, <fault>_truth.json and mask PNGs (body, door) at rest position.
"""
import json, subprocess, sys
from pathlib import Path
import cv2
import numpy as np

FPS, SECONDS = 240, 6
W, H = 1920, 1080
SS = 4  # supersampling for anti-aliased drawing
RPM = 1000.0
MACH_W, MACH_H, FEET = 500, 700, 14  # sprite size in px (machine is 600 mm wide -> 1.2 mm/px)
MM_PER_PX = 600.0 / MACH_W
X0, Y0 = (W - MACH_W) // 2, 900 - (MACH_H + FEET)  # top-left of sprite at rest
DOOR_C, DOOR_R, GLASS_R = (250, 390), 175, 148

# displacement components of the cabinet, in px: list of (harmonic order, x_amp, y_amp, phase)
FAULTS = {
    "healthy":   [(1, 0.035, 0.015, 0.0)],
    "imbalance": [(1, 0.30, 0.12, 0.0), (2, 0.02, 0.01, 0.7)],
    "loose":     [(1, 0.14, 0.10, 0.0), (2, 0.09, 0.07, 1.1), (3, 0.06, 0.05, 2.0), (0.5, 0.05, 0.03, 0.4)],
}


def rr(img, p1, p2, r, color):
    """filled rounded rectangle at supersampled scale"""
    (x1, y1), (x2, y2) = p1, p2
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def s(v):
    return int(round(v * SS))


def draw_sprite():
    """machine body without the drum contents; returns BGRA at 1x"""
    img = np.zeros((s(MACH_H + FEET), s(MACH_W), 4), np.uint8)
    for fx in (40, MACH_W - 80):  # feet
        cv2.rectangle(img, (s(fx), s(MACH_H - 5)), (s(fx + 40), s(MACH_H + FEET)), (50, 50, 55, 255), -1)
    rr(img, (0, 0), (s(MACH_W), s(MACH_H)), s(16), (226, 229, 230, 255))
    # side shading for a bit of depth
    for i in range(s(10)):
        a = 1 - i / s(10)
        img[s(16):s(MACH_H - 16), i, :3] = (img[s(16):s(MACH_H - 16), i, :3] * (1 - 0.18 * a)).astype(np.uint8)
        img[s(16):s(MACH_H - 16), -1 - i, :3] = (img[s(16):s(MACH_H - 16), -1 - i, :3] * (1 - 0.25 * a)).astype(np.uint8)
    # control panel
    cv2.line(img, (s(8), s(112)), (s(MACH_W - 8), s(112)), (170, 172, 175, 255), s(2), cv2.LINE_AA)
    rr(img, (s(24), s(26)), (s(172), s(88)), s(6), (205, 208, 210, 255))
    cv2.line(img, (s(60), s(70)), (s(136), s(70)), (150, 150, 152, 255), s(3), cv2.LINE_AA)
    rr(img, (s(228), s(36)), (s(334), s(78)), s(5), (35, 38, 40, 255))
    cv2.putText(img, "1:05", (s(246), s(68)), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * SS, (120, 230, 140, 255), s(2), cv2.LINE_AA)
    cv2.circle(img, (s(420), s(57)), s(31), (175, 178, 182, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (s(420), s(57)), s(25), (208, 211, 214, 255), -1, cv2.LINE_AA)
    cv2.line(img, (s(420), s(57)), (s(420), s(36)), (90, 90, 95, 255), s(3), cv2.LINE_AA)
    for bx in (360, 375, 390):
        cv2.circle(img, (s(bx - 170), s(98)), s(5), (150, 152, 155, 255), -1, cv2.LINE_AA)
    # door ring + handle
    cx, cy = DOOR_C
    cv2.circle(img, (s(cx), s(cy)), s(DOOR_R), (150, 154, 158, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (s(cx), s(cy)), s(DOOR_R - 8), (200, 203, 206, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (s(cx), s(cy)), s(GLASS_R + 6), (120, 124, 128, 255), -1, cv2.LINE_AA)
    rr(img, (s(cx + DOOR_R - 20), s(cy - 40)), (s(cx + DOOR_R + 6), s(cy + 40)), s(6), (140, 143, 147, 255))
    # kick panel and filter door
    cv2.line(img, (s(8), s(640)), (s(MACH_W - 8), s(640)), (175, 177, 180, 255), s(2), cv2.LINE_AA)
    rr(img, (s(36), s(652)), (s(110), s(688)), s(4), (210, 212, 215, 255))
    small = cv2.resize(img, (MACH_W, MACH_H + FEET), interpolation=cv2.INTER_AREA)
    return small


def draw_drum():
    """drum contents seen through the glass, drawn once and rotated per frame (BGR, square)"""
    n = 2 * GLASS_R
    img = np.full((s(n), s(n), 3), (70, 72, 76), np.uint8)
    c = s(GLASS_R)
    rng = np.random.default_rng(3)
    for r in range(18, GLASS_R, 16):  # perforations
        k = int(2 * np.pi * r / 16)
        for j in range(k):
            a = 2 * np.pi * j / k
            cv2.circle(img, (c + s(r * np.cos(a)), c + s(r * np.sin(a))), s(3), (40, 42, 45), -1, cv2.LINE_AA)
    for a0 in (0, 120, 240):  # lifters
        a = np.deg2rad(a0)
        p = np.array([[np.cos(a - 0.12), np.sin(a - 0.12)], [np.cos(a + 0.12), np.sin(a + 0.12)]]) * GLASS_R
        q = np.array([[np.cos(a + 0.05), np.sin(a + 0.05)], [np.cos(a - 0.05), np.sin(a - 0.05)]]) * (GLASS_R - 40)
        pts = (np.vstack([p, q]) * SS + c).astype(np.int32)
        cv2.fillPoly(img, [pts], (150, 152, 156), cv2.LINE_AA)
    for col in [(140, 70, 40), (60, 60, 170), (90, 140, 60)]:  # clothes plastered to the drum wall
        a = rng.uniform(0, 2 * np.pi)
        cv2.ellipse(img, (c + s(110 * np.cos(a)), c + s(110 * np.sin(a))), (s(55), s(28)), np.rad2deg(a) + 90, 0, 360, col, -1, cv2.LINE_AA)
    return cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)


def draw_background():
    rng = np.random.default_rng(7)
    bg = np.zeros((H, W, 3), np.float32)
    floor_y = 900
    bg[:floor_y] = (196, 212, 222)  # wall
    bg[:floor_y] += rng.normal(0, 4, (floor_y, W, 1))  # paint texture
    bg[:floor_y] = cv2.GaussianBlur(bg[:floor_y], (0, 0), 1.2)
    bg[floor_y - 18:floor_y] = (170, 180, 188)  # skirting
    bg[floor_y:] = (178, 182, 186)  # floor tiles
    bg[floor_y:] += rng.normal(0, 5, (H - floor_y, W, 1))
    for x in range(-200, W + 200, 160):
        cv2.line(bg, (x, floor_y), (int(W / 2 + (x - W / 2) * 1.6), H), (130, 134, 138), 2, cv2.LINE_AA)
    for y in (940, 1000, 1070):
        cv2.line(bg, (0, y), (W, y), (130, 134, 138), 2, cv2.LINE_AA)
    # socket board and a shelf: static texture for camera-shake reference
    cv2.rectangle(bg, (1500, 420), (1640, 520), (235, 238, 240), -1, cv2.LINE_AA)
    cv2.rectangle(bg, (1500, 420), (1640, 520), (160, 165, 170), 2, cv2.LINE_AA)
    for hx in (1545, 1595):
        cv2.circle(bg, (hx, 460), 7, (60, 60, 60), -1, cv2.LINE_AA)
    cv2.rectangle(bg, (1556, 485), (1584, 500), (60, 60, 60), -1, cv2.LINE_AA)
    cv2.rectangle(bg, (200, 300), (560, 318), (90, 120, 150), -1, cv2.LINE_AA)
    for i, bx in enumerate(range(220, 520, 50)):
        cv2.rectangle(bg, (bx, 220 + 12 * (i % 3)), (bx + 34, 300), (60 + 30 * i, 90, 160 - 15 * i), -1, cv2.LINE_AA)
    sh = np.zeros((H, W), np.float32)  # soft shadow under the machine
    cv2.ellipse(sh, (W // 2, floor_y + 6), (MACH_W // 2 + 30, 22), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    sh = cv2.GaussianBlur(sh, (0, 0), 14)
    bg *= (1 - 0.35 * sh)[..., None]
    return bg


def main(fault, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    n = FPS * SECONDS
    t = np.arange(n) / FPS
    # spin speed drifts slightly, like a real motor
    f_rot = RPM / 60 * (1 + 0.004 * np.sin(2 * np.pi * 0.2 * t))
    phase_t = np.cumsum(f_rot) / FPS  # revolutions
    comps = FAULTS[fault]
    dx = np.zeros(n)
    dy = np.zeros(n)
    for order, ax, ay, ph in comps:
        a = 2 * np.pi * order * phase_t
        dx += ax * np.sin(a + ph)
        dy += ay * np.sin(a + ph + np.pi / 2)
    rock = (0.04 if fault == "loose" else 0.01) * np.sin(2 * np.pi * phase_t)  # degrees
    # camera on a stool: picks up some floor vibration plus slow tremor
    cam_x = 0.12 * dx + np.cumsum(rng.normal(0, 0.004, n))
    cam_y = 0.12 * dy + np.cumsum(rng.normal(0, 0.004, n))
    cam_x -= cam_x.mean()
    cam_y -= cam_y.mean()

    sprite = draw_sprite().astype(np.float32)
    drum = draw_drum()
    bg = draw_background()
    noise_bank = [rng.normal(0, 1.3, (H, W, 1)).astype(np.float32) for _ in range(16)]
    gx, gy = np.meshgrid(np.arange(2 * GLASS_R), np.arange(2 * GLASS_R))
    glass_mask = (((gx - GLASS_R + 0.5) ** 2 + (gy - GLASS_R + 0.5) ** 2) <= (GLASS_R - 0.5) ** 2).astype(np.float32)[..., None]
    glare = np.zeros((2 * GLASS_R, 2 * GLASS_R), np.float32)
    cv2.ellipse(glare, (GLASS_R - 40, GLASS_R - 50), (90, 40), -35, 200, 300, 1.0, 10, cv2.LINE_AA)
    glare = cv2.GaussianBlur(glare, (0, 0), 6)[..., None]

    sh_, sw_ = sprite.shape[:2]
    pad = 8
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(out / f"{fault}.mp4")], stdin=subprocess.PIPE)
    gx0, gy0 = DOOR_C[0] - GLASS_R, DOOR_C[1] - GLASS_R
    for i in range(n):
        spr = sprite.copy()
        ang = (phase_t[i] * 360.0) % 360.0
        M = cv2.getRotationMatrix2D((GLASS_R - 0.5, GLASS_R - 0.5), -ang, 1.0)
        d = cv2.warpAffine(drum, M, (2 * GLASS_R, 2 * GLASS_R), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        d = d * 0.85 + glare * 90
        region = spr[gy0:gy0 + 2 * GLASS_R, gx0:gx0 + 2 * GLASS_R]
        region[..., :3] = region[..., :3] * (1 - glass_mask) + d * glass_mask
        # cabinet motion: sub-pixel translate + small rocking about the feet
        canvas = np.zeros((sh_ + 2 * pad, sw_ + 2 * pad, 4), np.float32)
        canvas[pad:pad + sh_, pad:pad + sw_] = spr
        M = cv2.getRotationMatrix2D((pad + sw_ / 2, pad + sh_), rock[i], 1.0)
        M[0, 2] += dx[i]
        M[1, 2] += dy[i]
        moved = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_CUBIC)
        frame = bg.copy()
        ys, xs = Y0 - pad, X0 - pad
        roi = frame[ys:ys + canvas.shape[0], xs:xs + canvas.shape[1]]
        a = np.clip(moved[..., 3:4] / 255.0, 0, 1)
        roi[:] = roi * (1 - a) + moved[..., :3] * a
        Mc = np.float32([[1, 0, cam_x[i]], [0, 1, cam_y[i]]])
        frame = cv2.warpAffine(frame, Mc, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frame *= 1 + 0.015 * np.sin(2 * np.pi * 100 * t[i])  # LED flicker from the 50 Hz mains
        frame += noise_bank[i % 16]
        ffmpeg.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()

    body = np.zeros((H, W), np.uint8)
    body[Y0:Y0 + MACH_H, X0:X0 + MACH_W] = (sprite[:MACH_H, :, 3] > 128).astype(np.uint8) * 255
    door = np.zeros((H, W), np.uint8)
    cv2.circle(door, (X0 + DOOR_C[0], Y0 + DOOR_C[1]), DOOR_R, 255, -1)
    cv2.imwrite(str(out / "mask_body.png"), body)
    cv2.imwrite(str(out / "mask_door.png"), door)
    truth = {"fault": fault, "fps": FPS, "rpm": RPM, "f_rot_hz": RPM / 60, "mm_per_px": MM_PER_PX,
             "components": [{"order": o, "hz": round(o * RPM / 60, 3), "x_amp_px": ax, "y_amp_px": ay,
                             "x_amp_mm": round(ax * MM_PER_PX, 4)} for o, ax, ay, _ in comps]}
    (out / f"{fault}_truth.json").write_text(json.dumps(truth, indent=2))
    print("wrote", out / f"{fault}.mp4")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

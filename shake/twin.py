"""Digital twin: a front-loading washing machine rendered at 240 fps with an injected vibration fault.

Ported from reference/spike/twin.py (Oct 27). Changes from the spike:
- faults: healthy, imbalance, looseness, misalignment;
- duration, seed and rpm are parameters;
- the truth file records every motion the renderer applies (cabinet translation, cabinet rocking,
  camera shake, drum speed drift, flicker, noise, encoding), and gives each part's expected translation
  per order as phasors that include the rocking, which the spike left out;
- the renderer's exact part masks are written for offline evaluation.

usage: python -m shake.twin <fault> <out_dir> [--seconds S] [--seed N] [--rpm R]
"""
import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

FPS = 240
W, H = 1920, 1080
SS = 4  # supersampling for anti-aliased drawing
MACH_W, MACH_H, FEET = 500, 700, 14  # sprite size in px; the machine is 600 mm wide, so 1.2 mm per px
MACHINE_WIDTH_MM = 600.0
MM_PER_PX = MACHINE_WIDTH_MM / MACH_W
X0, Y0 = (W - MACH_W) // 2, 900 - (MACH_H + FEET)  # top-left of the sprite at rest
DOOR_C, DOOR_R, GLASS_R = (250, 390), 175, 148
PAD = 8  # canvas padding around the sprite for sub-pixel warps

# Cabinet displacement components in px: (order, x amplitude, y amplitude, phase in rad).
# x is A sin(theta + phase); y is A sin(theta + phase + pi/2), theta = 2 pi * revolutions.
FAULTS = {
    "healthy": [(1, 0.035, 0.015, 0.0)],
    "imbalance": [(1, 0.30, 0.12, 0.0), (2, 0.02, 0.01, 0.7)],
    "looseness": [(1, 0.14, 0.10, 0.0), (2, 0.09, 0.07, 1.1), (3, 0.06, 0.05, 2.0), (0.5, 0.05, 0.03, 0.4)],
    "misalignment": [(1, 0.12, 0.05, 0.0), (2, 0.20, 0.08, 0.6), (3, 0.03, 0.015, 1.3)],
}
# Expected rule verdict for each injected fault.
EXPECTED_VERDICT = {"healthy": "none", "imbalance": "imbalance", "looseness": "looseness",
                    "misalignment": "misalignment"}
# Cabinet rocking about the feet: (amplitude in degrees, order, phase in rad). Positive is counter-clockwise on screen.
ROCKING = {"looseness": (0.04, 1, 0.0)}
ROCKING_DEFAULT = (0.01, 1, 0.0)

RPM_DRIFT = 0.004  # fractional speed wobble
RPM_DRIFT_HZ = 0.2
CAMERA_COUPLING = 0.12  # camera on a stool picks up this fraction of the cabinet's motion
CAMERA_WALK_SIGMA = 0.004  # px per frame random-walk step
FLICKER_HZ, FLICKER_DEPTH = 100.0, 0.015  # LED flicker from 50 Hz mains
NOISE_SIGMA = 1.3
NOISE_BANK = 16
CRF, PRESET = 16, "medium"

PARTS = ("washing machine", "door")


@dataclass
class TwinConfig:
    fault: str
    seconds: float = 6.0
    seed: int = 11
    rpm: float = 1000.0
    fps: int = FPS
    preset: str = PRESET  # x264 preset; tests may use a faster one
    components: list = field(default=None)

    def __post_init__(self):
        if self.fault not in FAULTS:
            raise ValueError(f"unknown fault {self.fault!r}, pick one of {sorted(FAULTS)}")
        if self.components is None:
            self.components = list(FAULTS[self.fault])


def slug(name):
    return name.replace(" ", "_")


def _rr(img, p1, p2, r, color):
    """filled rounded rectangle at supersampled scale"""
    (x1, y1), (x2, y2) = p1, p2
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def _s(v):
    return int(round(v * SS))


def draw_sprite():
    """machine body without the drum contents; BGRA at 1x"""
    s = _s
    img = np.zeros((s(MACH_H + FEET), s(MACH_W), 4), np.uint8)
    for fx in (40, MACH_W - 80):  # feet
        cv2.rectangle(img, (s(fx), s(MACH_H - 5)), (s(fx + 40), s(MACH_H + FEET)), (50, 50, 55, 255), -1)
    _rr(img, (0, 0), (s(MACH_W), s(MACH_H)), s(16), (226, 229, 230, 255))
    for i in range(s(10)):  # side shading for a bit of depth
        a = 1 - i / s(10)
        img[s(16):s(MACH_H - 16), i, :3] = (img[s(16):s(MACH_H - 16), i, :3] * (1 - 0.18 * a)).astype(np.uint8)
        img[s(16):s(MACH_H - 16), -1 - i, :3] = (img[s(16):s(MACH_H - 16), -1 - i, :3] * (1 - 0.25 * a)).astype(np.uint8)
    # control panel
    cv2.line(img, (s(8), s(112)), (s(MACH_W - 8), s(112)), (170, 172, 175, 255), s(2), cv2.LINE_AA)
    _rr(img, (s(24), s(26)), (s(172), s(88)), s(6), (205, 208, 210, 255))
    cv2.line(img, (s(60), s(70)), (s(136), s(70)), (150, 150, 152, 255), s(3), cv2.LINE_AA)
    _rr(img, (s(228), s(36)), (s(334), s(78)), s(5), (35, 38, 40, 255))
    cv2.putText(img, "1:05", (s(246), s(68)), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * SS, (120, 230, 140, 255), s(2), cv2.LINE_AA)
    cv2.circle(img, (s(420), s(57)), s(31), (175, 178, 182, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (s(420), s(57)), s(25), (208, 211, 214, 255), -1, cv2.LINE_AA)
    cv2.line(img, (s(420), s(57)), (s(420), s(36)), (90, 90, 95, 255), s(3), cv2.LINE_AA)
    for bx in (360, 375, 390):
        cv2.circle(img, (s(bx - 170), s(98)), s(5), (150, 152, 155, 255), -1, cv2.LINE_AA)
    # door ring and handle
    cx, cy = DOOR_C
    cv2.circle(img, (s(cx), s(cy)), s(DOOR_R), (150, 154, 158, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (s(cx), s(cy)), s(DOOR_R - 8), (200, 203, 206, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (s(cx), s(cy)), s(GLASS_R + 6), (120, 124, 128, 255), -1, cv2.LINE_AA)
    _rr(img, (s(cx + DOOR_R - 20), s(cy - 40)), (s(cx + DOOR_R + 6), s(cy + 40)), s(6), (140, 143, 147, 255))
    # kick panel and filter door
    cv2.line(img, (s(8), s(640)), (s(MACH_W - 8), s(640)), (175, 177, 180, 255), s(2), cv2.LINE_AA)
    _rr(img, (s(36), s(652)), (s(110), s(688)), s(4), (210, 212, 215, 255))
    return cv2.resize(img, (MACH_W, MACH_H + FEET), interpolation=cv2.INTER_AREA)


def draw_drum(seed=3):
    """drum contents seen through the glass, drawn once and rotated per frame (BGR, square)"""
    s = _s
    n = 2 * GLASS_R
    img = np.full((s(n), s(n), 3), (70, 72, 76), np.uint8)
    c = s(GLASS_R)
    rng = np.random.default_rng(seed)
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
        cv2.ellipse(img, (c + s(110 * np.cos(a)), c + s(110 * np.sin(a))), (s(55), s(28)), np.rad2deg(a) + 90, 0, 360,
                    col, -1, cv2.LINE_AA)
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
    # socket board and a shelf: static texture for the camera-shake reference
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


def part_masks(sprite=None):
    """The renderer's exact masks at rest, in frame coordinates (uint8 0/255)."""
    sprite = draw_sprite() if sprite is None else sprite
    machine = np.zeros((H, W), np.uint8)
    machine[Y0:Y0 + MACH_H + FEET, X0:X0 + MACH_W] = (sprite[..., 3] > 128).astype(np.uint8) * 255
    door = np.zeros((H, W), np.uint8)
    cv2.circle(door, (X0 + DOOR_C[0], Y0 + DOOR_C[1]), DOOR_R, 255, -1)
    glass = np.zeros((H, W), np.uint8)
    cv2.circle(glass, (X0 + DOOR_C[0], Y0 + DOOR_C[1]), GLASS_R, 255, -1)
    return {"washing machine": machine, "door": door}, glass


def pivot_px():
    """Rocking pivot in frame coordinates: bottom centre of the feet."""
    return (X0 + MACH_W / 2.0, Y0 + MACH_H + FEET)


def motion(cfg, rng):
    """Every per-frame motion the renderer applies."""
    n = int(round(cfg.fps * cfg.seconds))
    t = np.arange(n) / cfg.fps
    f_rot = cfg.rpm / 60 * (1 + RPM_DRIFT * np.sin(2 * np.pi * RPM_DRIFT_HZ * t))
    rev = np.cumsum(f_rot) / cfg.fps  # revolutions
    dx, dy = np.zeros(n), np.zeros(n)
    for order, ax, ay, ph in cfg.components:
        a = 2 * np.pi * order * rev
        dx += ax * np.sin(a + ph)
        dy += ay * np.sin(a + ph + np.pi / 2)
    r_amp, r_order, r_ph = ROCKING.get(cfg.fault, ROCKING_DEFAULT)
    rock_deg = r_amp * np.sin(2 * np.pi * r_order * rev + r_ph)
    cam_x = CAMERA_COUPLING * dx + np.cumsum(rng.normal(0, CAMERA_WALK_SIGMA, n))
    cam_y = CAMERA_COUPLING * dy + np.cumsum(rng.normal(0, CAMERA_WALK_SIGMA, n))
    cam_x -= cam_x.mean()
    cam_y -= cam_y.mean()
    gain = 1 + FLICKER_DEPTH * np.sin(2 * np.pi * FLICKER_HZ * t)
    return dict(t=t, f_rot=f_rot, rev=rev, dx=dx, dy=dy, rock_deg=rock_deg, cam_x=cam_x, cam_y=cam_y, gain=gain)


def rocking_displacement(rock_deg, x, y):
    """Displacement of frame point (x, y) for a rotation of rock_deg about the pivot (small-angle, as cv2 applies it)."""
    px, py = pivot_px()
    b = np.sin(np.deg2rad(rock_deg))
    return b * (y - py), -b * (x - px)


def part_truth(cfg, masks):
    """Expected translation per order and axis at each part's centroid, as phasors.

    Each axis signal is sum over orders of amp * sin(theta * order + phase). Translation and rocking are summed
    as phasors, so the amplitude here is what a translation tracker on that part should read.
    """
    px, py = pivot_px()
    r_amp, r_order, r_ph = ROCKING.get(cfg.fault, ROCKING_DEFAULT)
    r_rad = np.deg2rad(r_amp)
    out = {}
    for name, m in masks.items():
        ys, xs = np.nonzero(m)
        cx, cy = float(xs.mean()), float(ys.mean())
        ph = {}  # order -> [x phasor, y phasor]
        for order, ax, ay, p in cfg.components:
            v = ph.setdefault(float(order), [0j, 0j])
            v[0] += ax * np.exp(1j * p)
            v[1] += ay * np.exp(1j * (p + np.pi / 2))
        v = ph.setdefault(float(r_order), [0j, 0j])
        v[0] += r_rad * (cy - py) * np.exp(1j * r_ph)
        v[1] += -r_rad * (cx - px) * np.exp(1j * r_ph)
        comps = []
        for order in sorted(ph):
            zx, zy = ph[order]
            comps.append({"order": order, "hz": round(order * cfg.rpm / 60, 4),
                          "x_amp_px": round(abs(zx), 5), "y_amp_px": round(abs(zy), 5),
                          "x_phase": round(float(np.angle(zx)), 4), "y_phase": round(float(np.angle(zy)), 4),
                          "x_amp_mm": round(abs(zx) * MM_PER_PX, 5), "y_amp_mm": round(abs(zy) * MM_PER_PX, 5)})
        out[name] = {"mask": f"mask_{slug(name)}.png", "centroid_px": [round(cx, 2), round(cy, 2)], "components": comps}
    return out


def render(cfg, out_dir, encode=True):
    """Render the clip and write clip.mp4, truth.json, truth_series.npz and the masks. Returns the truth dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    mo = motion(cfg, rng)
    n = len(mo["t"])

    sprite_u8 = draw_sprite()
    sprite = sprite_u8.astype(np.float32)
    drum = draw_drum()
    bg = draw_background()
    # same noise on every channel, as in the spike; stored 3-channel so cv2 can add it in one pass
    noise_bank = []
    for _ in range(NOISE_BANK):
        g = rng.standard_normal((H, W), dtype=np.float32) * np.float32(NOISE_SIGMA)
        noise_bank.append(cv2.merge([g, g, g]))
    gx, gy = np.meshgrid(np.arange(2 * GLASS_R), np.arange(2 * GLASS_R))
    glass_mask = (((gx - GLASS_R + 0.5) ** 2 + (gy - GLASS_R + 0.5) ** 2) <= (GLASS_R - 0.5) ** 2)
    glass_mask = glass_mask.astype(np.float32)[..., None]
    glare = np.zeros((2 * GLASS_R, 2 * GLASS_R), np.float32)
    cv2.ellipse(glare, (GLASS_R - 40, GLASS_R - 50), (90, 40), -35, 200, 300, 1.0, 10, cv2.LINE_AA)
    glare = cv2.GaussianBlur(glare, (0, 0), 6)[..., None]

    sh_, sw_ = sprite.shape[:2]
    clip = out / "clip.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(cfg.fps), "-i", "-", "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(CRF),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clip)], stdin=subprocess.PIPE) if encode else None
    gx0, gy0 = DOOR_C[0] - GLASS_R, DOOR_C[1] - GLASS_R
    ys, xs = Y0 - PAD, X0 - PAD
    canvas = np.zeros((sh_ + 2 * PAD, sw_ + 2 * PAD, 4), np.float32)
    scene = bg.copy()  # only the machine's roi changes per frame, and it is fully rewritten each time
    roi_bg = bg[ys:ys + canvas.shape[0], xs:xs + canvas.shape[1]]
    roi = scene[ys:ys + canvas.shape[0], xs:xs + canvas.shape[1]]
    try:
        for i in range(n):
            spr = sprite.copy()
            ang = (mo["rev"][i] * 360.0) % 360.0
            M = cv2.getRotationMatrix2D((GLASS_R - 0.5, GLASS_R - 0.5), -ang, 1.0)
            d = cv2.warpAffine(drum, M, (2 * GLASS_R, 2 * GLASS_R), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT).astype(np.float32)
            d = d * 0.85 + glare * 90
            region = spr[gy0:gy0 + 2 * GLASS_R, gx0:gx0 + 2 * GLASS_R]
            region[..., :3] = region[..., :3] * (1 - glass_mask) + d * glass_mask
            # cabinet motion: sub-pixel translation plus a small rocking about the feet
            canvas[PAD:PAD + sh_, PAD:PAD + sw_] = spr
            M = cv2.getRotationMatrix2D((PAD + sw_ / 2, PAD + sh_), float(mo["rock_deg"][i]), 1.0)
            M[0, 2] += mo["dx"][i]
            M[1, 2] += mo["dy"][i]
            moved = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_CUBIC)
            a = np.clip(moved[..., 3:4] / 255.0, 0, 1)
            roi[:] = roi_bg * (1 - a) + moved[..., :3] * a
            Mc = np.float32([[1, 0, mo["cam_x"][i]], [0, 1, mo["cam_y"][i]]])
            frame = cv2.warpAffine(scene, Mc, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            frame = cv2.scaleAdd(frame, float(mo["gain"][i]), noise_bank[i % NOISE_BANK])  # flicker, then noise
            if ff:
                ff.stdin.write(cv2.convertScaleAbs(frame).tobytes())  # saturating cast to uint8
    finally:
        if ff:
            ff.stdin.close()
            ff.wait()

    masks, glass = part_masks(sprite_u8)
    for name, m in masks.items():
        cv2.imwrite(str(out / f"mask_{slug(name)}.png"), m)
    cv2.imwrite(str(out / "mask_glass.png"), glass)
    r_amp, r_order, r_ph = ROCKING.get(cfg.fault, ROCKING_DEFAULT)
    truth = {
        "fault": cfg.fault,
        "expected_verdict": EXPECTED_VERDICT[cfg.fault],
        "seed": cfg.seed,
        "fps": cfg.fps,
        "seconds": cfg.seconds,
        "frames": n,
        "size": [W, H],
        "rpm": cfg.rpm,
        "f_rot_hz": round(cfg.rpm / 60, 5),
        "mm_per_px": MM_PER_PX,
        "reference": {"part": "washing machine", "width_mm": MACHINE_WIDTH_MM, "width_px": MACH_W},
        "cabinet": {
            "translation": [{"order": o, "hz": round(o * cfg.rpm / 60, 4), "x_amp_px": ax, "y_amp_px": ay,
                             "phase": ph, "y_phase_offset": round(np.pi / 2, 6)} for o, ax, ay, ph in cfg.components],
            "rocking": {"amp_deg": r_amp, "order": r_order, "phase": r_ph,
                        "pivot_px": [round(v, 2) for v in pivot_px()], "positive": "counter-clockwise on screen"},
        },
        "drum": {"rpm": cfg.rpm, "drift_frac": RPM_DRIFT, "drift_hz": RPM_DRIFT_HZ},
        "camera": {"coupling": CAMERA_COUPLING, "walk_sigma_px_per_frame": CAMERA_WALK_SIGMA,
                   "series": "truth_series.npz:cam_x,cam_y"},
        "flicker": {"hz": FLICKER_HZ, "depth": FLICKER_DEPTH},
        "noise": {"sigma": NOISE_SIGMA, "bank": NOISE_BANK},
        "encoding": {"codec": "libx264", "crf": CRF, "preset": cfg.preset, "pix_fmt": "yuv420p"} if encode else None,
        "parts": part_truth(cfg, masks),
        "rotating_mask": "mask_glass.png",
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    np.savez(out / "truth_series.npz", **mo)
    return truth


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("fault", choices=sorted(FAULTS))
    p.add_argument("out_dir")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--rpm", type=float, default=1000.0)
    p.add_argument("--preset", default=PRESET, help="x264 preset (baseline: medium)")
    a = p.parse_args(argv)
    render(TwinConfig(a.fault, a.seconds, a.seed, a.rpm, preset=a.preset), a.out_dir)
    print("wrote", Path(a.out_dir) / "clip.mp4")


if __name__ == "__main__":
    main()

"""Digital twin of a pedestal fan, seen from a 3/4 angle, rendered at 240 fps with an injected fault.

The second virtual machine for the demo, next to the washing machine (shake.twin, whose renders and accuracy numbers
this module does not touch). Visible: the motor housing behind the guard, the guard rim and grill, three blades
spinning behind the grill (motion-blurred over a 180-degree shutter at 240 fps), the pole with its collars, and
the base on the floor. The head (housing, guard, blades) moves as one rigid body; the pole moves with a share of
the head's motion; the base stays put.

Measured parts: "fan motor housing", "fan guard rim" (the rim ring only) and "fan stand" (the pole). The blade
disc is the rotating region: it is never a part, and measurement drops it as rotating content.

Faults (head translation in px at 1920x1080; x is A sin(theta + phase), y is A sin(theta + phase + pi/2)):
- healthy: a small 1x plus a small blade-pass 3x (three blades);
- imbalance: a strong 1x wobble of the head;
- looseness: a loose head mount: 0.5x, 1x and a row of harmonics, which the pole hardly shares.

The truth file records every motion applied (head, pole, camera, speed drift, flicker, noise, encoding). Each part
is a rigid translation, so its expected motion per order is exact.

usage: python -m shake.twin_fan <fault> <out_dir> [--seconds S] [--seed N] [--rpm R] [--scale 0.5]
"""
import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import twin as washer  # background, noise, flicker and encoding are shared with the washer twin

FPS = 240
W, H = washer.W, washer.H
SS = washer.SS
RPM = 1290.0  # 21.5 Hz: clear of the 20 Hz flicker pattern at 240 fps, so 1x to 4x do not sit on flicker artifacts
BLADES = 3
YAW = 0.78  # cos of the 3/4 view: the guard's circle appears this narrow
GUARD_R = 230  # guard rim radius (fan plane, px)
RIM_W = 12
BLADE_R = 206  # blade tip radius: inside the rim with a clear gap
HUB_R = 38
HEAD_C = (930, 330)  # guard centre (frame px)
HOUSING_C = (1100, 342)
HOUSING_RX, HOUSING_RY = 74, 90
POLE_X0, POLE_X1, POLE_TOP, POLE_BOT = 1112, 1134, 424, 872  # under the motor housing, clear of the guard
BASE_C = (1123, 880)
BASE_RX, BASE_RY = 165, 34
GUARD_MM = 430.0  # the rim's outer height: a 16 inch fan
SHUTTER = 0.5  # fraction of the frame interval the shutter is open (180 degrees)
BLUR_SAMPLES = 6
POLE_SHARE = 0.45  # the pole moves with this share of the head's motion
LOOSE_POLE_SHARE = 0.15  # with a loose head mount the pole hardly shares the rattle (orders other than 1x)
CAMERA_COUPLING = 0.05  # camera on a separate stool: little of the fan's motion reaches it
# An induction fan motor holds its speed far better than a washer drum under a sloshing load. The washer's 0.4% wobble
# at 0.2 Hz would spread each harmonic of a 21.5 Hz fan into sidebands (4x keeps about 40% of its peak over 6 s).
RPM_DRIFT = 0.001
PAD = 8

FAULTS = {
    "healthy": [(1, 0.030, 0.020, 0.0), (3, 0.010, 0.008, 0.5)],
    "imbalance": [(1, 0.30, 0.16, 0.0), (3, 0.012, 0.010, 0.5)],
    "looseness": [(0.5, 0.06, 0.05, 0.3), (1, 0.15, 0.10, 0.0), (2, 0.10, 0.07, 1.0), (3, 0.07, 0.05, 2.0),
                  (4, 0.045, 0.03, 0.7)],
}
EXPECTED_VERDICT = {"healthy": "none", "imbalance": "imbalance", "looseness": "looseness"}
PARTS = ("fan motor housing", "fan guard rim", "fan stand")
MACHINE = "fan"

# head canvas: covers the guard, the housing behind it and the bracket down to the pole top
HX0 = HEAD_C[0] - int(GUARD_R * YAW) - 16
HY0 = HEAD_C[1] - GUARD_R - 16
HX1 = HOUSING_C[0] + HOUSING_RX + 16
HY1 = max(POLE_TOP + 40, HEAD_C[1] + GUARD_R + 16)
HW, HH = HX1 - HX0, HY1 - HY0
GC = (HEAD_C[0] - HX0, HEAD_C[1] - HY0)  # guard centre in head-canvas px
# pole canvas
PX0, PY0 = POLE_X0 - 20, POLE_TOP - 10
PW, PH = (POLE_X1 + 20) - PX0, (POLE_BOT + 6) - PY0


@dataclass
class FanConfig:
    fault: str
    seconds: float = 6.0
    seed: int = 11
    rpm: float = RPM
    fps: int = FPS
    preset: str = washer.PRESET
    out_scale: float = 1.0
    encoding: str = "grain"
    components: list = None

    def __post_init__(self):
        if self.fault not in FAULTS:
            raise ValueError(f"unknown fan fault {self.fault!r}, pick one of {sorted(FAULTS)}")
        if self.components is None:
            self.components = list(FAULTS[self.fault])


def _s(v):
    return int(round(v * SS))


def _ell(img, c, r, color, thickness=-1, start=0, end=360):
    """An ellipse for a circle of radius r in the fan plane, seen at the 3/4 angle."""
    cv2.ellipse(img, (_s(c[0]), _s(c[1])), (_s(r * YAW), _s(r)), 0, start, end, color, thickness if thickness < 0
                else max(1, _s(thickness)), cv2.LINE_AA)


def _down(img):
    return cv2.resize(img, (img.shape[1] // SS, img.shape[0] // SS), interpolation=cv2.INTER_AREA)


def draw_head_back():
    """Motor housing behind the guard, and the bracket down to the pole (RGBA, head canvas)."""
    img = np.zeros((HH * SS, HW * SS, 4), np.uint8)
    hc = (HOUSING_C[0] - HX0, HOUSING_C[1] - HY0)
    # neck from the guard centre to the housing
    cv2.rectangle(img, (_s(GC[0]), _s(GC[1] - 26)), (_s(hc[0]), _s(GC[1] + 26)), (70, 72, 76, 255), -1, cv2.LINE_AA)
    # bracket from the housing down to the pole top
    pts = np.array([[hc[0] - 22, hc[1] + HOUSING_RY - 12], [hc[0] + 26, hc[1] + HOUSING_RY - 12],
                    [POLE_X1 - HX0 + 7, POLE_TOP - HY0 + 34], [POLE_X0 - HX0 - 7, POLE_TOP - HY0 + 34]],
                   np.float64) * SS
    cv2.fillPoly(img, [pts.astype(np.int32)], (58, 60, 64, 255), cv2.LINE_AA)
    # housing body with shading and vent slots
    cv2.ellipse(img, (_s(hc[0]), _s(hc[1])), (_s(HOUSING_RX), _s(HOUSING_RY)), 0, 0, 360, (205, 208, 212, 255), -1,
                cv2.LINE_AA)
    cv2.ellipse(img, (_s(hc[0] + 18), _s(hc[1])), (_s(HOUSING_RX - 30), _s(HOUSING_RY - 14)), 0, 0, 360,
                (226, 229, 232, 255), -1, cv2.LINE_AA)
    for k in range(-4, 5):
        y = hc[1] + k * 15
        cv2.line(img, (_s(hc[0] + 4), _s(y)), (_s(hc[0] + 52), _s(y + 3)), (95, 98, 104, 255), _s(3), cv2.LINE_AA)
    cv2.ellipse(img, (_s(hc[0]), _s(hc[1])), (_s(HOUSING_RX), _s(HOUSING_RY)), 0, 0, 360, (120, 124, 130, 255),
                _s(3), cv2.LINE_AA)
    cv2.circle(img, (_s(hc[0] + 40), _s(hc[1] - 62)), _s(9), (40, 120, 200, 255), -1, cv2.LINE_AA)  # speed knob
    return _down(img)


def draw_head_front():
    """Guard grill, rim and hub badge, in front of the blades (RGBA, head canvas)."""
    img = np.zeros((HH * SS, HW * SS, 4), np.uint8)
    for r in (62, 100, 140, 180, 214):  # grill rings
        _ell(img, GC, r, (215, 218, 222, 230), 2)
    for k in range(28):  # grill spokes
        a = 2 * np.pi * k / 28
        p0 = (GC[0] + HUB_R * YAW * np.cos(a), GC[1] + HUB_R * np.sin(a))
        p1 = (GC[0] + (GUARD_R - RIM_W) * YAW * np.cos(a), GC[1] + (GUARD_R - RIM_W) * np.sin(a))
        cv2.line(img, (_s(p0[0]), _s(p0[1])), (_s(p1[0]), _s(p1[1])), (210, 213, 217, 200), _s(1.6), cv2.LINE_AA)
    _ell(img, GC, GUARD_R - RIM_W / 2, (236, 238, 240, 255), RIM_W)  # rim
    _ell(img, GC, GUARD_R, (150, 154, 160, 255), 2)
    _ell(img, GC, GUARD_R - RIM_W, (165, 168, 174, 255), 2)
    for k in range(4):  # rim clips
        a = np.pi / 4 + k * np.pi / 2
        c = (GC[0] + (GUARD_R - RIM_W / 2) * YAW * np.cos(a), GC[1] + (GUARD_R - RIM_W / 2) * np.sin(a))
        cv2.circle(img, (_s(c[0]), _s(c[1])), _s(5), (90, 94, 100, 255), -1, cv2.LINE_AA)
    _ell(img, GC, HUB_R, (40, 120, 200, 255))  # hub badge
    _ell(img, GC, HUB_R - 9, (235, 238, 240, 255))
    _ell(img, GC, HUB_R - 20, (40, 120, 200, 255))
    return _down(img)


def draw_blades():
    """The three blades in the fan plane (premultiplied RGBA float32, square, centred), drawn once."""
    n = 2 * (BLADE_R + 4)
    img = np.zeros((n * SS, n * SS, 4), np.uint8)
    c = n / 2
    r0, r1 = HUB_R * 0.7, BLADE_R
    shades = [(178, 132, 92), (168, 124, 86), (186, 140, 98)]
    for b in range(BLADES):
        a = 2 * np.pi * b / BLADES
        rs = np.linspace(r0, r1, 16)
        t = (rs - r0) / (r1 - r0)
        w = 0.22 + 0.38 * np.sin(np.pi * np.clip(t * 0.9 + 0.05, 0, 1)) ** 0.8
        sweep = 0.35 * t ** 1.4
        lead = np.stack([c + rs * np.cos(a + sweep + w), c + rs * np.sin(a + sweep + w)], 1)
        trail = np.stack([c + rs * np.cos(a + sweep - w * 0.8), c + rs * np.sin(a + sweep - w * 0.8)], 1)
        poly = (np.vstack([lead, trail[::-1]]) * SS).astype(np.int32)
        cv2.fillPoly(img, [poly], (*shades[b], 215), cv2.LINE_AA)
        cv2.polylines(img, [poly], True, (120, 88, 60, 235), _s(1.5), cv2.LINE_AA)
        mid = (np.stack([c + rs * np.cos(a + sweep + 0.1 * w), c + rs * np.sin(a + sweep + 0.1 * w)], 1) * SS)
        cv2.polylines(img, [mid.astype(np.int32)[3:]], False, (220, 196, 160, 200), _s(2), cv2.LINE_AA)
    cv2.circle(img, (_s(c), _s(c)), _s(HUB_R * 0.8), (70, 72, 76, 255), -1, cv2.LINE_AA)
    small = _down(img).astype(np.float32) / 255.0
    small[..., :3] *= small[..., 3:4]  # premultiply
    return small


def blades_at(blades, angles_deg):
    """Blades rotated to each angle, seen at the 3/4 angle and averaged (motion blur), in the head canvas."""
    n = blades.shape[0]
    acc = np.zeros((HH, HW, 4), np.float32)
    for ang in angles_deg:
        R = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), -float(ang), 1.0)
        R = np.vstack([R, [0, 0, 1]])
        S = np.array([[YAW, 0, GC[0] - YAW * (n / 2 - 0.5)], [0, 1, GC[1] - (n / 2 - 0.5)], [0, 0, 1]])
        M = (S @ R)[:2].astype(np.float32)
        acc += cv2.warpAffine(blades, M, (HW, HH), flags=cv2.INTER_LINEAR, borderValue=0)
    return acc / len(angles_deg)


def draw_pole():
    """Pole with height collars and a clamp knob (RGBA, pole canvas): edges in both directions for tracking."""
    img = np.zeros((PH * SS, PW * SS, 4), np.uint8)
    x0, x1 = POLE_X0 - PX0, POLE_X1 - PX0
    top, bot = POLE_TOP - PY0, POLE_BOT - PY0
    # a dark pole with a chrome highlight: strong edges against the light wall (a pale pole on a pale wall leaves
    # the static wall texture to dominate the tracked region and reads its motion low)
    cv2.rectangle(img, (_s(x0), _s(top)), (_s(x1), _s(bot)), (58, 60, 64, 255), -1)
    cv2.rectangle(img, (_s(x0 + 6), _s(top)), (_s(x0 + 10), _s(bot)), (175, 178, 182, 255), -1)
    cv2.rectangle(img, (_s(x1 - 3), _s(top)), (_s(x1), _s(bot)), (36, 38, 40, 255), -1)
    for y, h in ((top + 150, 22), (top + 330, 16)):  # collars
        cv2.rectangle(img, (_s(x0 - 6), _s(y)), (_s(x1 + 6), _s(y + h)), (200, 203, 207, 255), -1, cv2.LINE_AA)
        cv2.line(img, (_s(x0 - 6), _s(y + h / 2)), (_s(x1 + 6), _s(y + h / 2)), (110, 112, 116, 255), _s(2))
    cv2.circle(img, (_s(x1 + 12), _s(top + 161)), _s(7), (40, 42, 46, 255), -1, cv2.LINE_AA)  # clamp knob
    return _down(img)


def draw_scene():
    """Static background (the washer twin's room) with the fan's base on the floor."""
    bg = washer.draw_background()
    base = np.zeros(((2 * BASE_RY + 20) * SS, (2 * BASE_RX + 20) * SS, 4), np.uint8)
    c = (BASE_RX + 10, BASE_RY + 10)
    cv2.ellipse(base, (_s(c[0]), _s(c[1] + 6)), (_s(BASE_RX), _s(BASE_RY)), 0, 0, 360, (40, 42, 46, 255), -1,
                cv2.LINE_AA)
    cv2.ellipse(base, (_s(c[0]), _s(c[1])), (_s(BASE_RX), _s(BASE_RY)), 0, 0, 360, (62, 64, 68, 255), -1, cv2.LINE_AA)
    cv2.ellipse(base, (_s(c[0]), _s(c[1] - 4)), (_s(BASE_RX * 0.35), _s(BASE_RY * 0.4)), 0, 0, 360,
                (95, 98, 104, 255), -1, cv2.LINE_AA)
    base = _down(base).astype(np.float32)
    y0, x0 = BASE_C[1] - BASE_RY - 10, BASE_C[0] - BASE_RX - 10
    roi = bg[y0:y0 + base.shape[0], x0:x0 + base.shape[1]]
    a = base[..., 3:4] / 255.0
    roi[:] = roi * (1 - a) + base[..., :3] * a
    return bg


def _premult(rgba_u8):
    f = rgba_u8.astype(np.float32) / 255.0
    f[..., :3] *= f[..., 3:4]
    return f


def _over(src, dst):
    return src + dst * (1 - src[..., 3:4])


def part_masks(back=None, front=None, pole=None):
    """The renderer's exact part masks at rest (frame px, uint8 0/255) and the rotating blade disc."""
    back = draw_head_back() if back is None else back
    front = draw_head_front() if front is None else front
    pole = draw_pole() if pole is None else pole
    disc = np.zeros((H, W), np.uint8)
    cv2.ellipse(disc, HEAD_C, (int(round(GUARD_R * YAW)), GUARD_R), 0, 0, 360, 255, -1)
    blades = np.zeros((H, W), np.uint8)
    cv2.ellipse(blades, HEAD_C, (int(round((GUARD_R - RIM_W) * YAW)), GUARD_R - RIM_W), 0, 0, 360, 255, -1)
    rim = np.zeros((H, W), np.uint8)
    cv2.ellipse(rim, HEAD_C, (int(round(GUARD_R * YAW)), GUARD_R), 0, 0, 360, 255, -1)
    rim[blades > 0] = 0  # the ring only: never the blade disc
    head_alpha = np.zeros((H, W), np.uint8)
    head_alpha[HY0:HY1, HX0:HX1] = np.maximum(back[..., 3], front[..., 3])
    housing = np.zeros((H, W), np.uint8)
    hc = np.zeros((HH, HW), np.uint8)
    cv2.ellipse(hc, (HOUSING_C[0] - HX0, HOUSING_C[1] - HY0), (HOUSING_RX, HOUSING_RY), 0, 0, 360, 255, -1)
    housing[HY0:HY1, HX0:HX1] = hc
    housing[disc > 0] = 0  # what is visible beside the guard
    stand = np.zeros((H, W), np.uint8)
    stand[PY0:PY0 + PH, PX0:PX0 + PW] = (pole[..., 3] > 128).astype(np.uint8) * 255
    stand[head_alpha > 128] = 0  # the pole top inside the bracket is the head's
    stand[disc > 0] = 0  # and never anything seen through the blades
    return {"fan motor housing": housing, "fan guard rim": rim, "fan stand": stand}, blades


def motion(cfg, rng):
    """Every per-frame motion the renderer applies."""
    n = int(round(cfg.fps * cfg.seconds))
    t = np.arange(n) / cfg.fps
    f_rot = cfg.rpm / 60 * (1 + RPM_DRIFT * np.sin(2 * np.pi * washer.RPM_DRIFT_HZ * t))
    rev = np.cumsum(f_rot) / cfg.fps
    head, pole = np.zeros((n, 2)), np.zeros((n, 2))
    for order, ax, ay, ph, share in _components(cfg):
        a = 2 * np.pi * order * rev
        mx, my = ax * np.sin(a + ph), ay * np.sin(a + ph + np.pi / 2)
        head += np.stack([mx, my], 1)
        pole += share * np.stack([mx, my], 1)
    cam = CAMERA_COUPLING * head + np.cumsum(rng.normal(0, washer.CAMERA_WALK_SIGMA, (n, 2)), axis=0)
    cam -= cam.mean(axis=0)
    gain = 1 + washer.FLICKER_DEPTH * np.sin(2 * np.pi * washer.FLICKER_HZ * t)
    return dict(t=t, f_rot=f_rot, rev=rev, head=head, pole=pole, cam=cam, gain=gain)


def _components(cfg):
    out = []
    for order, ax, ay, ph in cfg.components:
        share = POLE_SHARE if (cfg.fault != "looseness" or order == 1) else LOOSE_POLE_SHARE
        out.append((order, ax, ay, ph, share))
    return out


def part_truth(cfg, masks, sc):
    """Expected translation per order and axis for each part (exact: every part is a rigid translation)."""
    out = {}
    for name, m in masks.items():
        ys, xs = np.nonzero(m)
        comps = []
        for order, ax, ay, ph, share in _components(cfg):
            k = sc * (share if name == "fan stand" else 1.0)
            comps.append({"order": float(order), "hz": round(order * cfg.rpm / 60, 4),
                          "x_amp_px": round(ax * k, 6), "y_amp_px": round(ay * k, 6),
                          "x_phase": round(ph, 4), "y_phase": round(ph + np.pi / 2, 4)})
        out[name] = {"mask": f"mask_{washer.slug(name)}.png", "centroid_px": [round(float(xs.mean()), 2),
                                                                            round(float(ys.mean()), 2)],
                     "components": comps, "components_centroid": comps}
    return out


def render(cfg, out_dir, encode=True):
    """Render the clip and write clip.mp4, truth.json, truth_series.npz and the masks. Returns the truth dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    mo = motion(cfg, rng)
    n = len(mo["t"])
    back_u8, front_u8, pole_u8 = draw_head_back(), draw_head_front(), draw_pole()
    back, front, pole = _premult(back_u8), _premult(front_u8), _premult(pole_u8)
    blades = draw_blades()
    bg = draw_scene()
    sc = float(cfg.out_scale)
    OW, OH = int(round(W * sc)), int(round(H * sc))
    noise_bank = []
    for _ in range(washer.NOISE_BANK):
        g = rng.standard_normal((OH, OW), dtype=np.float32) * np.float32(washer.NOISE_SIGMA)
        noise_bank.append(cv2.merge([g, g, g]))
    # the region the moving parts can touch, rewritten from the background every frame
    ry0, ry1 = min(HY0, PY0) - PAD, max(HY1, PY0 + PH) + PAD
    rx0, rx1 = min(HX0, PX0) - PAD, max(HX1, PX0 + PW) + PAD
    scene = bg.copy()
    roi_bg = bg[ry0:ry1, rx0:rx1]
    roi = scene[ry0:ry1, rx0:rx1]
    sweep = 360.0 * SHUTTER / cfg.fps  # blade sweep per frame in degrees = sweep x speed in rev/s
    clip = out / "clip.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
         "-r", str(cfg.fps), "-i", "-", "-c:v", "libx264", *washer.encoder_args(cfg.encoding, cfg.preset)[0],
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clip)], stdin=subprocess.PIPE) if encode else None

    def layer(src, x0, y0, d):
        """Premultiplied sprite moved by d (sub-pixel) into the roi."""
        M = np.float32([[1, 0, x0 - rx0 + d[0]], [0, 1, y0 - ry0 + d[1]]])
        return cv2.warpAffine(src, M, (rx1 - rx0, ry1 - ry0), flags=cv2.INTER_CUBIC, borderValue=0)

    ok = False
    try:
        for i in range(n):
            ang0 = mo["rev"][i] * 360.0
            span = sweep * mo["f_rot"][i]
            angles = ang0 + span * (np.arange(BLUR_SAMPLES) + 0.5) / BLUR_SAMPLES
            head = _over(front, _over(blades_at(blades, angles), back))
            comp = roi_bg / 255.0
            for spr, x0, y0, d in ((pole, PX0, PY0, mo["pole"][i]), (head, HX0, HY0, mo["head"][i])):
                # no clipping here: cubic overshoot at sharp edges must survive the shift, or each edge's
                # apparent position bends with the sub-pixel phase (a thin dark pole read 75% of its motion)
                L = layer(spr, x0, y0, d)
                comp = L[..., :3] + comp * (1 - L[..., 3:4])
            roi[:] = comp * 255.0
            Mc = np.float32([[1, 0, mo["cam"][i, 0]], [0, 1, mo["cam"][i, 1]]])
            frame = cv2.warpAffine(scene, Mc, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if sc != 1.0:
                frame = cv2.resize(frame, (OW, OH), interpolation=cv2.INTER_AREA)
            frame = cv2.scaleAdd(frame, float(mo["gain"][i]), noise_bank[i % washer.NOISE_BANK])
            if ff:
                ff.stdin.write(cv2.convertScaleAbs(frame).tobytes())
        ok = True
    finally:
        if ff:
            try:
                ff.stdin.close()
            except BrokenPipeError:
                pass
            try:
                ff.wait(timeout=washer.ENCODE_WAIT_S if ok else 5)
            except subprocess.TimeoutExpired:
                ff.kill()
                ff.wait()
                raise RuntimeError(f"ffmpeg did not finish encoding {clip}")
            if ok and ff.returncode != 0:
                raise RuntimeError(f"ffmpeg exited with code {ff.returncode} while encoding {clip}")

    masks, rotating = part_masks(back_u8, front_u8, pole_u8)
    if sc != 1.0:
        rs = lambda m: ((cv2.resize(m, (OW, OH), interpolation=cv2.INTER_AREA) > 127) * 255).astype(np.uint8)  # noqa
        masks, rotating = {k: rs(m) for k, m in masks.items()}, rs(rotating)
    for name, m in masks.items():
        cv2.imwrite(str(out / f"mask_{washer.slug(name)}.png"), m)
    cv2.imwrite(str(out / "mask_blades.png"), rotating)
    rim_rows = np.nonzero((masks["fan guard rim"] > 0).any(1))[0]
    rim_h = int(rim_rows.max() - rim_rows.min() + 1)
    truth = {
        "machine": MACHINE,
        "fault": cfg.fault,
        "expected_verdict": EXPECTED_VERDICT[cfg.fault],
        "seed": cfg.seed, "fps": cfg.fps, "seconds": cfg.seconds, "frames": n, "size": [OW, OH], "out_scale": sc,
        "series_units": "truth_series.npz motions are in full-resolution (1920x1080) px; multiply by out_scale",
        "rpm": cfg.rpm, "f_rot_hz": round(cfg.rpm / 60, 5), "blades": BLADES,
        "blade_pass_hz": round(BLADES * cfg.rpm / 60, 4),
        "mm_per_px": GUARD_MM / rim_h,
        "reference": {"part": "fan guard rim", "axis": "height", "height_mm": GUARD_MM, "height_px": rim_h},
        "head": {"translation": [{"order": o, "hz": round(o * cfg.rpm / 60, 4), "x_amp_px": ax, "y_amp_px": ay,
                                  "phase": ph, "y_phase_offset": round(np.pi / 2, 6)}
                                 for o, ax, ay, ph in cfg.components]},
        "pole": {"share": POLE_SHARE, "loose_share_other_orders": LOOSE_POLE_SHARE},
        "base": "static",
        "blades_render": {"shutter": SHUTTER, "blur_samples": BLUR_SAMPLES, "view_cos": YAW},
        "motor": {"rpm": cfg.rpm, "drift_frac": RPM_DRIFT, "drift_hz": washer.RPM_DRIFT_HZ},
        "camera": {"coupling": CAMERA_COUPLING, "walk_sigma_px_per_frame": washer.CAMERA_WALK_SIGMA,
                   "series": "truth_series.npz:cam"},
        "flicker": {"hz": washer.FLICKER_HZ, "depth": washer.FLICKER_DEPTH},
        "noise": {"sigma": washer.NOISE_SIGMA, "bank": washer.NOISE_BANK},
        "encoding": {"name": cfg.encoding, "codec": "libx264", **washer.encoder_args(cfg.encoding, cfg.preset)[1],
                     "pix_fmt": "yuv420p"} if encode else None,
        "parts": part_truth(cfg, masks, sc),
        "rotating_mask": "mask_blades.png",
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
    p.add_argument("--rpm", type=float, default=RPM)
    p.add_argument("--scale", type=float, default=1.0, help="output scale (0.5 gives 960x540)")
    a = p.parse_args(argv)
    render(FanConfig(a.fault, a.seconds, a.seed, a.rpm, out_scale=a.scale), a.out_dir)
    print("wrote", Path(a.out_dir) / "clip.mp4")


if __name__ == "__main__":
    main()

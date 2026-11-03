"""Fingerprint: vibration features per part and the fixed rules that give the official verdict.

Rules, first match wins, applied to the part that vibrates most:
1. none          overall velocity below the alert level
2. looseness     0.5x present, or a row of 3+ significant harmonics (2x to 6x, each at least 10% of the
                 largest running-speed component; tiny distortion products far below it do not count)
3. misalignment  2x at least half of 1x
4. imbalance     1x holds at least 60% of the vibration energy
5. unclear       anything else, and always when running speed is no real peak (1x at the noise floor, so no order
                 can be read)
When the blade count of a fan is known, blades x running speed is blade pass: it stays in the velocity (severity is
overall vibration) but is left out of every pattern feature (energy share, 2x/1x, harmonics), so it cannot push a
verdict toward imbalance, misalignment or looseness. If the velocity crosses the alert only because of blade pass,
the verdict is blade_pass: high vibration at blade pass, check the blades and the guard.
Each verdict cites the measured peaks it rests on. Thresholds are the builder's calls, tuned on the twin and
documented here; they are not ISO limits.
"""
import math

import numpy as np

ALERT_MM_S = 7.0  # RMS velocity below this is reported as no fault
ALERT_PX_S = 5.0  # fallback when there is no mm scale (resolution-dependent; flagged as such)
HALF_MIN_RATIO = 0.15  # 0.5x must reach this fraction of 1x
R21_MISALIGN = 0.5
E1_IMBALANCE = 0.6
HARMONIC_ORDERS = ["2x", "3x", "4x", "5x", "6x"]
MIN_HARMONICS = 3
HARMONIC_MIN_RATIO = 0.10  # of the largest of 1x..6x; a long, clean clip resolves products 30 dB down

# Compression and flicker artifacts (see artifact_kind). A part peak is an artifact when the background moves
# as much at that frequency (a whole-frame effect), or when flicker is present and the peak sits on the flicker
# pattern's frequencies or their sidebands with running speed. Orders below 5x, which the rules rest on, are never
# taken for flicker sidebands, and 1x is never discounted.
BG_RATIO = 0.3
SIDEBAND_TOL_BINS = 1.5
FLICKER_SAFE_ORDER = 5  # orders below this are never flicker sidebands
MAINS_DEFAULT = 50.0
# Orders that land on the flicker pattern (2x and 4x at 3000 rpm on 50 Hz mains, 3x and 5x on its sidebands) cannot
# be told from flicker at that frequency. They count only when strong: at least this share of 1x, which still lets a
# real 2x (misalignment needs 2x at half of 1x or more) through.
FLICKER_ORDER_MIN = 0.5
# An axis whose edges carry under this share of the other axis's information is left out (a horizontal shaft's x).
AXIS_WEAK = 0.05

FAULT_ORDERS = {"imbalance": {"1x"}, "misalignment": {"2x"}, "looseness": {"0.5x"} | set(HARMONIC_ORDERS)}


def flicker_pattern(result):
    """(fundamental in Hz of the flicker pattern at this frame rate, whether flicker shows in this clip).

    Light flickers at twice the mains frequency; sampled at the capture rate, the per-frame brightness repeats with
    a fundamental of gcd(fps, 2 x mains): 20 Hz for 100 Hz flicker at 240 or 120 fps."""
    fps = float(result.get("fps") or 0)
    mains = float(result.get("mains_hz") or MAINS_DEFAULT)
    g = math.gcd(int(round(fps)), int(round(2 * mains))) if fps else 0
    entries = list(result["parts"].values()) + ([result["background"]] if "background" in result else [])
    seen = any(p["label"] == "flicker" for e in entries for ax in "xy" for p in e["axes"][ax]["peaks"])
    return float(g), seen


def artifact_kind(p, result, axis, pattern):
    """'whole frame', 'flicker sideband' or None for one part peak (see BG_RATIO above)."""
    if p["label"] == "flicker":
        return "flicker"
    bin_hz, f1 = result["bin_hz"], result["running_speed"]["hz"]
    bg = p.get("bg_amp_px")
    if bg is None:  # older measurements: the background's peaks near this frequency
        near = [q["amp_px"] for q in result.get("background", {}).get("axes", {}).get(axis, {}).get("peaks", [])
                if abs(q["hz"] - p["hz"]) <= bin_hz + 1e-9]
        bg = max(near, default=0.0)
    if p.get("order") != 1 and bg >= BG_RATIO * p["amp_px"]:
        return "whole frame"
    g, seen = pattern
    if seen and g and (p.get("order") is None or p["order"] >= FLICKER_SAFE_ORDER):
        fps, tol = result["fps"], SIDEBAND_TOL_BINS * bin_hz
        for k in range(1, int((fps / 2 + f1) / g) + 2):
            for c in (k * g, k * g - f1, k * g + f1):
                c = abs(c)
                if c > fps / 2:
                    c = fps - c  # folds back below Nyquist
                if abs(p["hz"] - c) <= tol:
                    return "flicker sideband"
    return None


def on_flicker(hz, result, pattern):
    """Whether a frequency sits on the flicker pattern (multiples of its fundamental) or a sideband with 1x."""
    g, seen = pattern
    if not (seen and g):
        return False
    fps, f1 = result["fps"], result["running_speed"]["hz"]
    tol = SIDEBAND_TOL_BINS * result["bin_hz"]
    for k in range(1, int((fps / 2 + f1) / g) + 2):
        for c in (k * g, k * g - f1, k * g + f1):
            c = abs(c)
            if c > fps / 2:
                c = fps - c
            if abs(hz - c) <= tol:
                return True
    return False


def _vec(axes, key, skip=()):
    """Vector amplitude (px) of an order over both axes, and whether either axis is above the peak threshold.
    Axes listed in skip (where an artifact peak sits on this order) count as zero."""
    ox, oy = axes["x"]["orders"][key], axes["y"]["orders"][key]
    ax_ = 0.0 if "x" in skip else ox["amp_px"]
    ay_ = 0.0 if "y" in skip else oy["amp_px"]
    present = ("x" not in skip and ox["above_floor"]) or ("y" not in skip and oy["above_floor"])
    return float(np.hypot(ax_, ay_)), bool(present)


def _wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def part_features(entry, scale_mm_per_px, result=None, blades=None):
    axes = entry["axes"]
    artifacts = []
    if result is not None:
        pattern = flicker_pattern(result)
        for ax in "xy":
            for p in axes[ax]["peaks"]:
                kind = artifact_kind(p, result, ax, pattern)
                if kind:  # flicker too: an order that sits on it (6x at 100 Hz) must not count as present
                    artifacts.append({"axis": ax, "hz": p["hz"], "label": p["label"], "amp_px": p["amp_px"],
                                      "bg_amp_px": p.get("bg_amp_px"), "kind": kind})
    bin_hz = result["bin_hz"] if result is not None else 0.0
    conf = entry.get("axis_confidence") or {}
    weak = {ax for ax in "xy" if conf.get(ax, 1.0) < AXIS_WEAK}
    pattern = flicker_pattern(result) if result is not None else (0.0, False)
    amp, present, flick = {}, {}, {}
    for key in axes["x"]["orders"]:
        hz = axes["x"]["orders"][key]["hz"]
        order = float(key[:-1])
        if order >= 2 and result is not None and on_flicker(hz, result, pattern):
            flick[key] = _vec(axes, key, weak)  # judged by strength below, once 1x is known
            amp[key], present[key] = 0.0, False
            continue
        skip = {a["axis"] for a in artifacts if abs(a["hz"] - hz) <= SIDEBAND_TOL_BINS * bin_hz + 1e-9} if key != "1x" else set()
        amp[key], present[key] = _vec(axes, key, skip | weak)
    a1 = amp["1x"]
    on_flicker_orders = {}
    for key, (a_k, pres) in flick.items():
        strong = bool(pres and a1 > 0 and a_k >= FLICKER_ORDER_MIN * a1)
        on_flicker_orders[key] = "counted" if strong else "left out"
        if strong:
            amp[key], present[key] = a_k, True
    flagged = {(a["axis"], a["hz"]) for a in artifacts}
    peaks = [p for ax in "xy" for p in axes[ax]["peaks"]
             if p["label"] != "flicker" and (ax, p["hz"]) not in flagged and ax not in weak
             and not (p["order"] is not None and f"{p['order']:g}x" in flick)]
    # an order on the flicker that is strong enough counts as one peak at its vector amplitude
    f1_hz = result["running_speed"]["hz"] if result is not None else 0.0
    peaks += [{"hz": float(k_[:-1]) * f1_hz, "amp_px": amp[k_], "order": float(k_[:-1]), "label": k_}
              for k_, v in on_flicker_orders.items() if v == "counted"]
    k = scale_mm_per_px or 1.0
    vel_of = lambda ps: float(np.sqrt(sum((2 * np.pi * p["hz"] * p["amp_px"] * k) ** 2 / 2 for p in ps)))  # noqa: E731
    blade_pass = None
    if blades:
        bp_key = f"{float(blades):g}x"
        bp_peaks = [p for p in peaks if p["order"] is not None and abs(p["order"] - blades) < 1e-6]
        if bp_peaks or amp.get(bp_key):
            blade_pass = {"blades": int(blades), "order": bp_key,
                          "hz": round(bp_peaks[0]["hz"] if bp_peaks else blades * result["running_speed"]["hz"], 3),
                          "amp_px": round(amp.get(bp_key, 0.0) or max(p["amp_px"] for p in bp_peaks), 5),
                          "velocity_rms": round(vel_of(bp_peaks), 3)}
        # out of every pattern feature, kept in the velocity
        if bp_key in amp:
            amp[bp_key], present[bp_key] = 0.0, False
        pattern_peaks = [p for p in peaks if p not in bp_peaks]
    else:
        pattern_peaks = peaks
    # energy: every peak left after the artifacts, with orders in the order table (0.5x to 6x) counted once at their
    # amplitude over both axes, and other orders (3.5x, 7x and up) and non-synchronous peaks at their own amplitude
    sync = [p for p in pattern_peaks if p["order"] is not None]
    other_energy = sum(p["amp_px"] ** 2 for p in pattern_peaks
                       if p["order"] is None or f"{p['order']:g}x" not in amp)
    order_keys = {f"{p['order']:g}x" for p in sync} & set(amp)
    energy = sum(amp[k_] ** 2 for k_ in order_keys) + other_energy
    vel = vel_of(peaks)
    a_max = max(amp[k] for k in ["1x"] + HARMONIC_ORDERS)
    harmonics = [h for h in HARMONIC_ORDERS if present.get(h) and amp[h] >= HARMONIC_MIN_RATIO * a_max]
    faint = [h for h in HARMONIC_ORDERS if present.get(h) and h not in harmonics]
    ax1, ay1 = axes["x"]["orders"]["1x"]["amp_px"], axes["y"]["orders"]["1x"]["amp_px"]
    return {
        "amp_1x_px": round(a1, 5),
        "e1_share": round(a1 ** 2 / energy, 3) if energy > 0 else 0.0,
        "r21": round(amp["2x"] / a1, 3) if a1 > 0 else None,
        "r31": round(amp["3x"] / a1, 3) if a1 > 0 else None,
        "harmonics": harmonics,
        "n_harmonics": len(harmonics),
        "faint_harmonics": faint,
        "half_order": bool(present.get("0.5x") and a1 > 0 and amp["0.5x"] >= HALF_MIN_RATIO * a1),
        "half_ratio": round(amp["0.5x"] / a1, 3) if a1 > 0 else None,
        "h_over_v_1x": round(ax1 / ay1, 3) if ay1 > 0 else None,
        "velocity_rms": round(vel, 3),
        "velocity_unit": "mm/s" if scale_mm_per_px else "px/s",
        "present": {k2: v for k2, v in present.items() if v},
        "artifacts": artifacts,
        "blade_pass": blade_pass,
        "weak_axes": sorted(weak),
        "on_flicker_orders": on_flicker_orders,
        "velocity_without_blade_pass": round(vel_of(pattern_peaks), 3) if blade_pass else None,
    }


def phase_between(result, a, b, order="1x", axis="x"):
    """Phase of part b relative to part a at an order (rad, wrapped to [-pi, pi))."""
    pa = result["parts"][a]["axes"][axis]["orders"][order]["phase"]
    pb = result["parts"][b]["axes"][axis]["orders"][order]["phase"]
    return _wrap(pb - pa) if pa is not None and pb is not None else None


def _evidence(entry, labels, artifacts=()):
    flagged = {(a["axis"], a["hz"]) for a in artifacts}
    out = []
    for ax in "xy":
        for p in entry["axes"][ax]["peaks"]:
            if p["label"] in labels and (ax, p["hz"]) not in flagged:
                out.append({"axis": ax, "hz": p["hz"], "order": p["order"], "label": p["label"], "amp_px": p["amp_px"]})
    return sorted(out, key=lambda e: (e["order"], e["axis"]))


def classify(feat, alert, running_speed_clear=True):
    """(verdict, confidence, reason) from one part's features. running_speed_clear: whether the running speed is a
    real peak; when it is not, no order can be read and the verdict above the alert is unclear."""
    v = feat["velocity_rms"]
    if v < alert:
        conf = "high" if v < 0.5 * alert else "medium"
        return "none", conf, f"velocity {v:.2f} {feat['velocity_unit']} is below the alert level {alert:g}"
    bp = feat.get("blade_pass")
    if bp and feat["velocity_without_blade_pass"] < alert:
        rest = feat["velocity_without_blade_pass"]
        conf = "high" if rest < 0.5 * alert else "medium"
        return "blade_pass", conf, (f"high vibration at blade pass ({bp['blades']} blades x running speed, "
                                    f"{bp['hz']:.1f} Hz): {v:.2f} {feat['velocity_unit']}, {rest:.2f} without it; "
                                    "check the blades and the guard")
    if not running_speed_clear:
        return "unclear", "low", ("no clear peak at the running speed (1x is at the noise floor), so the fault "
                                  "pattern cannot be read")
    if feat["half_order"] or feat["n_harmonics"] >= MIN_HARMONICS:
        why = []
        if feat["half_order"]:
            why.append(f"0.5x at {feat['half_ratio']:.2f} of 1x")
        if feat["n_harmonics"] >= MIN_HARMONICS:
            why.append(f"{feat['n_harmonics']} harmonics ({', '.join(feat['harmonics'])})")
        conf = "high" if feat["half_order"] and feat["n_harmonics"] >= 2 else "medium"
        return "looseness", conf, "; ".join(why)
    if feat["r21"] is not None and feat["r21"] >= R21_MISALIGN:
        conf = "high" if feat["r21"] >= 2 * R21_MISALIGN else "medium"
        return "misalignment", conf, f"2x is {feat['r21']:.2f} of 1x"
    if feat["e1_share"] >= E1_IMBALANCE:
        conf = "high" if feat["e1_share"] >= 0.85 else "medium"
        return "imbalance", conf, f"1x holds {feat['e1_share']:.0%} of the vibration energy"
    return "unclear", "low", "no rule matched"


def fingerprint(result, alert=None, blades=None):
    """Features for every part, the rule verdict for each, and the official verdict.
    blades: a fan's blade count when known (entered, or from the plan); None for machines without blades."""
    mm = result["scale"]["mm_per_px"]
    alert = alert if alert is not None else (ALERT_MM_S if mm else ALERT_PX_S)
    parts = {}
    for name, entry in result["parts"].items():
        feat = part_features(entry, mm, result, blades)
        verdict, conf, reason = classify(feat, alert, result["running_speed"].get("clear", True))
        labels = {"none": set(), "imbalance": {"1x"}, "misalignment": {"1x", "2x"},
                  "looseness": {"0.5x", "1x"} | set(feat["harmonics"]), "unclear": set(),
                  "blade_pass": {feat["blade_pass"]["order"]} if feat.get("blade_pass") else set()}[verdict]
        parts[name] = {"features": feat, "verdict": verdict, "confidence": conf, "reason": reason,
                       "evidence": _evidence(entry, labels, feat["artifacts"])}
    names = list(parts)
    lead = max(names, key=lambda n: parts[n]["features"]["velocity_rms"])
    phases = {f"{a} -> {b}": round(phase_between(result, a, b), 3)
              for i, a in enumerate(names) for b in names[i + 1:] if phase_between(result, a, b) is not None}
    return {"verdict": parts[lead]["verdict"], "confidence": parts[lead]["confidence"], "reason": parts[lead]["reason"],
            "lead_part": lead, "evidence": parts[lead]["evidence"], "alert_level": alert,
            "alert_unit": "mm/s" if mm else "px/s", "scaled": bool(mm), "phase_1x_x_rad": phases,
            "blades": int(blades) if blades else None, "notes": _notes(parts, result), "parts": parts}


def _notes(parts, result):
    """Plain-words notes for the page: running speed on the flicker, orders on it, and axes left out."""
    notes = []
    g, seen = flicker_pattern(result)
    f1 = result["running_speed"]["hz"]
    if g and any(abs(f1 - k * g) <= SIDEBAND_TOL_BINS * result["bin_hz"] for k in range(1, int(f1 / g) + 2)):
        notes.append(f"Running speed ({f1:.1f} Hz) coincides with the light flicker at {result['fps']:g} fps: 1x is "
                     "kept, but it may include some flicker. Film in daylight or under DC lighting to be sure.")
    flick = sorted({k for p in parts.values() for k in p["features"].get("on_flicker_orders", {})},
                   key=lambda k: float(k[:-1]))
    if flick:
        rpm = result["running_speed"]["rpm"]
        notes.append(f"At {rpm:.0f} rpm, {', '.join(flick)} land on the light flicker ({2 * float(result.get('mains_hz') or MAINS_DEFAULT):g} Hz "
                     f"from the mains) or its sidebands; they count only when at least {FLICKER_ORDER_MIN:.0%} of 1x. "
                     "Film in daylight or under DC lighting to be sure.")
    for name, p in parts.items():
        for ax in p["features"].get("weak_axes", []):
            notes.append(f"The {name}'s {'horizontal' if ax == 'x' else 'vertical'} motion is left out: its edges "
                         f"barely fix that direction, so the reading there is mostly texture and noise.")
    return notes

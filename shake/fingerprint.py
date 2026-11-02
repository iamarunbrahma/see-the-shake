"""Fingerprint: vibration features per part and the fixed rules that give the official verdict.

Rules, first match wins, applied to the part that vibrates most:
1. none          overall velocity below the alert level
2. looseness     0.5x present, or a row of 3+ significant harmonics (2x to 6x, each at least 10% of the
                 largest running-speed component; tiny distortion products far below it do not count)
3. misalignment  2x at least half of 1x
4. imbalance     1x holds at least 60% of the vibration energy
5. unclear       anything else
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


def part_features(entry, scale_mm_per_px, result=None):
    axes = entry["axes"]
    artifacts = []
    if result is not None:
        pattern = flicker_pattern(result)
        for ax in "xy":
            for p in axes[ax]["peaks"]:
                kind = artifact_kind(p, result, ax, pattern)
                if kind and kind != "flicker":
                    artifacts.append({"axis": ax, "hz": p["hz"], "label": p["label"], "amp_px": p["amp_px"],
                                      "bg_amp_px": p.get("bg_amp_px"), "kind": kind})
    bin_hz = result["bin_hz"] if result is not None else 0.0
    amp, present = {}, {}
    for key in axes["x"]["orders"]:
        hz = axes["x"]["orders"][key]["hz"]
        skip = {a["axis"] for a in artifacts if abs(a["hz"] - hz) <= SIDEBAND_TOL_BINS * bin_hz + 1e-9} if key != "1x" else set()
        amp[key], present[key] = _vec(axes, key, skip)
    a1 = amp["1x"]
    flagged = {(a["axis"], a["hz"]) for a in artifacts}
    peaks = [p for ax in "xy" for p in axes[ax]["peaks"] if p["label"] != "flicker" and (ax, p["hz"]) not in flagged]
    # energy: every non-flicker peak, with orders counted once from the order table
    sync = [p for p in peaks if p["order"] is not None]
    nonsync_energy = sum(p["amp_px"] ** 2 for p in peaks if p["order"] is None)
    order_keys = {f"{p['order']:g}x" for p in sync}
    energy = sum(amp[k] ** 2 for k in order_keys if k in amp) + nonsync_energy
    k = scale_mm_per_px or 1.0
    vel = float(np.sqrt(sum((2 * np.pi * p["hz"] * p["amp_px"] * k) ** 2 / 2 for p in peaks)))
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


def classify(feat, alert):
    """(verdict, confidence, reason) from one part's features."""
    v = feat["velocity_rms"]
    if v < alert:
        conf = "high" if v < 0.5 * alert else "medium"
        return "none", conf, f"velocity {v:.2f} {feat['velocity_unit']} is below the alert level {alert:g}"
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


def fingerprint(result, alert=None):
    """Features for every part, the rule verdict for each, and the official verdict."""
    mm = result["scale"]["mm_per_px"]
    alert = alert if alert is not None else (ALERT_MM_S if mm else ALERT_PX_S)
    parts = {}
    for name, entry in result["parts"].items():
        feat = part_features(entry, mm, result)
        verdict, conf, reason = classify(feat, alert)
        labels = {"none": set(), "imbalance": {"1x"}, "misalignment": {"1x", "2x"},
                  "looseness": {"0.5x", "1x"} | set(feat["harmonics"]), "unclear": set()}[verdict]
        parts[name] = {"features": feat, "verdict": verdict, "confidence": conf, "reason": reason,
                       "evidence": _evidence(entry, labels, feat["artifacts"])}
    names = list(parts)
    lead = max(names, key=lambda n: parts[n]["features"]["velocity_rms"])
    phases = {f"{a} -> {b}": round(phase_between(result, a, b), 3)
              for i, a in enumerate(names) for b in names[i + 1:] if phase_between(result, a, b) is not None}
    return {"verdict": parts[lead]["verdict"], "confidence": parts[lead]["confidence"], "reason": parts[lead]["reason"],
            "lead_part": lead, "evidence": parts[lead]["evidence"], "alert_level": alert,
            "alert_unit": "mm/s" if mm else "px/s", "scaled": bool(mm), "phase_1x_x_rad": phases, "parts": parts}

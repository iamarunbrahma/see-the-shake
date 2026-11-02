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
import numpy as np

ALERT_MM_S = 7.0  # RMS velocity below this is reported as no fault
ALERT_PX_S = 5.0  # fallback when there is no mm scale (resolution-dependent; flagged as such)
HALF_MIN_RATIO = 0.15  # 0.5x must reach this fraction of 1x
R21_MISALIGN = 0.5
E1_IMBALANCE = 0.6
HARMONIC_ORDERS = ["2x", "3x", "4x", "5x", "6x"]
MIN_HARMONICS = 3
HARMONIC_MIN_RATIO = 0.10  # of the largest of 1x..6x; a long, clean clip resolves products 30 dB down

FAULT_ORDERS = {"imbalance": {"1x"}, "misalignment": {"2x"}, "looseness": {"0.5x"} | set(HARMONIC_ORDERS)}


def _vec(axes, key):
    """Vector amplitude (px) of an order over both axes, and whether either axis is above the peak threshold."""
    ox, oy = axes["x"]["orders"][key], axes["y"]["orders"][key]
    return float(np.hypot(ox["amp_px"], oy["amp_px"])), bool(ox["above_floor"] or oy["above_floor"])


def _wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def part_features(entry, scale_mm_per_px):
    axes = entry["axes"]
    amp, present = {}, {}
    for key in axes["x"]["orders"]:
        amp[key], present[key] = _vec(axes, key)
    a1 = amp["1x"]
    peaks = [p for ax in "xy" for p in axes[ax]["peaks"] if p["label"] != "flicker"]
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
    }


def phase_between(result, a, b, order="1x", axis="x"):
    """Phase of part b relative to part a at an order (rad, wrapped to [-pi, pi))."""
    pa = result["parts"][a]["axes"][axis]["orders"][order]["phase"]
    pb = result["parts"][b]["axes"][axis]["orders"][order]["phase"]
    return _wrap(pb - pa) if pa is not None and pb is not None else None


def _evidence(entry, labels):
    out = []
    for ax in "xy":
        for p in entry["axes"][ax]["peaks"]:
            if p["label"] in labels:
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
        feat = part_features(entry, mm)
        verdict, conf, reason = classify(feat, alert)
        labels = {"none": set(), "imbalance": {"1x"}, "misalignment": {"1x", "2x"},
                  "looseness": {"0.5x", "1x"} | set(feat["harmonics"]), "unclear": set()}[verdict]
        parts[name] = {"features": feat, "verdict": verdict, "confidence": conf, "reason": reason,
                       "evidence": _evidence(entry, labels)}
    names = list(parts)
    lead = max(names, key=lambda n: parts[n]["features"]["velocity_rms"])
    phases = {f"{a} -> {b}": round(phase_between(result, a, b), 3)
              for i, a in enumerate(names) for b in names[i + 1:] if phase_between(result, a, b) is not None}
    return {"verdict": parts[lead]["verdict"], "confidence": parts[lead]["confidence"], "reason": parts[lead]["reason"],
            "lead_part": lead, "evidence": parts[lead]["evidence"], "alert_level": alert,
            "alert_unit": "mm/s" if mm else "px/s", "scaled": bool(mm), "phase_1x_x_rad": phases, "parts": parts}

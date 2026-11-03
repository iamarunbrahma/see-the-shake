"""Compare a measurement with the twin's truth: frequency error in bins, amplitude error, verdict.

Only truth components that stand clearly above the measured noise floor are scored (EVAL_SNR x floor);
smaller ones are listed as below the evaluation threshold rather than silently dropped.

Scored truth is the twin's structure-weighted translation of each part (translation plus rocking, with the rocking
taken as the least-squares translation of its motion field over the part's rest image; see twin.rocking_lever).
The plain centroid value is reported next to it in every row, with its own error, and is not tuned to anything.

The twin's drum (or fan motor) speed drifts slowly (0.4% at 0.2 Hz), so a component is not one pure line: its
energy spreads over neighbouring bins, more for higher orders (3x reads about 9% under its applied amplitude in any
spectrum). Scored amplitudes are therefore the applied amplitude times the peak a unit sinusoid at that order of the
drifting speed reads in the pipeline's own spectrum over the measured window (drift_factor). Which components are
scored still follows the applied amplitude, and every row keeps it (true_amp_applied_px).
"""
import numpy as np

from . import measure

EVAL_SNR = 10.0
FREQ_TOL_BINS = 1.0
AMP_TOL = 0.20


def drift_factor(truth, result, order, phase=0.0):
    """Peak amplitude the pipeline's spectrum reads for a unit sinusoid at this order of the twin's drifting speed,
    over the measured window, against a drift-free line (1.0 when the truth records no drift; never above 1, since
    drift only spreads a line)."""
    sp = truth.get("drum") or truth.get("motor") or {}
    if not sp.get("drift_frac") or not result.get("frames"):
        return 1.0
    fps, start, n = float(truth["fps"]), int(result.get("start_frame") or 0), int(result["frames"])
    t = np.arange(start + n) / fps
    f_rot = sp["rpm"] / 60 * (1 + sp["drift_frac"] * np.sin(2 * np.pi * sp["drift_hz"] * t))
    rev = np.cumsum(f_rot) / fps
    hz = order * sp["rpm"] / 60

    def read(x):
        f, Z = measure.spectrum(x, fps)
        return float(measure.amplitude_at(f, np.abs(Z), hz)[0])
    # against a drift-free line read the same way, so the estimator's own small bias cancels
    return min(1.0, read(np.sin(2 * np.pi * order * rev[start:] + phase)) / read(np.sin(2 * np.pi * hz * t[start:] + phase)))


def compare(truth, result, verdict=None):
    """Rows per part, axis and truth component, plus pass flags."""
    rows = []
    bin_hz = result["bin_hz"]
    f1_true = truth["f_rot_hz"]
    factors = {}

    def factor(order, phase):
        key = (order, round(phase, 3))
        if key not in factors:
            factors[key] = drift_factor(truth, result, order, phase)
        return factors[key]
    for part, pt in truth["parts"].items():
        entry = result["parts"].get(part)
        if entry is None:
            continue
        centroid = {c["order"]: c for c in pt.get("components_centroid", [])}
        for c in pt["components"]:
            for ax in "xy":
                a_applied = c[f"{ax}_amp_px"]
                k = factor(c["order"], c.get(f"{ax}_phase", 0.0))
                a_true = a_applied * k
                axis = entry["axes"][ax]
                floor = axis["noise_floor_px"]
                cen = centroid.get(c["order"])
                a_cen = cen[f"{ax}_amp_px"] * factor(c["order"], cen.get(f"{ax}_phase", 0.0)) if cen else None
                row = {"part": part, "axis": ax, "order": c["order"], "true_hz": c["hz"], "true_amp_px": round(a_true, 6),
                       "true_amp_applied_px": a_applied, "drift_factor": round(k, 4),
                       "true_amp_centroid_px": round(a_cen, 6) if a_cen is not None else None, "floor_px": floor,
                       "scored": a_applied >= EVAL_SNR * floor}
                hz_true = c["order"] * f1_true
                cand = [p for p in axis["peaks"] if abs(p["hz"] - hz_true) <= FREQ_TOL_BINS * bin_hz + 1e-9]
                if cand:
                    p = max(cand, key=lambda q: q["amp_px"])
                    row.update(found=True, meas_hz=p["hz"], meas_amp_px=p["amp_px"],
                               freq_err_bins=round(abs(p["hz"] - hz_true) / bin_hz, 3),
                               amp_err=round((p["amp_px"] - a_true) / a_true, 4) if a_true > 0 else None,
                               amp_err_centroid=round((p["amp_px"] - a_cen) / a_cen, 4) if a_cen else None)
                else:
                    row.update(found=False, meas_hz=None, meas_amp_px=None, freq_err_bins=None, amp_err=None,
                               amp_err_centroid=None)
                row["pass"] = bool(row["found"] and row["freq_err_bins"] <= FREQ_TOL_BINS
                                   and abs(row["amp_err"]) <= AMP_TOL) if row["scored"] else None
                rows.append(row)
    scored = [r for r in rows if r["scored"]]
    out = {"rows": rows, "scored": len(scored), "passed": sum(r["pass"] for r in scored),
           "running_speed_err_bins": round(abs(result["running_speed"]["hz"] - f1_true) / bin_hz, 3)}
    if verdict is not None:
        out["verdict"] = verdict
        out["expected_verdict"] = truth["expected_verdict"]
        out["verdict_ok"] = verdict == truth["expected_verdict"]
    return out

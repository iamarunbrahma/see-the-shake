"""Compare a measurement with the twin's truth: frequency error in bins, amplitude error, verdict.

Only truth components that stand clearly above the measured noise floor are scored (EVAL_SNR x floor);
smaller ones are listed as below the evaluation threshold rather than silently dropped.

Scored truth is the twin's structure-weighted translation of each part (translation plus rocking, with the rocking
taken as the least-squares translation of its motion field over the part's rest image; see twin.rocking_lever).
The plain centroid value is reported next to it in every row, with its own error, and is not tuned to anything.
"""
EVAL_SNR = 10.0
FREQ_TOL_BINS = 1.0
AMP_TOL = 0.20


def compare(truth, result, verdict=None):
    """Rows per part, axis and truth component, plus pass flags."""
    rows = []
    bin_hz = result["bin_hz"]
    f1_true = truth["f_rot_hz"]
    for part, pt in truth["parts"].items():
        entry = result["parts"].get(part)
        if entry is None:
            continue
        centroid = {c["order"]: c for c in pt.get("components_centroid", [])}
        for c in pt["components"]:
            for ax in "xy":
                a_true = c[f"{ax}_amp_px"]
                axis = entry["axes"][ax]
                floor = axis["noise_floor_px"]
                a_cen = centroid[c["order"]][f"{ax}_amp_px"] if c["order"] in centroid else None
                row = {"part": part, "axis": ax, "order": c["order"], "true_hz": c["hz"], "true_amp_px": a_true,
                       "true_amp_centroid_px": a_cen, "floor_px": floor, "scored": a_true >= EVAL_SNR * floor}
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

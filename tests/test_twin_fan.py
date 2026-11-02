"""The fan twin: geometry, truth, and the rule verdicts on its clips (2 s, 960x540, cached like the washer's)."""
import hashlib
import json

import cv2
import numpy as np
import pytest

from shake import evaluate, fingerprint, measure, twin_fan

from .conftest import CACHE, ROOT

SECONDS, SCALE, SEED = 2.0, 0.5, 5


def fan_clip(fault):
    h = hashlib.sha1((ROOT / "shake" / "twin_fan.py").read_bytes() + (ROOT / "shake" / "twin.py").read_bytes())
    d = CACHE / f"fan_{h.hexdigest()[:10]}_{SECONDS:g}s_{SCALE:g}x_seed{SEED}" / fault
    if not (d / "truth.json").exists():
        twin_fan.render(twin_fan.FanConfig(fault, seconds=SECONDS, seed=SEED, out_scale=SCALE), d)
    return d


@pytest.fixture(scope="module")
def fan_runs():
    runs = {}
    for fault in twin_fan.FAULTS:
        d = fan_clip(fault)
        truth = json.loads((d / "truth.json").read_text())
        masks = {n: cv2.imread(str(d / truth["parts"][n]["mask"]), 0) for n in truth["parts"]}
        blades = cv2.imread(str(d / truth["rotating_mask"]), 0)
        ref = truth["reference"]
        sc = measure.Scale.from_reference(masks, ref["part"], ref["height_mm"], axis="height")
        rpm = truth["rpm"]
        res, _ = measure.measure(d / "clip.mp4", masks, truth["fps"], rpm_range=(0.9 * rpm, 1.1 * rpm), scale=sc,
                                 ignore=blades)
        fp = fingerprint.fingerprint(res)
        runs[fault] = dict(dir=d, truth=truth, masks=masks, blades=blades, result=res, fp=fp,
                           cmp=evaluate.compare(truth, res, fp["verdict"]))
    return runs


def test_measured_parts_never_include_the_blade_disc():
    masks, blades = twin_fan.part_masks()
    assert set(masks) == set(twin_fan.PARTS)
    for name, m in masks.items():
        assert (m > 0).sum() > 2000, name
        assert not ((m > 0) & (blades > 0)).any(), f"{name} overlaps the spinning blades"
    # the guard rim is a ring around the blades, not the disc
    ys, xs = np.nonzero(masks["fan guard rim"])
    assert (masks["fan guard rim"] > 0).sum() < 0.2 * (blades > 0).sum()


def test_truth_phasors_match_applied_motion():
    for fault in twin_fan.FAULTS:
        cfg = twin_fan.FanConfig(fault, seconds=1.0)
        mo = twin_fan.motion(cfg, np.random.default_rng(0))
        masks, _ = twin_fan.part_masks()
        parts = twin_fan.part_truth(cfg, masks, 1.0)
        for name, pt in parts.items():
            series = mo["pole"] if name == "fan stand" else mo["head"]
            orders = [c["order"] for c in pt["components"]]
            cols = [np.ones_like(mo["rev"])]
            for o in orders:
                cols += [np.sin(2 * np.pi * o * mo["rev"]), np.cos(2 * np.pi * o * mo["rev"])]
            B = np.column_stack(cols)
            for ai, axis in enumerate("xy"):
                coef, *_ = np.linalg.lstsq(B, series[:, ai], rcond=None)
                for i, c in enumerate(pt["components"]):
                    assert abs(np.hypot(coef[1 + 2 * i], coef[2 + 2 * i]) - c[f"{axis}_amp_px"]) < 1e-4, (fault, name, c)


def test_render_writes_a_complete_truth(fan_runs):
    t = fan_runs["looseness"]["truth"]
    d = fan_runs["looseness"]["dir"]
    for f in ("clip.mp4", "truth_series.npz", "mask_blades.png", *(t["parts"][p]["mask"] for p in t["parts"])):
        assert (d / f).exists(), f
    assert t["machine"] == "fan" and t["blades"] == 3 and abs(t["blade_pass_hz"] - 3 * t["f_rot_hz"]) < 1e-3
    assert 1000 <= t["rpm"] <= 1400 and t["reference"]["axis"] == "height"
    assert {"coupling", "walk_sigma_px_per_frame"} <= set(t["camera"]) and t["flicker"]["hz"] == 100.0
    assert set(np.load(d / "truth_series.npz").files) >= {"head", "pole", "cam", "gain", "rev"}


@pytest.mark.parametrize("fault", list(twin_fan.FAULTS))
def test_rule_verdict_per_fan_fault(fan_runs, fault):
    run = fan_runs[fault]
    assert run["fp"]["verdict"] == twin_fan.EXPECTED_VERDICT[fault], run["fp"]["reason"]


def test_blade_pass_does_not_make_a_healthy_fan_look_faulty(fan_runs):
    fp = fan_runs["healthy"]["fp"]
    assert fp["verdict"] == "none"
    for part in fp["parts"].values():
        f = part["features"]
        assert part["verdict"] == "none" and f["velocity_rms"] < 0.6 * fp["alert_level"]
        assert not f["half_order"] and f["n_harmonics"] < 3


def test_running_speed_and_the_strong_components(fan_runs):
    """The 1x of every part, and the looseness row, within one bin and 20% of truth (at 960x540, 2 s)."""
    for fault in ("imbalance", "looseness"):
        cmp = fan_runs[fault]["cmp"]
        assert cmp["running_speed_err_bins"] <= 1.0
        strong = [r for r in cmp["rows"] if r["scored"] and r["true_amp_px"] >= 0.04]
        assert strong and all(r["pass"] for r in strong), [r for r in strong if not r["pass"]]
    rotating = {p: e["excluded_rotating_px"] for p, e in fan_runs["looseness"]["result"]["parts"].items()}
    assert rotating["fan stand"] == rotating["fan motor housing"] == 0


def test_blade_pass_cannot_push_a_verdict(fan_runs):
    """A healthy fan whose blade pass grows past the alert: without the blade count the rules called it imbalance;
    with it, blade pass is left out of the pattern and the verdict says so. Severity keeps it."""
    import copy
    base = fan_runs["healthy"]["result"]
    f3 = 3 * base["running_speed"]["hz"]
    for amp in np.arange(0.002, 0.2, 0.002):  # a blade-pass peak of growing size on every part
        m = copy.deepcopy(base)
        for e in m["parts"].values():
            for ax in "xy":
                a = e["axes"][ax]
                a["peaks"] = [p for p in a["peaks"] if p["order"] != 3] + [
                    {"hz": round(f3, 3), "amp_px": float(amp), "order": 3.0, "label": "3x", "phase": 0.0, "snr": 50.0,
                     "bg_amp_px": 0.0}]
                a["orders"]["3x"].update(amp_px=float(amp), above_floor=True)
        with_count = fingerprint.fingerprint(m, blades=3)
        if with_count["verdict"] != "none":
            break
    without = fingerprint.fingerprint(m)
    assert with_count["verdict"] == "blade_pass" and "check the blades and the guard" in with_count["reason"]
    assert without["verdict"] in ("imbalance", "unclear")  # the vulnerability the blade count closes
    lead = with_count["parts"][with_count["lead_part"]]["features"]
    assert lead["velocity_rms"] == without["parts"][with_count["lead_part"]]["features"]["velocity_rms"]
    assert lead["velocity_without_blade_pass"] < with_count["alert_level"] <= lead["velocity_rms"]
    assert {e["label"] for e in with_count["evidence"]} == {"3x"}


@pytest.mark.parametrize("fault", list(twin_fan.FAULTS))
def test_fan_verdicts_hold_with_the_blade_count(fan_runs, fault):
    fp = fingerprint.fingerprint(fan_runs[fault]["result"], blades=3)
    assert fp["verdict"] == twin_fan.EXPECTED_VERDICT[fault], fp["reason"]

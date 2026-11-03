import json

import numpy as np

from shake import twin

from .conftest import twin_clip


def _project(signal, rev, orders):
    """{order: amplitude} of signal = sum A sin(2 pi order rev + phase), by one joint least-squares fit."""
    cols = [np.ones_like(rev)]
    for o in orders:
        a = 2 * np.pi * o * rev
        cols += [np.sin(a), np.cos(a)]
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), signal, rcond=None)
    return {o: float(np.hypot(coef[1 + 2 * i], coef[2 + 2 * i])) for i, o in enumerate(orders)}


def test_truth_phasors_match_applied_motion():
    """Every applied motion (translation + rocking) is in the truth: the phasors reproduce the per-frame series."""
    for fault in twin.FAULTS:
        cfg = twin.TwinConfig(fault, seconds=1.0)
        mo = twin.motion(cfg, np.random.default_rng(0))
        masks, _ = twin.part_masks()
        lever = {"door": (123.0, -4.0), "washing machine": (-250.0, 7.0)}
        parts = twin.part_truth(cfg, masks, lever)
        for name, pt in parts.items():
            lx, ly = lever[name]
            b = np.sin(np.deg2rad(mo["rock_deg"]))
            x, y = mo["dx"] + b * lx, mo["dy"] + b * ly
            orders = [c["order"] for c in pt["components"]]
            ax, ay = _project(x, mo["rev"], orders), _project(y, mo["rev"], orders)
            for c in pt["components"]:
                assert abs(ax[c["order"]] - c["x_amp_px"]) < 1e-4, (fault, name, c)
                assert abs(ay[c["order"]] - c["y_amp_px"]) < 1e-4, (fault, name, c)


def test_rocking_displacement_matches_cv2_rotation():
    """rocking_displacement is what cv2.getRotationMatrix2D applies about the pivot (to first order: the
    neglected cos term is about 1e-4 px at 0.04 degrees and 500 px)."""
    import cv2
    px, py = twin.pivot_px()
    M = cv2.getRotationMatrix2D((px, py), 0.04, 1.0)
    for x, y in [(px, py - 500), (px - 200, py - 100), (px + 250, py)]:
        moved = M @ np.array([x, y, 1.0])
        dx, dy = twin.rocking_displacement(0.04, x, y)
        assert np.allclose(moved - [x, y], [dx, dy], atol=5e-4)


def test_rocking_lever_equals_centroid_for_uniform_texture():
    rng = np.random.default_rng(1)
    gray = (rng.normal(128, 30, (twin.H, twin.W))).astype(np.float32)
    w = np.zeros((twin.H, twin.W), bool)
    w[300:500, 800:1000] = True
    lx, ly = twin.rocking_lever(gray, w, 1.0)
    px, py = twin.pivot_px()
    ys, xs = np.nonzero(w)
    assert abs(lx - (ys.mean() - py)) < 0.05 * abs(ys.mean() - py)
    assert abs(ly - -(xs.mean() - px)) < 5


def test_render_writes_clip_truth_and_masks():
    d = twin_clip("looseness")
    truth = json.loads((d / "truth.json").read_text())
    assert set(truth["parts"]) == {"washing machine", "door"}
    for f in ("clip.mp4", "truth_series.npz", "mask_glass.png", *(p["mask"] for p in truth["parts"].values())):
        assert (d / f).exists()
    # the truth records every motion the renderer applies
    assert truth["cabinet"]["rocking"]["amp_deg"] == 0.04
    assert {"coupling", "walk_sigma_px_per_frame"} <= set(truth["camera"])
    assert truth["flicker"]["hz"] == 100.0 and truth["noise"]["sigma"] > 0 and truth["encoding"]["codec"] == "libx264"
    assert truth["size"] == [960, 540] and truth["mm_per_px"] == 2.4
    s = np.load(d / "truth_series.npz")
    assert {"dx", "dy", "rock_deg", "cam_x", "cam_y", "gain", "rev"} <= set(s.files)
    # rocking changes the whole machine's 1x away from the bare cabinet translation
    wm = {c["order"]: c for c in truth["parts"]["washing machine"]["components"]}
    assert abs(wm[1.0]["x_amp_px"] - 0.14 * 0.5) > 0.01


def test_misalignment_has_strong_2x():
    comps = {o: (ax, ay) for o, ax, ay, _ in twin.FAULTS["misalignment"]}
    assert comps[2][0] > comps[1][0]


def test_scored_truth_follows_the_drum_drift():
    """The drum's 0.4% speed drift spreads 3x over neighbouring bins: over 6 s at 240 fps a pure 3x line reads about
    11% under its applied amplitude in the pipeline's spectrum, and the scored truth says so (1x barely moves)."""
    from shake import evaluate
    truth = {"fps": 240, "drum": {"rpm": 1000.0, "drift_frac": twin.RPM_DRIFT, "drift_hz": twin.RPM_DRIFT_HZ}}
    res = {"frames": 1440, "start_frame": 0}
    k1, k3 = evaluate.drift_factor(truth, res, 1.0), evaluate.drift_factor(truth, res, 3.0, 0.7)
    assert 0.98 < k1 <= 1.0 and 0.85 < k3 < 0.93
    assert evaluate.drift_factor({"fps": 240}, res, 3.0) == 1.0  # no drift recorded: applied amplitude

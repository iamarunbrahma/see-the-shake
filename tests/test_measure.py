import numpy as np
import pytest

from shake import evaluate, measure, twin


def test_refine_recovers_off_bin_sinusoid():
    fps, n = 240.0, 480
    t = np.arange(n) / fps
    for hz, amp in [(16.73, 0.2), (33.41, 0.05), (8.33, 0.03)]:
        f, Z = measure.spectrum(amp * np.sin(2 * np.pi * hz * t + 0.3), fps)
        A = np.abs(Z)
        rf, ra = measure.refine(f, A, int(np.argmax(A)))
        assert abs(rf - hz) < 0.1 and abs(ra - amp) / amp < 0.02


def test_order_and_flicker_labels():
    assert measure.label_order(16.7, 16.67, 0.5) == (1.0, "1x")
    assert measure.label_order(8.3, 16.67, 0.5) == (0.5, "0.5x")
    assert measure.label_order(23.0, 16.67, 0.5)[1] == "non-synchronous"
    assert measure.label_order(100.0, 16.667, 0.5, flicker=100.0) == (None, "flicker")
    assert measure.flicker_hz(240) == 100 and measure.flicker_hz(120) == 20


@pytest.mark.parametrize("fault", list(twin.FAULTS))
def test_amplitudes_and_frequencies_match_truth(twin_runs, fault):
    run = twin_runs[fault]
    cmp = evaluate.compare(run["truth"], run["result"])
    assert cmp["running_speed_err_bins"] <= 1.0
    scored = [r for r in cmp["rows"] if r["scored"]]
    assert scored
    bad = [r for r in scored if not r["pass"]]
    assert not bad, bad


def test_camera_shake_removed(twin_runs):
    run = twin_runs["imbalance"]
    bg = run["result"]["background"]["axes"]["x"]["orders"]["1x"]["amp_px"]
    assert bg > 0.05 * 0.30 * 0.5  # the camera really shakes at 1x (coupling 0.12)
    # and the part reading still matches truth after subtracting it (scored above); raw tracking would read high
    raw = run["arrays"]["series"]["door"][:, 0]
    f, Z = measure.spectrum(raw, 240.0)
    raw_1x = measure.amplitude_at(f, np.abs(Z), run["result"]["running_speed"]["hz"])[0]
    rel_1x = run["result"]["parts"]["door"]["axes"]["x"]["orders"]["1x"]["amp_px"]
    assert raw_1x > rel_1x * 1.05


def test_flicker_rejected(twin_runs):
    """1.5% 100 Hz flicker: gain normalisation leaves a residue far below any fault, and it is labelled, not an order."""
    for fault, run in twin_runs.items():
        for part in run["result"]["parts"].values():
            for ax in "xy":
                for p in part["axes"][ax]["peaks"]:
                    if abs(p["hz"] - 100) < 1:
                        assert p["label"] == "flicker" and p["order"] is None, (fault, p)
                        assert p["amp_px"] < 0.01, (fault, p)


def test_rotating_drum_excluded(twin_runs):
    run = twin_runs["healthy"]
    glass_px = np.pi * (twin.GLASS_R * 0.5) ** 2
    for part in ("door", "washing machine"):
        ex = run["result"]["parts"][part]["excluded_rotating_px"]
        assert 0.7 * glass_px < ex < 1.6 * glass_px, (part, ex, glass_px)


def test_mm_scale(twin_runs):
    r = twin_runs["healthy"]["result"]
    assert abs(r["scale"]["mm_per_px"] - 2.4) < 0.01


def _mean_pass_before_parallel(path, start, count):
    """mean_and_motion_mask as it was when the accuracy table was made (sequential, one decode loop), verbatim."""
    import cv2
    from shake.ingest import read_gray
    acc = acc2 = None
    n = 0
    for g in read_gray(path, start=start, count=count, blur=measure.BLUR):
        gm = float(g.mean())
        gn = g * (1.0 / gm)
        if acc is None:
            acc, acc2, gsum = g.astype(np.float64), (gn.astype(np.float64)) ** 2, gm
            gnsum = gn.astype(np.float64)
        else:
            acc += g
            acc2 += gn.astype(np.float64) ** 2
            gnsum += gn
            gsum += gm
        n += 1
    ref = (acc / n).astype(np.float32)
    scale = gsum / n
    std = np.sqrt(np.maximum(acc2 / n - (gnsum / n) ** 2, 0)).astype(np.float32) * scale
    gy, gx = np.gradient(ref)
    grad = np.sqrt(gx * gx + gy * gy)
    noise = float(np.median(std))
    rotating = std > 4 * noise + measure.MAX_VIB_PX * grad
    rotating = measure._dilate(cv2.morphologyEx(rotating.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0, 7)
    return ref, rotating, n, noise


def test_parallel_measurement_is_bit_identical(twin_runs):
    """The threaded decode, strip-parallel mean pass and frame-parallel tracking change nothing, bit for bit,
    so the accuracy table made with the sequential code still stands."""
    run = twin_runs["looseness"]
    clip = run["dir"] / "clip.mp4"
    new, old = measure.mean_and_motion_mask(clip, 0, 480), _mean_pass_before_parallel(clip, 0, 480)
    assert np.array_equal(new[0], old[0]) and np.array_equal(new[1], old[1]) and new[2:] == old[2:]
    seq, _ = measure.track(clip, run["masks"], 0, 480, workers=1)
    par, _ = measure.track(clip, run["masks"], 0, 480, workers=4)
    assert all(np.array_equal(seq[k], par[k]) for k in seq)


def test_running_speed_takes_1x_under_a_big_2x_and_says_it_is_ambiguous():
    """The rotor rig at 1000 rpm: 2x slightly larger than 1x, and a wide range that holds both."""
    fps, n = 500.0, 500
    t = np.arange(n) / fps
    x = 0.9 * np.sin(2 * np.pi * 16.67 * t) + 1.0 * np.sin(2 * np.pi * 33.33 * t + 0.4)
    f, Z = measure.spectrum(x, fps)
    hz, source, info = measure.running_speed(f, np.abs(Z) ** 2, rpm_range=(300, 3600), detail=True)
    assert abs(hz - 16.67) < 0.3 and "half" in source and info["ambiguous"]
    # a looseness spectrum (0.5x at 40% of 1x) keeps 1x, and is not ambiguous
    y = 0.4 * np.sin(2 * np.pi * 8.33 * t) + 1.0 * np.sin(2 * np.pi * 16.67 * t) + 0.6 * np.sin(2 * np.pi * 33.33 * t)
    f, Z = measure.spectrum(y, fps)
    hz, source, info = measure.running_speed(f, np.abs(Z) ** 2, rpm_range=(300, 3600), detail=True)
    assert abs(hz - 16.67) < 0.3 and source == "rpm_range"


def test_axis_confidence_finds_the_weak_axis_of_a_horizontal_bar():
    import cv2
    img = np.full((200, 400), 200, np.float32)
    img[90:110, 20:380] = 60  # a horizontal shaft: edges along x only fix y
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    w = np.zeros_like(img, bool)
    w[80:120, 60:340] = True  # the middle of the shaft, ends out of view
    c = measure.axis_confidence(img, w)
    assert c["y"] == 1.0 and c["x"] < 0.05


def _part_with_a_patch_of_its_own(path, fps=240.0, n=480, f1=10.0, amp=0.2, patch_amp=1.0, seed=0):
    """A rigid textured part moving vertically by amp px at f1, with a patch of its surface moving on its own at
    3x f1, as marks on a turning shaft or disc do. Lossless (FFV1), with sensor noise."""
    import cv2
    rng = np.random.default_rng(seed)
    H, W = 320, 320
    bg = cv2.GaussianBlur(rng.uniform(60, 200, (H, W)).astype(np.float32), (0, 0), 2.0)
    part = cv2.GaussianBlur(rng.uniform(40, 220, (H, W)).astype(np.float32), (0, 0), 1.5)
    pm = np.zeros((H, W), np.float32)
    pm[130:190, 110:210] = 1
    patch = np.zeros((H, W), np.float32)
    patch[140:180, 115:155] = 1
    pm_s, patch_s = cv2.GaussianBlur(pm, (0, 0), 0.7), cv2.GaussianBlur(patch, (0, 0), 0.7)

    def shift(img, dy):
        return cv2.warpAffine(img, np.float32([[1, 0, 0], [0, 1, dy]]), (W, H), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (W, H), False)
    for i in range(n):
        t = i / fps
        d = amp * np.sin(2 * np.pi * f1 * t)
        m, pt = shift(pm_s, d), shift(patch_s, d)
        img = bg * (1 - m) + shift(part, d) * m
        img = img * (1 - pt) + shift(part, d + patch_amp * np.sin(2 * np.pi * 3 * f1 * t)) * pt
        wr.write(np.clip(img + rng.normal(0, 1.0, (H, W)), 0, 255).astype(np.uint8))
    wr.release()
    return {"part": (pm > 0).astype(np.uint8) * 255}


def test_surface_texture_turning_with_a_part_is_left_out(tmp_path, monkeypatch):
    """Without the texture test the patch reads as a 3x nearly as large as 1x; with it, the patch is left out and
    the part's own 1x stays right."""
    clip = tmp_path / "part.avi"
    masks = _part_with_a_patch_of_its_own(clip)
    res, _ = measure.measure(clip, masks, 240.0, rpm=600)
    e = res["parts"]["part"]
    y = e["axes"]["y"]["orders"]
    monkeypatch.setattr(measure, "TEXTURE_MIN", 2.0)  # never re-track: what the tracker read before
    before = measure.measure(clip, masks, 240.0, rpm=600)[0]["parts"]["part"]
    yb = before["axes"]["y"]["orders"]
    assert before["excluded_texture_px"] == 0 and yb["3x"]["amp_px"] > 0.5 * yb["1x"]["amp_px"]
    assert e["excluded_texture_px"] > 0.6 * 40 * 40
    assert abs(y["1x"]["amp_px"] - 0.2) < 0.02 and y["3x"]["amp_px"] < 0.25 * yb["3x"]["amp_px"]


def _made_up(f1, comps, fps=240.0, seconds=10.0, noise=0.003, seed=0, extra=()):
    """Tracked series for one part ('machine') and a still background: comps [(order, amp_px)] on both axes at f1,
    extra [(hz, amp_px)] at other frequencies, white tracking noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fps * seconds)) / fps
    x = sum(a * np.sin(2 * np.pi * o * f1 * t + o) for o, a in comps) + sum(a * np.sin(2 * np.pi * h * t) for h, a in extra)
    y = sum(a * np.cos(2 * np.pi * o * f1 * t + o) for o, a in comps) + sum(a * np.cos(2 * np.pi * h * t) for h, a in extra)
    part = np.column_stack([x, y]) + rng.normal(0, noise, (len(t), 2))
    return {"machine": part, "background": rng.normal(0, noise, (len(t), 2))}


def _rules(series, fps=240.0, **kw):
    from shake import fingerprint
    res, _ = measure.analyse(series, ["machine"], fps, scale=measure.Scale(1.0, "test"), **kw)
    return res, fingerprint.fingerprint(res)


IMBALANCE = [(1, 0.3), (2, 0.02)]


def test_a_range_with_no_real_peak_is_set_aside():
    """QA H2: Spark said 20 to 200 rpm (clamped to 126 to 200) for a machine at 998 rpm. The range used to give a
    noise bin at 3.33 Hz as 1x, the true 1x read as 5x and the verdict misalignment."""
    res, fp = _rules(_made_up(16.64, IMBALANCE, seconds=6.0), rpm_range=(126, 200))
    rs = res["running_speed"]
    assert abs(rs["hz"] - 16.64) < 0.05 and rs["clear"] and rs["range_set_aside"] == [126, 200]
    assert "set aside" in rs["source"] and any("holds no clear vibration peak" in n for n in rs["notes"])
    assert fp["verdict"] == "imbalance"


def test_a_much_stronger_peak_outside_the_range_sets_it_aside_unless_it_is_a_harmonic():
    # the range holds a small real peak (a belt, say) while the machine's 1x, 25 times stronger, sits outside it
    s = _made_up(16.64, IMBALANCE, extra=[(3.0, 0.012)])
    res, fp = _rules(s, rpm_range=(150, 200))
    assert abs(res["running_speed"]["hz"] - 16.64) < 0.05 and fp["verdict"] == "imbalance"
    assert any("times stronger" in n for n in res["running_speed"]["notes"])
    # a fan: the range holds 1x and blade pass (3x) is five times stronger outside it; the range stands
    res, _ = _rules(_made_up(21.5, [(1, 0.05), (3, 0.25)]), rpm_range=(1200, 1400))
    assert abs(res["running_speed"]["hz"] - 21.5) < 0.05 and res["running_speed"]["source"] == "rpm_range"
    assert not res["running_speed"]["notes"]


def test_an_entered_rpm_is_a_nameplate_value():
    """QA H3: an induction motor turning 3% under the rpm entered read every peak as non-synchronous and the
    verdict unclear. The running speed is now the strongest peak within 5%, and the page says where it settled."""
    for true_rpm, entered in ((1450, 1500), (2900, 3000)):
        res, fp = _rules(_made_up(true_rpm / 60, IMBALANCE), rpm=entered)
        rs = res["running_speed"]
        assert abs(rs["rpm"] - true_rpm) < 2 and rs["source"].startswith("entered rpm") and rs["clear"]
        assert any(f"settled on {true_rpm} rpm" in n and f"the {entered} rpm entered" in n for n in rs["notes"])
        assert fp["verdict"] == "imbalance", fp["reason"]
    # an rpm entered right: no note
    res, _ = _rules(_made_up(25.0, IMBALANCE), rpm=1500)
    assert abs(res["running_speed"]["rpm"] - 1500) < 2 and not res["running_speed"]["notes"]


def test_no_real_peak_at_the_running_speed_reads_unclear():
    """Strong vibration nowhere near the rpm entered: no order can be read, so the verdict is unclear, not a fault."""
    res, fp = _rules(_made_up(16.67, [(1, 0.0)], extra=[(37.3, 0.3)]), rpm=1000)
    assert not res["running_speed"]["clear"] and any("noise floor" in n for n in res["running_speed"]["notes"])
    assert fp["verdict"] == "unclear" and "noise floor" in fp["reason"]


def test_running_speed_on_the_flicker_is_still_1x():
    """QA M1: at 120 fps, 100 Hz flicker folds to 20 Hz; a 1x there was labelled flicker and the verdict was
    none at 0.00 mm/s, while the same machine at 22 Hz read imbalance."""
    for hz in (20.0, 22.0):
        res, fp = _rules(_made_up(hz, IMBALANCE, fps=120.0), fps=120.0, rpm=hz * 60)
        y = [p for p in res["parts"]["machine"]["axes"]["y"]["peaks"] if abs(p["hz"] - hz) < 0.2]
        assert y and y[0]["label"] == "1x" and y[0]["order"] == 1.0
        assert fp["verdict"] == "imbalance" and fp["parts"]["machine"]["features"]["velocity_rms"] > 7
        assert any("coincides with the light flicker" in n for n in fp["notes"]) == (hz == 20.0)
    # a peak at the flicker that is no low order is still labelled flicker
    assert measure.label_order(20.0, 7.0, 0.1, flicker=20.0) == (None, "flicker")


def test_the_energy_share_counts_orders_beyond_the_table():
    """A 7x is a peak like any other: it lowers 1x's share of the energy (it used to be skipped)."""
    res, fp = _rules(_made_up(10.0, [(1, 0.3), (7, 0.2)]), rpm=600)
    f = fp["parts"]["machine"]["features"]
    assert any(p["label"] == "7x" for p in res["parts"]["machine"]["axes"]["x"]["peaks"])
    assert abs(f["e1_share"] - 0.09 / 0.13) < 0.03


def test_one_velocity_per_part_and_named_for_what_it_is():
    res, fp = _rules(_made_up(25.0, IMBALANCE), rpm=1500)
    entry = res["parts"]["machine"]
    assert "velocity_rms" not in entry and entry["velocity_rms_all_peaks"] > 0
    assert fp["parts"]["machine"]["features"]["velocity_rms"] > 0

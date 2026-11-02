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

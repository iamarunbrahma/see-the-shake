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

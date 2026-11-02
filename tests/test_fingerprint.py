import pytest

from shake import fingerprint, twin


@pytest.mark.parametrize("fault", list(twin.FAULTS))
def test_rule_verdict_per_fault(twin_runs, fault):
    fp = twin_runs[fault]["fp"]
    assert fp["verdict"] == twin.EXPECTED_VERDICT[fault], fp["reason"]
    if fp["verdict"] != "none":
        assert fp["evidence"], "a fault verdict must cite peaks"
        orders = {e["label"] for e in fp["evidence"]}
        assert orders & fingerprint.FAULT_ORDERS[fp["verdict"]]


def _feat(**kw):
    base = dict(velocity_rms=20.0, velocity_unit="mm/s", half_order=False, half_ratio=0.0, n_harmonics=0,
                harmonics=[], r21=0.1, e1_share=0.9)
    base.update(kw)
    return base


def test_rule_order():
    assert fingerprint.classify(_feat(velocity_rms=2.0), 7.0)[0] == "none"
    assert fingerprint.classify(_feat(half_order=True, half_ratio=0.3, r21=0.8), 7.0)[0] == "looseness"
    assert fingerprint.classify(_feat(n_harmonics=3, harmonics=["2x", "3x", "4x"], r21=0.9), 7.0)[0] == "looseness"
    assert fingerprint.classify(_feat(r21=0.6, e1_share=0.7), 7.0)[0] == "misalignment"
    assert fingerprint.classify(_feat(r21=0.2, e1_share=0.9), 7.0)[0] == "imbalance"
    assert fingerprint.classify(_feat(r21=0.2, e1_share=0.3), 7.0)[0] == "unclear"


def _entry(amps, floor=0.0002):
    """A minimal measured part: amplitudes per order on x, nothing on y."""
    def axis(a):
        orders = {k: {"hz": 16.67 * float(k[:-1]), "amp_px": v, "above_floor": v > 8 * floor, "phase": 0.0}
                  for k, v in a.items()}
        peaks = [{"hz": 16.67 * float(k[:-1]), "amp_px": v, "order": float(k[:-1]), "label": k}
                 for k, v in a.items() if v > 8 * floor]
        return {"noise_floor_px": floor, "orders": orders, "peaks": peaks}
    keys = ["0.5x", "1x", "1.5x", "2x", "2.5x", "3x", "4x", "5x", "6x"]
    return {"axes": {"x": axis({k: amps.get(k, 0.0) for k in keys}), "y": axis({k: 0.0 for k in keys})}}


def test_faint_harmonics_do_not_make_looseness():
    """Misalignment measured on a long clean clip: a 4x at 2% of 2x is above the noise floor but is not a row."""
    f = fingerprint.part_features(_entry({"1x": 0.07, "2x": 0.19, "3x": 0.03, "4x": 0.0046}), 1.2)
    assert f["harmonics"] == ["2x", "3x"] and f["faint_harmonics"] == ["4x"]
    assert fingerprint.classify(f, 7.0)[0] == "misalignment"
    loose = fingerprint.part_features(_entry({"1x": 0.10, "2x": 0.08, "3x": 0.05, "4x": 0.03}), 1.2)
    assert loose["n_harmonics"] == 3 and fingerprint.classify(loose, 7.0)[0] == "looseness"


def test_features_are_reported(twin_runs):
    fp = twin_runs["looseness"]["fp"]
    f = fp["parts"][fp["lead_part"]]["features"]
    assert f["half_order"] and f["n_harmonics"] >= 2 and f["velocity_unit"] == "mm/s"
    assert "washing machine -> door" in fp["phase_1x_x_rad"]


FIXTURES = __import__("pathlib").Path(__file__).parent / "api" / "fixtures"


def _recorded(kind, fault):
    import json
    return json.loads((FIXTURES / kind / fault / "measured.json").read_text())


def test_compression_and_flicker_artifacts_are_discounted():
    """The live phone-like healthy run (720p, default x264): flicker sidebands near 5x and 7x and the 20 Hz flicker
    pattern read 6.1 mm/s against the 7 mm/s alert. Discounted, the reading is the real 1x motion."""
    m = _recorded("live_phone", "healthy")
    fp = fingerprint.fingerprint(m)
    f = fp["parts"][fp["lead_part"]]["features"]
    assert fp["verdict"] == "none" and f["velocity_rms"] < 3.5
    kinds = {round(a["hz"]): a["kind"] for a in f["artifacts"]}
    assert kinds[83] == kinds[117] == "flicker sideband" and "5x" not in f["harmonics"]


@pytest.mark.parametrize("kind", ["live", "live_phone"])
def test_artifacts_never_touch_running_speed_or_low_orders(kind):
    for fault in twin.FAULTS:
        fp = fingerprint.fingerprint(_recorded(kind, fault))
        assert fp["verdict"] == twin.EXPECTED_VERDICT[fault]
        for part in fp["parts"].values():
            for a in part["features"]["artifacts"]:
                assert a["label"] != "1x"
                if a["kind"] == "flicker sideband":
                    assert a["label"] in ("non-synchronous",) or float(a["label"][:-1]) >= fingerprint.FLICKER_SAFE_ORDER


def test_flicker_on_an_order_is_not_a_harmonic():
    """At 1000 rpm 6x is exactly 100 Hz, the light flicker; that peak must not count as a sixth harmonic."""
    entry = _entry({"1x": 0.07, "2x": 0.02, "3x": 0.015, "6x": 0.012})
    for p in entry["axes"]["x"]["peaks"]:
        if p["label"] == "6x":
            p.update(label="flicker", order=None, hz=100.0)
    result = {"fps": 240.0, "bin_hz": 0.1667, "mains_hz": 50.0, "running_speed": {"hz": 16.67}, "parts": {"m": entry}}
    with_rules = fingerprint.part_features(entry, 1.2, result)
    assert with_rules["harmonics"] == ["2x", "3x"]
    assert any(a["kind"] == "flicker" and a["hz"] == 100.0 for a in with_rules["artifacts"])
    assert "6x" in fingerprint.part_features(entry, 1.2)["harmonics"]  # what happened before


def test_spectrum_plots_skip_artifact_peaks(monkeypatch, tmp_path):
    """The phone-like healthy spectrum Spark is sent: flicker sidebands near 5x and 7x get no order label."""
    import matplotlib.axes
    import numpy as np
    from shake import measure
    m = _recorded("live_phone", "healthy")
    fp = fingerprint.fingerprint(m)
    skip = {p: {(a["axis"], a["hz"]) for a in e["features"]["artifacts"]} for p, e in fp["parts"].items()}
    f = np.linspace(0, 120, 721)
    arrays = {"spectra": {(p, ax): (f, np.full_like(f, 1e-3)) for p in m["parts"] for ax in "xy"}}
    labels = []
    monkeypatch.setattr(matplotlib.axes.Axes, "annotate", lambda self, text, *a, **k: labels.append(text))
    measure.plot_spectra(m, arrays, tmp_path / "all.png")
    before = list(labels)
    labels.clear()
    measure.plot_spectra(m, arrays, tmp_path / "skipped.png", skip=skip)
    assert "5x" in before and "7x" in before
    assert "5x" not in labels and "7x" not in labels and "1x" in labels

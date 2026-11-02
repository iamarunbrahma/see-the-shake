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


def test_features_are_reported(twin_runs):
    fp = twin_runs["looseness"]["fp"]
    f = fp["parts"][fp["lead_part"]]["features"]
    assert f["half_order"] and f["n_harmonics"] >= 2 and f["velocity_unit"] == "mm/s"
    assert "washing machine -> door" in fp["phase_1x_x_rad"]

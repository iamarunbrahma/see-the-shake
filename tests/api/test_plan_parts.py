"""Inspection plan validation, offline."""
import pytest

from shake import plan_parts
from shake.plan_parts import validate


def raw(parts, lo=400, hi=1600, machine="washing machine"):
    return {"machine_type": machine, "rpm_min": lo, "rpm_max": hi, "rpm_reason": "spin", "parts": parts,
            "rotating_parts_seen": ["drum"]}


def test_machine_first_and_phrases_normalised():
    p = validate(raw(["The Front Door", "control panel!", "washing machine", "front door"]), 240)
    assert p["parts"] == ["washing machine", "front door", "control panel"]
    assert p["fallback_phrase"] == "washing machine" and p["rpm_range"] == [400.0, 1600.0]


def test_rotating_parts_dropped_while_a_static_one_remains():
    p = validate(raw(["fan blades", "motor housing", "downrod", "blade guard"], 150, 400, "ceiling fan"), 240)
    assert p["parts"] == ["ceiling fan", "motor housing", "downrod", "blade guard"]
    assert any("rotating" in w for w in p["warnings"])
    assert plan_parts.is_rotating("drum") and plan_parts.is_rotating("fan blades")
    assert not plan_parts.is_rotating("ceiling fan") and not plan_parts.is_rotating("belt guard")


def test_at_most_four_parts():
    p = validate(raw(["door", "control panel", "kick panel", "detergent drawer", "top panel"]), 240)
    assert len(p["parts"]) == 4 and p["parts"][0] == "washing machine"


def test_rpm_sorted_widened_and_clamped():
    assert validate(raw([], 1600, 400), 240)["rpm_range"] == [400.0, 1600.0]
    narrow = validate(raw([], 990, 1010), 240)
    assert narrow["rpm_range"] == [800.0, 1200.0] and narrow["warnings"]
    clamped = validate(raw([], 3000, 9000), 240)
    assert clamped["rpm_range"][1] == pytest.approx(60 * 119) and clamped["rpm_range"][0] == 3000.0


def test_unmeasurable_rpm_range_falls_back_to_strongest_peak():
    p = validate(raw([], 20, 45), 240)
    assert p["rpm_range"] is None and "strongest peak" in p["warnings"][-1]


def test_missing_rpm_uses_default():
    p = validate({"machine_type": "pump", "parts": ["pump"]}, 240)
    assert p["rpm_range"] == list(plan_parts.DEFAULT_RPM)


def test_user_parts_override():
    p = validate(raw(["door"]), 240, user_parts=["Washing Machine", "detergent drawer"])
    assert p["parts"] == ["washing machine", "detergent drawer"] and p["parts_source"] == "user"


def test_prompt_has_no_em_dash():
    assert "—" not in plan_parts.PROMPT

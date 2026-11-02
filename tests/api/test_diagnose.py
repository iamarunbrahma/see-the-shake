"""Code verification of Spark's diagnosis, offline: evidence within one bin, orders that fit the claimed fault,
rule verdict as the headline, confidence capped at low on disagreement, dropped evidence or no support."""
import copy
import json
from pathlib import Path

import pytest

from shake import diagnose
from shake.meta_api import SPARK, Cassette, MetaAPI
from tests.api.live_fixtures import load_fixture

BIN = 0.1667
F1 = 16.667


def _peak(order, amp=0.2, label=None, hz=None):
    hz = hz if hz is not None else (order * F1 if order else 50.0)
    return {"hz": round(hz, 3), "amp_px": amp, "amp_mm": amp * 1.2, "order": order,
            "label": label or (f"{order:g}x" if order else "non-synchronous"), "phase": 0.0, "snr": 100.0}


def _result(peaks_x, peaks_y=()):
    axes = {"x": {"peaks": list(peaks_x), "orders": {}, "noise_floor_px": 1e-3},
            "y": {"peaks": list(peaks_y), "orders": {}, "noise_floor_px": 1e-3}}
    return {"bin_hz": BIN, "fps": 240, "seconds": 6.0, "nyquist_hz": 120, "flicker_hz": 100.0,
            "running_speed": {"hz": F1, "rpm": 1000, "source": "rpm_range"},
            "scale": {"mm_per_px": 1.2, "basis": "washing machine width = 600 mm over 500 px"},
            "parts": {"washing machine": {"axes": axes}, "door": {"axes": copy.deepcopy(axes)}}}


def _fp(verdict, conf="high"):
    return {"verdict": verdict, "confidence": conf, "reason": "test", "lead_part": "washing machine",
            "evidence": [], "parts": {}, "alert_level": 7.0, "alert_unit": "mm/s"}


def _claim(fault, evidence, conf="high", **kw):
    return {"fault": fault, "confidence": conf, "evidence": evidence, "explanation": kw.get("explanation", "x"),
            "fix": kw.get("fix", ["a"]), "check_next": ["b"]}


def ev(hz, order, part="washing machine"):
    return {"part": part, "frequency_hz": hz, "order": order}


IMB = _result([_peak(1, 0.3), _peak(2, 0.02)], [_peak(1, 0.12)])


def test_agreeing_supported_claim_keeps_its_confidence():
    card = diagnose.verify(_claim("imbalance", [ev(16.667, "1x"), ev(16.70, "1x", "washing machine")]), IMB,
                           _fp("imbalance"))
    sp = card["spark"]
    assert card["headline"]["fault"] == "imbalance" and card["headline"]["source"] == "fixed rules"
    assert sp["counts"] == {"verified": 2, "context": 0, "dropped": 0}
    assert sp["supported"] and sp["agrees"] and sp["confidence"] == "high" and not sp["capped"]
    assert not card["show_both"]


def test_peak_more_than_one_bin_away_is_dropped_and_caps_confidence():
    card = diagnose.verify(_claim("imbalance", [ev(16.667, "1x"), ev(16.667 + 1.5 * BIN, "1x")]), IMB,
                           _fp("imbalance"))
    sp = card["spark"]
    assert [i["status"] for i in sp["evidence"]] == ["verified", "dropped"]
    assert "within one bin" in sp["evidence"][1]["reason"]
    assert sp["confidence"] == "low" and sp["capped"] and card["show_both"]


def test_fabricated_frequency_and_unknown_part_are_dropped():
    card = diagnose.verify(_claim("imbalance", [ev(12.0, "1x"), ev(16.667, "1x", "motor")]), IMB, _fp("imbalance"))
    sp = card["spark"]
    assert sp["counts"]["dropped"] == 2 and not sp["supported"]
    assert sp["confidence"] == "low" and len(sp["cap_reasons"]) == 2


def test_wrong_order_label_is_dropped():
    card = diagnose.verify(_claim("imbalance", [ev(16.667, "1x"), ev(33.334, "1x")]), IMB, _fp("imbalance"))
    item = card["spark"]["evidence"][1]
    assert item["status"] == "dropped" and "measured peak is 2x" in item["reason"]


def test_real_peak_cited_for_the_wrong_fault_does_not_count():
    # looseness claimed on a 1x peak: real, but looseness does not rest on 1x
    card = diagnose.verify(_claim("looseness", [ev(16.667, "1x")]), IMB, _fp("imbalance"))
    sp = card["spark"]
    assert sp["evidence"][0]["status"] == "context"
    assert not sp["supported"] and not sp["agrees"] and sp["confidence"] == "low"
    assert any("needs" in r for r in sp["cap_reasons"]) and any("disagrees" in r for r in sp["cap_reasons"])


def test_disagreement_caps_even_with_real_evidence():
    mis = _result([_peak(1, 0.12), _peak(2, 0.2)])
    card = diagnose.verify(_claim("misalignment", [ev(33.334, "2x"), ev(16.667, "1x")], "medium"), mis,
                           _fp("imbalance"))
    sp = card["spark"]
    assert [i["status"] for i in sp["evidence"]] == ["verified", "context"]
    assert sp["supported"] and not sp["agrees"]
    assert sp["confidence"] == "low" and sp["confidence_reported"] == "medium"
    assert card["headline"]["fault"] == "imbalance" and card["show_both"]


def test_looseness_needs_half_order_or_a_row_of_harmonics():
    loose = _result([_peak(1), _peak(2, 0.1), _peak(3, 0.06), _peak(4, 0.03), _peak(0.5, 0.05)])
    two = diagnose.verify(_claim("looseness", [ev(33.334, "2x"), ev(50.0, "3x")]), loose, _fp("looseness"))
    assert not two["spark"]["supported"] and two["spark"]["confidence"] == "low"
    row = diagnose.verify(_claim("looseness", [ev(33.334, "2x"), ev(50.0, "3x"), ev(66.668, "4x")]), loose,
                          _fp("looseness"))
    assert row["spark"]["supported"] and row["spark"]["confidence"] == "high"
    half = diagnose.verify(_claim("looseness", [ev(8.334, "0.5x")]), loose, _fp("looseness"))
    assert half["spark"]["supported"]


def test_flicker_peak_is_dropped():
    r = _result([_peak(1), _peak(None, 0.01, "flicker", hz=100.0)])
    card = diagnose.verify(_claim("imbalance", [ev(16.667, "1x"), ev(100.0, "6x")]), r, _fp("imbalance"))
    assert card["spark"]["evidence"][1]["status"] == "dropped"
    assert "flicker" in card["spark"]["evidence"][1]["reason"]


def test_none_verdict_needs_no_evidence_and_text_has_no_em_dash():
    card = diagnose.verify(_claim("healthy", [], "medium", explanation="Fine \u2014 nothing to do",
                                  fix=["Keep \u2014 as is"]), IMB, _fp("none"))
    sp = card["spark"]
    assert sp["fault"] == "none" and sp["agrees"] and sp["supported"] and sp["confidence"] == "medium"
    assert "\u2014" not in sp["explanation"] and "\u2014" not in sp["fix"][0]


def test_order_parsing_and_fault_synonyms():
    assert diagnose.parse_order("1x") == 1.0 and diagnose.parse_order("0.5X") == 0.5
    assert diagnose.parse_order("2") == 2.0 and diagnose.parse_order("non-synchronous") is None
    assert diagnose.normalise_fault("Unbalance") == "imbalance"
    assert diagnose.normalise_fault("mechanical looseness") == "looseness"
    assert diagnose.normalise_fault("shaft misalignment") == "misalignment"
    assert diagnose.normalise_fault("???") == "unclear"


def test_failed_spark_call_keeps_the_rule_headline(tmp_path):
    c = Cassette(tmp_path / "c.json", "record")
    c.add("spark", SPARK, "diag", {}, {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                                       "output": [], "usage": {"input_tokens": 10, "output_tokens": 16384}})
    api = MetaAPI("replay", cassette=tmp_path / "c.json")
    fp = _fp("imbalance")
    fp["parts"] = {}
    card = diagnose.diagnose(api, IMB, {**fp, "phase_1x_x_rad": {}}, label="diag")
    assert card["headline"]["fault"] == "imbalance" and "error" in card["spark"] and card["show_both"]
    assert card["usd"] > 0


def test_peak_table_lists_every_peak():
    fx = load_fixture("imbalance")
    text = diagnose.peak_table(fx["measured"], fx["fingerprint"])
    n = sum(len(e["axes"][a]["peaks"]) for e in fx["measured"]["parts"].values() for a in "xy")
    assert text.count(" | ") >= 5 * n and "Official rule verdict" in text and "\u2014" not in text


def _obs(part, direction):
    return {"most_moving_part": part, "direction": direction, "description": "It sways — mostly sideways."}


def test_visual_observation_checked_against_in_band_motion():
    r = _result([_peak(1, 0.30), _peak(2, 0.05)], [_peak(1, 0.10)])  # in band: x 0.30, y 0.10 -> horizontal
    r["parts"]["door"]["axes"]["x"]["peaks"] = [_peak(1, 0.10)]
    v = diagnose.verify_visual(_obs("washing machine", "horizontal"), r)
    assert v["status"] == "verified" and [c["status"] for c in v["checks"]] == ["verified", "verified"]
    assert "—" not in v["description"]
    wrong = diagnose.verify_visual(_obs("door", "vertical"), r)
    assert wrong["status"] == "not verified"
    rock = diagnose.verify_visual(_obs("washing machine", "rocking"), r)
    assert [c["status"] for c in rock["checks"]] == ["verified", "unverifiable"] and rock["status"] == "partly verified"


def test_visual_observation_ignores_peaks_outside_the_band():
    # a big 2x is outside the magnified band, so the video shows the 1x motion: mostly vertical here
    r = _result([_peak(1, 0.05), _peak(2, 0.4)], [_peak(1, 0.2)])
    v = diagnose.verify_visual(_obs("washing machine", "vertical"), r)
    assert v["checks"][1]["status"] == "verified" and v["checks"][1]["measured"]["horizontal_share"] < 0.4


def test_visual_observation_missing_or_unknown_part():
    assert diagnose.verify_visual(None, IMB)["status"] == "not provided"
    v = diagnose.verify_visual(_obs("drum", "horizontal"), IMB)
    assert v["checks"][0]["status"] == "not_verified"


def test_video_schema_only_when_the_video_is_sent():
    assert "visual_observation" not in diagnose.SCHEMA["properties"]
    assert "visual_observation" in diagnose.SCHEMA_WITH_VIDEO["required"]
    from shake import magnify
    assert diagnose.BAND_ORDERS == tuple(magnify.DEFAULT_BAND)


VISUAL = Path(__file__).parent / "fixtures" / "visual"


@pytest.mark.parametrize("fault", ["imbalance", "looseness", "misalignment"])
def test_replay_diagnosis_with_video(fault):
    """Recorded live diagnoses that sent the magnified clip: the visual observation and its checks replay."""
    d = VISUAL / fault
    if not (d / "card.json").exists():
        pytest.skip("no recording")
    load = lambda n: json.loads((d / n).read_text())  # noqa: E731
    api = MetaAPI("replay", cassette=d / "cassette.json.gz")
    card = diagnose.diagnose(api, load("measured.json"), load("fingerprint.json"), plan=load("plan.json"),
                             magnified_clip=Path("side_by_side.mp4"), label=f"{fault}:diagnose")
    want = load("card.json")
    assert card["spark"] == want["spark"] and card["spark"]["visual"]["checks"]
    assert card["spark"]["visual"]["checks"][1]["status"] == "verified"  # direction, read from the video


# ---------------------------------------------------------------- compression and flicker artifacts

def _fp_with_artifacts(verdict, artifacts, part="washing machine"):
    fp = _fp(verdict)
    fp["parts"] = {part: {"features": {"artifacts": artifacts}}}
    return fp


ART_RESULT = _result([_peak(1, 0.3), _peak(5, 0.01, hz=83.36)], [_peak(1, 0.1)])


def test_claim_on_an_artifact_peak_is_not_verified():
    fp = _fp_with_artifacts("imbalance", [{"axis": "x", "hz": 83.36, "label": "5x", "amp_px": 0.01,
                                           "bg_amp_px": 0.008, "kind": "whole frame"}])
    card = diagnose.verify(_claim("imbalance", [ev(16.667, "1x"), ev(83.36, "5x")]), ART_RESULT, fp)
    item = card["spark"]["evidence"][1]
    assert item["status"] == "dropped" and item["matched"]["artifact"] == "whole frame"
    assert "background moves the same way" in item["reason"]
    assert card["spark"]["confidence"] == "low" and "1 evidence item(s) dropped" in card["spark"]["cap_reasons"]


def test_a_real_peak_in_the_same_bin_still_counts():
    # the artifact is on the horizontal axis; the vertical peak at the same frequency is real
    r = _result([_peak(1, 0.3), _peak(5, 0.01, hz=83.36)], [_peak(1, 0.1), _peak(5, 0.02, hz=83.36)])
    fp = _fp_with_artifacts("imbalance", [{"axis": "x", "hz": 83.36, "label": "5x", "amp_px": 0.01,
                                           "kind": "flicker sideband"}])
    item = diagnose.verify(_claim("imbalance", [ev(83.36, "5x")]), r, fp)["spark"]["evidence"][0]
    assert item["status"] == "context" and item["matched"]["axis"] == "y"


def test_prompt_lists_artifacts_apart_without_order_labels():
    fp = _fp_with_artifacts("imbalance", [{"axis": "x", "hz": 83.36, "label": "5x", "amp_px": 0.01,
                                           "kind": "flicker sideband"}])
    fp["parts"]["washing machine"].update(verdict="imbalance", confidence="high", reason="r", features={
        "artifacts": fp["parts"]["washing machine"]["features"]["artifacts"], "velocity_rms": 9.0,
        "velocity_unit": "mm/s", "e1_share": 0.9, "r21": 0.1, "r31": 0.0, "harmonics": [], "half_order": False,
        "half_ratio": 0.0, "h_over_v_1x": 3.0})
    text = diagnose.peak_table(ART_RESULT, {**fp, "phase_1x_x_rad": {}})
    labelled = text.split("Left out by the rules")[0]
    assert "washing machine | x (horizontal) | 83.360 | 5x" not in labelled
    assert "door | x (horizontal) | 83.360 | 5x" in labelled  # the door's peak was not left out
    assert "washing machine | x (horizontal) | 83.360 | flicker sideband" in text


def test_visual_check_ignores_artifacts_in_the_band():
    r = _result([_peak(1, 0.1), _peak(None, 0.5, "non-synchronous", hz=18.0)], [_peak(1, 0.2)])
    obs = {"most_moving_part": "washing machine", "direction": "vertical", "description": ""}
    art = {"washing machine": {("x", 18.0): "whole frame"}}
    assert diagnose.verify_visual(obs, r)["checks"][1]["status"] == "not_verified"  # the artifact dominates x
    assert diagnose.verify_visual(obs, r, artifacts=art)["checks"][1]["status"] == "verified"


def test_live_upload_artifact_citations_are_dropped():
    """A live diagnosis of the phone-like healthy clip (Nov 2) cited encoder peaks labelled 5x and 7x. With the
    artifact rules, they are dropped and Spark's confidence is capped; its 1x items stay context."""
    from shake import fingerprint
    d = Path(__file__).parent / "fixtures" / "artifacts"
    res = json.loads((d / "healthy_phone_upload_measured.json").read_text())
    claim = json.loads((d / "healthy_phone_upload_spark.json").read_text())
    fp = fingerprint.fingerprint(res)
    assert fp["verdict"] == "none" and fp["parts"][fp["lead_part"]]["features"]["velocity_rms"] < 0.5 * 7.0
    sp = diagnose.verify(claim, res, fp)["spark"]
    status = {round(e["frequency_hz"]): e["status"] for e in sp["evidence"]}
    assert status[83] == status[117] == "dropped" and status[17] == "context"
    assert sp["confidence"] == "low" and sp["agrees"]


# ---------------------------------------------------------------- blade pass on a fan

F1_FAN = 21.5


def _fan_result():
    r = _result([_peak(1, 0.05, hz=F1_FAN), _peak(3, 0.4, hz=3 * F1_FAN)], [_peak(1, 0.02, hz=F1_FAN)])
    r["running_speed"]["hz"] = F1_FAN
    return r


def _fan_fp(verdict, blades=3):
    fp = _fp(verdict)
    fp["blades"] = blades
    return fp


def test_spark_can_agree_with_blade_pass():
    card = diagnose.verify(_claim("blade pass", [ev(3 * F1_FAN, "3x")]), _fan_result(), _fan_fp("blade_pass"))
    sp = card["spark"]
    assert sp["fault"] == "blade_pass" and sp["agrees"] and sp["supported"]
    assert sp["evidence"][0]["status"] == "verified" and sp["confidence"] == "high" and not card["show_both"]
    assert "blade_pass" in diagnose.SCHEMA["properties"]["fault"]["enum"]


def test_blade_pass_is_not_evidence_for_a_fault_pattern():
    # Spark calls the 3x blade pass a harmonic for looseness: it is context, and the claim is unsupported
    card = diagnose.verify(_claim("looseness", [ev(3 * F1_FAN, "3x")]), _fan_result(), _fan_fp("blade_pass"))
    item = card["spark"]["evidence"][0]
    assert item["status"] == "context" and "blade pass" in item["reason"]
    assert not card["spark"]["supported"] and card["spark"]["confidence"] == "low"


def test_blade_pass_needs_the_blade_count():
    card = diagnose.verify(_claim("blade_pass", [ev(3 * F1_FAN, "3x")]), _fan_result(), _fan_fp("none", blades=None))
    sp = card["spark"]
    assert not sp["supported"] and sp["confidence"] == "low"
    assert any("blade count" in r for r in sp["cap_reasons"])


def test_prompt_explains_blade_pass_when_the_count_is_known():
    r = _fan_result()
    assert "3x (64.5 Hz) is blade pass" in diagnose.blade_note(_fan_fp("none"), r)
    assert diagnose.blade_note(_fan_fp("none", blades=None), r) == ""

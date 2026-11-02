"""Code verification of Spark's diagnosis, offline: evidence within one bin, orders that fit the claimed fault,
rule verdict as the headline, confidence capped at low on disagreement, dropped evidence or no support."""
import copy

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
    card = diagnose.verify(_claim("healthy", [], "medium", explanation="Fine — nothing to do",
                                  fix=["Keep — as is"]), IMB, _fp("none"))
    sp = card["spark"]
    assert sp["fault"] == "none" and sp["agrees"] and sp["supported"] and sp["confidence"] == "medium"
    assert "—" not in sp["explanation"] and "—" not in sp["fix"][0]


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
    assert text.count(" | ") >= 5 * n and "Official rule verdict" in text and "—" not in text

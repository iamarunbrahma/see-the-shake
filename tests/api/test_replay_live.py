"""Replay each recorded live twin run (scripts/live_twin.py) offline and check it reproduces the report:
the plan, the SAM masks' IoU against the renderer's masks, and the verified diagnosis card. No network calls."""
import json
from pathlib import Path

import pytest

from shake import diagnose, plan_parts, segment, twin
from shake.meta_api import MetaAPI
from tests.api.live_fixtures import LIVE, load_fixture


@pytest.fixture(scope="module")
def renderer_masks():
    return twin.part_masks()[0]


@pytest.mark.parametrize("fault", list(twin.FAULTS))
def test_replay_live_run(fault, renderer_masks):
    fx = load_fixture(fault)
    rep = fx["report"]
    api = MetaAPI("replay", cassette=fx["cassette_path"])

    fid = api.upload(Path("preview.mp4"), label=f"{fault}:plan:preview")
    plan = plan_parts.plan_from_file(api, fid, rep["fps"], 4.0, label=f"{fault}:plan")
    assert plan["parts"] == rep["plan"]["parts"] and plan["rpm_range"] == rep["plan"]["rpm_range"]

    fid = api.upload(Path("sam_subclip.mp4"), label=f"{fault}:segment:subclip")
    shape, frames = (twin.H, twin.W), rep["segment"]["subclip"]["frames"]
    masks, infos = segment.segment_file(api, fid, plan["parts"], shape, frames, plan["fallback_phrase"])
    for p in rep["segment"]["added_for_iou"]:
        m, _ = segment.segment_phrase(api, fid, p, shape, frames)
        if m is not None:
            masks.setdefault(p, m)
    for part, gt in renderer_masks.items():
        want = rep["iou"][part]
        got = max((segment.iou(m, gt) for m in masks.values()), default=0.0)
        assert got == pytest.approx(want["iou"], abs=2e-3), part
        if want["phrase"]:
            assert segment.iou(masks[want["phrase"]], gt) == pytest.approx(want["iou"], abs=2e-3)

    card = diagnose.diagnose(api, fx["measured"], fx["fingerprint"], plan=plan, label=f"{fault}:diagnose")
    assert card["headline"] == rep["card"]["headline"]
    assert card["spark"] == rep["card"]["spark"]
    assert api.cassette.pos == len(api.cassette.items), "every recorded call is replayed"
    assert api.ledger.total() == pytest.approx(rep["costs_usd"]["total"], abs=1e-5)


def test_live_rows_meet_the_design():
    """The recorded runs: masks close to the renderer's, the rule verdict as the headline, Spark capped at low
    whenever it disagrees or evidence was dropped, and a small cost per run. (Whether the rule verdict is right
    is reported in the live table, not asserted: see rule_ok in report.json.)"""
    reps = [json.loads((LIVE / f / "report.json").read_text()) for f in twin.FAULTS if (LIVE / f / "report.json").exists()]
    if not reps:
        pytest.skip("no live recordings")
    for r in reps:
        assert r["iou"]["washing machine"]["iou"] > 0.9 and r["iou"]["door"]["iou"] > 0.9
        assert r["costs_usd"]["total"] < 0.25
        assert r["card"]["headline"]["fault"] == r["rule_verdict"]
        if not r["spark"]["agrees"] or r["spark"]["counts"]["dropped"]:
            assert r["spark"]["confidence"] == "low"

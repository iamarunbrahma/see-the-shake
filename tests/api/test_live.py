"""Live Meta Model API tests on the short test twin clips. They cost a few cents; run with: pytest -m live"""
import json

import pytest

from shake import diagnose, plan_parts, segment
from shake.meta_api import MetaAPI
from tests.conftest import load_masks, twin_clip

pytestmark = pytest.mark.live


@pytest.fixture()
def api(tmp_path):
    return MetaAPI("live", ledger=tmp_path / "ledger.jsonl")


def test_live_sam_masks_the_twin(api):
    d = twin_clip("imbalance")
    gt = load_masks(d, json.loads((d / "truth.json").read_text()))
    # the bare phrase "door" finds nothing on the twin; Spark's phrases ("round front door") do
    door = "round front door"
    masks, seg = segment.segment_parts(api, d / "clip.mp4", ["washing machine", door], gt["door"].shape,
                                       start_frame=60)
    assert segment.iou(masks["washing machine"], gt["washing machine"]) > 0.9
    assert segment.iou(masks[door], gt["door"]) > 0.85
    assert seg["parts"][door]["frames_billed"] == segment.SUB_FRAMES
    assert seg["usd"] == pytest.approx(2 * segment.SUB_FRAMES * 0.0002)


def test_live_plan_names_the_machine(api):
    d = twin_clip("imbalance")
    plan = plan_parts.plan_parts(api, d / "clip.mp4", 240)
    assert "washing machine" in plan["machine_type"] and plan["parts"][0] == plan["machine_type"]
    assert 1 <= len(plan["parts"]) <= 4
    assert plan["usd"] < 0.05


def test_live_diagnosis_is_verified(api, twin_runs):
    run = twin_runs["imbalance"]
    card = diagnose.diagnose(api, run["result"], run["fp"], label="live_imbalance")
    assert "error" not in card["spark"], card["spark"]
    assert card["headline"]["fault"] == "imbalance"
    if not card["spark"]["agrees"] or card["spark"]["counts"]["dropped"]:
        assert card["spark"]["confidence"] == "low"

"""Replay the live fan twin runs (pipeline, Nov 2) offline: Spark's plan with its blade count, the SAM 3.1 masks,
and the diagnosis with the magnified clip. These are the recordings the web app offers as recorded fan runs."""
import gzip
import json
from pathlib import Path

import pytest

from shake import diagnose, plan_parts, segment, twin_fan
from shake.meta_api import MetaAPI

LIVE = Path(__file__).parent / "fixtures" / "live"


@pytest.mark.parametrize("fault", list(twin_fan.FAULTS))
def test_replay_fan_recording(fault):
    d = LIVE / f"fan-{fault}"
    if not (d / "results.json").exists():
        pytest.skip(f"no recording for fan-{fault}")
    r = json.loads((d / "results.json").read_text())
    cas = json.loads(gzip.decompress((d / "cassette.json.gz").read_bytes()))
    label = f"fan-{fault}"
    assert cas[0]["key"] == f"{label}:plan:preview"
    api = MetaAPI("replay", cassette=d / "cassette.json.gz")
    m = r["measurement"]

    fid = api.upload(Path("preview.mp4"), label=f"{label}:plan:preview")
    plan = plan_parts.plan_from_file(api, fid, m["fps"], 4.0, label=f"{label}:plan")
    assert plan["parts"] == r["plan"]["parts"] and plan["blades"] == twin_fan.BLADES

    fid = api.upload(Path("sam_subclip.mp4"), label=f"{label}:segment:subclip")
    shape = tuple(r["ingest"]["clip"][k] for k in ("height", "width"))
    masks, infos = segment.segment_file(api, fid, plan["parts"], shape, r["segment"]["subclip"]["frames"],
                                        plan["fallback_phrase"])
    for p, info in r["segment"]["parts"].items():
        if info.get("found"):
            assert infos[p]["area_px"] == info["area_px"], p

    card = diagnose.diagnose(api, m, r["fingerprint"], plan=plan, magnified_clip=Path("side_by_side.mp4"),
                             label=f"{label}:diagnose", band_orders=tuple(r["magnify"]["band_orders"]))
    assert card["spark"] == r["diagnosis"]["spark"]
    assert api.cassette.pos == len(api.cassette.items), "every recorded call is replayed"
    # the recorded run against the twin's truth
    assert r["fingerprint"]["verdict"] == r["truth"]["expected_verdict"] == twin_fan.EXPECTED_VERDICT[fault]
    assert card["spark"]["agrees"] and card["spark"]["visual"]["status"] == "verified"
    assert r["sam_iou"]["fan guard rim"]["iou"] > 0.6 and r["sam_iou"]["fan stand"]["iou"] > 0.8

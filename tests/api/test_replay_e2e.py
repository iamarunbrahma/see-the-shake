"""Replay the live end-to-end pipeline runs recorded after the artifact rules were merged (Nov 2, round 2):
the plan, the SAM masks against the renderer's, and the diagnosis with the magnified clip and its visual check.
No network calls."""
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from shake import diagnose, fingerprint, plan_parts, segment, twin
from shake.meta_api import MetaAPI

E2E = Path(__file__).parent / "fixtures" / "live_e2e"
RUNS = ["imbalance", "healthy", "looseness", "misalignment", "healthy_phone"]


def load(name):
    d = E2E / name
    if not (d / "results.json").exists():
        pytest.skip(f"no recording for {name}")
    r = json.loads((d / "results.json").read_text())
    cas = json.loads(gzip.decompress((d / "cassette.json.gz").read_bytes()))
    return d, r, cas


@pytest.fixture(scope="module")
def full_res_masks():
    return twin.part_masks()[0]


def masks_at(full, size):
    w, h = size
    if (w, h) == (twin.W, twin.H):
        return full
    return {k: ((cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA) > 127) * 255).astype(np.uint8)
            for k, m in full.items()}


@pytest.mark.parametrize("name", RUNS)
def test_replay_pipeline_recording(name, full_res_masks):
    d, r, cas = load(name)
    label = cas[0]["key"].rsplit(":plan:preview", 1)[0]
    api = MetaAPI("replay", cassette=d / "cassette.json.gz")

    fid = api.upload(Path("preview.mp4"), label=f"{label}:plan:preview")
    plan = plan_parts.plan_from_file(api, fid, r["measurement"]["fps"], 4.0, label=f"{label}:plan")
    assert plan["parts"] == r["plan"]["parts"] and plan["rpm_range"] == r["plan"]["rpm_range"]

    size = (r.get("clip") or {}).get("size", [twin.W, twin.H])
    fid = api.upload(Path("sam_subclip.mp4"), label=f"{label}:segment:subclip")
    masks, _ = segment.segment_file(api, fid, plan["parts"], (size[1], size[0]), r["segment"]["subclip"]["frames"],
                                    plan["fallback_phrase"])
    want = r.get("iou_vs_renderer") or {k: v["iou"] for k, v in r["sam_iou"].items()}
    for part, gt in masks_at(full_res_masks, size).items():
        got = max(segment.iou(m, gt) for m in masks.values())
        assert got == pytest.approx(want[part], abs=2e-3), part

    card = diagnose.diagnose(api, r["measurement"], r["fingerprint"], plan=plan,
                             magnified_clip=Path("side_by_side.mp4"), label=f"{label}:diagnose",
                             band_orders=tuple(r["magnify"]["band_orders"]))
    assert card["spark"] == r["diagnosis"]["spark"] and card["spark"]["visual"]["checks"]
    assert card["headline"]["fault"] == r["fingerprint"]["verdict"]
    assert api.cassette.pos == len(api.cassette.items), "every recorded call is replayed"


def test_phone_like_healthy_reads_well_below_the_alert_and_ignores_artifacts():
    _, r, cas = load("healthy_phone")
    fp = fingerprint.fingerprint(r["measurement"])  # the current rules give the recorded result
    assert fp["verdict"] == r["fingerprint"]["verdict"] == "none"
    lead = fp["parts"][fp["lead_part"]]["features"]
    assert lead["velocity_rms"] < 0.5 * fp["alert_level"] and lead["artifacts"]
    # Spark was shown the artifacts apart from the labelled peaks, and none of its evidence rests on one
    prompt = json.dumps(next(c for c in cas if c["op"] == "spark" and c["key"].endswith(":diagnose"))["request"])
    assert "Left out by the rules" in prompt
    assert not [e for e in r["diagnosis"]["spark"]["evidence"]
                if (e.get("matched") or {}).get("artifact") and e["status"] != "dropped"]

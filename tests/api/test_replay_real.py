"""Replay the live runs on real footage (samples/real/) offline: Spark's plan, the SAM 3.1 masks (their recorded
areas), and the diagnosis with its visual check, against the recorded measurement. No network calls."""
import gzip
import json
from pathlib import Path

import pytest

from shake import diagnose, plan_parts, segment
from shake.meta_api import MetaAPI

REAL = Path(__file__).parent / "fixtures" / "real"
RUNS = sorted(p.name for p in REAL.iterdir() if (p / "results.json").exists()) if REAL.exists() else []


@pytest.mark.parametrize("name", RUNS)
def test_replay_real_recording(name):
    d = REAL / name
    r = json.loads((d / "results.json").read_text())
    cas = json.loads(gzip.decompress((d / "cassette.json.gz").read_bytes()))
    label = cas[0]["key"].rsplit(":plan:preview", 1)[0]
    api = MetaAPI("replay", cassette=d / "cassette.json.gz")
    m = r["measurement"]

    fid = api.upload(Path("preview.mp4"), label=f"{label}:plan:preview")
    user = r["options"].get("user_parts")
    plan = plan_parts.plan_from_file(api, fid, m["fps"], 4.0, user_parts=user, label=f"{label}:plan")
    assert plan["parts"] == r["plan"]["parts"]

    fid = api.upload(Path("sam_subclip.mp4"), label=f"{label}:segment:subclip")
    seg = r["segment"]
    shape = tuple(r["ingest"]["clip"][k] for k in ("height", "width"))
    # the pipeline also segments the whole machine when it is not one of the parts (recordings from 54492b8 on)
    machine = plan["fallback_phrase"]
    context = [machine] if machine not in plan["parts"] and any(
        c["op"] == "sam" and c["key"] == machine for c in cas) else []
    masks, infos = segment.segment_file(api, fid, plan["parts"] + context, shape, seg["subclip"]["frames"],
                                        plan["fallback_phrase"])
    for p, info in seg["parts"].items():
        if info.get("found"):
            assert infos[p]["area_px"] == info["area_px"], p

    card = diagnose.diagnose(api, m, r["fingerprint"], plan=plan, magnified_clip=Path("side_by_side.mp4"),
                             label=f"{label}:diagnose", band_orders=tuple(r["magnify"]["band_orders"]))
    assert card["spark"] == r["diagnosis"]["spark"]
    assert card["headline"]["fault"] == r["fingerprint"]["verdict"]
    assert api.cassette.pos == len(api.cassette.items), "every recorded call is replayed"

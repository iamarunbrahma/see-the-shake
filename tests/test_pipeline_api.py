"""The pipeline with Muse Spark and SAM 3.1, end to end, on recorded live responses only (no network).

The recorded runs (tests/api/fixtures/live, written by scripts/live_twin.py) were made on 6 s, 1920x1080, seed 1
twin clips. Replay checks each call's operation, model and label, so this also proves the pipeline issues the
same calls in the same order. A 1 s clip of the same render keeps the test fast: SAM's masks and Spark's answers
come from the recording, and measurement runs on the shorter clip.
"""
import json

import pytest

from shake import pipeline
from shake.meta_api import APIError, CreditExhausted

from .conftest import CACHE, twin_clip

RUNS = CACHE / "runs"
FAULT = "imbalance"


@pytest.fixture(scope="module")
def replayed():
    seen = []
    r = pipeline.run_twin(FAULT, RUNS, seconds=1.0, api="replay", progress=lambda k, s, d: seen.append((k, s)))
    report = json.loads((pipeline.RECORDINGS / FAULT / "report.json").read_text())
    return r, report, seen


def test_all_seven_steps_run_on_recorded_calls(replayed):
    r, report, _ = replayed
    assert r["state"] == "done"
    assert {s["key"]: s["status"] for s in r["steps"]} == {k: "done" for k, _ in pipeline.STEPS}
    assert r["plan"]["parts"] == report["plan"]["parts"] and r["plan"]["rpm_range"] == report["plan"]["rpm_range"]
    assert r["segment"]["missing"] == report["segment"]["missing"]
    assert r["mask_source"] == "SAM 3.1" and set(r["files"]["masks"]) == set(report["plan"]["parts"]) - set(
        report["segment"]["missing"])


def test_sam_masks_match_the_recording_and_the_renderer(replayed):
    r, report, _ = replayed
    for part, want in report["iou"].items():
        assert r["sam_iou"][part]["iou"] == pytest.approx(want["iou"], abs=2e-3)
        assert r["sam_iou"][part]["iou"] > 0.9
    tc = r["truth_comparison"]
    assert tc["parts_matched_by_iou"]["door"] == report["iou"]["door"]["phrase"]
    assert tc["verdict_ok"] and tc["scored"] > 0


def test_diagnosis_is_verified_and_the_rules_stay_the_headline(replayed):
    r, report, _ = replayed
    dg = r["diagnosis"]
    assert dg["status"] == "done" and dg["headline"]["fault"] == r["fingerprint"]["verdict"] == "imbalance"
    sp = dg["spark"]
    assert sp["fault"] == report["card"]["spark"]["fault"]
    assert {e["status"] for e in sp["evidence"]} <= {"verified", "context", "dropped"}
    assert sp["counts"]["verified"] >= 1 and all(e["reason"] for e in sp["evidence"])
    if not sp["agrees"] or sp["counts"]["dropped"]:
        assert sp["confidence"] == "low"


def test_cost_is_the_recorded_cost(replayed):
    r, report, _ = replayed
    api = r["api"]
    assert api["mode"] == "replay" and api["calls"]["calls"] == 7
    assert api["usd"]["total"] == pytest.approx(report["costs_usd"]["total"], abs=1e-5)
    for k in ("plan", "segment", "diagnose"):
        assert api["usd"][k] == pytest.approx(report["costs_usd"][k], abs=1e-5)


def _small_twin():
    d = twin_clip(FAULT)
    truth = json.loads((d / "truth.json").read_text())
    return d / "clip.mp4", {n: d / truth["parts"][n]["mask"] for n in truth["parts"]}, truth


def test_supplied_masks_are_the_fallback_when_sam_fails(monkeypatch, tmp_path):
    from shake import segment

    def broken(*a, **k):
        raise APIError("SAM 3.1 unavailable (test)")
    monkeypatch.setattr(segment, "segment_parts", broken)
    clip, masks, truth = _small_twin()
    opts = pipeline.twin_options(truth, api="replay", cassette=str(pipeline.recording(FAULT)), api_label=FAULT,
                                 spark_video=False, video_seconds=0.5, max_seconds=1.0)
    r = pipeline.run(clip, masks, tmp_path, opts, truth=truth, mask_source="twin renderer")
    status = {s["key"]: s["status"] for s in r["steps"]}
    assert status["plan"] == "done" and status["segment"] == "fallback"
    assert r["mask_source"] == "twin renderer" and set(r["files"]["masks"]) == set(masks)
    assert any("SAM 3.1 unavailable" in w for w in r["warnings"])
    assert r["fingerprint"]["verdict"] == "imbalance"
    # the recording no longer lines up after the skipped SAM calls, so Spark's diagnosis fails; the rules stand
    assert status["diagnose"] == "fallback" and r["diagnosis"]["headline"]["fault"] == "imbalance"


def test_402_stops_the_run(monkeypatch, tmp_path):
    from shake import plan_parts

    def no_credit(*a, **k):
        raise CreditExhausted("402 payment required", status=402)
    monkeypatch.setattr(plan_parts, "plan_parts", no_credit)
    clip, masks, truth = _small_twin()
    opts = pipeline.twin_options(truth, api="replay", cassette=str(pipeline.recording(FAULT)), api_label=FAULT)
    with pytest.raises(CreditExhausted):
        pipeline.run(clip, masks, tmp_path, opts, truth=truth, mask_source="twin renderer")
    res = json.loads(next(tmp_path.glob("*/results.json")).read_text())
    assert res["state"] == "failed" and res["error"]["step"] == "plan" and res["error"]["message"] == pipeline.CREDIT_MESSAGE
    assert "Arun" not in res["error"]["message"] and "trace" not in res["error"]


def test_worker_processes_start_from_stdin():
    """segment decodes masks in a forkserver pool; from stdin (no __main__ file) the workers used to die."""
    import subprocess
    import sys
    from .conftest import ROOT
    code = ("import multiprocessing as mp\n"
            "from concurrent.futures import ProcessPoolExecutor\n"
            "from shake import pipeline\n"
            "with pipeline._workers_can_start():\n"
            "    with ProcessPoolExecutor(2, mp_context=mp.get_context('forkserver')) as ex:\n"
            "        print(sorted(ex.map(abs, [-1, 2, -3])))\n"
            "import __main__; print(__main__.__file__)\n")
    r = subprocess.run([sys.executable, "-"], input=code, capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["[1, 2, 3]", "<stdin>"] and "Traceback" not in r.stderr


def test_recorded_rule_verdict_is_read_from_the_recording():
    """A replayed diagnosis carries the rule verdict Spark was shown; the page notes when today's rules differ."""
    from shake.meta_api import MetaAPI
    seen = {f: pipeline._recorded_rule_verdict(MetaAPI("replay", cassette=pipeline.recording(f)), f"{f}:diagnose")
            for f in ("imbalance", "misalignment")}
    assert seen == {"imbalance": "imbalance", "misalignment": "looseness"}  # misalignment predates the rule fix


def test_the_whole_machine_stays_out_of_the_background(monkeypatch, tmp_path):
    """When the parts are not the whole machine, it is segmented too and left out of the camera-shake reference,
    not measured."""
    from shake import plan_parts, segment
    clip, masks, truth = _small_twin()
    m = pipeline.load_masks(masks)
    plan = {"machine_type": "washing machine", "fallback_phrase": "washing machine", "parts": ["door"],
            "rpm_range": [900, 1100], "usd": 0.0}
    monkeypatch.setattr(plan_parts, "plan_parts", lambda *a, **k: dict(plan))
    asked = []

    def fake_segment(api, clip, phrases, shape=None, **k):
        asked.extend(phrases)
        return {p: m[p] for p in phrases}, {"parts": {}, "missing": [], "usd": 0.0}
    monkeypatch.setattr(segment, "segment_parts", fake_segment)
    opts = pipeline.Options(api="live", reference=("door", 350.0), min_seconds=1.0, max_seconds=1.0,
                            video_seconds=0.5, spark_video=False)
    monkeypatch.setattr(pipeline, "make_api", lambda o, out: _NoCallAPI())
    r = pipeline.run(clip, None, tmp_path, opts)
    assert asked == ["door", "washing machine"]
    assert list(r["measurement"]["parts"]) == ["door"] and r["segment"]["background_excludes"] == ["washing machine"]


class _NoCallAPI:
    """Stands in for MetaAPI where every model call is patched out; Spark's diagnosis fails and the rules stand."""
    from shake.meta_api import Ledger
    ledger = Ledger()
    cassette = None

    def spark(self, *a, **k):
        raise APIError("no API in this test")

    def upload(self, *a, **k):
        raise APIError("no API in this test")


def test_reference_by_height():
    assert pipeline.parse_reference("large metal disc=150:height") == ("large metal disc", 150.0, "height")
    assert pipeline.parse_reference("washing machine=600") == ("washing machine", 600.0)
    with pytest.raises(ValueError):
        pipeline.parse_reference("disc=150:depth")


def _replay_real():
    import importlib.util
    spec = importlib.util.spec_from_file_location("replay_real", pipeline.ROOT / "scripts" / "replay_real.py")
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    return rr


def test_real_rotor_replay_reads_imbalance_at_3000_rpm(tmp_path):
    """The rig at 3000 rpm with the eddy-current probes showing a clean 1x: replayed through today's pipeline
    (Spark and SAM 3.1 from the recording), the rules say imbalance and the frames check is inconclusive."""
    rec, now = _replay_real().replay("rotor-3000rpm-Unb1", tmp_path, rotor_ref_height=True)
    assert rec["fingerprint"]["verdict"] == "looseness"  # what the recording said
    assert now["fingerprint"]["verdict"] == "imbalance"
    assert now["ingest"]["interpolation"]["status"] == "inconclusive"
    assert "height" in now["measurement"]["scale"]["basis"]


def test_real_rotor_replay_at_1000_rpm_matches_the_probes(tmp_path):
    """At 1000 rpm the probes show 2x as large as 1x (2x/1x 0.96 to 1.48), 3x to 6x under 6% of 1x, and 3.7 to
    8.5 mm/s on one vertical axis. The recording read looseness from 3x to 6x that turning texture made up."""
    rr = _replay_real()
    rec, now = rr.replay("rotor-1000rpm-Unb2", tmp_path, rotor_ref_height=True)
    assert rec["fingerprint"]["verdict"] == "looseness"
    parts = now["measurement"]["parts"]
    shaft, disc = (next(e for p_, e in parts.items() if w in p_) for w in ("shaft", "disc"))
    assert shaft["excluded_texture_px"] > 0 and disc["excluded_texture_px"] > 0
    assert shaft["axis_confidence"]["x"] < 0.05  # a horizontal shaft has no edges to fix its horizontal motion
    y = shaft["axes"]["y"]["orders"]
    assert all(y[k]["amp_px"] < 0.1 * y["1x"]["amp_px"] for k in ("3x", "4x", "5x", "6x")), y
    fp = now["fingerprint"]
    lead = fp["parts"][fp["lead_part"]]["features"]
    assert 0.8 <= lead["r21"] <= 1.6 and 3.7 <= lead["velocity_rms"] <= 8.6, lead
    assert rr.pattern_verdict(now).startswith("misalignment")


def test_live_twin_checks_its_recordings_before_rendering(tmp_path, monkeypatch):
    """scripts/live_twin.py with the committed recordings in place used to render for 1 min 43 s, then stop."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("live_twin", pipeline.ROOT / "scripts" / "live_twin.py")
    lt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lt)
    monkeypatch.setattr(lt.twin, "render", lambda *a, **k: pytest.fail("rendered before checking the recordings"))
    assert lt.main(["--out", str(tmp_path), "--max-usd", "1.0"]) == 2

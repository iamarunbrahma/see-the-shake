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
    assert res["state"] == "failed" and res["error"]["step"] == "plan" and "tell Arun" in res["error"]["message"]


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

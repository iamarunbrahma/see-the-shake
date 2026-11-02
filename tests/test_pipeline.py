import json
import time

import pytest

from shake import pipeline

from .conftest import CACHE, twin_clip

RUNS = CACHE / "runs"


def twin_inputs(fault):
    d = twin_clip(fault)
    truth = json.loads((d / "truth.json").read_text())
    masks = {n: d / truth["parts"][n]["mask"] for n in truth["parts"]}
    return d / "clip.mp4", masks, truth


@pytest.fixture(scope="module")
def imbalance_run():
    clip, masks, truth = twin_inputs("imbalance")
    seen = []
    r = pipeline.run(clip, masks, RUNS, pipeline.twin_options(truth), truth=truth, mask_source="twin renderer",
                     progress=lambda k, s, d: seen.append((k, s)))
    return r, seen, (clip, masks, truth)


def test_results_and_files(imbalance_run):
    r, seen, _ = imbalance_run
    assert r["state"] == "done" and r["fingerprint"]["verdict"] == "imbalance"
    assert r["truth_comparison"]["verdict_ok"] and r["truth_comparison"]["passed"] == r["truth_comparison"]["scored"]
    out = RUNS / r["run_id"]
    for f in [r["files"]["side_by_side"], r["files"]["slice"], *r["files"]["spectra"].values(), *r["files"]["masks"].values()]:
        assert (out / f).stat().st_size > 500, f
    assert json.loads((out / "results.json").read_text())["run_id"] == r["run_id"]
    assert json.loads((out / "status.json").read_text())["state"] == "done"


def test_seven_steps_with_spark_and_sam_slots(imbalance_run):
    r, seen, _ = imbalance_run
    status = {s["key"]: s["status"] for s in r["steps"]}
    assert list(status) == [k for k, _ in pipeline.STEPS]
    assert {status[k] for k in ("ingest", "measure", "fingerprint", "magnify")} == {"done"}
    assert status["plan"] == status["segment"] == status["diagnose"] == "skipped"
    assert r["plan"]["status"] == "not_connected" and r["diagnosis"]["status"] == "not_connected"
    assert {k for k, _ in seen} == set(status)


def test_slots_take_callables(imbalance_run, tmp_path):
    """A plan and a diagnosis callable plug in without other changes (stand-ins here; no model calls)."""
    _, _, (clip, masks, truth) = imbalance_run
    plan = lambda clip, ing: {"machine": "washing machine", "rpm_range": [900, 1100], "parts": list(masks)}  # noqa: E731
    diag = lambda res: {"fault": res["fingerprint"]["verdict"], "source": "stand-in"}  # noqa: E731
    r = pipeline.run(clip, masks, tmp_path, pipeline.Options(reference=("washing machine", 600.0), min_seconds=2.0,
                                                              video_seconds=0.5), plan=plan, diagnose=diag)
    status = {s["key"]: s["status"] for s in r["steps"]}
    assert status["plan"] == "done" and status["diagnose"] == "done"
    assert r["measurement"]["running_speed"]["source"] == "rpm_range" and r["diagnosis"]["source"] == "stand-in"


def test_cached_rerun_is_instant(imbalance_run):
    r, _, (clip, masks, truth) = imbalance_run
    t = time.time()
    again = pipeline.run(clip, masks, RUNS, pipeline.twin_options(truth), truth=truth, mask_source="twin renderer")
    assert again["run_id"] == r["run_id"] and time.time() - t < 2.0


def test_bad_clip_fails_at_ingest(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    _, masks, _ = twin_inputs("healthy")
    with pytest.raises(Exception):
        pipeline.run(bad, masks, tmp_path)
    runs = [p for p in tmp_path.iterdir() if p.is_dir()]
    res = json.loads((runs[0] / "results.json").read_text())
    assert res["state"] == "failed" and res["error"]["step"] == "ingest"

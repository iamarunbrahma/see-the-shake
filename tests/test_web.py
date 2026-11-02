"""Web app: page, samples, an upload run, a twin run, video serving, input errors. No model calls."""
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shake.web.app import create_app

from .conftest import CACHE, ROOT, twin_clip

EM_DASH = "—"
UI_SOURCES = ["shake/web/static/index.html", "shake/web/app.py", "shake/pipeline.py", "shake/magnify.py",
              "shake/fingerprint.py"]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(CACHE / "runs")) as c:
        yield c


def wait(client, job_id, timeout=240):
    t = time.time()
    while time.time() - t < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "failed"):
            return j
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not finish in {timeout} s: {j}")


def test_page_has_slots_and_no_em_dashes(client):
    r = client.get("/")
    assert r.status_code == 200 and "See the Shake" in r.text
    for slot in ('id="plan-slot"', 'id="diagnosis-slot"', 'id="video"', 'id="spectra"', 'id="evidence"'):
        assert slot in r.text
    for f in UI_SOURCES:
        assert EM_DASH not in (ROOT / f).read_text(), f


def test_samples(client):
    s = client.get("/api/samples").json()
    assert [f["key"] for f in s["twin_faults"]] == ["healthy", "imbalance", "looseness", "misalignment"]
    assert len(s["steps"]) == 7


def test_upload_run(client):
    d = twin_clip("imbalance")
    truth = json.loads((d / "truth.json").read_text())
    files = [("clip", ("clip.mp4", (d / "clip.mp4").read_bytes(), "video/mp4"))]
    names = list(truth["parts"])
    for n in names:
        files.append(("masks", (f"{n}.png", (d / truth["parts"][n]["mask"]).read_bytes(), "image/png")))
    form = {"mask_names": ", ".join(names), "rpm_min": "900", "rpm_max": "1100", "ref_part": "washing machine",
            "ref_mm": "600", "min_seconds": "2"}
    r = client.post("/api/runs", files=files, data=form)
    assert r.status_code == 200, r.text
    j = wait(client, r.json()["job_id"])
    assert j["state"] == "done", j
    res = j["results"]
    assert res["fingerprint"]["verdict"] == "imbalance" and res["mask_source"] == "uploaded"
    assert res["plan"]["status"] == "not_connected" and res["diagnosis"]["status"] == "not_connected"
    # the page fetches these
    video = client.get(j["base"] + res["files"]["side_by_side"], headers={"Range": "bytes=0-99"})
    assert video.status_code == 206 and len(video.content) == 100
    for f in [res["files"]["slice"], *res["files"]["spectra"].values(), "results.json"]:
        assert client.get(j["base"] + f).status_code == 200
    assert client.get(f"/api/runs/{res['run_id']}").json()["results"]["run_id"] == res["run_id"]


def test_box_parts_and_input_errors(client):
    d = twin_clip("healthy")
    clip = ("clip", ("clip.mp4", (d / "clip.mp4").read_bytes(), "video/mp4"))
    r = client.post("/api/runs", files=[clip], data={})
    assert r.status_code == 400 and "at least one part" in r.json()["detail"]
    r = client.post("/api/runs", files=[clip], data={"boxes": "machine 10 10 20 20"})
    assert r.status_code == 400 and "Cannot read the box" in r.json()["detail"]
    r = client.post("/api/runs", files=[clip], data={"boxes": "machine: 10, 10, 20, 20", "ref_part": "door", "ref_mm": "5"})
    assert r.status_code == 400 and "reference part" in r.json()["detail"]
    r = client.post("/api/twin", json={"fault": "wobble"})
    assert r.status_code == 400


def test_twin_run(client):
    r = client.post("/api/twin", json={"fault": "looseness", "seed": 2, "seconds": 1, "scale": 0.25})
    assert r.status_code == 200
    j = wait(client, r.json()["job_id"])
    assert j["state"] == "done", j
    res = j["results"]
    assert res["mask_source"] == "twin renderer" and res["truth"]["fault"] == "looseness"
    assert "truth_comparison" in res and res["truth_comparison"]["scored"] > 0


def test_twin_replay_run(client):
    """The twin demo on recorded Muse Spark and SAM 3.1 responses: same run as tests/test_pipeline_api.py."""
    r = client.post("/api/twin", json={"fault": "imbalance", "seconds": 1, "api": "replay"})
    assert r.status_code == 200, r.text
    j = wait(client, r.json()["job_id"], timeout=400)
    assert j["state"] == "done", j
    res = j["results"]
    assert res["mask_source"] == "SAM 3.1" and res["api"]["mode"] == "replay" and res["api"]["usd"]["total"] > 0
    assert res["diagnosis"]["spark"]["evidence"] and res["plan"]["status"] == "done"
    assert client.get(j["base"] + res["files"]["masks"]["washing machine"]).status_code == 200


def test_api_mode_errors(client):
    assert client.post("/api/twin", json={"fault": "imbalance", "api": "sometimes"}).status_code == 400
    assert client.post("/api/twin", json={"fault": "imbalance", "api": "replay", "seconds": 8}).status_code == 400
    d = twin_clip("healthy")
    clip = ("clip", ("clip.mp4", (d / "clip.mp4").read_bytes(), "video/mp4"))
    r = client.post("/api/runs", files=[clip], data={"api": "maybe", "boxes": "machine: 1, 1, 5, 5"})
    assert r.status_code == 400

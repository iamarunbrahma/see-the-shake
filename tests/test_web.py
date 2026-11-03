"""Web app: page, samples, an upload run, a twin run, video serving, input errors. No model calls."""
import json
import re
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


def test_slowed_phone_upload_asks_for_the_capture_rate_then_measures_the_same(client, tmp_path):
    """A phone's slow motion saved as a slowed 30 fps file: refused until the capture rate is given, then measured
    exactly like the original 240 fps clip (the frames are the same)."""
    import subprocess
    d = twin_clip("imbalance")
    truth = json.loads((d / "truth.json").read_text())
    slowed = CACHE / "ingest" / "imbalance_slowed30.mp4"
    slowed.parent.mkdir(parents=True, exist_ok=True)
    if not slowed.exists():
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-itsscale", "8", "-i", str(d / "clip.mp4"),
                        "-c", "copy", str(slowed)], check=True, stdin=subprocess.DEVNULL, timeout=120)
    names = list(truth["parts"])
    masks = [("masks", (f"{n}.png", (d / truth["parts"][n]["mask"]).read_bytes(), "image/png")) for n in names]
    form = {"mask_names": ", ".join(names), "rpm_min": "900", "rpm_max": "1100", "ref_part": "washing machine",
            "ref_mm": "600", "min_seconds": "2"}
    up = lambda path, **extra: client.post("/api/runs", data={**form, **extra},  # noqa: E731
                                           files=[("clip", ("clip.mp4", path.read_bytes(), "video/mp4")), *masks])
    r = up(slowed)
    assert r.status_code == 400 and r.json()["detail"]["field"] == "capture_fps"
    assert "plays at 30" in r.json()["detail"]["message"]
    j = wait(client, up(slowed, capture_fps="240").json()["job_id"])
    assert j["state"] == "done", j
    clip = j["results"]["ingest"]["clip"]
    assert clip["capture_fps"] == 240 and clip["fps_source"] == "override" and abs(clip["duration_s"] - 2.0) < 1e-6
    orig = wait(client, up(d / "clip.mp4").json()["job_id"])  # the same frames at a 240 fps container
    assert j["results"]["measurement"]["parts"] == orig["results"]["measurement"]["parts"]
    assert j["results"]["fingerprint"]["verdict"] == "imbalance"


def test_fan_twin_run(client):
    """The fan in the twin picker: a short, small clip through the whole pipeline with renderer masks."""
    s = client.get("/api/samples").json()
    fan = next(m for m in s["machines"] if m["key"] == "fan")
    assert [f["key"] for f in fan["faults"]] == ["healthy", "imbalance", "looseness"]
    assert client.post("/api/twin", json={"machine": "fan", "fault": "misalignment"}).status_code == 400
    assert all(f["recorded"] for f in fan["faults"])  # recorded live fan runs: tests/api/fixtures/live/fan-*
    assert client.post("/api/twin", json={"machine": "fan", "fault": "imbalance", "api": "replay",
                                          "seconds": 8}).status_code == 400
    r = client.post("/api/twin", json={"machine": "fan", "fault": "imbalance", "seed": 3, "seconds": 1, "scale": 0.5})
    assert r.status_code == 200, r.text
    j = wait(client, r.json()["job_id"], timeout=300)
    assert j["state"] == "done", j
    res = j["results"]
    assert res["truth"]["machine"] == "fan" and set(res["measurement"]["parts"]) == {
        "fan motor housing", "fan guard rim", "fan stand"}
    assert res["fingerprint"]["verdict"] == "imbalance" and res["truth_comparison"]["verdict_ok"]
    assert "height" in res["measurement"]["scale"]["basis"]
    assert res["blades"] == {"count": 3, "source": "given"} and res["fingerprint"]["blades"] == 3


def test_blade_count_is_checked(client):
    d = twin_clip("healthy")
    clip = ("clip", ("clip.mp4", (d / "clip.mp4").read_bytes(), "video/mp4"))
    for bad in ("three", "1", "40"):
        r = client.post("/api/runs", files=[clip], data={"boxes": "machine: 10, 10, 200, 200", "blades": bad})
        assert r.status_code == 400 and "blades" in r.json()["detail"].lower()


def test_reference_axis_is_checked(client):
    d = twin_clip("healthy")
    clip = ("clip", ("clip.mp4", (d / "clip.mp4").read_bytes(), "video/mp4"))
    r = client.post("/api/runs", files=[clip], data={"boxes": "machine: 10, 10, 200, 200", "ref_part": "machine",
                                                    "ref_mm": "600", "ref_axis": "depth"})
    assert r.status_code == 400 and "width or height" in r.json()["detail"]


# ---------------------------------------------------------------- hostile or careless input

def _clip_file():
    d = twin_clip("healthy")
    return ("clip", ("clip.mp4", (d / "clip.mp4").read_bytes(), "video/mp4"))


@pytest.fixture()
def fresh(tmp_path):
    """An app with its own runs folder, to see exactly which files a request leaves behind."""
    with TestClient(create_app(tmp_path / "runs")) as c:
        c.uploads = tmp_path / "runs" / "uploads"
        yield c


def test_cross_site_posts_are_refused(fresh):
    """A page on another site could make the browser POST a run (a paid live one, too): CSRF."""
    box = {"boxes": "machine: 10, 10, 200, 200"}
    for headers in ({"Origin": "http://evil.example"}, {"Origin": "http://localhost:9000"}, {"Origin": "null"},
                    {"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"}):
        r = fresh.post("/api/runs", files=[_clip_file()], data=box, headers=headers)
        assert r.status_code == 403, headers
        assert fresh.post("/api/twin", json={"fault": "imbalance"}, headers=headers).status_code == 403
    assert not list(fresh.uploads.iterdir())
    same = {"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"}
    r = fresh.post("/api/runs", files=[_clip_file()], data=box, headers=same)
    assert r.status_code == 200 and wait(fresh, r.json()["job_id"])["state"] == "done"


@pytest.mark.parametrize("form, field", [
    ({"capture_fps": "nan"}, "capture_fps"), ({"capture_fps": "inf"}, "capture_fps"),
    ({"capture_fps": "-240"}, "capture_fps"),
    ({"rpm": "nan"}, "rpm"), ({"rpm": "0"}, "rpm"), ({"rpm": "-600"}, "rpm"), ({"rpm": "1e9"}, "rpm"),
    ({"rpm_min": "600"}, "rpm_max"), ({"rpm_min": "3000", "rpm_max": "600"}, "rpm_min"),
    ({"ref_part": "machine", "ref_mm": "nan"}, "ref_mm"), ({"ref_part": "machine", "ref_mm": "0"}, "ref_mm"),
    ({"ref_part": "machine", "ref_mm": "-5"}, "ref_mm"), ({"ref_mm": "600"}, "ref_part"),
    ({"alpha": "nan"}, "alpha"), ({"alpha": "1e9"}, "alpha"), ({"min_seconds": "0"}, "min_seconds"),
    ({"boxes": "/tmp/x/proof: 10, 10, 200, 200"}, "boxes"), ({"boxes": "../../x: 10, 10, 200, 200"}, "boxes"),
    ({"boxes": "machine: 5000, 10, 10, 10"}, "boxes"), ({"boxes": "machine: 10, 10, 0, 50"}, "boxes"),
    ({"boxes": "machine: 0, 0, 960, 540"}, "boxes")])
def test_bad_values_are_refused_before_a_job_starts(fresh, form, field):
    data = {"boxes": "machine: 10, 10, 200, 200", **form}
    r = fresh.post("/api/runs", files=[_clip_file()], data=data)
    assert r.status_code == 400, r.text
    assert r.json()["detail"]["field"] == field
    assert not list(fresh.uploads.iterdir())  # a rejected upload is deleted


def test_mask_names_cannot_hold_a_path(fresh, tmp_path):
    d = twin_clip("healthy")
    truth = json.loads((d / "truth.json").read_text())
    mask = (d / truth["parts"]["washing machine"]["mask"]).read_bytes()
    r = fresh.post("/api/runs", files=[_clip_file(), ("masks", ("m.png", mask, "image/png"))],
                   data={"mask_names": str(tmp_path / "escape")})
    assert r.status_code == 400 and r.json()["detail"]["field"] == "mask_names"
    assert not (tmp_path / "escape.png").exists()


def test_files_that_are_not_clips(fresh, tmp_path):
    import subprocess
    png = tmp_path / "still.png"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240", "-frames:v",
                    "1", str(png)], check=True, timeout=60)
    box = {"boxes": "machine: 10, 10, 200, 200"}
    cases = [(("x.png", png.read_bytes()), "ends in '.png'"), (("x.mp4", png.read_bytes()), "still image"),
             (("x.mp4", b""), "empty"), (("x.mp4", bytes(range(256)) * 100), "not a video"),
             (("a." + "b" * 250, b"x"), "Choose a video file")]
    for (name, body), words in cases:
        r = fresh.post("/api/runs", files=[("clip", (name, body, "video/mp4"))], data=box)
        assert r.status_code == 400, (name[:20], r.text)
        msg = r.json()["detail"]["message"]
        assert words in msg and str(fresh.uploads) not in msg and "@ 0x" not in msg, msg
    assert not list(fresh.uploads.iterdir())  # no leftover upload or incoming_ file


def test_odd_file_names_are_stored_by_hash(fresh):
    clip = _clip_file()[1][1]
    jobs = []
    for name in ("../../../../tmp/evil.mp4", "ünïcödé 振動.mp4", "a;$(touch PWNED)`id`&.mp4", "CLIP.MOV", "noext"):
        r = fresh.post("/api/runs", files=[("clip", (name, clip, "video/mp4"))], data={"boxes": "m: 10, 10, 200, 200"})
        assert r.status_code == 200, (name, r.text)
        jobs.append(r.json()["job_id"])
    assert all(wait(fresh, j)["state"] == "done" for j in jobs)
    assert all(re.fullmatch(r"[0-9a-f]{16}\.(mp4|mov)", p.name) for p in fresh.uploads.iterdir())


def test_upload_size_is_enforced_while_the_body_streams(tmp_path):
    small = TestClient(create_app(tmp_path / "runs", max_upload=4 * 1024 * 1024, max_body=6 * 1024 * 1024))
    small.uploads, fresh = tmp_path / "runs" / "uploads", small
    big = b"\0" * (7 * 1024 * 1024)
    r = fresh.post("/api/runs", files=[("clip", ("big.mp4", big, "video/mp4"))], data={"boxes": "m: 1, 1, 9, 9"})
    assert r.status_code == 413  # declared Content-Length over the limit: refused before reading
    boundary = "shakeboundary"
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"clip\"; filename=\"big.mp4\"\r\n"
            "Content-Type: video/mp4\r\n\r\n").encode()

    def chunks():  # no Content-Length: the limit is counted as the body arrives
        yield head
        for _ in range(8):
            yield b"\0" * (1024 * 1024)
        yield f"\r\n--{boundary}--\r\n".encode()
    r = fresh.post("/api/runs", content=chunks(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert r.status_code == 413 and "larger than" in r.json()["detail"]
    r = fresh.post("/api/runs", files=[("clip", ("big.mp4", b"\0" * (5 * 1024 * 1024), "video/mp4"))],
                   data={"boxes": "m: 1, 1, 9, 9"})
    assert r.status_code == 413 and "clip is larger than 4 MB" in r.json()["detail"]["message"]
    assert not list(fresh.uploads.iterdir())


def test_live_runs_need_a_key(fresh, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    r = fresh.post("/api/runs", files=[_clip_file()], data={"api": "live"})
    assert r.status_code == 400 and r.json()["detail"]["field"] == "clip_api"
    assert "MODEL_API_KEY" in r.json()["detail"]["message"]
    r = fresh.post("/api/twin", json={"fault": "imbalance", "api": "live"})
    assert r.status_code == 400 and r.json()["detail"]["field"] == "twin_api"
    assert fresh.post("/api/twin", json={"fault": "imbalance", "seed": -1}).json()["detail"]["field"] == "seed"


def test_a_failed_job_reports_one_line_and_drops_its_upload(fresh, monkeypatch):
    from shake import pipeline as pl

    def boom(*a, **k):
        raise RuntimeError("could not open /srv/secret/place/frame.png")
    monkeypatch.setattr(pl, "run", boom)
    r = fresh.post("/api/runs", files=[_clip_file()], data={"boxes": "m: 10, 10, 200, 200"})
    j = wait(fresh, r.json()["job_id"])
    assert j["state"] == "failed" and "trace" not in j and "\n" not in j["error"]
    assert "frame.png" in j["error"] and "/srv" not in j["error"]
    assert not list(fresh.uploads.iterdir())
    assert fresh.get("/api/jobs/000000000000").status_code == 404


def test_a_cached_rerun_is_marked(client):
    d = twin_clip("imbalance")
    truth = json.loads((d / "truth.json").read_text())
    files = [("clip", ("clip.mp4", (d / "clip.mp4").read_bytes(), "video/mp4"))]
    files += [("masks", (f"{n}.png", (d / truth["parts"][n]["mask"]).read_bytes(), "image/png"))
              for n in truth["parts"]]
    form = {"mask_names": ", ".join(truth["parts"]), "rpm_min": "900", "rpm_max": "1100", "ref_part": "washing machine",
            "ref_mm": "600", "min_seconds": "2"}
    first = wait(client, client.post("/api/runs", files=files, data=form).json()["job_id"])
    again = wait(client, client.post("/api/runs", files=files, data=form).json()["job_id"])
    assert first["state"] == again["state"] == "done" and again["from_cache"]
    assert again["results"]["run_id"] == first["results"]["run_id"]


def test_page_handles_failed_requests():
    """The page must never stay on "Queued": every request is read through getJSON, and a lost job is reported."""
    page = (ROOT / "shake/web/static/index.html").read_text()
    assert page.count("getJSON(") >= 3 and "r.status === 404" in page and "MAX_MISSES" in page
    assert 'accept="video/*,.mp4,.mov,.m4v,.webm,.avi,.mkv"' in page and "WebM or AVI" in page

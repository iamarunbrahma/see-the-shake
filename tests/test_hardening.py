"""Inputs a user (or a hostile page) can get wrong: part names, numbers, masks, files, a missing key, and how the
command line reports them. No model calls."""
import json
import math
import re

import numpy as np
import pytest

from shake import ingest, pipeline, twin
from shake.meta_api import APIError, CreditExhausted, MissingKey

from .conftest import twin_clip

SAFE = re.compile(r"^[a-z0-9_-]+$")


def small_twin():
    d = twin_clip("healthy")
    truth = json.loads((d / "truth.json").read_text())
    return d / "clip.mp4", {n: d / truth["parts"][n]["mask"] for n in truth["parts"]}, truth


# ---------------------------------------------------------------- part names and files (H1)

@pytest.mark.parametrize("name", ["washing machine", "../../etc/passwd", "/tmp/x/proof", "a\\b", "Door", "door",
                                  "<img src=x onerror=alert(1)>", "", "...", "振動"])
def test_slug_is_a_safe_file_name(name):
    s = twin.slug(name)
    assert SAFE.match(s) and len(s) <= 47 and s == twin.slug(name)


def test_slug_keeps_different_names_apart():
    names = ["Door", "door", "door!", "door?", "washing machine", "washing_machine"]
    assert len({twin.slug(n) for n in names}) == len(names)


@pytest.mark.parametrize("name", ["/tmp/x/proof", "../../x", "a/b", "a\\b", "..", " ", "x" * 81, "a\x00b"])
def test_part_names_that_hold_a_path_are_refused(name):
    with pytest.raises(pipeline.OptionsError):
        pipeline.check_part_name(name)


def test_writes_stay_in_the_run_folder(tmp_path):
    assert pipeline._in_run(tmp_path, "masks/door_abc123.png") == (tmp_path / "masks/door_abc123.png").resolve()
    for rel in ("../escape.png", "masks/../../escape.png", "/tmp/escape.png"):
        with pytest.raises(pipeline.OptionsError):
            pipeline._in_run(tmp_path, rel)


def test_a_part_named_like_a_path_writes_nothing(tmp_path):
    """The old repro: a box named with an absolute path made the server write <name>.png there."""
    clip, masks, _ = small_twin()
    target = tmp_path / "outside" / "proof"
    m = pipeline.load_masks({"washing machine": masks["washing machine"]})["washing machine"]
    with pytest.raises(pipeline.OptionsError, match="cannot contain"):
        pipeline.run(clip, {str(target): m}, tmp_path / "runs", pipeline.Options(rpm_range=(900, 1100)))
    assert not target.with_suffix(".png").exists() and not (tmp_path / "runs").exists()


# ---------------------------------------------------------------- numbers (M3)

@pytest.mark.parametrize("change, field", [
    (dict(rpm=float("nan")), "rpm"), (dict(rpm=float("inf")), "rpm"), (dict(rpm=-600), "rpm"), (dict(rpm=0), "rpm"),
    (dict(capture_fps=float("nan")), "capture_fps"), (dict(capture_fps=-240), "capture_fps"),
    (dict(rpm_range=(3000, 600)), "rpm_min"), (dict(rpm_range=(600, float("nan"))), "rpm_max"),
    (dict(reference=("machine", 0.0)), "ref_mm"), (dict(reference=("machine", -5.0)), "ref_mm"),
    (dict(reference=("machine", float("nan"))), "ref_mm"), (dict(reference=("machine", 600.0, "depth")), "ref_axis"),
    (dict(alpha=float("nan")), "alpha"), (dict(alpha=1e9), "alpha"), (dict(alpha=-5.0), "alpha"),
    (dict(min_seconds=0), "min_seconds"), (dict(min_seconds=float("nan")), "min_seconds"),
    (dict(blades=1), "blades"), (dict(blades=40), "blades")])
def test_bad_numbers_are_refused_with_the_input_named(change, field):
    with pytest.raises(pipeline.OptionsError) as e:
        pipeline.validate_options(pipeline.Options(**change))
    assert e.value.field == field


def test_running_speed_must_be_measurable_at_the_clip_rate():
    pipeline.validate_options(pipeline.Options(rpm=1000, rpm_range=(600, 1600)), fps=240)
    for o in (pipeline.Options(rpm=1e9), pipeline.Options(rpm=60), pipeline.Options(rpm_range=(8000, 9000))):
        with pytest.raises(pipeline.OptionsError, match="what 240 fps can measure"):
            pipeline.validate_options(o, fps=240)


def test_good_options_pass():
    pipeline.validate_options(pipeline.Options(capture_fps=240, rpm=1000, reference=("machine", 600.0, "height"),
                                               alpha=40.0, min_seconds=2.0, blades=3), fps=240)


# ---------------------------------------------------------------- masks and boxes

def test_masks_must_fit_the_frame_and_leave_a_background():
    shape = (360, 640)
    part = np.zeros(shape, np.uint8)
    part[60:260, 100:400] = 255
    pipeline.check_masks({"m": part}, shape)
    with pytest.raises(pipeline.OptionsError, match="is 960x540, but the clip's frames are 640x360"):
        pipeline.check_masks({"m": np.zeros((540, 960), np.uint8)}, shape)
    with pytest.raises(pipeline.OptionsError, match="empty"):
        pipeline.check_masks({"m": np.zeros(shape, np.uint8)}, shape)
    with pytest.raises(pipeline.OptionsError, match="whole frame"):
        pipeline.check_masks({"m": np.full(shape, 255, np.uint8)}, shape)
    tiny = np.zeros(shape, np.uint8)
    tiny[5, 5] = 255
    with pytest.raises(pipeline.OptionsError, match="too small"):
        pipeline.check_masks({"m": tiny}, shape)


def test_a_wrong_size_mask_is_explained_not_a_numpy_error(tmp_path):
    clip, _, _ = small_twin()
    with pytest.raises(pipeline.OptionsError, match="Make the mask the same size as the clip"):
        pipeline.run(clip, {"m": np.full((100, 100), 255, np.uint8)}, tmp_path, pipeline.Options(rpm_range=(900, 1100)))
    res = json.loads(next(tmp_path.glob("*/results.json")).read_text())
    assert res["state"] == "failed" and "trace" not in res["error"] and "100x100" in res["error"]["message"]


# ---------------------------------------------------------------- the key (M6)

def test_a_live_run_without_a_key_stops_before_ingest(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    clip, masks, _ = small_twin()
    with pytest.raises(MissingKey, match="MODEL_API_KEY is not set"):
        pipeline.run(clip, masks, tmp_path, pipeline.Options(api="live"))
    res = json.loads(next(tmp_path.glob("*/results.json")).read_text())
    steps = {s["key"]: s["status"] for s in res["steps"]}
    assert steps["ingest"] == "pending" and steps["plan"] == "failed" and "ingest" not in res


def test_a_live_twin_without_a_key_renders_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(MissingKey):
        pipeline.run_twin("imbalance", tmp_path, api="live")
    assert not (tmp_path / "twin_clips").exists()


def test_messages_for_people():
    msg = pipeline.user_message(FileNotFoundError("No such file: '/srv/app/runs/abc/spectrum_x.png'"))
    assert "/srv" not in msg and "spectrum_x.png" in msg
    assert "did not accept the key" in pipeline.user_message(APIError("upload: HTTP 401: Unauthorized", 401))
    credit = pipeline.user_message(CreditExhausted("HTTP 402", 402))
    assert credit == pipeline.CREDIT_MESSAGE and "Arun" not in credit and "card" not in credit


# ---------------------------------------------------------------- files ingest cannot use

def test_files_that_are_not_clips_get_plain_messages(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(bytes(range(256)) * 400)
    import subprocess
    png, audio = tmp_path / "still.png", tmp_path / "sound.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240", "-frames:v",
                    "1", str(png)], check=True, timeout=60)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-c:a", "aac", str(audio)], check=True, timeout=60)
    for path, words in ((empty, "empty"), (junk, "not a video"), (png, "still image"), (audio, "no video")):
        with pytest.raises(ingest.IngestError) as e:
            ingest.clip_info(path)
        assert words in str(e.value) and str(tmp_path) not in str(e.value) and "0x" not in str(e.value), str(e.value)


# ---------------------------------------------------------------- the command line (M3, M5)

def cli(capsys, *argv):
    try:
        code = pipeline.main(list(argv))
    except SystemExit as e:
        code = e.code
    out = capsys.readouterr()
    assert "Traceback" not in out.out + out.err
    return code, out.out + out.err


@pytest.mark.parametrize("argv, words", [
    (["clip.mp4", "--mask", "nomask"], "--mask needs a part name"),
    (["clip.mp4", "--mask", "../x=m.png"], "cannot contain"),
    (["clip.mp4", "--api", "live", "--ref", "washing machine"], "--ref needs a part and a length"),
    (["clip.mp4", "--api", "live", "--ref", "washing machine=-5"], "greater than 0"),
    (["missing.mp4", "--mask", "m=m.png"], "There is no clip"),
    (["missing.mp4", "--api", "replay"], "A replay needs a recorded run"),
])
def test_cli_errors_are_one_line(capsys, argv, words):
    code, text = cli(capsys, *argv)
    assert code == 1 and words in text and text.strip().startswith("error:")


@pytest.mark.parametrize("argv", [["--twin", "imbalance", "--seed", "-1"], ["x.mp4", "--rpm", "nan"],
                                  ["x.mp4", "--alpha", "abc"], ["x.mp4", "--capture-fps", "0"],
                                  ["x.mp4", "--rpm-range", "600", "inf"]])
def test_cli_refuses_bad_numbers(capsys, argv):
    code, text = cli(capsys, *argv)
    assert code == 2 and "error: argument --" in text


def test_cli_without_a_key_stops_at_once(capsys, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    clip, _, _ = small_twin()
    code, text = cli(capsys, str(clip), "--api", "live")
    assert code == 1 and "MODEL_API_KEY is not set" in text and "[running ] ingest" not in text


def test_cost_line_says_when_nothing_was_charged():
    api = {"mode": "replay", "usd": {"total": 0.0626}}
    assert "made no calls" in pipeline.cost_line({"api": api})
    assert "from the cache" in pipeline.cost_line({"api": {**api, "mode": "record"}, "from_cache": True})
    assert pipeline.cost_line({"api": {**api, "mode": "record"}}) == "API cost: $0.0626"
    assert math.isclose(api["usd"]["total"], 0.0626)


def test_a_clip_too_short_to_measure_is_explained(tmp_path):
    import subprocess
    clip = tmp_path / "short.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=240",
                    "-t", "0.1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True, timeout=60)
    box = np.zeros((240, 320), np.uint8)
    box[40:160, 60:220] = 255
    with pytest.raises(pipeline.OptionsError, match=r"only 24 frames \(0.10 s at 240 fps\)"):
        pipeline.run(clip, {"m": box}, tmp_path / "runs", pipeline.Options(rpm_range=(900, 1100)))

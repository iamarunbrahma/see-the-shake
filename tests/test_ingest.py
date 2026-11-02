import subprocess

import numpy as np
import pytest

from shake import ingest

from .conftest import CACHE, ROOT, twin_clip


def _ffmpeg(*args):
    out = args[-1]
    if not out.exists() or out.stat().st_mtime < args[args.index("-i") + 1].stat().st_mtime:
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *map(str, args)], check=True,
                       stdin=subprocess.DEVNULL, timeout=180)


@pytest.fixture(scope="module")
def clips():
    src = twin_clip("healthy") / "clip.mp4"
    d = CACHE / "ingest"
    d.mkdir(parents=True, exist_ok=True)
    slowed = d / "slowed_tagged.mp4"  # what a phone does: 240 fps capture saved as a 30 fps file
    _ffmpeg("-itsscale", 8, "-i", src, "-c", "copy", "-movflags", "use_metadata_tags",
            "-metadata", "com.android.capture.fps=240.000000", slowed)
    untagged = d / "slowed_untagged.mp4"
    _ffmpeg("-itsscale", 8, "-i", src, "-c", "copy", untagged)
    blend = d / "synth_from_60.mp4"  # fake 240 fps: 60 fps frames with blended in-betweens
    _ffmpeg("-i", src, "-vf", "fps=60,minterpolate=fps=240:mi_mode=blend", "-c:v", "libx264", "-tune", "grain",
            "-crf", "16", "-preset", "veryfast", blend)
    mci = d / "synth_mci.mp4"  # fake 240 fps by motion-compensated interpolation
    _ffmpeg("-i", src, "-vf", "fps=60,minterpolate=fps=240:mi_mode=mci", "-c:v", "libx264", "-tune", "grain",
            "-crf", "16", "-preset", "veryfast", mci)
    return dict(src=src, slowed=slowed, untagged=untagged, blend=blend, mci=mci)


def test_stream_rate(clips):
    info = ingest.clip_info(clips["src"])
    assert info.capture_fps == 240 and info.fps_source == "stream" and info.codec == "h264"
    assert info.frames == 480 and abs(info.duration_s - 2.0) < 1e-6


def test_capture_fps_from_metadata(clips):
    info = ingest.clip_info(clips["slowed"])
    assert info.capture_fps == 240 and info.fps_source.startswith("metadata:")
    assert abs(info.stream_fps - 30) < 0.5 and abs(info.time_scale - 8) < 0.2
    assert abs(info.duration_s - 2.0) < 1e-6  # capture time, not playback time


def test_override_and_reject(clips):
    with pytest.raises(ingest.IngestError, match="at least 120"):
        ingest.clip_info(clips["untagged"])  # 30 fps playback, no metadata: refused
    info = ingest.clip_info(clips["untagged"], capture_fps=240)
    assert info.capture_fps == 240 and info.fps_source == "override" and abs(info.time_scale - 8) < 0.2
    with pytest.raises(ingest.IngestError):
        ingest.clip_info(clips["src"], capture_fps=60)


def test_real_frames_not_flagged(clips):
    c = ingest.interpolation_check(clips["src"])
    assert c["status"] == "ok" and not c["flagged"], c


@pytest.mark.parametrize("name", ["blend", "mci"])
def test_synthesised_frames_flagged(clips, name):
    c = ingest.interpolation_check(clips[name])
    assert c["flagged"] and c["status"] == "synthesised", c


def test_steady_window_avoids_a_bump():
    fps = 240
    e = np.ones(12 * fps) + 0.01 * np.random.default_rng(0).standard_normal(12 * fps)
    e[: 2 * fps] += 5  # someone touched the phone in the first 2 s
    s, n = ingest.steady_window(e, fps, min_s=4, max_s=10)
    assert n == 10 * fps and s >= 2 * fps - 1


def test_ingest_end_to_end(clips):
    r = ingest.ingest(clips["slowed"], min_s=1.0).to_dict()
    assert r["clip"]["capture_fps"] == 240 and r["interpolation"]["status"] == "ok"
    assert r["window"]["frames"] == 480


# ---------------------------------------------------------------- phone slow-motion containers

@pytest.fixture(scope="module")
def phone(clips):
    """What phones do with slow motion, made from a 240 fps twin clip: a slowed 30 fps container (same frames),
    a speed ramp (real-time timestamps, then slowed), jittered 30 fps timestamps, and synthesised frames in a
    slowed container."""
    d = CACHE / "ingest"
    out = {"slowed30": d / "phone_slowed30.mp4", "ramp": d / "phone_vfr_ramp.mp4", "jitter": d / "phone_vfr_jitter.mp4",
           "synth30": d / "phone_synth_slowed30.mp4"}
    _ffmpeg("-itsscale", 8, "-i", clips["src"], "-c", "copy", out["slowed30"])
    enc = ["-fps_mode", "vfr", "-c:v", "libx264", "-preset", "veryfast", "-tune", "grain", "-crf", "16",
           "-pix_fmt", "yuv420p"]
    _ffmpeg("-i", clips["src"], "-vf", "setpts='if(lt(N,60),N/240,0.25+(N-60)/30)/TB'", *enc, out["ramp"])
    _ffmpeg("-i", clips["src"], "-vf", "setpts='(N/30+0.004*sin(N*1.7))/TB'", *enc, out["jitter"])
    _ffmpeg("-itsscale", 8, "-i", clips["blend"], "-c", "copy", out["synth30"])
    return out


def test_slowed_container_asks_for_the_capture_rate(phone, clips):
    with pytest.raises(ingest.IngestError) as e:
        ingest.clip_info(phone["slowed30"])
    assert e.value.field == "capture_fps" and "plays at 30" in str(e.value) and "captured at" in str(e.value)
    info = ingest.clip_info(phone["slowed30"], capture_fps=240)
    assert info.capture_fps == 240 and info.fps_source == "override" and abs(info.time_scale - 8) < 0.1
    assert info.frames == 480 and abs(info.duration_s - 2.0) < 1e-6 and not info.variable_rate
    a = list(ingest.read_gray(clips["src"], start=200, count=4))
    b = list(ingest.read_gray(phone["slowed30"], start=200, count=4))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))  # the same frames, so the same measurement


@pytest.mark.parametrize("name", ["ramp", "jitter"])
def test_variable_rate_containers(phone, name):
    with pytest.raises(ingest.IngestError) as e:
        ingest.clip_info(phone[name])  # its average rate is about 30 fps, whatever r_frame_rate claims
    assert e.value.field == "capture_fps"
    r = ingest.ingest(phone[name], capture_fps=240, min_s=1.0)
    assert r.info.variable_rate and any("variable frame rate" in w for w in r.info.warnings)
    assert r.info.frames == 480 and r.window == (0, 480) and r.interpolation["status"] == "ok"


def test_synthesised_frames_flagged_in_a_slowed_container(phone):
    r = ingest.ingest(phone["synth30"], capture_fps=240, min_s=1.0)
    assert r.interpolation["flagged"] and any("synthesised" in w for w in r.info.warnings)


# ---------------------------------------------------------------- real footage and phone tags

REAL = ROOT / "samples" / "real"


@pytest.mark.parametrize("clip,fps", [("aeg-washer-120fps/aeg-washer-1400rpm-120fps.webm", None),
                                      ("xjtu-rotor-500fps/3000rpm-Unb1.avi", 500)])
def test_real_compressed_clips_are_inconclusive_not_synthesised(clip, fps):
    """A web re-encode (VP9) and a lab camera (MPEG-4 Part 2) lose the sensor noise the check needs; the probes
    show their spectra are right, so the check must not call them synthesised."""
    info = ingest.clip_info(REAL / clip, capture_fps=fps)
    c = ingest.interpolation_check(REAL / clip, info=info)
    assert c["status"] == "inconclusive" and not c["flagged"] and "compression" in c["reason"]
    assert not any("SAM" in w for w in info.warnings)  # the SAM sub-clip is re-encoded anyway


def test_phone_tag_at_960_fps_is_flagged(clips):
    tagged = CACHE / "ingest" / "phone_tag_960.mp4"
    _ffmpeg("-i", clips["src"], "-c", "copy", "-movflags", "use_metadata_tags",
            "-metadata", "com.android.capture.fps=960.000000", tagged)
    c = ingest.interpolation_check(tagged)
    assert c["flagged"] and "inventing frames" in c["reason"]

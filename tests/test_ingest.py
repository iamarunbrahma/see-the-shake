import subprocess

import numpy as np
import pytest

from shake import ingest

from .conftest import CACHE, twin_clip


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

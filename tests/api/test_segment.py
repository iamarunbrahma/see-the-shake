"""SAM 3.1 stream handling, offline on a recorded stream: thinning, parsing, stable masks."""
import numpy as np
import pytest

from shake import segment, twin
from tests.api.live_fixtures import load_fixture, sam_calls


@pytest.fixture(scope="module")
def door_events():
    fx = load_fixture("imbalance")
    calls = sam_calls(fx)
    name = fx["report"]["iou"]["door"]["phrase"]
    return next(c["response"] for c in calls if c["key"] == name)


def test_thinned_stream_parses_to_the_selected_frames(door_events):
    full = segment.frame_stats(segment.stream_text(door_events))
    assert len(full) == 1 and len(next(iter(full.values()))["frames"]) == 60
    thin = segment.thin_events(door_events, keep=4)
    recs = segment.mask_records(segment.parse_events(thin))
    assert [r[0] for r in recs] == [0, 20, 39, 59]
    one = segment.thin_events(door_events, select={7})
    assert [r[0] for r in segment.mask_records(segment.parse_events(one))] == [7]


def test_rasters_fill_their_boxes(door_events):
    for fi, oid, (x0, y0, x1, y1), r, conf in segment.mask_records(segment.parse_events(
            segment.thin_events(door_events, keep=2))):
        assert r.shape == (y1 - y0, x1 - x0) and r.max() == 1 and 0 < conf <= 1


def test_parallel_and_serial_decode_agree(door_events):
    shape = (twin.H, twin.W)
    a, ia = segment.stable_mask(door_events, shape, 60, mask_frames=4, workers=1)
    b, ib = segment.stable_mask(door_events, shape, 60, mask_frames=4, workers=2)
    assert np.array_equal(a, b) and ia["frames_returned"] == 60 and ia["frames_decoded"] == 4
    door = twin.part_masks()[0]["door"]
    assert segment.iou(a, door) > 0.95


def test_empty_response_is_a_valid_miss():
    fx = load_fixture("imbalance")
    missing = fx["report"]["segment"]["missing"]
    if not missing:
        pytest.skip("this recording has no phrase without a match")
    ev = next(c["response"] for c in sam_calls(fx) if c["key"] == missing[0])
    m, info = segment.stable_mask(ev, (twin.H, twin.W), 60)
    assert m is None and info["found"] is False and info["objects"] == 0


def test_iou():
    a = np.zeros((10, 10), np.uint8)
    a[:5] = 255
    b = np.zeros((10, 10), np.uint8)
    b[:, :5] = 255
    assert segment.iou(a, a) == 1.0 and segment.iou(a, b) == pytest.approx(25 / 75)


def test_mask_shape_must_match_what_sam_saw(door_events):
    with pytest.raises(segment.SegmentError, match="SAM saw 1920x1080"):
        segment.stable_mask(door_events, (1920, 1080), 60)  # transposed, as a rotated phone clip would be


def test_rotated_clip_decodes_the_same_in_subclip_and_measurement(tmp_path):
    """Phones store portrait clips as landscape plus rotation metadata. The SAM sub-clip (ffmpeg) and measurement
    (OpenCV) must both apply it, so masks and frames line up."""
    import subprocess

    from shake.ingest import read_gray
    src, rot = tmp_path / "src.mp4", tmp_path / "rot.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=30", "-frames:v", "60", "-pix_fmt", "yuv420p", str(src)], check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-display_rotation", "90", "-i", str(src),
                    "-c", "copy", str(rot)], check=True)
    sub = segment.subclip(rot, tmp_path / "sub.mp4", 0, 50)
    assert segment.decoded_shape(sub["path"]) == next(read_gray(rot)).shape == (320, 180)

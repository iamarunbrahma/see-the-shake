"""Magnification: the magnified clip moves about alpha times more in the band, and nothing outside a part's zone changes.

Tolerance for the gain check is 30%. Magnification is for display, and the check needs a second tracker
(magnify.track_large) because the magnified parts move several px. Its error budget: second-generation H.264
(under 6% on a plain re-encode), the motion-compensated template and coarse-to-fine tracking of large motion,
which on this clip reads up to about 27% high at 2x.
"""
import hashlib
import json

import numpy as np
import pytest

from shake import magnify

from .conftest import CACHE, ROOT, _key

ALPHA = 40.0
TOL = 0.30


@pytest.fixture(scope="module")
def magnified(twin_runs):
    run = twin_runs["misalignment"]  # strong 2x outside the 1x band: tests band selectivity too
    f1 = run["result"]["running_speed"]["hz"]
    h = hashlib.sha1((ROOT / "shake" / "magnify.py").read_bytes()).hexdigest()[:10]
    d = CACHE / _key() / f"magnify_{h}_a{ALPHA:g}"
    vpath = d / "verify.json"
    clip = run["dir"] / "clip.mp4"
    if not vpath.exists():
        info = magnify.magnify(clip, run["masks"], run["arrays"]["rel"], 240.0, f1, d, alpha=ALPHA, seconds=2.0)
        rot = magnify.rotating_mask(clip, 0, info["frames"])
        v = magnify.verify(d / "magnified_native.mp4", run["masks"], run["arrays"]["rel"], 240.0, f1, info, rot)
        vpath.write_text(json.dumps({"info": info, "verify": v}, indent=2))
    out = json.loads(vpath.read_text())
    return dict(run=run, f1=f1, dir=d, **out)


def test_in_band_motion_is_alpha_times_original(magnified):
    rows = [r for r in magnified["verify"]["rows"] if r["order"] == 1 and r["judged"]]
    assert len(rows) == 4  # both parts, both axes
    for r in rows:
        assert r["expected_gain"] == pytest.approx(ALPHA, rel=0.01)
        assert abs(r["ratio"] - 1) <= TOL, r


def test_out_of_band_motion_is_barely_changed(magnified):
    rows = [r for r in magnified["verify"]["rows"] if r["order"] == 2 and r["judged"]]
    assert rows
    for r in rows:
        assert r["expected_gain"] < 2.0  # the 1x band leaves 2x nearly alone
        assert abs(r["ratio"] - 1) <= TOL, r
        assert r["gain"] < 0.1 * ALPHA


def test_outputs_written(magnified):
    d, info = magnified["dir"], magnified["info"]
    for k in ("side_by_side", "native", "slice"):
        assert (d / info["files"][k]).stat().st_size > 1000
    assert info["alpha"] == ALPHA and info["layers"][0]["outer"] and not info["layers"][1]["outer"]


def test_pixels_outside_part_zones_untouched(twin_runs):
    run = twin_runs["imbalance"]
    masks, rel = run["masks"], run["arrays"]["rel"]
    alpha, sh, _, _ = magnify.shifts(rel, 240.0, run["result"]["running_speed"]["hz"], ALPHA)
    shape = next(iter(masks.values())).shape
    layers = magnify.build_layers(masks, sh, shape)
    plate = magnify.make_plate(run["dir"] / "clip.mp4", 0, 480, layers, shape)
    zone = np.zeros(shape, bool)
    for L in layers:
        y0, y1, x0, x1 = L.box
        zone[y0:y1, x0:x1] |= L.zone
    # the zone is the masks grown by the largest shift plus the feather, not the whole frame
    grown_by = int(np.ceil(max(np.abs(L.shift).max() for L in layers))) + 3 + 2 + 2
    union = np.zeros(shape, bool)
    for m in masks.values():
        union |= m > 0
    assert zone.sum() < union.sum() * 1.3 and zone.sum() > union.sum()
    changed_inside = 0
    for i, f in enumerate(magnify.read_bgr(run["dir"] / "clip.mp4", 0, 8)):
        g = magnify.magnify_frame(f, i, layers, plate)
        assert np.array_equal(g[~zone], f[~zone]), f"frame {i}: pixels outside the zones changed"
        changed_inside += int((g[zone] != f[zone]).any(-1).sum())
    assert changed_inside > 0 and grown_by < 40

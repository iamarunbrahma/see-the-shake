"""Shared fixtures: short, downscaled twin clips (2 s, 960x540), cached on disk between runs.

The cache key includes a hash of shake/twin.py, so changing the renderer re-renders.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from shake import fingerprint, measure, twin

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "out" / "test_cache"
SECONDS, SCALE, SEED = 2.0, 0.5, 5


def _key():
    h = hashlib.sha1((ROOT / "shake" / "twin.py").read_bytes()).hexdigest()[:10]
    return f"{h}_{SECONDS:g}s_{SCALE:g}x_seed{SEED}"


def twin_clip(fault):
    d = CACHE / _key() / fault
    if not (d / "truth.json").exists():
        twin.render(twin.TwinConfig(fault, seconds=SECONDS, seed=SEED, out_scale=SCALE), d)
    return d


def load_masks(d, truth):
    return {n: cv2.imread(str(d / truth["parts"][n]["mask"]), 0) for n in truth["parts"]}


@pytest.fixture(scope="session")
def twin_runs():
    """{fault: dict(dir, truth, masks, result, arrays, fp)} for every twin fault."""
    runs = {}
    for fault in twin.FAULTS:
        d = twin_clip(fault)
        truth = json.loads((d / "truth.json").read_text())
        masks = load_masks(d, truth)
        sc = measure.Scale.from_reference(masks, truth["reference"]["part"], truth["reference"]["width_mm"])
        rpm = truth["rpm"]
        res, arrays = _measured(d, masks, truth, sc, rpm)
        runs[fault] = dict(dir=d, truth=truth, masks=masks, result=res, arrays=arrays, fp=fingerprint.fingerprint(res))
    return runs


def _measured(d, masks, truth, sc, rpm):
    """measure.measure on a test clip, cached on disk by a hash of measure.py (re-measured whenever it changes)."""
    h = hashlib.sha1((ROOT / "shake" / "measure.py").read_bytes()).hexdigest()[:10]
    jpath, npath = d / f"measured_{h}.json", d / f"measured_{h}.npz"
    if jpath.exists() and npath.exists():
        z = np.load(npath)
        series = {k[7:]: z[k] for k in z.files if k.startswith("series_")}
        rel = {k[4:]: z[k] for k in z.files if k.startswith("rel_")}
        return json.loads(jpath.read_text()), {"series": series, "rel": rel}
    res, arrays = measure.measure(str(d / "clip.mp4"), masks, truth["fps"], rpm_range=(0.9 * rpm, 1.1 * rpm), scale=sc)
    np.savez(npath, **{f"series_{k}": v for k, v in arrays["series"].items()},
             **{f"rel_{k}": v for k, v in arrays["rel"].items()})
    jpath.write_text(json.dumps(res))
    return json.loads(jpath.read_text()), arrays

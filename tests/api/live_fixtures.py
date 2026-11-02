"""Recorded live twin runs (written by scripts/live_twin.py): cassette, measurement, fingerprint and report."""
import gzip
import json
from pathlib import Path

import pytest

LIVE = Path(__file__).parent / "fixtures" / "live"  # 1920x1080, near-lossless encoding
LIVE_PHONE = Path(__file__).parent / "fixtures" / "live_phone"  # 1280x720, phone-like bitrate
SETS = {"grain": LIVE, "phone": LIVE_PHONE}


def load_fixture(fault, kind="grain"):
    d = SETS[kind] / fault
    if not (d / "report.json").exists():
        pytest.skip(f"no live recording for {fault}; run scripts/live_twin.py")
    return {"dir": d, "cassette_path": d / "cassette.json.gz",
            "cassette": json.loads(gzip.decompress((d / "cassette.json.gz").read_bytes())),
            "report": json.loads((d / "report.json").read_text()),
            "measured": json.loads((d / "measured.json").read_text()),
            "fingerprint": json.loads((d / "fingerprint.json").read_text())}


def sam_calls(fx):
    return [c for c in fx["cassette"] if c["op"] == "sam"]

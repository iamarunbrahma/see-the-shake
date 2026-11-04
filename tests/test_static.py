"""The static site (scripts/export_static.py): every file a run references is exported, and the page in static mode
loads the prepared runs without ever calling /api. No model calls; the browser check needs Node's Playwright."""
import importlib.util
import json
import shutil
import socket
import subprocess
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from .conftest import ROOT, twin_clip

RESULTS = ROOT / "tests" / "api" / "fixtures" / "real" / "aeg-washer-54492b8" / "results.json"
RUNS = [("washer-imbalance", "Washing machine twin: imbalance", "washer", "imbalance", "a1b2c3d4e5f6"),
        ("real-washer", "Real washing machine", "real", None, "0f1e2d3c4b5a")]


def _export_static():
    spec = importlib.util.spec_from_file_location("export_static", ROOT / "scripts" / "export_static.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def site(tmp_path):
    """A small prepared set like scripts/prepare_hosted.py writes (a real run's results.json, stand-in media and a
    real clip as the video), exported with a video limit low enough to force a re-encode."""
    ex = _export_static()
    runs = tmp_path / "hosted_runs"
    res = json.loads(RESULTS.read_text())
    for key, label, machine, fault, rid in RUNS:
        d = runs / rid
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps(res))
        for name in ex.referenced(res["files"]):
            (d / name).parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".mp4"):
                shutil.copy(twin_clip("healthy") / "clip.mp4", d / name)
            else:
                cv2.imwrite(str(d / name), np.full((40, 60, 3), 128, np.uint8))
    (runs / "prepared.json").write_text(json.dumps({"runs": [
        {"key": k, "label": label, "machine": m, "fault": f, "run_id": r} for k, label, m, f, r in RUNS]}))
    out = tmp_path / "site"
    video = (runs / RUNS[0][4] / res["files"]["side_by_side"]).stat().st_size
    ex.export(runs, out, max_video_mb=video / 2e6)  # half the clip's size: it must be re-encoded
    return ex, out, res, video


def test_export_writes_every_file_the_runs_reference(site):
    ex, out, res, video = site
    index = json.loads((out / "runs" / "index.json").read_text())
    assert [p["key"] for p in index["prepared"]] == [k for k, *_ in RUNS]
    assert len(index["steps"]) == 7 and index["quickstart_url"].endswith("#quickstart")
    washer = {f["key"]: f["recorded"] for f in index["machines"][0]["faults"]}
    assert washer == {"healthy": False, "imbalance": True, "looseness": False, "misalignment": False}
    for *_, rid in RUNS:
        d = out / "runs" / rid
        assert json.loads((d / "results.json").read_text()) == res
        for name in ex.referenced(res["files"]):
            assert (d / name).is_file(), name
        mp4 = d / res["files"]["side_by_side"]
        assert mp4.stat().st_size <= video / 2 and cv2.VideoCapture(str(mp4)).read()[0]  # re-encoded and playable
    page = (out / "index.html").read_text()
    assert page.count('<html lang="en" data-mode="static">') == 1 and '<html lang="en">' not in page
    card = (out / "README.md").read_text()
    assert card.startswith("---\ntitle: See the Shake\n") and "sdk: static" in card and "license: mit" in card
    assert "https://youtu.be/08ZK__uc-3g" in card and "github.com/iamarunbrahma/see-the-shake" in card


def test_export_credits_the_real_clips(site):
    """The real runs republish videos made from a CC BY-SA 4.0 and a CC BY 4.0 clip: the page's footer and the Space
    card credit both, and the app's own page is left without the footer."""
    ex, out, _, _ = site
    page = (out / "index.html").read_text()
    assert page.count("<footer") == 1 and page.index("</main>") < page.index("<footer") < page.index("<script>")
    footer = page[page.index("<footer"):page.index("</footer>")]
    card = (out / "README.md").read_text()
    credits = card[card.index("## Credits"):]
    for text in (footer, credits):
        for need in ("Washing machine 120fps", "Golha", "CC BY-SA 4.0", "Kang, Zheng, Liu", "CC BY 4.0",
                     "commons.wikimedia.org/wiki/File:Washing_machine_120fps.webm", "47y7rwc2yg.1"):
            assert need in text, need
    assert 'href="https://github.com/iamarunbrahma/see-the-shake"' in footer
    assert 'href="https://youtu.be/08ZK__uc-3g"' in footer
    for text in (page, card):
        assert "\u2014" not in text and "\u2013" not in text  # no em or en dashes
    assert "<footer" not in ex.PAGE.read_text()


def _has_playwright():
    return shutil.which("node") and subprocess.run(["node", "-e", "require('playwright')"], capture_output=True,
                                                   cwd=ROOT).returncode == 0


BROWSER = r"""
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const pg = await b.newPage();
  const urls = [], errors = [];
  pg.on('request', r => urls.push(r.url()));
  pg.on('pageerror', e => errors.push(String(e)));
  await pg.goto(process.argv[1]);
  await pg.waitForSelector('#prepared_list button');
  const tab = await pg.isDisabled('#tab-clip');
  const choices = await pg.$$eval('#twin_api option', os => os.map(o => o.value));
  await pg.click('#prepared_list button >> nth=1');
  await pg.waitForSelector('#results:not(.hidden)');
  const verdict = await pg.innerText('#results');
  await pg.click('#form-twin button[type=submit]');  // the twin's recorded run
  await pg.waitForFunction(() => document.querySelector('#video').src.includes('a1b2c3d4e5f6'));
  console.log(JSON.stringify({ urls, errors, tab, choices, verdict: verdict.slice(0, 400),
                               note: await pg.innerText('#static_note'), footer: await pg.innerText('footer') }));
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
"""


@pytest.mark.skipif(not _has_playwright(), reason="needs Node and its playwright package")
def test_static_page_never_calls_the_api(site):
    _, out, _, _ = site
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), partial(SimpleHTTPRequestHandler, directory=str(out)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        r = subprocess.run(["node", "-e", BROWSER, f"http://127.0.0.1:{port}/"], capture_output=True, text=True,
                           cwd=ROOT, timeout=120)
    finally:
        httpd.shutdown()
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert not [u for u in got["urls"] if "/api/" in u], got["urls"]
    assert not got["errors"], got["errors"]
    assert got["tab"] and got["choices"] == ["replay"]  # no upload, no live run, no rendering
    assert "Imbalance" in got["verdict"] and "Quickstart" in got["note"]
    assert "CC BY-SA 4.0" in got["footer"] and "CC BY 4.0" in got["footer"] and "Demo video" in got["footer"]
    assert any(u.endswith("/runs/index.json") for u in got["urls"])
    assert any("/runs/0f1e2d3c4b5a/results.json" in u for u in got["urls"])  # a run opens from its own folder

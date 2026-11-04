"""Export the prepared runs (scripts/prepare_hosted.py) as a static site for a Hugging Face Static Space.

site/ holds:
- index.html: the app's page in static mode (data-mode="static"), which never calls /api;
- runs/index.json: what /api/samples gives the page (steps, machines and faults), plus the prepared runs;
- runs/<id>/results.json and every file it references (videos re-encoded when larger than --max-video-mb);
- README.md: the Space's card.

usage: python scripts/export_static.py [--runs hosted_runs] [--out site] [--max-video-mb 5]
"""
import argparse
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAGE = ROOT / "shake" / "web" / "static" / "index.html"
GITHUB = "https://github.com/iamarunbrahma/see-the-shake"
QUICKSTART_URL = GITHUB + "#quickstart"
DEMO_VIDEO = "https://youtu.be/08ZK__uc-3g"
SPACE_README = f"""---
title: See the Shake
emoji: 📳
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
short_description: Phone slow-motion video to machine vibration diagnosis
---

Code: [github.com/iamarunbrahma/see-the-shake]({GITHUB})
Demo video: [youtu.be/08ZK__uc-3g]({DEMO_VIDEO})
"""


def _labels(name):
    """A dict literal assigned at the top of shake/web/app.py (read, not imported: importing builds the app)."""
    tree = ast.parse((ROOT / "shake" / "web" / "app.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def referenced(files):
    """Every file name in a results.json "files" entry (strings, possibly nested in dicts)."""
    if isinstance(files, str):
        return [files]
    if isinstance(files, dict):
        return [f for v in files.values() for f in referenced(v)]
    return []


def small_video(src, dst, max_bytes):
    """Copy a video, re-encoding it (H.264, no audio, fast start) when it is larger than max_bytes."""
    if src.stat().st_size <= max_bytes:
        shutil.copy2(src, dst)
        return
    for crf in (26, 29, 32, 35):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "libx264",
                        "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(dst)], check=True, stdin=subprocess.DEVNULL)
        if dst.stat().st_size <= max_bytes:
            return


def samples(prepared):
    """The page's settings, as /api/samples gives them, with a twin's recorded run offered where one is prepared."""
    from shake import pipeline, twin
    ready = {(p["machine"], p["fault"]) for p in prepared}
    labels = {"washer": _labels("FAULT_LABELS"), "fan": _labels("FAN_FAULT_LABELS")}
    faults = {"washer": list(twin.FAULTS), "fan": list(pipeline.twin_faults("fan"))}
    machines = [{"key": m, "label": pipeline.MACHINES[m],
                 "faults": [{"key": k, "label": labels[m][k], "recorded": (m, k) in ready} for k in faults[m]]}
                for m in ("washer", "fan")]
    return {"machines": machines, "steps": [{"key": k, "label": label} for k, label in pipeline.STEPS],
            "max_upload_bytes": 0, "quickstart_url": QUICKSTART_URL,
            "prepared": [{k: p.get(k) for k in ("key", "label", "machine", "fault", "run_id")} for p in prepared]}


def export(runs, out, max_video_mb=5.0):
    runs, out = Path(runs), Path(out)
    prepared = json.loads((runs / "prepared.json").read_text())["runs"]
    shutil.rmtree(out, ignore_errors=True)
    (out / "runs").mkdir(parents=True)
    for p in prepared:
        src, dst = runs / p["run_id"], out / "runs" / p["run_id"]
        dst.mkdir()
        res = json.loads((src / "results.json").read_text())
        shutil.copy2(src / "results.json", dst / "results.json")
        for name in referenced(res.get("files", {})):
            if not (src / name).is_file():
                raise SystemExit(f"{p['key']}: results.json references {name}, which is missing")
            (dst / name).parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".mp4"):
                small_video(src / name, dst / name, max_video_mb * 1e6)
            else:
                shutil.copy2(src / name, dst / name)
    (out / "runs" / "index.json").write_text(json.dumps(samples(prepared), indent=1))
    page = PAGE.read_text()
    if page.count('<html lang="en">') != 1:
        raise SystemExit("index.html no longer starts with <html lang=\"en\">; update export_static.py")
    (out / "index.html").write_text(page.replace('<html lang="en">', '<html lang="en" data-mode="static">'))
    (out / "README.md").write_text(SPACE_README)
    return prepared


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--runs", default=str(ROOT / "hosted_runs"))
    a.add_argument("--out", default=str(ROOT / "site"))
    a.add_argument("--max-video-mb", type=float, default=5.0)
    a = a.parse_args(argv)
    prepared = export(a.runs, a.out, a.max_video_mb)
    out = Path(a.out)
    videos = sorted(out.rglob("*.mp4"))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{len(prepared)} runs in {out}: {size / 1e6:.1f} MB in all; videos "
          + ", ".join(f"{v.stat().st_size / 1e6:.1f} MB" for v in videos))
    return 0


if __name__ == "__main__":
    sys.exit(main())

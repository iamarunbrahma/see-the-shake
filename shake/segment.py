"""Segment (SAM 3.1): one stable mask per part phrase.

Vibration is sub-pixel, so a part's mask barely changes from frame to frame. Rather than paying for every frame,
a short sub-clip (SUB_FRAMES frames of the measurement window, at the source resolution) is uploaded once and
each phrase gets one streaming Responses request on it:
- input_video by file_id, one noun-phrase concept per request;
- metadata mask_encoding "one_bit" and include_confidence "true";
- the stream is parsed with meta_sam_parser (box-local masks, <Nf> source frame indices, frames with no object
  left out).
The stable mask is the per-pixel majority over MASK_FRAMES evenly spaced frames of the chosen object (the object
seen in the most frames, weighted by confidence). meta_sam_parser decodes and re-encodes every mask in pure Python
(about 1 s per 1080p whole-machine mask), so only those frames are parsed, each as its own consistent one-line
stream in a process pool: the parser still validates every mask it is given, and presence and confidence come
from every frame's header. A phrase that finds nothing is reported missing; if every phrase finds
nothing, the whole-machine fallback phrase is tried. SAM bills at least 50 frames per request, so SUB_FRAMES = 60
costs $0.012 per phrase.
"""
import asyncio
import copy
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .meta_api import SAM_MIN_FRAMES, MetaAPI

SUB_FRAMES = 60
SUB_PLAYBACK_FPS = 30  # the sub-clip keeps every source frame; only its playback rate changes
MAJORITY = 0.5
MASK_FRAMES = 8  # frames decoded per phrase for the majority mask
MIN_PRESENT = 0.5  # warn when the chosen object is in fewer than this share of the sub-clip frames


log = logging.getLogger("shake.segment")


class SegmentError(RuntimeError):
    pass


def subclip(clip, out, start_frame, frames=SUB_FRAMES, playback_fps=SUB_PLAYBACK_FPS):
    """frames consecutive source frames from start_frame, at the source resolution, as H.264 MP4."""
    if frames < SAM_MIN_FRAMES:
        frames = SAM_MIN_FRAMES  # billed anyway
    vf = f"select='between(n\\,{start_frame}\\,{start_frame + frames - 1})',setpts=N/{playback_fps}/TB"
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(clip), "-vf", vf, "-r", str(playback_fps),
           "-frames:v", str(frames), "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise SegmentError(f"ffmpeg could not cut the sub-clip: {r.stderr.strip()[:300]}")
    return {"path": str(out), "start_frame": int(start_frame), "frames": int(frames)}


# ---------------------------------------------------------------- parsing

def _lanes_closed(events):
    """The cookbook's adapter: the API streams response.content_part.done but not response.output_text.done,
    which the parser uses to close a lane, so synthesise it from the content part."""
    done = set()
    for d in events:
        lane = (d.get("item_id"), d.get("output_index"), d.get("content_index"))
        if d.get("type") == "response.output_text.done":
            if lane not in done:
                done.add(lane)
                yield d
            continue
        yield d
        part = d.get("part")
        if (d.get("type") == "response.content_part.done" and lane not in done and isinstance(part, dict)
                and part.get("type") == "output_text" and isinstance(part.get("text"), str)):
            done.add(lane)
            yield {"type": "response.output_text.done", "item_id": d.get("item_id"),
                   "output_index": d.get("output_index"), "content_index": d.get("content_index"),
                   "text": part["text"]}


_HEADER = re.compile(r"^<([0-9]+)f(?:;[^>]*)?>")
_OBJECT = re.compile(r"(?:^|,|>)([0-9]+)<\|box;([^|]*)\|>")


def stream_text(events):
    return "".join(d.get("delta", "") for d in events if d.get("type") == "response.output_text.delta")


def frame_stats(text):
    """{object id: {"frames": [frame indices], "conf": [box confidences]}} from every frame line's header."""
    stats = {}
    for line in text.splitlines():
        h = _HEADER.match(line)
        if not h:
            continue
        for oid, fields in _OBJECT.findall(line[h.end():]):
            o = stats.setdefault(oid, {"frames": [], "conf": []})
            o["frames"].append(int(h.group(1)))
            c = re.search(r"(?:^|;)c=([0-9.eE+-]+)", fields)
            if c:
                o["conf"].append(float(c.group(1)))
    return stats


def _set_text(obj, text):
    """Replace the output_text text inside a response or output item (single text lane)."""
    for item in ([obj] if obj.get("type") == "message" else obj.get("output") or []):
        for c in item.get("content") or []:
            if c.get("type") == "output_text":
                c["text"] = text


def _deltas(events):
    """Positions of the delta events, or None when the stream is not one lane of whole frame lines."""
    idx = [i for i, d in enumerate(events) if d.get("type") == "response.output_text.delta"]
    lanes = {(events[i].get("item_id"), events[i].get("content_index")) for i in idx}
    if len(lanes) != 1 or not all(events[i]["delta"].endswith("\n") for i in idx[:-1]):
        return None
    return idx


def thin_events(events, keep=MASK_FRAMES, select=None):
    """A consistent copy of the stream that carries only `keep` evenly spaced frame lines (or the lines whose
    positions are in `select`).

    Each delta is one frame line; the final text is repeated in output_text.done, content_part.done,
    output_item.done and response.completed, and the parser checks it against the deltas, so all of them are
    rewritten. Streams that do not have that shape (several lanes, deltas that are not whole lines) pass
    through unchanged.
    """
    idx = _deltas(events)
    if idx is None or (select is None and len(idx) <= keep):
        return events
    picks = select if select is not None else np.unique(np.linspace(0, len(idx) - 1, keep).round().astype(int))
    sel = {idx[j] for j in picks}
    kept = "".join(events[i]["delta"] for i in idx if i in sel)
    out = []
    for i, d in enumerate(events):
        t = d.get("type")
        if t == "response.output_text.delta" and i not in sel:
            continue
        if t in ("response.output_text.done", "response.content_part.done", "response.output_item.done",
                 "response.completed", "response.incomplete"):
            d = copy.deepcopy(d)
            if t == "response.output_text.done":
                d["text"] = kept
            elif t == "response.content_part.done" and (d.get("part") or {}).get("type") == "output_text":
                d["part"]["text"] = kept
            elif t == "response.output_item.done":
                _set_text(d.get("item") or {}, kept)
            else:
                _set_text(d.get("response") or {}, kept)
        out.append(d)
    return out


def parse_events(events):
    """meta_sam_parser's VideoSegmentationResult for a list of recorded Responses stream events."""
    from meta_sam_parser import parse_responses_stream, video_segmentation_format

    async def source():
        for d in _lanes_closed(events):
            yield d

    async def run():
        parsed = parse_responses_stream(source(), video_segmentation_format())
        async with parsed:
            return await parsed.final_result()

    return asyncio.run(run())


def _check(result):
    if result.outcome.status != "completed":
        raise SegmentError(f"SAM response incomplete: {getattr(result.outcome, 'reason', '?')} "
                           f"({getattr(result.outcome, 'detail', None)})")
    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        raise SegmentError("parser rejected records: " + "; ".join(f"line {d.line}: {d.message}" for d in errors))


def mask_records(result):
    """[(frame index, object id, (x0, y0, x1, y1), raster uint8 rows x cols, confidence)] for the latest revision
    of each (frame, object) mask. Masks are box-local rasters; bounds are half-open source pixels."""
    from meta_sam_parser import decode_mask_to_raster

    latest = {}
    for rec in result.records:
        if rec.kind == "mask":
            k = (rec.frame.frame_index if rec.frame else 0, rec.object_id)
            if k not in latest or rec.revision >= latest[k].revision:
                latest[k] = rec
    out = []
    for (fi, oid), rec in sorted(latest.items()):
        m, b = rec.mask, rec.bounds
        r = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)
        out.append((fi, oid, (int(b.left), int(b.top), int(b.right), int(b.bottom)), r, rec.confidence))
    return out


def _decode_stream(events):
    result = parse_events(events)
    _check(result)
    return mask_records(result)


def decode_frames(events, keep=MASK_FRAMES, workers=None):
    """Mask records of `keep` evenly spaced frames. Each frame line is parsed and decoded as its own consistent
    one-line stream, in a process pool (the parser is pure Python and CPU-bound)."""
    idx = _deltas(events)
    if not keep or idx is None or len(idx) <= 1:
        return _decode_stream(events)
    picks = np.unique(np.linspace(0, len(idx) - 1, min(keep, len(idx))).round().astype(int))
    streams = [thin_events(events, select={int(j)}) for j in picks]
    workers = workers if workers is not None else min(len(streams), os.cpu_count() or 1, 8)
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        # never plain fork: the caller may be multi-threaded (a web server, pytest's timeout thread)
        ctx = mp.get_context("forkserver" if sys.platform.startswith("linux") else "spawn")
        try:
            with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
                return [r for part in ex.map(_decode_stream, streams) for r in part]
        except (OSError, RuntimeError, EOFError, BrokenProcessPool) as e:  # no worker processes: decode here
            log.warning("parallel mask decode failed (%s); decoding serially", e)
    return [r for st in streams for r in _decode_stream(st)]


def object_masks(records, shape):
    """{object id: {"count": per-pixel frames covered, "frames": n, "conf": [..], "areas": [..]}} in frame coords.
    Each box-local raster is pasted into a full frame of shape (H, W)."""
    H, W = shape
    objs = {}
    for fi, oid, (x0, y0, x1, y1), r, conf in records:
        if (y1 - y0, x1 - x0) != r.shape:  # not seen so far: rasters match their box exactly
            import cv2
            r = cv2.resize(r, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        o = objs.setdefault(oid, {"count": np.zeros((H, W), np.uint16), "frames": 0, "conf": [], "areas": [],
                                  "frame_indices": []})
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        o["count"][cy0:cy1, cx0:cx1] += r[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        o["frames"] += 1
        o["frame_indices"].append(fi)
        o["areas"].append(int(r.sum()))
        if conf is not None:
            o["conf"].append(float(conf))
    return objs


def stable_mask(events, shape, frames_sent, mask_frames=MASK_FRAMES, workers=None):
    """(uint8 0/255 mask or None, info) for one phrase's stream."""
    stats = frame_stats(stream_text(events))
    info = {"objects": len(stats), "frames_sent": frames_sent, "per_object": {
        oid: {"frames": len(o["frames"]), "confidence": round(float(np.mean(o["conf"])), 4) if o["conf"] else None}
        for oid, o in stats.items()}}
    objs = object_masks(decode_frames(events, mask_frames, workers), shape)
    if not objs:
        info.update(found=False)
        return None, info
    for oid, o in objs.items():
        info["per_object"].setdefault(oid, {})["mean_area_px"] = int(np.mean(o["areas"]))

    def score(oid):
        st = stats.get(oid, {"frames": [], "conf": []})
        return len(st["frames"]) * (np.mean(st["conf"]) if st["conf"] else 0.5), np.mean(objs[oid]["areas"])

    oid = max(objs, key=score)
    o, st = objs[oid], stats.get(oid, {"frames": [], "conf": []})
    n_seen = len(st["frames"]) or o["frames"]
    mask = (o["count"] >= MAJORITY * o["frames"]).astype(np.uint8) * 255
    ys, xs = np.nonzero(mask)
    area = int(len(ys))
    unstable = float(np.sum((o["count"] > 0) & (o["count"] < o["frames"]))) / max(area, 1)  # edge pixels that flip
    conf = st["conf"] or o["conf"]
    info.update(found=area > 0, object_id=oid, frames_returned=n_seen, frames_decoded=o["frames"],
                present_share=round(n_seen / max(frames_sent, 1), 3),
                confidence=round(float(np.mean(conf)), 4) if conf else None,
                area_px=area, bbox=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if area else None,
                unstable_share=round(unstable, 4))
    if n_seen < MIN_PRESENT * frames_sent:
        info.setdefault("warnings", []).append(f"object seen in only {n_seen} of {frames_sent} frames")
    return (mask if area else None), info


# ---------------------------------------------------------------- requests

def segment_phrase(api: MetaAPI, file_id, phrase, shape, frames_sent):
    events = api.sam_video(file_id, phrase, frames_sent=frames_sent)
    mask, info = stable_mask(events, shape, frames_sent)
    info["usd"] = api.ledger.calls[-1]["usd"]
    info["frames_billed"] = api.ledger.calls[-1].get("frames_billed")
    return mask, info


def segment_file(api: MetaAPI, file_id, phrases, shape, frames_sent, fallback_phrase=None):
    """Masks for each phrase on an uploaded sub-clip. Returns ({phrase: mask}, {phrase: info})."""
    masks, infos = {}, {}
    for p in phrases:
        m, info = segment_phrase(api, file_id, p, shape, frames_sent)
        infos[p] = info
        if m is not None:
            masks[p] = m
    if not masks and fallback_phrase:
        for p in [fallback_phrase, "machine"]:
            if p in infos:
                continue
            m, info = segment_phrase(api, file_id, p, shape, frames_sent)
            info["fallback"] = True
            infos[p] = info
            if m is not None:
                masks[p] = m
                break
    return masks, infos


def segment_parts(api: MetaAPI, clip, phrases, shape, start_frame=0, frames=SUB_FRAMES, fallback_phrase=None,
                  workdir=None, label="segment"):
    """Cut the sub-clip, upload it once, and run one SAM 3.1 request per phrase."""
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        sub = subclip(clip, Path(td) / "sam_subclip.mp4", start_frame, frames)
        fid = api.upload(sub["path"], label=f"{label}:subclip")
    masks, infos = segment_file(api, fid, phrases, shape, sub["frames"], fallback_phrase)
    return masks, {"file_id": fid, "subclip": {k: sub[k] for k in ("start_frame", "frames")}, "parts": infos,
                   "missing": [p for p in phrases if p not in masks],
                   "usd": round(sum(i.get("usd", 0.0) for i in infos.values()), 6)}


def iou(a, b):
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0

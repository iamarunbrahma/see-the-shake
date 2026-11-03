"""Shared Meta Model API client: every model call in See the Shake goes through here.

- OpenAI SDK on MODEL_API_BASE (default https://api.meta.ai/v1) with the key from MODEL_API_KEY. In cloud sessions
  the key may be a network secret, so MODEL_API_KEY can read "proxy-injected".
- Retries with exponential backoff and jitter on 429, 5xx and connection errors, honouring Retry-After.
  400, 401, 403, 404 and 413 are not retried. 402 raises CreditExhausted: the account is out of credit, so all API
  work stops.
- A live run checks first that MODEL_API_KEY is set (require_key), so it fails before any work is done.
- Files API uploads (purpose "user_data", for SAM 3.1 and Muse Spark alike).
- Muse Spark through the streaming Responses API, with a large output budget (it reasons before it answers)
  and a check that the response completed.
- SAM 3.1 through the streaming Responses API; every raw event is kept so meta_sam_parser can parse it, and so
  a run can be recorded and replayed.
- Every call is costed and appended to a ledger.
- Record / replay: a Cassette stores each call's response (never headers or the key). Tests replay cassettes,
  so the default suite makes no network calls.

The key is never logged: log records from this module, openai and httpx pass through a redacting filter.
"""
import gzip
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

BASE_DEFAULT = "https://api.meta.ai/v1"
SPARK = "muse-spark-1.3"
SAM = "sam-3.1"
# USD per token (dev.meta.ai/docs/pricing-rate-limits, standard tier; the contributor tier trains on your content).
SPARK_PRICE = {"input": 1.25e-6, "cached_input": 0.15e-6, "output": 4.25e-6}
SAM_USD_PER_FRAME = 0.20 / 1000
SAM_MIN_FRAMES = 50
SPARK_MIN_OUTPUT_TOKENS = 4096  # Spark reasons first; small budgets end incomplete with no answer
SPARK_OUTPUT_TOKENS = 16384
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
ATTEMPTS = 5
BACKOFF_S, BACKOFF_CAP_S = 0.5, 30.0
TIMEOUT_S = 600.0

log = logging.getLogger("shake.api")


class APIError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(redact(message))
        self.status = status


class CreditExhausted(APIError):
    """HTTP 402: the account has no credit left. All API work stops."""


class IncompleteResponse(APIError):
    pass


class MissingKey(APIError):
    """MODEL_API_KEY is not set, so a live run cannot start."""


KEY_MISSING_MESSAGE = ("MODEL_API_KEY is not set. Live runs call the Meta Model API with your key: run "
                       "export MODEL_API_KEY=<your key> and start again. Recorded runs and runs with Muse Spark "
                       "and SAM 3.1 off need no key.")


def require_key():
    """Raise MissingKey unless MODEL_API_KEY holds something (a key, or a proxy's placeholder such as
    "proxy-injected" where a network proxy adds the real key)."""
    if not os.environ.get("MODEL_API_KEY", "").strip():
        raise MissingKey(KEY_MISSING_MESSAGE)


# ---------------------------------------------------------------- key hygiene

def _key_values():
    k = os.environ.get("MODEL_API_KEY", "")
    return [k] if len(k) >= 8 and k != "proxy-injected" else []


_TOKENISH = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}|\bLLM_[A-Za-z0-9._\-]{8,}")


def redact(text):
    """Remove the API key (and anything shaped like a bearer token) from text."""
    s = str(text)
    for k in _key_values():
        s = s.replace(k, "[redacted]")
    return _TOKENISH.sub(lambda m: (m.group(1) or "") + "[redacted]", s)


class RedactFilter(logging.Filter):
    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        clean = redact(msg)
        if clean != msg:
            record.msg, record.args = clean, None
        return True


_FILTER = RedactFilter()
for _name in ("shake.api", "openai", "httpx", "httpx2", "httpcore"):
    logging.getLogger(_name).addFilter(_FILTER)


def make_client(timeout=TIMEOUT_S):
    """OpenAI SDK client on the Meta Model API. SDK retries are off; call_with_retries does them."""
    import openai
    return openai.OpenAI(api_key=os.environ.get("MODEL_API_KEY", "proxy-injected"),
                         base_url=os.environ.get("MODEL_API_BASE", BASE_DEFAULT), max_retries=0, timeout=timeout)


# ---------------------------------------------------------------- retries

def _retry_after(err):
    resp = getattr(err, "response", None)
    try:
        v = resp.headers.get("retry-after") if resp is not None else None
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def call_with_retries(fn, what, attempts=ATTEMPTS, sleep=time.sleep):
    """Run fn(); retry transient failures with backoff. Raises APIError / CreditExhausted otherwise."""
    import openai
    for i in range(attempts):
        try:
            return fn()
        except openai.APIStatusError as e:
            status = e.status_code
            msg = f"{what}: HTTP {status}: {_error_message(e)}"
            if status == 402:
                raise CreditExhausted(msg + ". The Meta Model API account has no credit left, so API work has "
                                      "stopped.", 402) from None
            if status not in RETRY_STATUS or i == attempts - 1:
                raise APIError(msg, status) from None
            wait = _retry_after(e)
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            if i == attempts - 1:
                raise APIError(f"{what}: connection failed after {attempts} attempts: {e}") from None
            msg, wait = f"{what}: {type(e).__name__}", None
        except _StreamFailed as e:  # response.failed or an error event mid-stream
            if not e.retryable or i == attempts - 1:
                raise APIError(f"{what}: {e}", e.status) from None
            msg, wait = f"{what}: {e}", None
        backoff = min(BACKOFF_CAP_S, BACKOFF_S * 2 ** i) * (0.5 + random.random())
        wait = max(wait or 0.0, backoff)
        log.warning("%s; retry %d/%d in %.1f s", msg, i + 1, attempts - 1, wait)
        sleep(wait)


def _error_message(e):
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            return f"{err.get('message')} ({err.get('code') or err.get('type')})"
    return str(getattr(e, "message", e))[:300]


class _StreamFailed(Exception):
    def __init__(self, message, retryable=True, status=None):
        super().__init__(message)
        self.retryable, self.status = retryable, status


# ---------------------------------------------------------------- record / replay

class Cassette:
    """Record of API interactions. mode 'record' appends and saves; 'replay' serves them back.

    Each interaction is {"op", "model", "key", "request", "response"}. request is a summary (prompt text,
    file ids, metadata), never headers or credentials. Replay keeps the recorded order of steps but, within a run of
    consecutive calls of the same op (the SAM phrases, which are requested concurrently), serves the unused call
    with the same model and key in any order. A call out of step or never recorded fails loudly instead of using
    the wrong response. Thread-safe.
    """

    def __init__(self, path, mode="replay"):
        if mode not in ("record", "replay"):
            raise ValueError(mode)
        self.path, self.mode = Path(path), mode
        self.items = _read_json(self.path) if mode == "replay" else []
        self.pos = 0  # index of the first unused recording; len(items) once all are replayed
        self._used = set()
        self._lock = threading.Lock()

    def add(self, op, model, key, request, response):
        with self._lock:
            self.items.append({"op": op, "model": model, "key": key, "request": request, "response": response})
            _write_json(self.path, self.items)

    def next(self, op, model, key):
        with self._lock:
            # candidates: the unused calls in the run of same-op calls that starts at the first unused one
            i = self.pos
            while i < len(self.items) and self.items[i]["op"] == op:
                it = self.items[i]
                if i not in self._used and (it["model"], it["key"]) == (model, key):
                    self._used.add(i)
                    while self.pos in self._used:
                        self.pos += 1
                    return it["response"]
                i += 1
            nxt = self.items[self.pos] if self.pos < len(self.items) else None
            have = f"next unused is {nxt['op']} {nxt['model']} {nxt['key']!r}" if nxt else "none left"
            raise APIError(f"cassette {self.path.name}: no recorded call for {op} {model} {key!r} ({have})")

    def has(self, op, model, key):
        """Whether next(op, model, key) would find a recording."""
        with self._lock:
            i = self.pos
            while i < len(self.items) and self.items[i]["op"] == op:
                it = self.items[i]
                if i not in self._used and (it["model"], it["key"]) == (model, key):
                    return True
                i += 1
            return False

    def save(self):
        with self._lock:
            _write_json(self.path, self.items)


def _read_json(path):
    raw = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, separators=(",", ":")).encode()
    if redact(data.decode()) != data.decode():
        raise APIError("refusing to write a recording that contains the API key")
    path.write_bytes(gzip.compress(data, mtime=0) if str(path).endswith(".gz") else data)


# ---------------------------------------------------------------- cost

def spark_cost(usage):
    """USD for one Spark response from its usage block (Responses API field names)."""
    if not usage:
        return 0.0
    inp = usage.get("input_tokens") or 0
    out = usage.get("output_tokens") or 0
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
    return (inp - cached) * SPARK_PRICE["input"] + cached * SPARK_PRICE["cached_input"] + out * SPARK_PRICE["output"]


def sam_cost(frames_billed):
    return max(SAM_MIN_FRAMES, int(frames_billed or 0)) * SAM_USD_PER_FRAME


@dataclass
class Ledger:
    """Costed calls of one run, optionally appended to a JSONL file."""
    path: Path = None
    calls: list = field(default_factory=list)

    _lock: object = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, entry):
        entry = {"ts": round(time.time(), 3), **entry}
        with self._lock:
            self.calls.append(entry)
            if self.path:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as fh:
                    fh.write(json.dumps(entry) + "\n")

    def total(self, **match):
        return round(sum(c.get("usd", 0.0) for c in self.calls if all(c.get(k) == v for k, v in match.items())), 6)

    def summary(self):
        by = {}
        for c in self.calls:
            k = (c["op"], c.get("label"))
            b = by.setdefault(k, {"op": c["op"], "label": c.get("label"), "calls": 0, "usd": 0.0})
            b["calls"] += 1
            b["usd"] = round(b["usd"] + c.get("usd", 0.0), 6)
        return {"usd": self.total(), "calls": len(self.calls), "by_step": list(by.values())}


# ---------------------------------------------------------------- the client

@dataclass
class SparkReply:
    text: str
    status: str
    usage: dict
    usd: float
    response_id: str = None
    incomplete_reason: str = None

    def json(self):
        """The reply parsed as JSON (tolerates a ```json fence)."""
        return parse_json_text(self.text)


def parse_json_text(text):
    s = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    if not s.startswith(("{", "[")):
        i = min([j for j in (s.find("{"), s.find("[")) if j >= 0], default=-1)
        if i < 0:
            raise ValueError(f"no JSON in reply: {s[:200]!r}")
        s = s[i:]
    obj, _ = json.JSONDecoder().raw_decode(s)
    return obj


class MetaAPI:
    """One object per run. mode: 'live' (network), 'record' (network, saved to cassette), 'replay' (no network)."""

    def __init__(self, mode="live", cassette=None, ledger=None, client=None, sleep=time.sleep):
        if mode not in ("live", "record", "replay"):
            raise ValueError(mode)
        if mode != "live" and cassette is None:
            raise ValueError(f"mode {mode!r} needs a cassette path")
        self.mode = mode
        self.cassette = None if cassette is None else (
            cassette if isinstance(cassette, Cassette) else Cassette(cassette, "replay" if mode == "replay" else "record"))
        self.ledger = ledger if isinstance(ledger, Ledger) else Ledger(ledger)
        self._client = client
        self._sleep = sleep

    @property
    def client(self):
        if self.mode == "replay":
            raise APIError("replay mode makes no network calls")
        if self._client is None:
            self._client = make_client()
        return self._client

    def will_call(self, op, key):
        """Whether an optional call (a retry) should be made: always live; in replay, only if it was recorded."""
        if self.mode != "replay":
            return True
        return self.cassette.has(op, {"sam": SAM, "spark": SPARK, "upload": "files"}[op], key)

    def _call(self, op, model, key, request, live_fn):
        if self.mode == "replay":
            return self.cassette.next(op, model, key), True
        resp = call_with_retries(live_fn, f"{op} {model} {key}", sleep=self._sleep)
        if self.mode == "record":
            self.cassette.add(op, model, key, request, resp)
        return resp, False

    # Files API
    def upload(self, path, purpose="user_data", label=None, expires_s=7 * 24 * 3600):
        """Upload a file; returns its file id. Files expire after expires_s (the API keeps them forever otherwise)."""
        path = Path(path)
        key = label or path.name
        size = path.stat().st_size if path.exists() else None

        def live():
            with open(path, "rb") as fh:
                kw = {"expires_after": {"anchor": "created_at", "seconds": int(expires_s)}} if expires_s else {}
                obj = self.client.files.create(file=(path.name, fh, _mime(path)), purpose=purpose, **kw)
            return obj.model_dump(mode="json")

        resp, replayed = self._call("upload", "files", key, {"filename": path.name, "bytes": size, "purpose": purpose},
                                    live)
        fid = resp.get("id")
        if not isinstance(fid, str) or not fid.startswith("file"):
            raise APIError(f"upload {path.name}: unexpected file id {fid!r}")
        self.ledger.add({"op": "upload", "model": "files", "label": key, "bytes": size, "file_id": fid, "usd": 0.0,
                         "replayed": replayed})
        log.info("uploaded %s (%s bytes) as %s", path.name, size, fid)
        return fid

    # Muse Spark
    def spark(self, content, label, schema=None, schema_name="answer", max_output_tokens=SPARK_OUTPUT_TOKENS,
              reasoning_effort=None, instructions=None):
        """One Muse Spark turn on the Responses API (streamed). content: a list of input_* blocks."""
        if max_output_tokens < SPARK_MIN_OUTPUT_TOKENS:
            raise ValueError(f"Spark needs max_output_tokens >= {SPARK_MIN_OUTPUT_TOKENS}; it reasons first")
        kw = {"model": SPARK, "input": [{"type": "message", "role": "user", "content": content}],
              "max_output_tokens": int(max_output_tokens), "stream": True}
        if instructions:
            kw["instructions"] = instructions
        if schema is not None:
            kw["text"] = {"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}}
        if reasoning_effort:
            kw["reasoning"] = {"effort": reasoning_effort}

        def live():
            final = None
            text = []
            for ev in self.client.responses.create(**kw):
                t = getattr(ev, "type", "")
                if t == "response.output_text.delta":
                    text.append(ev.delta)
                elif t in ("response.completed", "response.incomplete"):
                    final = ev.response.model_dump(mode="json")
                elif t == "response.failed":
                    err = (ev.response.model_dump(mode="json").get("error") or {})
                    raise _StreamFailed(f"response.failed: {err.get('message')} ({err.get('code')})",
                                        retryable=err.get("code") in (None, "server_error", "service_overloaded"))
                elif t == "error":
                    raise _StreamFailed(f"stream error: {getattr(ev, 'message', ev)}")
            if final is None:
                raise _StreamFailed("stream ended without response.completed")
            final["_streamed_text"] = "".join(text)
            return final

        request = {"content": _summarise_content(content), "schema": schema_name if schema else None,
                   "max_output_tokens": max_output_tokens, "reasoning_effort": reasoning_effort}
        resp, replayed = self._call("spark", SPARK, label, request, live)
        text = _output_text(resp)
        usage = resp.get("usage") or {}
        usd = spark_cost(usage)
        status = resp.get("status")
        reason = (resp.get("incomplete_details") or {}).get("reason")
        self.ledger.add({"op": "spark", "model": SPARK, "label": label, "usage": _usage_brief(usage),
                         "usd": round(usd, 6), "status": status, "replayed": replayed})
        log.info("spark %s: %s, %s in / %s out tokens, $%.4f", label, status, usage.get("input_tokens"),
                 usage.get("output_tokens"), usd)
        reply = SparkReply(text, status, usage, usd, resp.get("id"), reason)
        if status != "completed" or not text.strip():
            raise IncompleteResponse(f"spark {label}: status {status!r} ({reason}), {len(text)} chars of text; "
                                     f"raise max_output_tokens (was {max_output_tokens})")
        return reply

    # SAM 3.1
    def sam_video(self, file_id, phrase, frames_sent=None, include_confidence=True, mask_encoding="one_bit"):
        """One SAM 3.1 concept request on an uploaded video. Returns the raw stream events (dicts)."""
        phrase = phrase.strip()
        if not phrase:
            raise ValueError("SAM needs a short noun phrase")
        metadata = {"mask_encoding": mask_encoding, "include_confidence": "true" if include_confidence else "false"}
        kw = {"model": SAM, "stream": True, "metadata": metadata,
              "input": [{"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": phrase},
                                     {"type": "input_video", "file_id": file_id}]}]}

        def live():
            events = []
            for ev in self.client.responses.create(**kw):
                d = ev.model_dump(mode="json") if hasattr(ev, "model_dump") else dict(ev)
                events.append(d)
                if d.get("type") == "response.failed":
                    err = (d.get("response") or {}).get("error") or {}
                    raise _StreamFailed(f"response.failed: {err.get('message')} ({err.get('code')})",
                                        retryable=err.get("code") in (None, "server_error", "service_overloaded"))
                if d.get("type") == "error":
                    raise _StreamFailed(f"stream error: {d.get('message')}")
            return events

        events, replayed = self._call("sam", SAM, phrase, {"file_id": file_id, "metadata": metadata}, live)
        usage = _final_usage(events)
        billed = usage.get("video_frames_processed") if usage else None
        usd = sam_cost(billed if billed is not None else frames_sent)
        self.ledger.add({"op": "sam", "model": SAM, "label": phrase, "frames_billed": billed,
                         "frames_sent": frames_sent, "usd": round(usd, 6), "replayed": replayed})
        log.info("sam %r: %d events, %s frames billed, $%.4f", phrase, len(events), billed, usd)
        return events


def _mime(path):
    return {".mp4": "video/mp4", ".mov": "video/quicktime", ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".pdf": "application/pdf"}.get(Path(path).suffix.lower(), "application/octet-stream")


def _summarise_content(content):
    out = []
    for c in content:
        c = dict(c)
        if c.get("type") == "input_image" and str(c.get("image_url", "")).startswith("data:"):
            c["image_url"] = f"<data url, {len(c['image_url'])} chars>"
        out.append(c)
    return out


def _output_text(resp):
    parts = []
    for item in resp.get("output") or []:
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    parts.append(c.get("text") or "")
    return "".join(parts) or resp.get("output_text") or resp.get("_streamed_text") or ""


def _usage_brief(u):
    return {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
            "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
            "cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens")}


def _final_usage(events):
    for d in reversed(events):
        if d.get("type") in ("response.completed", "response.incomplete"):
            return (d.get("response") or {}).get("usage") or {}
    return {}


def image_block(png_path):
    """input_image content block with a PNG as a data URL."""
    import base64
    data = base64.b64encode(Path(png_path).read_bytes()).decode("ascii")
    return {"type": "input_image", "image_url": f"data:{_mime(png_path)};base64,{data}"}

"""The shared client, offline: retries and backoff, 402 handling, key redaction, cassettes, cost."""
import logging

import httpx2
import openai
import pytest

from shake import meta_api
from shake.meta_api import APIError, Cassette, CreditExhausted, IncompleteResponse, MetaAPI

KEY = "LLM_test_secret_key_0123456789abcdef"
REQ = httpx2.Request("POST", "https://api.meta.ai/v1/responses")


def status_error(code, retry_after=None, message="boom", err_code=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    cls = {400: openai.BadRequestError, 401: openai.AuthenticationError, 402: openai.APIStatusError,
           429: openai.RateLimitError, 500: openai.InternalServerError, 503: openai.InternalServerError}[code]
    return cls(message, response=httpx2.Response(code, headers=headers, request=REQ),
               body={"error": {"message": message, "code": err_code}})


def flaky(*errors, result="ok"):
    seq = list(errors)
    calls = []

    def fn():
        calls.append(1)
        if seq:
            raise seq.pop(0)
        return result

    fn.calls = calls
    return fn


def test_retries_transient_errors_with_backoff():
    sleeps = []
    fn = flaky(status_error(429, retry_after=3), status_error(503), openai.APIConnectionError(request=REQ))
    assert meta_api.call_with_retries(fn, "t", sleep=sleeps.append) == "ok"
    assert len(fn.calls) == 4 and len(sleeps) == 3
    assert sleeps[0] >= 3  # honours Retry-After
    assert all(s > 0 for s in sleeps)


def test_gives_up_after_attempts():
    fn = flaky(*[status_error(500)] * 9)
    with pytest.raises(APIError) as e:
        meta_api.call_with_retries(fn, "t", attempts=3, sleep=lambda s: None)
    assert e.value.status == 500 and len(fn.calls) == 3


@pytest.mark.parametrize("code", [400, 401])
def test_no_retry_on_client_errors(code):
    fn = flaky(status_error(code))
    with pytest.raises(APIError) as e:
        meta_api.call_with_retries(fn, "t", sleep=lambda s: pytest.fail("must not sleep"))
    assert e.value.status == code and len(fn.calls) == 1


def test_402_stops_immediately():
    fn = flaky(status_error(402, message="Billing account issue.", err_code="billing_not_configured"))
    with pytest.raises(CreditExhausted) as e:
        meta_api.call_with_retries(fn, "t", sleep=lambda s: pytest.fail("must not sleep"))
    assert len(fn.calls) == 1 and "tell Arun" in str(e.value)


def test_key_never_logged_or_raised(monkeypatch, caplog):
    monkeypatch.setenv("MODEL_API_KEY", KEY)
    assert KEY not in meta_api.redact(f"Authorization: Bearer {KEY}")
    assert KEY not in meta_api.redact(f"key={KEY}")
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("openai").debug("headers %s", {"Authorization": f"Bearer {KEY}"})
        logging.getLogger("shake.api").warning("oops %s", KEY)
    assert caplog.records and KEY not in caplog.text
    fn = flaky(status_error(400, message=f"bad key {KEY}"))
    with pytest.raises(APIError) as e:
        meta_api.call_with_retries(fn, "t", sleep=lambda s: None)
    assert KEY not in str(e.value)


def test_cassette_refuses_to_store_the_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_API_KEY", KEY)
    c = Cassette(tmp_path / "c.json", "record")
    with pytest.raises(APIError):
        c.add("spark", "m", "k", {}, {"echo": KEY})


def test_cassette_replays_in_order_and_checks_calls(tmp_path):
    c = Cassette(tmp_path / "c.json.gz", "record")
    c.add("upload", "files", "a.mp4", {}, {"id": "file-1"})
    c.add("sam", "sam-3.1", "door", {}, [{"type": "response.completed", "response": {"usage": {}}}])
    api = MetaAPI("replay", cassette=tmp_path / "c.json.gz")
    assert api.upload(tmp_path / "a.mp4") == "file-1"
    with pytest.raises(APIError, match="no recorded call for sam sam-3.1 'washing machine'"):
        api.sam_video("file-1", "washing machine")
    api = MetaAPI("replay", cassette=tmp_path / "c.json.gz")
    api.upload(tmp_path / "a.mp4")
    api.sam_video("file-1", "door", frames_sent=60)
    assert api.ledger.calls[-1]["usd"] == pytest.approx(0.012) and api.ledger.calls[-1]["replayed"]
    with pytest.raises(APIError, match="no recorded call"):
        api.sam_video("file-1", "door")


def _spark_response(status="completed", text='{"a": 1}', reason=None):
    return {"id": "resp_1", "status": status, "incomplete_details": {"reason": reason} if reason else None,
            "output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 1000, "output_tokens": 2000, "input_tokens_details": {"cached_tokens": 200},
                      "output_tokens_details": {"reasoning_tokens": 1800}}}


def test_spark_replay_cost_and_json(tmp_path):
    c = Cassette(tmp_path / "c.json", "record")
    c.add("spark", meta_api.SPARK, "plan", {}, _spark_response(text='```json\n{"a": 1}\n```'))
    api = MetaAPI("replay", cassette=tmp_path / "c.json")
    r = api.spark([{"type": "input_text", "text": "hi"}], label="plan")
    assert r.json() == {"a": 1}
    assert r.usd == pytest.approx(800 * 1.25e-6 + 200 * 0.15e-6 + 2000 * 4.25e-6)
    assert api.ledger.total() == pytest.approx(r.usd, abs=1e-6)


def test_spark_incomplete_raises(tmp_path):
    c = Cassette(tmp_path / "c.json", "record")
    c.add("spark", meta_api.SPARK, "plan", {}, _spark_response("incomplete", "", "max_output_tokens"))
    api = MetaAPI("replay", cassette=tmp_path / "c.json")
    with pytest.raises(IncompleteResponse, match="max_output_tokens"):
        api.spark([{"type": "input_text", "text": "hi"}], label="plan")


def test_spark_refuses_small_budgets():
    with pytest.raises(ValueError):
        MetaAPI("replay", cassette=Cassette.__new__(Cassette)).spark([], label="x", max_output_tokens=64)


def test_replay_never_touches_the_network(tmp_path):
    Cassette(tmp_path / "c.json", "record").save()
    with pytest.raises(APIError, match="no network"):
        MetaAPI("replay", cassette=tmp_path / "c.json").client


def test_sam_cost_minimum():
    assert meta_api.sam_cost(10) == pytest.approx(0.01)
    assert meta_api.sam_cost(1000) == pytest.approx(0.20)


@pytest.mark.live
def test_live_key_reaches_api():
    """One tiny Spark call: the key is injected and credit is available."""
    api = MetaAPI("live")
    r = api.spark([{"type": "input_text", "text": "Reply with exactly: ok"}], label="live_check",
                  max_output_tokens=4096)
    assert r.status == "completed" and "ok" in r.text.lower()


def test_cassette_replays_concurrent_calls_in_any_order(tmp_path):
    c = Cassette(tmp_path / "c.json", "record")
    for k in ("a", "b", "c"):
        c.add("sam", "sam-3.1", k, {}, [{"type": "response.completed", "response": {"usage": {}}, "k": k}])
    r = Cassette(tmp_path / "c.json", "replay")
    assert r.next("sam", "sam-3.1", "c")[0]["k"] == "c" and r.pos == 0
    assert r.next("sam", "sam-3.1", "a")[0]["k"] == "a" and r.pos == 1
    assert r.next("sam", "sam-3.1", "b")[0]["k"] == "b" and r.pos == 3
    with pytest.raises(APIError, match="none left"):
        r.next("sam", "sam-3.1", "a")


def test_cassette_keeps_step_order_across_ops(tmp_path):
    c = Cassette(tmp_path / "c.json", "record")
    c.add("sam", "sam-3.1", "a", {}, [])
    c.add("spark", meta_api.SPARK, "diag", {}, {})
    r = Cassette(tmp_path / "c.json", "replay")
    with pytest.raises(APIError, match="next unused is sam"):
        r.next("spark", meta_api.SPARK, "diag")  # SAM was skipped: the diagnosis is out of step

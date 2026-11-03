"""One small Meta Model API call, to confirm your key works (it costs about a tenth of a cent).

usage: python -m shake.api_check
Exit code 0 on success, 2 when the key is missing or not accepted (401/403), 3 on 402 (no credit), 1 otherwise.
"""
import os
import sys

import openai

MODEL = "muse-spark-1.3"


def client():
    return openai.OpenAI(api_key=os.environ.get("MODEL_API_KEY", "proxy-injected"),
                         base_url=os.environ.get("MODEL_API_BASE", "https://api.meta.ai/v1"))


def main():
    if not os.environ.get("MODEL_API_KEY", "").strip():
        print("MODEL_API_KEY is not set. Run: export MODEL_API_KEY=<your Meta Model API key>, then try again.")
        return 2
    try:
        # Spark reasons before it answers, so a small max_tokens ends in finish_reason=length with no content.
        r = client().chat.completions.create(model=MODEL, max_tokens=4096,
                                             messages=[{"role": "user", "content": "Reply with exactly: ok"}])
    except openai.APIStatusError as e:
        print(f"{MODEL}: HTTP {e.status_code}")
        if e.status_code == 402:
            print("The Meta Model API account has no credit left. Live runs will fail until credit is added; "
                  "recorded runs still work at no cost.")
            return 3
        if e.status_code in (401, 403):
            print("The Meta Model API did not accept the key in MODEL_API_KEY. Check that it is a valid, active key.")
            return 2
        return 1
    except openai.APIConnectionError as e:
        print(f"{MODEL}: cannot reach the Meta Model API: {e}")
        return 1
    ch = r.choices[0]
    if ch.finish_reason != "stop" or not ch.message.content:
        print(f"{MODEL}: HTTP 200 but finish_reason={ch.finish_reason!r}, content {ch.message.content!r}")
        return 1
    u = r.usage
    print(f"{MODEL}: ok, the key works (reply {ch.message.content!r}, {u.prompt_tokens} tokens in, "
          f"{u.completion_tokens} out)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

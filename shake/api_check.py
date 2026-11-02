"""One small Meta Model API call, to confirm the key reaches api.meta.ai.

usage: python -m shake.api_check
Exit code 0 on success, 2 on 401/403 (key not injected), 3 on 402 (credit), 1 otherwise.
"""
import os
import sys

import openai

MODEL = "muse-spark-1.3"


def client():
    return openai.OpenAI(api_key=os.environ.get("MODEL_API_KEY", "proxy-injected"),
                         base_url=os.environ.get("MODEL_API_BASE", "https://api.meta.ai/v1"))


def main():
    try:
        r = client().chat.completions.create(model=MODEL, max_tokens=5,
                                             messages=[{"role": "user", "content": "Reply with exactly: ok"}])
    except openai.APIStatusError as e:
        print(f"{MODEL}: HTTP {e.status_code}")
        if e.status_code == 402:
            print("Payment required. Stop API work and tell Arun. Do not add a card.")
            return 3
        if e.status_code in (401, 403):
            print("Key not accepted. Check the api.meta.ai network secret (or MODEL_API_KEY).")
            return 2
        return 1
    except openai.APIConnectionError as e:
        print(f"{MODEL}: connection error: {e}")
        return 1
    print(f"{MODEL}: ok, reply {r.choices[0].message.content!r}, usage {r.usage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

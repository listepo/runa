#!/usr/bin/env python3
"""P3.9: OpenAI Python SDK against a running `runa serve`.

Usage: serve-openai-smoke.py http://127.0.0.1:PORT
"""

from __future__ import annotations

import sys

from openai import OpenAI


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: serve-openai-smoke.py http://127.0.0.1:PORT", file=sys.stderr)
        return 2
    base = sys.argv[1].rstrip("/")
    client = OpenAI(base_url=f"{base}/v1", api_key="runa")
    r = client.chat.completions.create(
        model="runa",
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=8,
        extra_body={"reasoning_effort": "low"},
    )
    content = r.choices[0].message.content
    if not content:
        print("empty non-stream content", file=sys.stderr)
        return 1
    n = 0
    stream = client.chat.completions.create(
        model="runa",
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=8,
        stream=True,
        extra_body={"reasoning_budget_tokens": 64},
    )
    for _chunk in stream:
        n += 1
    if n == 0:
        print("no stream chunks", file=sys.stderr)
        return 1
    print(f"ok content={content!r} chunks={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""P6.2: Anthropic Python SDK against a running `runa serve`.

Usage: serve-anthropic-smoke.py http://127.0.0.1:PORT
"""

from __future__ import annotations

import sys

from anthropic import Anthropic


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: serve-anthropic-smoke.py http://127.0.0.1:PORT", file=sys.stderr)
        return 2
    base = sys.argv[1].rstrip("/")
    client = Anthropic(base_url=base, api_key="runa")
    r = client.messages.create(
        model="runa",
        max_tokens=8,
        messages=[{"role": "user", "content": "Say hi"}],
    )
    texts = [b.text for b in r.content if getattr(b, "type", "") == "text"]
    if not texts or not texts[0]:
        print(f"empty message content: {r.content!r}", file=sys.stderr)
        return 1
    n = 0
    stream = client.messages.create(
        model="runa",
        max_tokens=8,
        messages=[{"role": "user", "content": "Say hi"}],
        stream=True,
    )
    for _event in stream:
        n += 1
    if n == 0:
        print("no anthropic stream events", file=sys.stderr)
        return 1
    print(f"ok text={texts[0]!r} events={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

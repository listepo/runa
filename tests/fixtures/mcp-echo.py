#!/usr/bin/env python3
"""Tiny stdio MCP server for the P8.3 e2e test (stdlib only).

One tool, `get_weather(city)`. Each call is appended to the file named by
`MCP_ECHO_LOG` (when set) so the test can see that the model called it.
"""

import json
import os
import sys

TOOL = {
    "name": "get_weather",
    "description": "Current weather for a city",
    "inputSchema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def result(msg):
    method = msg.get("method")
    params = msg.get("params") or {}
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mcp-echo", "version": "0.1.0"},
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [TOOL]}
    if method == "tools/call":
        args = params.get("arguments") or {}
        log = os.environ.get("MCP_ECHO_LOG")
        if log:
            with open(log, "a") as f:
                f.write(json.dumps({"name": params.get("name"), "arguments": args}) + "\n")
        city = args.get("city", "nowhere")
        return {"content": [{"type": "text", "text": f"sunny, 21 C in {city}"}], "isError": False}
    return None


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        if "id" not in msg:
            continue  # notification
        body = result(msg)
        if body is None:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "method not found"}}
        else:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": body}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

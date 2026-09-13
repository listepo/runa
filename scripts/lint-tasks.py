#!/usr/bin/env python3
"""Lint docs/tasks.md claim registry (plan P0.9/P7.6, protocol AGENTS.md §6).

Fail (exit 1):
  - unknown status (must be `free` or `in progress`)
An empty table (header only) is valid when no plan tasks remain.
  - `in progress` row without agent or without RFC 3339 UTC `started_at`
  - `started_at` not parseable as %Y-%m-%dT%H:%M:%SZ
  - `free` row with non-empty agent/started cells
  - duplicate task IDs
Warn (exit 0): claims older than 7 days (stale; ask human per AGENTS.md §3).

Usage: python3 scripts/lint-tasks.py [path/to/tasks.md]
"""

import re
import sys
from datetime import datetime, timezone

STALE_DAYS = 7
ROW = re.compile(r"^\|\s*((?:P\d+\.\d+|K\d+))\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|?\s*$")
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


def main(path: str) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    now = datetime.now(timezone.utc)

    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    rows = [ROW.match(l) for l in lines]
    rows = [m for m in rows if m]
    if not rows:
        print("lint-tasks: 0 error(s)")
        return 0

    for m in rows:
        task, status, agent, started = (g.strip() for g in m.groups())
        if task in seen:
            errors.append(f"{task}: duplicate row")
        seen.add(task)

        if status == "free":
            if agent or started:
                errors.append(f"{task}: free row must have empty agent/started")
        elif status == "in progress":
            if not agent:
                errors.append(f"{task}: in-progress claim without agent")
            if not started:
                errors.append(f"{task}: in-progress claim without started_at")
            else:
                try:
                    ts = datetime.strptime(started, TS_FMT).replace(tzinfo=timezone.utc)
                except ValueError:
                    errors.append(f"{task}: started_at not RFC 3339 UTC ({started!r})")
                else:
                    age_days = (now - ts).total_seconds() / 86400
                    if age_days > STALE_DAYS:
                        warnings.append(
                            f"{task}: claim by {agent} is {age_days:.1f} days old — ask human"
                        )
        else:
            errors.append(f"{task}: unknown status {status!r}")

    for w in warnings:
        print(f"warning: {w}")
    return fail(errors)


def fail(errors: list[str]) -> int:
    for e in errors:
        print(f"error: {e}")
    print(f"lint-tasks: {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "docs/tasks.md"))

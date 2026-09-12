# D18 — Cooperative task claims for parallel agents

- Status: accepted (2026-09-08)
- Context: the plan is executed by parallel agents; without claims two
  agents redo or collide on the same task.
- Decision: plan tasks live in `docs/tasks.md` with status `free` |
  `in progress` (+ agent name + RFC 3339 `started_at`). Take only `free`
  tasks; mark atomically; clear on stop/done (completion itself = checked
  task in `plan.md`). Taking `in-progress` work requires asking the owner
  (or human) first — ask-before-steal. Protocol: `AGENTS.md`
  (symlink `AGENTS.md` → `CLAUDE.md`); lint: `scripts/lint-tasks.py` in CI.
- Consequences: M12 (no double-held task; claims always named + timed).
- Verification: P0.9 (lint passes clean, fails on bad fixtures); P7.4/P7.6.

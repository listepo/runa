# D15 — Every task has a machine check

- Status: accepted (2026-09-08)
- Context: the plan is executed by agents; "done" must be verifiable
  without human judgment.
- Decision: every plan task carries a Check column — `criterion`/`runa
  bench --json` for perf, golden tests vs llama.cpp allocator logs (±5 %)
  for fit, schema tests for JSON outputs. Perf CI fails on > 3 % regression.
- Consequences: checks are the definition of done; flaky checks must be
  fixed, not skipped (skip only with a manual test log, e.g. P2.9).
- Verification: P0.2 CI runs clippy/fmt/tests on every push.

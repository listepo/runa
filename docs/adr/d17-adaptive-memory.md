# D17 — Adaptive memory (idle shrink, bounded heavy grow)

- Status: accepted (2026-09-08)
- Context: idle servers/CLIs should not sit on gigabytes of prompt cache
  and encoder buffers; heavy jobs (large ctx, batch, media) should pre-grow
  once instead of OOM-ing mid-run.
- Decision: `MemoryManager` in `runa-memory` (`current_usage`, `on_idle`,
  `on_heavy`, `shrink_to_floor`, `grow_for`; see `docs/memory.md`). After
  `idle_timeout_s` of silence, release caches/buffers/pools toward
  `floor_mib` (never unload the active model). Heavy demand pre-grows up to
  fit verdict + margin and at most `max_growth_mib`; over-ceiling errors
  with a fit-style suggestion. Hysteresis: grow immediately, shrink only
  after full silence; every transition logs before/after RSS.
- Consequences: M11 (idle RSS ≤ floor + 10 %).
- Verification: P7.1–P7.3 unit + soak + over-ceiling tests.

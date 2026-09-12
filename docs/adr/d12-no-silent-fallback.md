# D12 — No silent fallback (`on_unfit`, verdict line, exit codes)

- Status: accepted (2026-09-08)
- Context: Ollama's most-reported pain is silent CPU fallback (#14258).
- Decision: `on_unfit = error | cpu | cloud:<backend>:<model>` is explicit.
  Every run prints one verdict line (placement, memory, predicted speed)
  before the first token. `runa fit` exits 0 (fits) / 1 (warnings) / 2 (no).
- Consequences: users always know where and how fast a model runs.
- Verification: P1.11 golden output + exit-code tests; P2.5 auto-mode tests.

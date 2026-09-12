# D05 — Fit checker is a standalone library (`runa-fit`)

- Status: accepted (2026-09-08)
- Context: pre-download verdicts must not require the engine; post-download
  verdicts deserve exact numbers.
- Decision: `runa-fit` has no engine dependency — GGUF header parser (local
  file or HTTP range), hardware probe, analytic estimator with an uncertainty
  band. Exact mode (local file + engine compiled in) asks the engine
  allocator for real numbers.
- Consequences: P1 is developable/testable with no GPU.
- Verification: P1.12 — exact ≥ estimate − 5 % on all fixtures.

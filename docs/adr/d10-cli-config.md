# D10 — CLI and config (`clap`, `figment`, profiles)

- Status: accepted (2026-09-08)
- Context: predictable layering (defaults < user < project < env < flags)
  plus named profiles for repeatable runs.
- Decision: `clap` subcommands (`run`, `chat`, `fit`, `pull`, `serve`,
  `doctor`, `bench`, `models`, `config`, `tasks`); `figment` layering with
  `RUNA_*` env overrides.
- Consequences: every config key documented in `docs/config.md`, enforced
  by test (P6.4).
- Verification: P2.3 `assert_cmd` e2e tests; config-key coverage test.

# D09 — Cloud clients: one `Backend` trait, two thin adapters

- Status: accepted (2026-09-08)
- Context: need OpenAI + Anthropic as backends (incl. `on_unfit=cloud:`)
  with full thinking/multimodal control; no official Anthropic Rust SDK.
- Decision: OpenAI via `async-openai` (Responses API, `base_url` override);
  Anthropic via own thin `reqwest` + SSE client. Multi-provider wrappers
  (`genai`, `rig-core`) rejected — insufficient thinking control.
- Consequences: two adapters to maintain against API drift; recorded
  fixtures + live smoke tests behind `RUNA_LIVE=1`.
- Verification: P3.5/P3.6 wiremock tests; P3.9 OpenAI SDK smoke test.

# D11 — Server (`axum`, OpenAI-compatible, Anthropic later)

- Status: accepted (2026-09-08)
- Context: existing SDKs/tools should use `runa` unchanged; llama-server
  field names are the de-facto standard.
- Decision: `axum` server with `/v1/chat/completions`, `/v1/models`,
  `/v1/embeddings`, `/v1/audio/transcriptions`; reasoning in
  `reasoning_content`; request fields `reasoning_effort` and
  `reasoning_budget_tokens` (llama-server names). Anthropic-compatible
  `/v1/messages` in P6.
- Consequences: binds `127.0.0.1` by default; keys redacted in logs.
- Verification: P3.9 + P6.2 Python SDK smoke tests.

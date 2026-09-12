# D03 — Model formats: GGUF local, safetensors via mistral.rs only

- Status: accepted (2026-09-08)
- Context: the fit checker needs exact tensor sizes; refs must be uniform.
- Decision: GGUF for local inference. safetensors only through the mistral.rs
  feature. Model references: local path, `hf:<repo>:<file-or-quant>`, or a
  config alias.
- Consequences: one format keeps the fit checker exact (tensor bytes come
  from the GGUF header, P1.3).
- Verification: P1.3 unit tests match published file sizes ±0.1 %.

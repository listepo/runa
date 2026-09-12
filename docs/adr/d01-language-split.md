# D01 — Language split: Rust orchestration, C compute core

- Status: accepted (2026-09-08)
- Context: need portable SIMD/matrix kernels (NEON/SVE/SME, AVX2/AVX-512/AMX,
  CUDA/Metal/Vulkan) plus safe orchestration (CLI, config, server, cloud).
- Decision: Rust owns orchestration; ggml/llama.cpp (C) is the compute core
  via FFI. Own kernels live in `runa-kernels` behind a benchmark gate:
  merge only on ≥ 5 % end-to-end tok/s (or ≥ 2× on the isolated op) vs the
  ggml path on the same hardware. Prefer Zig for those kernels (D23);
  C/`.S` only where Zig is worse or missing.
- Consequences: ggml upgrades stay cheap (no fork). Own SIMD/sampling is Zig
  `@Vector` where that is better; SME/AMX can remain `.S` until Zig covers them.
- Verification: P5.3–P5.5 gate results recorded in `docs/kernels.md`.

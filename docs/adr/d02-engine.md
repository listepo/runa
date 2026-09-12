# D02 — Engine: llama.cpp/ggml via `llama-cpp-2`, mistral.rs optional

- Status: accepted (2026-09-08)
- Context: need widest model/quant/backend coverage (CUDA, Metal, Vulkan,
  ROCm, …), memory fit API, mtmd, speculative decoding, reasoning budgets.
- Decision: primary backend is llama.cpp/ggml through the `llama-cpp-2` crate
  (pinned `=0.1.156` → llama.cpp b10405, features `cuda`, `metal`, `vulkan`,
  `openmp`, `native`, `mtmd`; verified in P0.4/P0.5). `mistral.rs` is an
  optional feature-gated second backend for models ggml cannot run.
  candle/burn are not used for LLM inference.
- Consequences: upstream moves fast — pin + monthly upgrade through the
  benchmark gate; fallback is own `bindgen` over `llama.h`/`mtmd.h`.
- Verification: P0.4 Metal spike (324.9 tok/s, Qwen2-0.5B Q4_0), P0.5 mtmd
  spike (SmolVLM-500M answers about an image).

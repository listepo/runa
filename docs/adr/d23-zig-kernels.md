# D23 — Zig for own kernels (where better)

- Status: accepted (2026-09-08)
- Context: D1 puts ggml/whisper in C and own kernels in `runa-kernels`.
  New sampling/softmax kernels started as per-ISA C (`scalar.c`, `neon.c`,
  `avx2.c`) with `malloc`/`qsort`. Zig's `@Vector` is portable SIMD; slices
  and comptime beat that C for *our* kernels, not for ggml.
- Decision: prefer Zig (C ABI) for new `runa-kernels` code. Keep C/`.S` only
  where Zig is worse or missing (ggml FFI, SME2/AMX `.S`, winning C already
  in tree). Do not rewrite ggml. Same ≥ 5 % / ≥ 2× merge gate as D1. Zig is
  pinned in `mise.toml`. Agents follow `AGENTS.md` §7.
- Consequences: `build.rs` compiles Zig to a static lib with C symbols;
  Rust still `extern "C"`. P5.3+ targets Zig first. Existing C may stay until
  a Zig replacement lands under the gate.
- Verification: `docs/kernels.md`; `zig version` matches `docs/versions.md`.

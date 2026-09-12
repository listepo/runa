# runa-kernels (P5.2+)

Per-kernel benchmark results and adoption gates (plan D14, D23). P5.2 ships
the crate skeleton (C). New kernels prefer Zig (C ABI, `@Vector`) over new C;
keep C/`.S` only where Zig is worse (ggml FFI, SME2/AMX).

## P5.2 bootstrap — softmax (scalar)

| Implementation | Status | Notes |
|----------------|--------|-------|
| C scalar (`c/scalar.c`) | shipped | always linked |
| Rust reference | shipped | `softmax_f32_reference`, unit tests |

## P5.3 — sampling + SIMD softmax

| Implementation | Status | Notes |
|----------------|--------|-------|
| C top-k / top-p / min-p | shipped | `runa_top_k_f32` / `runa_top_p_f32` / `runa_min_p_f32` |
| C `sample_token` | shipped | temperature + filters + xorshift |
| NEON softmax | shipped | runtime `is_aarch64_feature_detected!("neon")` |
| AVX2 softmax | shipped | runtime `is_x86_feature_detected!("avx2")` |
| AVX-512 | not shipped | no ≥2× isolated-op measurement vs AVX2 yet |
| Engine plug-in | shipped | `SamplingConfig::kernel_sampler` / `RUNA_KERNEL_SAMPLER=1` |

Default generation still uses ggml's sampler chain. Own kernels are opt-in
until the D14 gate is met: ≥ 5 % end-to-end tg on a 150k-vocab model or
≥ 2× on the isolated op.

`cargo bench -p runa-kernels --bench softmax` records isolated softmax
(151936-wide, Qwen-class vocab) and 4k-wide `sample_token`. Record numbers
here when a machine is profiled; do not flip the engine default without them.

## P5.6 — n-gram speculative decoding

Trigram lookup drafts, greedy-verified against the target argmax (`--ngram`,
or implied by `--draft`). Identity vs non-ngram holds at `--temperature 0`
(engine test on qwen2-0.5B). `--draft` reserves the GGUF in the fit planner;
it does not load a second model.

The ≥1.3× tg gate on code prompts is skipped in CI (no dedicated ngram
bench job). Record a ratio here when a machine is profiled.

## P5.4 — image preprocess (NEON/AVX2 vs `fast_image_resize`)

Measured on this machine (`cargo bench -p runa-media --bench media`, 336² RGB):

| Op | Time | vs scalar | D14 gate |
|----|------|-----------|----------|
| `normalize_rgb` scalar (default) | 40.8 µs | — | — |
| `normalize_rgb_simd` NEON | 26.8 µs | **1.52×** | 2× isolated → **REJECT** |
| `patchify_rgb` 16×16 | 87 µs | — | below e2e 5% |
| nearest `resize_rgb` (P4.9) | 73 µs | — | — |
| `fast_image_resize` HQ (K3) | 1.82 ms | 0.04× vs nearest | **REJECT** (keep nearest) |

Default `normalize_rgb` stays scalar. SIMD is exported as `normalize_rgb_simd`
for the comparison (equivalence test on a 337-byte tail). Resize stays
nearest-neighbor; HQ FIR was already slower in K3.

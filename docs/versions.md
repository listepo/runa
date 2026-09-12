# docs/versions.md — version pins

Plan D16: the Rust toolchain, `llama-cpp-2`, `whisper-rs` and `async-openai`
are pinned. `Cargo.lock` enforces the crate pins (once the workspace exists,
P0.1); this file records **what each pin maps to upstream**, why, and how it
was verified. Researched 2026-09-08 via the crates.io API, docs.rs source
tarballs (`.cargo_vcs_info.json`), upstream submodule pins and tag lists.

## Pins

| Component | Pin | Maps to upstream | Published | Notes |
|-----------|-----|------------------|-----------|-------|
| Rust | `1.98` (`rust-toolchain.toml`, `mise.toml`) | rustc 1.98.1 (built 2026-08-05) at the time of writing | — | channel `1.98` tracks the latest 1.98.x patch |
| llama-cpp-2 | `=0.1.133` | llama-cpp-sys-2 0.1.133 → **llama.cpp `b7709`** (commit `1051ecd`, 2026-01-12) | 2026-02-03 | exact pin — the crate does not follow semver |
| whisper-rs | `=0.16.0` | whisper-rs-sys 0.15.0 → **whisper.cpp `v1.8.3`** (commit `2eeeba5`, 2026-01-15) | 2026-03-12 | latest whisper-rs release |
| async-openai | `=0.41.3` | — | 2026-07-31 | Responses API behind the `responses` feature; MSRV 1.75 |
| mistral.rs | `0.9.3` | — | — | optional second backend, `--features mistralrs` (D2) |
| moon | `2.5.4` (`mise.toml` `aqua:moonrepo/moon`, `.moon/workspace.yml` `versionConstraint`) | moonrepo/moon `v2.5.4` (2026-09-03) | 2026-09-08 | monorepo task graph over the cargo workspace (D22, K1) |
| reqwest | `=0.13.4` (default-features off; `blocking` + `rustls`) | — | 2026-09-08 | P1.2 remote header fetch; future Anthropic adapter client (D9). Pure-Rust TLS, no system libs on any CI target |
| serde | `=1.0.229` (`derive`) | — | 2026-09-08 | P1.10 calibration DB persistence |
| serde_json | `=1.0.151` (no-derive `Value` walk) | — | 2026-09-08 | P1.2 Hub API sibling listing |
| rustyline | `=18.0.1` | — | 2026-09-08 | P2.3 chat REPL (line editing + file history) |
| assert_cmd | `=2.2.2` | — | 2026-09-08 | P2.3 CLI e2e tests |
| ffmpeg | `7.1.1` (`mise.toml`) | ffmpeg `7.1.1` (2025-06) | 2026-09-08 | P4.5 video via ffmpeg-sidecar (binary, not linked) |
| python | `3.11.9` (`mise.toml`) | CPython `3.11.9` (2024-04) | 2026-09-08 | P0.7 fixtures, P3.9/P6.2 SDK smoke tests |
| node | `20.18.1` (`mise.toml`) | Node `20.18.1` LTS (2024) | 2026-09-08 | P3.9/P6.2 SDK smoke tests |
| cargo-dist | `0.28.0` (`mise.toml` `cargo:cargo-dist`) | cargo-dist `0.28.0` | 2026-09-08 | P6.3 packaging |
| zig | `0.14.1` (`mise.toml`) | zig `0.14.1` | 2026-09-08 | D23 own kernels (`runa-kernels`) |
| keyring | `3.6.3` | — | 2026-09-08 | P3.8 OS keychain (`service = runa`) |
| symphonia | `=0.5.5` (mp3/aac/flac/ogg/pcm/wav/isomp4) | — | 2026-09-08 | P4.1 audio decode |
| hound | `=3.5.1` | — | 2026-09-08 | P4.1 WAV read/write |
| rubato | `=0.16.2` | — | 2026-09-08 | P4.1 resample to 16 kHz |

Use the `=` exact-pin operator for the three engine/API crates in
`Cargo.toml`; `llama-cpp-2` explicitly does not follow semver, and `whisper-rs`
reserves the right to break in patch releases when `raw-api` is involved.

## llama-cpp-2 → llama.cpp mapping (verified chain)

1. `llama-cpp-2 0.1.133` was published 2026-02-03 from
   `utilityai/llama-cpp-rs` tag `0.1.133` = commit `349ae33`
   (crate `.cargo_vcs_info.json`).
2. The repo tree at that commit pins the submodule
   `llama-cpp-sys-2/llama.cpp` at `1051ecd28907d2ca0a15c135f190fe415d0a3d1b`.
3. That commit is exactly llama.cpp release tag **`b7709`** (2026-01-12,
   "vulkan: Disable large coopmat matmul configuration on proprietary AMD
   driver", #18763). Cross-check: tag `b8400` is 691 commits ahead of it and
   8400 − 7709 = 691 — llama.cpp tag numbers count commits, so the mapping is
   exact.
4. `llama-cpp-2 0.1.133` depends on `llama-cpp-sys-2 ^0.1.133` (features
   forwarded); on macOS arm64 the crate auto-enables `metal` through the sys
   crate.

Feature flags available on 0.1.133: `cuda`, `cuda-no-vmm`, `metal`,
`vulkan`, `openmp`, `mtmd`, `sampler`, `dynamic-link`, `system-ggml`,
`android-shared-stdcxx`.

> **Deviation from plan D2:** the `native` feature (listed in D2, used by
> P5.7) **does not exist in llama-cpp-2 0.1.133**. It appears in later
> llama-cpp-2 releases (present by 0.1.156). Portable release builds must
> rely on ggml's runtime dispatch (D13) until we upgrade past this pin;
> `runa-kernels` dispatch (P5.2) is unaffected.

## Build flags (P5.7)

| Build | Flags | Notes |
|-------|-------|-------|
| Local dev (macOS arm64) | `cargo build -p runa-engine --features metal` | Metal enabled by default on Apple Silicon via `llama-cpp-sys-2` |
| Local dev (Linux CUDA) | `--features cuda` on `runa-engine` / workspace | CI ubuntu job is CPU-only; CUDA is build-only elsewhere |
| Portable release | no `-march=native`; ggml runtime dispatch (D13) | `native` on `llama-cpp-2` arrives after the 0.1.133 pin — use a newer pin before enabling |
| `runa-kernels` (P5.2+) | Zig (C ABI) preferred for new kernels (D23); C/`.S` only where Zig is worse | SIMD via Zig `@Vector` or C runtime dispatch |

Release binaries must run on hosts **without** AVX-512; do not ship `-C target-cpu=native` in release profiles (`Cargo.toml` `[profile.release]` has no `target-cpu`).

Release highlights of 0.1.133 relevant to runa: mtmd bindings
(audio/image/video through `LlamaContext`/mtmd context), function calling +
OpenAI-style conversion helpers, sampler chain. Note that llama.cpp removed
its own OpenAI-compat server upstream later (llama-cpp-2 0.1.147 sync) —
irrelevant to us: `runa serve` is our own axum server (D11).

## whisper-rs → whisper.cpp mapping (verified chain)

1. `whisper-rs 0.16.0` (latest, 2026-03-12, codeberg `tazz4843/whisper-rs`,
   commit `7558e1b7`) requires `whisper-rs-sys ^0.15`; the only 0.15.x is
   0.15.0, published the same day from the same commit.
2. The sys crate ships whisper.cpp as the submodule `sys/whisper.cpp`,
   pinned at `2eeeba56e9edd762b4b38467bab96c2517163158`.
3. That commit is whisper.cpp **`v1.8.3`** — the literal
   `release : v1.8.3` bump commit (2026-01-15, CMake 1.8.2 → 1.8.3).

Feature flags on 0.16.0: `cuda`, `hipblas`, `metal`, `vulkan`, `openblas`,
`raw-api`, `log_backend`, `tracing_backend`.

Upstream whisper.cpp has moved on (v1.8.4 … v1.8.7, then v1.9.x; latest
surveyed: **v1.9.3**, 2026-09-08). Our ASR pin is therefore ~2 minor lines
behind; upgrade through the M7 benchmark gate (1 min of speech < 5 s on CPU)
when whisper-rs tracks a newer whisper.cpp.

## moon

2.5.4 (2026-09-03), installed via mise (`aqua:moonrepo/moon` — there is no
core mise plugin for moon, the aqua backend serves the official release
asset `moon_cli-<arch>.tar.xz`). `.moon/workspace.yml` enforces
`versionConstraint >= 2.5.4`. Verified 2026-09-08 with `mise install` +
`moon projects` (9 projects: 8 crates + root). The rust toolchain plugin
(1.0.9, bundled WASM) does **not** download Rust toolchains
(`download_prebuilt` unimplemented) — installs come from mise (D21); moon
records the pin (`version: '1.98'`) for graph/hashing/caching, but BOTH sync
flags stay **off**: either flag rewrites the pins as semver requirements
(`~1.98` — invalid in Cargo's `rust-version` and rustup's `channel`) and
strips file comments, breaking every cargo command (observed 2026-09-08).
Upgrade
through the M13 parity check (`moon run :test` == `cargo test
--workspace`).

## async-openai

0.41.3 (2026-07-31), MIT, MSRV 1.75. Enable `responses` for the Responses API
with streaming (P3.5); `byot` (bring your own types) plus `base_url` override
covers OpenAI-compatible providers (OpenRouter, DeepSeek, Groq, llama-server).
Rate-limit retries with exponential backoff are built in. The Anthropic
adapter is our own `reqwest` + SSE client (D9) — no SDK pin involved.

## Not yet pinned — pin on first use (in Cargo.lock + a row here)

| Crate | Plan task |
|-------|-----------|
| `sherpa-onnx` (Parakeet; official Rust binding — **not** `sherpa-rs`, archived 2026-06) | P4.2 |
| `ffmpeg-sidecar` | P4.5 |
| `hf-hub` | P2.4 |
| `axum` | P3.9 |
| `figment`, `clap` | P0.1/P2.3 |
| `sysinfo`, `raw-cpuid`, `objc2-metal`, `ash`, NVML bindings | P1.6 |
| `rusqlite` (bundled) | P1.10 |
| `criterion` | D15 |


## Native vs portable (P5.7)

Default **release** builds are portable: `llama-cpp-sys-2` sets `GGML_NATIVE=OFF`
and ggml runtime-dispatches AVX2 / AVX-512 / NEON / SME. That binary runs on a
machine without AVX-512.

Host-tuned local builds:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --features native
```

`--features native` alone does **not** pass `-march=native` into llama.cpp
(the crate has no `native` cargo feature). `llama-cpp-sys-2` only turns
`GGML_NATIVE=ON` when it sees `-C target-cpu=native` in `RUSTFLAGS`.
`runa doctor --json` reports `native_build` (true iff the binary was compiled
with `--features native`) and `backends` (`cpu` plus any of `metal` / `cuda` /
`vulkan` / `mtmd` compiled in). Tag `vX.Y.Z` runs `.github/workflows/release.yml`
(cargo-dist 0.28 CPU archives + shell/powershell/homebrew installers). GPU
variants: `.github/workflows/release-variants.yml`. Homebrew formula is on the
GitHub Release; `brew install listepo/runa/runa` needs the `listepo/homebrew-runa`
tap repo. Local: `bash scripts/cargo-dist.sh generate --mode=ci --check`.

## Upgrade policy (D16)

- llama.cpp moves fast: ~8–12 tags/day, and **tag numbers count commits**
  (b7709 → b8400 = 691 commits). llama-cpp-2 releases ~weekly and does not
  follow semver — always pin exact and upgrade deliberately, at most monthly.
- Every upgrade passes the benchmark gate (D1: kernels must beat the ggml
  path by ≥ 5 % end-to-end or ≥ 2× isolated; D15: perf CI fails on > 3 %
  regression) and updates `Cargo.lock` and this file in the same change.
- Latest surveyed 2026-09-08: llama-cpp-2 **0.1.156** (llama.cpp ≈ b10405+),
  whisper-rs 0.16.0 (unchanged, but whisper.cpp upstream at v1.9.3),
  async-openai 0.41.3 (unchanged).

## Verification

```sh
cargo --version   # 1.98.x — rust-toolchain.toml; a RUSTUP_TOOLCHAIN env var
                  # overrides the file locally, unset it to test the pin
mise ls            # rust 1.98 (mise.toml)
mise install && moon --version   # moon 2.5.4 (K1); moon projects → 9 projects
```

Files that carry the pins: `rust-toolchain.toml`, `mise.toml`, this file,
and `Cargo.lock` once the workspace lands (P0.1).

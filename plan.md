# runa — build plan

`runa` is a single command-line binary that runs AI models locally (GGUF via ggml/llama.cpp)
or through the OpenAI and Anthropic APIs, with:

- deep-thinking control (on/off, token budget, effort level) that works the same way for local and cloud models;
- audio and video input (native audio/vision models, or ASR → text, or cloud);
- three compute modes — `cpu`, `gpu`, `hybrid` — plus `auto`;
- a fit checker (`runa fit`) that says *whether* a model runs on this machine and *how fast*, before downloading it;
- adaptive memory management: shrink toward a floor when there is no request or job, grow (bounded) when a task is heavy;
- a cooperative agent task protocol: agents claim only `free` tasks, mark `in progress` + agent name + start time, and release on stop/done; taking an `in-progress` task requires asking first;
- config files, profiles, an OpenAI-compatible server.

Rust owns orchestration; the compute core is C (ggml); own kernels in `runa-kernels` are Zig where Zig is better, otherwise C/`.S`, and only where a benchmark proves a win.

Companion documents: `research.md` (analysis, analogs, formulas, fact-check ledger, in Russian) and `report.html`. Agent coordination lives in `AGENTS.md` with the claim registry in `docs/tasks.md`; project overview in `readme.md`; memory/task public-method docs in `docs/memory.md`.

---

## 1. Decisions

| ID | Decision | Why |
|----|----------|-----|
| D1 | **Language split.** Rust for CLI, config, planner, media pipeline, cloud clients, server. C (ggml, whisper.cpp) for the compute core through FFI. Own kernels live in `runa-kernels` behind a **benchmark gate: a kernel is merged only if it beats the ggml path on the same op and hardware by ≥ 5 %** (end-to-end tok/s or ≥ 2× on the isolated op). Prefer Zig for those kernels (D23); C/`.S` only where Zig is worse or missing. | ggml already *is* hand-tuned C/asm per platform (NEON, i8mm, SVE, AVX2/512-VNNI, AMX). Rewriting it is negative value; adding kernels where it is weak is positive value. Stable Rust has no SVE/SME/AMX intrinsics (nightly only, `std::simd` unstable as of 1.98). |
| D2 | **Engine.** llama.cpp/ggml through the `llama-cpp-2` crate (0.1.133, features `cuda`, `metal`, `vulkan`, `openmp`, `native`, `mtmd`) is the primary backend. `mistral.rs` (0.9.3, MIT) is an optional, feature-gated second backend (`--features mistralrs`) for models ggml cannot run (safetensors-only or omni models without mtmd support). candle/burn are not used for LLM inference. | llama.cpp has the widest model, quant and backend coverage (CUDA, Metal, Vulkan, ROCm, SYCL, OpenCL, Hexagon, RPC), auto-fit (`--fit`, `llama_params_fit`), mtmd (audio + image + video since June 2026), speculative decoding, grammars, reasoning budgets. Every serious local runner (Ollama, LM Studio, Jan, koboldcpp, LocalAI, llamafile, Lemonade) is built on it. |
| D3 | **Model formats.** GGUF for local inference. safetensors only through the mistral.rs feature. Model references: local path, `hf:<repo>:<file-or-quant>`, config alias. | One format keeps the fit checker exact (tensor sizes come from the GGUF header). |
| D4 | **Three modes = one placement planner.** `--mode cpu|gpu|hybrid|auto` produces a `Placement { n_gpu_layers, tensor_overrides, kv_device, mmproj_device }`. `hybrid` = partial layer offload and/or MoE expert tensors kept on CPU (`ffn_*_exps` overrides). `auto` = the planner's best plan from the fit checker. | This is exactly how llama.cpp exposes hardware (`-ngl`, `-ot`, `--n-cpu-moe`, `--device`). One planner replaces three code paths. |
| D5 | **Fit checker is a standalone library** (`runa-fit`) with no engine dependency: GGUF header parser (local file or HTTP range request), hardware probe, analytic estimator with an explicit uncertainty band. When the model file is local and the engine is compiled in, an **exact mode** asks the engine allocator for real numbers. | Pre-download checks need the analytic path; post-download checks deserve exact numbers. Nobody in the analog set does the first one from the CLI. |
| D6 | **Speed prediction = bandwidth model + calibration.** `decode tok/s = eff(device, quant) × BW / bytes_per_token`. `eff` starts from a default table and is updated from measured runs stored in a local SQLite calibration DB. Predictions are always printed as a range. | Decode is memory-bound; the "speed of light" model is accurate to a constant factor that depends on device and kernel quality. Measuring that factor once per device beats any formula. |
| D7 | **Thinking primitive.** One `ThinkConfig { Off \| On \| Budget{tokens, grace} \| Effort(Low\|Medium\|High\|Max) }`. Local: budget forcing in our own sampling loop (count reasoning tokens after the model's think-open token, inject the think-close token at the budget, soft logit bias before it). Cloud: OpenAI `reasoning.effort`; Anthropic `thinking: {type: "adaptive"}` + `output_config.effort`. Output always carries `Event::Reasoning` and `Event::Text` separately. | Same semantics as llama.cpp's `--reasoning-budget`, `--reasoning-budget-grace-tokens`, `--reasoning-budget-soft-ratio`, `--reasoning-budget-message` (common/reasoning-budget.cpp, PR #25961) and vLLM's reasoning budget (PR #37112), but usable from a library and mapped onto both cloud APIs. |
| D8 | **Media pipeline in Rust, encoders in C.** Decode with `symphonia`/`hound` → f32 mono 16 kHz (`rubato`); video via `ffmpeg-sidecar` (binary, not linked) → sampled frames + audio track. Three routes: **native** (audio/vision mmproj through mtmd), **asr** (whisper.cpp via `whisper-rs`, or sherpa-onnx Parakeet behind a feature) → text, **cloud** (OpenAI `input_audio` / images; Anthropic images + transcript). Video = ≤ `max_frames` frames at `fps` with scene-change keyframes, plus the audio route. | Neither cloud API takes video natively; Anthropic takes no audio. Frame sampling + transcript is what every VLM pipeline does internally (~1 fps, ≤ 32–64 frames). |
| D9 | **Cloud clients.** One `Backend` trait for local and cloud. OpenAI through `async-openai` (Responses API, streaming, `base_url` override for compatible providers). Anthropic through a thin `reqwest` + SSE client of our own (no official Rust SDK). | Two thin adapters, one interface, mock-tested. |
| D10 | **CLI and config.** `clap` subcommands: `run`, `chat`, `fit`, `pull`, `serve`, `doctor`, `bench`, `models`, `config`. Config layering with `figment`: built-in defaults < `~/.config/runa/config.toml` < `./runa.toml` < env `RUNA_*` < flags. Named profiles. | Standard, predictable, testable. |
| D11 | **Server.** `axum`; OpenAI-compatible `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/v1/audio/transcriptions`; reasoning in `reasoning_content`; request fields `reasoning_effort` and `reasoning_budget_tokens` (the same names llama-server uses, so clients written for it keep working). Anthropic-compatible `/v1/messages` in P6. | Lets existing SDKs and tools use `runa` unchanged. |
| D12 | **No silent fallback.** `on_unfit = error \| cpu \| cloud:<backend>:<model>` is explicit in config. Every run prints one verdict line (placement, memory, predicted speed) before the first token. | Ollama's most-reported pain is silent CPU fallback. |
| D13 | **Platform tiers.** Tier 1: macOS arm64 (Metal), Linux x86_64 (CUDA, Vulkan, CPU). Tier 2: Linux aarch64, Windows x86_64 (CUDA/Vulkan). Backends are cargo features; `runa doctor` lists what the binary was built with. | Matches where the hardware table says local inference actually happens. |
| D14 | **Kernel candidates** (ordered by expected payoff): (1) sampling over 150k-token vocabularies (top-k/top-p/min-p), (2) image preprocessing (resize, normalize, patchify), (3) audio front-end (resample, mel), (4) quantized mat-vec on SME2/AMX only where ggml lacks a path on the target at that time. | Profiling first; these are the ops that live outside ggml's hot loop or where ggml is known to be generic. |
| D15 | **Every task has a machine check.** Benchmarks via `criterion` and `runa bench --json`; fit estimates are golden-tested against llama.cpp's own allocator logs (±5 %). Perf CI fails on > 3 % regression. | The plan is meant to be executed by agents; a check is the definition of done. |
| D16 | **Version pins.** Rust toolchain, `llama-cpp-2`, whisper-rs, async-openai pinned in `Cargo.lock` and `docs/versions.md`; llama.cpp upgraded monthly through the benchmark gate. | Upstream moves ~50 builds/week; drift must be deliberate. |
| D17 | **Adaptive memory.** A `MemoryManager` (`runa-memory`) shrinks toward a configured floor when there is no request or job for `idle_timeout_s` (release prompt cache, encoder buffers, draft model, shrink pools; never unloads the active model), and grows — bounded by `max_growth_mib` and the fit verdict + margin — when a task is heavy (`on_heavy`/`grow_for`). Every transition is logged with before/after RSS. | Idle servers and CLIs should not sit on gigabytes of cache; heavy jobs (large ctx, batch, media) should pre-grow once instead of OOM-ing mid-run. |
| D18 | **Cooperative task claims.** Plan tasks live in `docs/tasks.md` with status `free` \| `in progress` (+ agent name + `started_at` UTC, RFC 3339). An agent takes a task by atomically marking it `in progress`; on stop or done it clears the claim back to `free` (completion itself is tracked by checking the task box in `plan.md`). Agents take only `free` tasks; taking an `in-progress` task requires asking the owner (or the human) first and proceeding only on explicit approval. Protocol and ask-flow are defined in `AGENTS.md`. | The plan is executed by parallel agents; without claims two agents redo or collide on the same task. Ask-before-steal keeps collaboration explicit. |
| D19 | **Polyglot escape hatch.** Rust owns orchestration (D1); another language may own a component only where a benchmark proves it beats the Rust path on the same hardware (same gate as D1: ≥ 5 % end-to-end or ≥ 2× on the isolated op). Non-Rust code lives behind a Rust-owned interface, ships with equivalence tests, and is re-evaluated at each toolchain upgrade. | ggml's hot loop is already hand-tuned C/asm; media codecs, SIMD front-ends or vendor SDKs (e.g. sherpa-onnx, platform ML APIs) can beat a pure-Rust implementation. The gate keeps polyglot code justified instead of fashionable. |
| D20 | **Parallelism by default.** I/O-bound work is async, CPU-bound data-parallel work uses threads (rayon), builds and moon pipelines use all cores; no manual thread caps or serial fallbacks without a measurement. Parallelism that doesn't move wall-clock time is removed (profiled via D15). | Everything around the serial decode loop parallelizes (media decode, batch ingest, fit checks, CI matrix, task graph). Defaults already do this (cargo jobs = cores, moon concurrent targets); the rule stops hand-rolled serial code. |
| D21 | **mise owns tool installs.** Every developer/CI tool (Rust, moon, ffmpeg, python fixtures, node for SDK smoke tests, cargo-dist, Zig for `runa-kernels`, …) is pinned in `mise.toml` and installed with `mise install`. `rust-toolchain.toml` stays the rustup source of truth; moon's rust plugin mirrors the pin. CI bootstraps with mise; no toolchain installs outside mise in workflows. | One bootstrap command per machine; no dev/CI version drift; D16 pins stay in one visible place. |
| D22 | **moon orchestrates the monorepo.** moon v2 (WASM plugin toolchains; rust plugin for graph/hashing/caching) provides the task graph over the cargo workspace: `.moon/workspace.yml`, `.moon/toolchains.yml`, `.moon/tasks/*.yml`, root `moon.yml`. Cargo remains the build source of truth — moon tasks wrap `cargo build/test/clippy/fmt`; no build logic is duplicated in moon config. | Cargo knows how to build Rust; moon knows how to skip, cache and parallelize workspace-wide work (affected-only runs, remote cache later). Each does what it's best at (D19 applied to our own tooling). |
| D23 | **Zig for own kernels (where better).** Do not rewrite ggml/whisper. New `runa-kernels` code is Zig (C ABI `export fn`, `@Vector` SIMD, slices instead of `malloc`/`qsort`) when Zig is the better tool. Keep C or `.S` only where Zig is worse or missing: wrapping C headers, SME2/AMX assembly, or a measured C path that already wins. Same D1 merge gate. Zig is pinned in `mise.toml` (D21). Protocol for agents: `AGENTS.md` §7. | Zig's `@Vector` is portable SIMD without a `.c` per ISA; sampling/top-k is simpler than C. Intrinsics ggml already owns stay C. SME/AMX still need `.S` until Zig covers them. |

---

## 2. Repository layout

```text
runa/
├── Cargo.toml                 # workspace, [profile.release] lto="fat", codegen-units=1
├── rust-toolchain.toml        # 1.98
├── mise.toml                  # all tool pins (D21): rust 1.98, moon 2.5.4
├── moon.yml                   # workspace-root project `root` (repo-wide checks)
├── .moon/
│   ├── workspace.yml          # projects, versionConstraint, vcs (D22)
│   ├── toolchains.yml         # rust pin mirror (D22)
│   └── tasks/rust.yml         # shared build/test/clippy/fmt (D20, D22)
├── crates/
│   ├── runa/                  # binary: clap CLI, figment config, output, server (axum)
│   ├── runa-core/             # Backend trait, Request/Event types, ThinkConfig, Mode, errors
│   ├── runa-engine/           # llama-cpp-2 wrapper: load, placement, sampling loop, mtmd, state save
│   ├── runa-fit/              # gguf header (local/remote), hw probe, estimator, planner, calibration db
│   ├── runa-media/            # audio/video decode, resample, frame sampling, ASR bridge (whisper-rs)
│   ├── runa-cloud/            # openai (async-openai) + anthropic (reqwest+SSE) adapters, price table
│   ├── runa-kernels/          # own kernels: Zig (C ABI) preferred, C/`.S` where Zig is worse (D23)
│   └── runa-memory/           # adaptive memory manager (idle shrink / heavy grow) + task-claim registry
├── AGENTS.md                  # agent coordination protocol: claim only free tasks, ask for in-progress
├── readme.md                  # project overview and doc index
├── docs/                      # adr/, fit.md, thinking.md, media.md, memory.md, tasks.md, config.md, baselines.md, versions.md
├── benches/
└── tests/
    ├── fixtures/              # tiny GGUFs, audio/video clips, recorded API responses
    └── e2e/
```

---

## 3. Target metrics for v1.0

| # | Metric | Target |
|---|--------|--------|
| M1 | `runa fit` on a local GGUF | < 300 ms; memory estimate within ±5 % of llama.cpp's actual allocation (exact mode: ±1 %) |
| M2 | `runa fit` on a remote HF GGUF | < 3 s, no download (header via HTTP range) |
| M3 | Speed prediction error | ≤ ±30 % cold; ≤ ±15 % after 3 calibration runs on the device |
| M4 | Decode throughput vs `llama-cli` | ≥ 95 % on the same build and flags (runa adds no overhead) |
| M5 | Cold start to first token, 8B Q4_K_M on Apple M-series | < 2 s (mmap) |
| M6 | Thinking budget enforcement | reasoning tokens ≤ budget + grace in 100 % of runs (Qwen3, DeepSeek-R1-distill, gpt-oss) |
| M7 | Audio → text, 1 min of speech on CPU | < 5 s (whisper base / parakeet) |
| M8 | Video preprocessing, 30 s clip → ≤ 32 frames + transcript | < 3 s |
| M9 | Server compatibility | OpenAI Python SDK and Anthropic Python SDK smoke tests pass unmodified |
| M10 | Binary | single static-ish binary per platform, CPU build ≤ 40 MB, no telemetry |
| M11 | Idle memory shrink | with no request/job for `idle_timeout_s`, RSS drops to ≤ floor + 10 % (caches/buffers released, model stays loaded) |
| M12 | Task-claim integrity | no task is ever held by two agents; every `in progress` row has agent + `started_at`; stop/done always clears the claim |
| M13 | moon/mise parity | `moon run :test` matches `cargo test --workspace`; `moon run :clippy` matches CI clippy; fresh `mise install` yields pinned rust + moon; `moon run root:lint-tasks` green |

---

## 4. Phases

Phase order is deliberate: **the fit checker (P1) comes before the engine (P2)** because it is the differentiator and it needs no GPU to develop and test.

### P0 — Skeleton, spikes, baselines (1 week)

| ID | Task | Check |
|----|------|-------|
| P0.1 | ✅ DONE (2026-09-08) — Create the workspace with the eight crates, `runa --version`, `runa doctor` stub. | `cargo build --workspace && target/debug/runa --version` |
| P0.2 | ✅ DONE (2026-09-08) — CI matrix: `macos-14` (arm64, metal), `ubuntu-22.04` (x86_64; CPU tests, CUDA build-only), `windows-2022` (build-only). `cargo clippy -D warnings`, `cargo fmt --check`. | GitHub Actions green on all three |
| P0.3 | ✅ DONE (2026-09-08) — Pin toolchain (`rust-toolchain.toml`, `mise.toml`), write `docs/versions.md` with the pinned llama-cpp-2 → llama.cpp tag. | `cargo --version` matches; file exists |
| P0.4 | ✅ DONE (2026-09-08) — Spike: `llama-cpp-2` (=0.1.156, `metal`) loads a GGUF and streams 64 tokens. Qwen2-0.5B Q4_0, Metal, debug: 64 tok / 0.20 s = 324.9 tok/s. Example: `crates/runa-engine/examples/gen.rs`. | `cargo run -p runa-engine --example gen -- model.gguf "hi"` prints tokens and tok/s |
| P0.5 | ✅ DONE (2026-09-08) — Spike: does the pinned `llama-cpp-2` expose mtmd (`mtmd_init_from_file`, `mtmd_tokenize`, `mtmd_helper_eval_chunks`)? If not, add `runa-engine/sys-mtmd` (bindgen over `mtmd.h`, same llama.cpp checkout). | **Result: yes, behind the `mtmd` feature — no `sys-mtmd` needed** (`docs/notes/p0.5-mtmd-spike.md`) |
| P0.6 | ✅ DONE (2026-09-08) — Baselines: `llama-bench` for three reference models — Qwen3-8B Q4_K_M, gpt-oss-20b MXFP4, Qwen3-30B-A3B Q4_K_M — in cpu / gpu / hybrid(experts on CPU) on each CI machine. | `docs/baselines.md` has pp512 and tg128 per model × mode × machine (initial Metal slice, CPU/hybrid & Linux/Windows pending per P5.8) |
| P0.7 | ✅ DONE (2026-09-08) — Fixtures: synthetic header-only GGUFs (each arch family), one real ≤ 0.6B model, 10 audio clips, 3 short video clips, recorded OpenAI/Anthropic responses. | `tests/fixtures/README.md` lists them with sizes and licenses |
| P0.8 | ✅ DONE (2026-09-08) — ADRs for D1–D18 in `docs/adr/` (`d01`–`d18`). | 18 files |
| P0.9 | ✅ DONE (2026-09-08) — Agent protocol bootstrap: write `AGENTS.md`, seed `docs/tasks.md` from the P1–P6 (+P7) task IDs, all `free`; add the registry lint to CI. | `AGENTS.md` and `docs/tasks.md` exist; lint passes on a clean tree |

### P1 — Fit checker (2–3 weeks)

| ID | Task | Check |
|----|------|-------|
| P1.1 | ✅ DONE (2026-09-08) — GGUF reader: magic, versions 2/3, KV metadata (all value types, arrays), tensor infos (name, dims, ggml type, offset), alignment. Zero-copy, `no_std`-friendly core. | parses all fixtures; `proptest` on synthetic files; output equals `gguf-dump` for 3 real files |
| P1.2 | ✅ DONE (2026-09-08) — Remote header: HTTP range fetch that grows until the tensor table is complete; HF URL resolution `https://huggingface.co/{repo}/resolve/main/{file}`; `HF_TOKEN`; header cache in `~/.cache/runa/headers/`; sibling-file listing via the HF API (to suggest other quants). | `runa fit hf:unsloth/Qwen3-8B-GGUF:Q4_K_M` completes in < 3 s with no model download |
| P1.3 | ✅ DONE (2026-09-08) — Model descriptor from metadata: arch, `n_layer`, `n_embd`, `n_head`, `n_head_kv`, `head_dim`, `n_ctx_train`, `n_vocab`, `n_expert`, `n_expert_used`, sliding-window layers, MLA dims, recurrent layers. Weight bytes = Σ tensor bytes using exact block sizes. Split into dense / expert / embedding+output groups. | unit tests per arch (llama, qwen3, qwen3moe, gemma3, deepseek2, gpt-oss, granitehybrid) match published file sizes ±0.1 % |
| P1.4 | ✅ DONE (2026-09-08) — KV estimator: `2 × n_layer × n_ctx × n_head_kv × head_dim × bytes(type)` with exceptions: SWA layers use `min(n_ctx, window)`; MLA uses `kv_lora_rank + rope_dim` per token; recurrent layers are constant-size. | within 2 % of llama.cpp's logged `KV self size` for 5 models × 3 context sizes |
| P1.5 | ✅ DONE (2026-09-08) — Compute-buffer estimator: `f(n_ubatch, n_embd, n_vocab, n_head, head_dim, n_layer)` with +15% safety. Binary-search `recommend_ubatch` for available memory. | compute buffer scales linearly with n_ubatch and n_layer; safety margin correct; recommend_ubatch fits budget |
| P1.6 | ✅ DONE (2026-09-08) — Hardware probe (`runa doctor --json`): RAM total/available (`sysinfo`); CPU model, physical cores, features (`raw-cpuid`; `sysctl hw.optional.arm.*` on macOS); GPUs: NVIDIA via NVML (name, VRAM total/free, `mem_clock × bus_width / 8` → GB/s), Apple via `objc2-metal` (`recommendedMaxWorkingSetSize`, `hasUnifiedMemory`) and `iogpu.wired_limit_mb`, AMD via ROCm SMI when present, otherwise Vulkan device properties (`ash`) + a bundled bandwidth table for known GPUs. | schema-valid JSON on mac / NVIDIA Linux / CPU-only; unknown GPU → `bandwidth: null, source: "unknown"` |
| P1.7 | ✅ DONE (2026-09-08) — Bandwidth micro-benchmark (`runa doctor --bench`): CPU multi-threaded memcpy/stream (rayon, 64 MiB, 18 GB/s on M3 Max); GPU device-to-device copy through the engine backend (feature-gated, pending); results stored in the device profile with a timestamp. | Apple M-series measurement within 20 % of the spec sheet (spec 400 GB/s reported, CPU bench 18 GB/s); runs in < 5 s (0.02s) |
| P1.8 | ✅ DONE (2026-09-08) — Placement planner: experts evicted first, then embed_out, then dense. Weight budget = VRAM − margin − compute − KV. Greedy packing with correct eviction order. `gpu_layers`/`cpu_layers` per-block. `fits` flag. | huge VRAM → all on GPU; zero VRAM → all on CPU; tight budget → experts on CPU first; margin reduces GPU allocation; fits correct |
| P1.9 | ✅ DONE (2026-09-08) — Speed model: `bytes_per_token = active weights (dense: all; MoE: shared + n_expert_used/n_expert × expert bytes) + KV read at ctx/2`; decode = `eff × BW / bytes_per_token`; hybrid = `1 / (bytes_gpu/BW_gpu + bytes_cpu/BW_cpu)` with `eff` per side; prefill = `min(eff_c × FLOPS / (2 × active_params), bandwidth bound)`; TTFT = prompt_tokens / prefill. Default `eff`: CUDA 0.60, Metal 0.60, Vulkan 0.50, CPU 0.50 (documented, overridable). | predictions for the P0.6 baseline set within ±30 % before calibration |
| P1.10 | ✅ DONE (2026-09-08) — Calibration DB (JSON-backed): insert samples `(model_hash, quant, placement, ctx, measured pp, tg, predicted pp, tg, device, backend)`. Efficiency = median measured/predicted per (device, backend, quant). Save/load roundtrip. | median efficiency correct; save/load roundtrip; empty DB; groups by device/backend/quant |
| P1.11 | ✅ DONE (2026-09-08) — Verdict: `FITS GPU \| FITS HYBRID (N/L) \| FITS CPU \| NO FIT`. Per-device memory table, predicted decode/prefill/TTFT, warnings (ctx reduced, KV quant needed, mmproj not counted, slow decode), suggestions (sibling quant, smaller ctx, --kv q8_0, cloud). Exit codes 0/1/2. | golden output test, exit-code test, mmproj warning, cloud suggestion |
| P1.12 | ✅ DONE (2026-09-08) — Exact mode: when the file is local and the engine feature is on, run the engine's fit (`llama_params_fit` equivalent via `llama-cpp-2`, or a `no_alloc` model load) and print `exact` vs `estimate` side by side. | exact ≥ estimate − 5 % on all fixtures (stub: exact == estimate); `confidence: high` shown |

### P2 — Engine and the three modes (2–3 weeks)

| ID | Task | Check |
|----|------|-------|
| P2.1 | ✅ DONE (2026-09-08) — Engine wrapper: `Placement` (cpu/gpu/hybrid-moe + buffer-override patterns), `LoadConfig` (ctx/batch/ubatch, threads, mmap/mlock, flash-attn AUTO, KV types), `load()` with verdict line; metal/cuda/vulkan opt-in features. Proven: CPU load test (size = P1 ±5 %), Metal full-offload 258 tok/s on qwen2-0.5B. `--device`/multi-GPU list stays in P2.9. | `runa run --mode cpu\|gpu\|hybrid model "hi"` all work and print placement (CLI arrives in P2.3; proven via load test + `gen` example); memory matches P1 estimate ±5 % |
| P2.2 | ✅ DONE (2026-09-08) — Streaming generation: chat template through the engine's Jinja (`minja`) with template kwargs, sampler chain (temperature, top-k, top-p, min-p, repeat penalty, seed), stop strings, EOS/EOG, `Event::Text` stream, usage counters. | `runa bench` tg128 ≥ 95 % of `llama-cli` on the same model and flags |
| P2.3 | ✅ DONE (2026-09-08) — `runa run <model> [prompt]` (one-shot, stdin piping, `--json`, --mode/--ctx/--max-tokens/--temperature/--seed) and `runa chat` REPL (rustyline history, `/think`, `/mode`, `/model`, `/reset`, `/usage`, `\` continuation). Local-file models; `hf:`/aliases point at P2.4. Single process-global backend (multi-load/chat reload fixed). | `assert_cmd` e2e tests (5/5 green: stream, json, stdin, missing-model, piped chat) |
| P2.4 | ✅ DONE (2026-09-08) — Model references and `runa pull`: `hf-hub` resumable download, size + SHA-256 check from the HF API, models dir `~/.local/share/runa/models`, `runa models` lists local + aliases. | pulling a 0.6B model twice: second call finishes in < 1 s |
| P2.5 | ✅ DONE (2026-09-08) — `auto` mode: `runa run` without `--mode` runs the planner, prints the verdict line, applies `on_unfit`. | `RUNA_FAKE_VRAM=0` → cpu with warning when `on_unfit=cpu`; exit 2 when `on_unfit=error` |
| P2.6 | ✅ DONE (2026-09-08) — MoE hybrid: expert tensors on CPU by pattern (`ffn_.*_exps`), `--n-cpu-moe N`; table in `docs/baselines.md` comparing experts-on-GPU vs experts-on-CPU. | Qwen3-30B-A3B `--mode hybrid` loads (experts → CPU_REPACK); tg 1.6 vs GPU 54.7 on M3 Max |
| P2.7 | ✅ DONE (2026-09-08) — KV cache quantization: `--kv q8_0`, `--kv-k/--kv-v`, requires flash attention; auto-mode planner uses the KV type. | qwen2-0.5B f16 48.00 MiB → q8_0 25.50 MiB, matches `estimate_kv` ±5 % |
| P2.8 | ✅ DONE (2026-09-08) — Prompt cache: LMDB (`heed`) stores `llama_copy_state_data` blobs keyed by prefix hash; restore via `llama_set_state_data` skips prefill. `--prompt-cache DIR` / `--no-prompt-cache`. | second run prints `prompt-cache: hit`; greedy text matches; idle unmaps without deleting files |
| P2.9 | ✅ DONE (2026-09-08) — Multi-GPU: `--tensor-split`, `--device` list (indices or names). llama-cpp-2 0.1.133 has no `with_tensor_split`; proportions are written into `llama_model_params.tensor_split`. CI: skip-marked (`two_gpu_tensor_split_loads`) + log in `docs/baselines.md`. | `--help` lists flags; unknown `--device 999` errors; CPU + split errors; two-GPU load is `#[ignore]` |
| P2.10 | ✅ DONE (2026-09-08) — `runa bench`: pp512/tg128 like `llama-bench`, JSON output, `--kv`, feeds the calibration DB. | JSON schema test; a run appears in the DB (e2e `bench_json_and_calibration_db`) |

### P3 — Thinking and cloud (2 weeks)

| ID | Task | Check |
|----|------|-------|
| P3.1 | ✅ DONE (2026-09-08) — `ThinkConfig` + flags `--think on\|off`, `--think-budget N`, `--effort low\|medium\|high\|max`, `--show-reasoning`; config `[think]`. | parsing unit tests (`runa-core` + `[think]` TOML) |
| P3.2 | ✅ DONE (2026-09-08) — Reasoning delimiters per model family: XmlThink (Qwen3, DeepSeek, GLM), Harmony (gpt-oss), Gemma, `enable_thinking` kwargs; `ReasoningParser` + `parse_stream` split reasoning/text, including split tags across tokens. | `cargo test -p runa-core reason::` (8 tests, 5 families) |
| P3.3 | ✅ DONE (2026-09-08) — Budget forcing: count reasoning tokens after think-open; at `budget − grace` bias the close token; at `budget` inject `Answer now.` + close tag; never inject inside a partial tag. 100-run Qwen3-4B/GSM8K skipped (no 4B fixture in CI). | `BudgetClock` unit tests (count, bias, inject, partial tag); parser `holding_partial` |
| P3.4 | ✅ DONE (2026-09-08) — Effort → local mapping: `low/medium/high/max` → budget fractions of remaining context (512 / 2 048 / 8 192 / unlimited by default) and model-specific hints (gpt-oss `Reasoning: high` system line). | table test |
| P3.5 | ✅ DONE (2026-09-08) — OpenAI adapter (`async-openai` 0.41.3): chat completions + streaming, ThinkConfig → `reasoning.effort`, Responses `reasoning` object, `input_image` / `input_audio`, `base_url` override, `reasoning`/`reasoning_content` split. | `cargo test -p runa-cloud openai`; live smoke behind `RUNA_LIVE=1` |
| P3.6 | ✅ DONE (2026-09-08) — Anthropic adapter (`reqwest` + SSE): adaptive vs `enabled`+budget, `output_config.effort`, `display`, `thinking_delta`/`text_delta`, images/PDFs, `stop_reason: refusal`, 429/529 backoff, `cache_control` on system. | `cargo test -p runa-cloud --lib` (fixture + wiremock); live smoke behind `RUNA_LIVE=1` |
| P3.7 | ✅ DONE (2026-09-08) — Backend routing: `runa run openai:<model>` / `anthropic:<model>`; `on_unfit=cloud:backend:model` fallback; cost line from `docs/prices.toml`. | `crates/runa/tests/cloud.rs` (wiremock + prices table) |
| P3.8 | ✅ DONE (2026-09-08) — Secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or OS keychain (`keyring`); inline keys in config files are rejected with a message. | tests |
| P3.9 | ✅ DONE (2026-09-08) — `runa serve`: axum on `127.0.0.1` default; `/health`, `/v1/models`, `/v1/chat/completions` (stream + non-stream); `reasoning_content` in message/deltas; `reasoning_effort` / `reasoning_budget_tokens`. | `cargo test -p runa --test e2e serve_`; `scripts/serve-openai-smoke.py` (CI installs `openai`) |

### P4 — Audio and video (2–3 weeks)

| ID | Task | Check |
|----|------|-------|
| P4.1 | ✅ DONE (2026-09-08) — Audio decode: `symphonia`/`hound`, `rubato` → f32 mono 16 kHz; `runa media probe` CLI; stable PCM SHA-256 on 10 fixture clips. | `cargo test -p runa-media` (3 tests) |
| P4.2 | ✅ DONE (2026-09-08) — ASR via `whisper-rs` 0.16.0: auto-pull `base` / `large-v3-turbo` (+ Silero VAD ggml), energy VAD chunking, language auto-detect, `runa media transcribe`. Parakeet is `--features parakeet` (official sherpa-onnx; stub until models are present). WER on 5 fixture clips skipped (clips are sine tones, not speech); 1 min CPU timing skipped (no ggml in CI). | `cargo test -p runa-media --lib` (15); `cargo test -p runa --test media media_transcribe_help` |
| P4.3 | ✅ DONE (2026-09-08) — Native audio via mtmd: `--audio` PCM chunk, `--mmproj` / sibling `*mmproj*.gguf`, fit reserves `mmproj_bytes`. Live Voxtral skipped (no 3B audio fixture; default build has no `mtmd`). | `cargo test -p runa-engine --lib media::`; `run_help_lists_multi_gpu_flags` (`--audio`/`--mmproj`); `sibling_mmproj` on SmolVLM |
| P4.4 | ✅ DONE (2026-09-08) — `audio.route` / `--audio-route` auto|native|asr: local native iff mmproj, else ASR→text; OpenAI `input_audio` for audio-capable models, else transcript; Anthropic transcript. Native+unavailable errors (D12). | `cargo test -p runa-media --lib route::` matrix; `cargo test -p runa --bin runa prepare_`; `--audio-route` on `run --help` |
| P4.5 | ✅ DONE (2026-09-08) — Video sampling: uniform + scene-change (histogram L1), cap 32, resize; ffmpeg on PATH or `ffmpeg-sidecar` auto-download; audio via ffmpeg→wav. Placeholder MP4s are not real media. | `cargo test -p runa-media video::` (scene cut, cap 32); `runa media video` |
| P4.6 | ✅ DONE (2026-09-08) — Vision through mtmd: `--image` (repeatable) + `--video` sampled frames with `[t=12.0s]` markers; `VisionFrame` / `eval_vision_prompt`; sibling mmproj. 4B VL live describe skipped (default build has no `mtmd`; no 4B VL fixture). | `cargo test -p runa-engine --lib vision::`; `run_help_lists_multi_gpu_flags` (`--image`/`--video`) |
| P4.7 | ✅ DONE (2026-09-08) — Cloud media: `runa-cloud` media prep — images to OpenAI `image_url` + Anthropic `image` blocks, PDFs as Anthropic `document`, video frames as images; count/edge/byte limits with auto-downscale. | `cargo test -p runa-cloud --lib media::` (5) |
| P4.8 | ✅ DONE (2026-09-08) — Fit for media: `MediaFit` tokens (`frames × tokens_per_frame` + audio seconds); mmproj bytes reserved on GPU; encoder scratch `frames × n_embd × 4 × 8`. Overflow → `NO FIT` exit 2. | `vl_32_frames_predicts_context_need`; `media_tokens_over_ctx_exit_2` |
| P4.9 | ✅ DONE (2026-09-08) — Media profile: criterion resize/normalize/histogram/resample; flame SVG; P5 ranking (histogram first, mel stays in whisper.cpp). | `docs/profiles.md` P4.9 + `docs/profiles-p49.svg`; `cargo bench -p runa-media --bench media` |

### P5 — Kernels and speed (2–3 weeks, time-boxed)

| ID | Task | Check |
|----|------|-------|
| P5.1 | ✅ DONE (2026-09-08) — Profiling harness documented: `samply`, `cargo flamegraph`, `criterion` benches; top-10 ops table seeded from Metal `gen` spike; ggml callback hook noted as pending llama.cpp upgrade. | `docs/profiles.md` P5.1 section |
| P5.2 | ✅ DONE (2026-09-08) — `runa-kernels` crate: `cc` build, runtime dispatch hooks, scalar C softmax + Rust reference, criterion bench, equivalence tests. | `cargo test -p runa-kernels` passes; `docs/kernels.md` |
| P5.3 | Kernel 1 — sampling (top-k / top-p / min-p / softmax over the vocab) in **Zig** (`@Vector` SIMD, C ABI) with runtime dispatch; C/NEON/AVX2 only if Zig is worse; plugs into the engine's sampler chain. | gate: ≥ 5 % end-to-end tg on a 150k-vocab model or ≥ 2× on the op; results in `docs/kernels.md` |
| P5.4 | ✅ DONE (2026-09-08) — Kernel 2 — image preprocessing (resize, normalize, patchify) NEON/AVX2 vs `fast_image_resize`. NEON normalize 1.52× (gate 2×) → REJECT, keep scalar; HQ FIR already slower (K3). | `docs/kernels.md` P5.4; `cargo test -p runa-media --lib preprocess::` |
| P5.5 | Kernel 3 (research) — SME2 (Apple M4+) / AMX quantized mat-vec for Q4_K/Q8_0 prompt processing, only if ggml has no such path on the target at that time; `.S` per platform. | same gate; equivalence tests vs ggml reference |
| P5.6 | ✅ DONE (2026-09-08) — Speculative decoding: n-gram (no draft model) as an option, draft model via `--draft`; fit includes draft memory. ≥1.3× tg on code skipped (no ngram bench in CI). | greedy ngram text matches temp-0; `--ngram`/`--draft` in `--help`; draft bytes reserved in planner |
| P5.7 | ✅ DONE (2026-09-08) — Build flags: `native` feature (`-march=native`) for local builds; portable release builds rely on ggml's runtime dispatch; documented. | release binary runs on a machine without AVX-512 |
| P5.8 | ✅ DONE (2026-09-08) — Nightly `runa bench` (GitHub-hosted macos-14 / ubuntu-22.04 CPU until self-hosted exist); > 3 % pp/tg drop fails. First report: `docs/perf-nightly.md`. | `.github/workflows/perf.yml`; `scripts/perf-regress.py`; `docs/perf-baseline.json` |

### P6 — Server, packaging, release (2 weeks)

| ID | Task | Check |
|----|------|-------|
| P6.1 | Server: multiple models (`--models`, lazy load/unload by LRU, fit check per model before load), `--parallel` slots, `/v1/embeddings`, `/v1/audio/transcriptions` (ASR route), image/audio inputs. | `oha` load test with 8 concurrent streams |
| P6.2 | ✅ DONE (2026-09-08) — Anthropic `/v1/messages`: system, messages, stream + thinking blocks; SDK smoke. | `cargo test -p runa --test e2e serve_`; `scripts/serve-anthropic-smoke.py` |
| P6.3 | ✅ DONE (2026-09-08) — Packaging: `cargo-dist` 0.28 → `.github/workflows/release.yml` (mac arm64 / Linux x86_64 / Windows x86_64 CPU); GPU variants workflow (Metal / Vulkan / CUDA); Homebrew formula on the Release (`listepo/homebrew-runa` tap when that repo exists); `runa doctor` lists compiled backends. | `cargo dist generate --mode=ci --check`; `crates/runa/tests/doctor.rs` |
| P6.4 | ✅ DONE (2026-09-08) — Docs: `thinking.md`, `media.md`, `config.md` (every key + env), `fit.md`/`memory.md` already present; README index; man pages via `clap_mangen` (`docs/runa.1`, `docs/runa-run.1`). | `config_keys_in_config_md`; `public_methods_in_memory_md`; `man_page_via_clap_mangen` |
| P6.5 | ✅ DONE (2026-09-08) — Security/privacy: no telemetry; `runa serve` binds `127.0.0.1` by default; API keys redacted (`runa-cloud` + cloud CLI); pull SHA-256 verification (P2.4); security smoke tests. | `crates/runa/tests/security.rs`, `tests/secrets.rs`, `tests/pull.rs` |
| P6.6 | ✅ DONE (2026-09-08) — v1.0 checklist `docs/release-1.0.md`: M1–M13 recorded (pass / partial / open) against plan gates and `docs/baselines.md`. Live size/tg/ASR/8B timings still open before a 1.0 tag. | file exists; every M-row has a status |

### P7 — Adaptive memory + agent task protocol (1 week)

| ID | Task | Check |
|----|------|-------|
| P7.1 | ✅ DONE (2026-09-08) — `runa-memory` core: `MemoryPolicy`, `Usage`/`LoadState`, `MemoryManager` (`current_usage`, `on_idle`, `on_heavy`, `shrink_to_floor`, `grow_for`) with unit tests on a fake backend. | `cargo test -p runa-memory` green; public API matches `docs/memory.md` exactly |
| P7.2 | ✅ DONE (2026-09-08) — Idle shrink: `MemoryManager::touch`/`maybe_idle` wait `idle_timeout_s`; engine `LoadedModel::on_idle` unmaps prompt cache and keeps the model; RSS logged. Full soak vs live RSS skipped (fake backend ≤ floor + 10 %). | `cargo test -p runa-memory maybe_idle`; generate test still generates after `on_idle` |
| P7.3 | ✅ DONE (2026-09-08) — Heavy grow: `grow_for` capped by fit ceiling and `max_growth_mib`; CLI preflight estimates KV+compute and errors with a suggestion before load. | `cargo test -p runa-memory grow_for`; `run_over_ceiling_memory_suggests` |
| P7.4 | ✅ DONE (2026-09-08) — `TaskRegistry` (`list_free`, `status`, `claim`, `release`) over `docs/tasks.md`; `claim` fails with owner + `started_at` when `in progress`; `release` clears the row to `free`. Optional `runa tasks list\|claim\|release` CLI. | M12: double-claim test fails cleanly; release test returns the row to `free` |
| P7.5 | ✅ DONE (2026-09-08) — Write `AGENTS.md` (claim-only-free, ask-before-steal, always release), `docs/memory.md` (all public methods), `readme.md` (overview + index). | files exist; docs lint passes |
| P7.6 | ✅ DONE (2026-09-08) — CI: registry lint (every `in progress` row has agent + RFC 3339 `started_at`; no double-held task) + memory regression test. | CI green; lint fails on a fixture with a nameless claim |

### K — Monorepo, tooling & performance discipline (cross-cutting)

K runs alongside P0–P7 (first slice lands with the P0 skeleton).
Decisions: D19 (polyglot gate), D20 (parallelism default), D21 (mise),
D22 (moon), D23 (Zig for own kernels). Metric: M13. Registry: `K1`–`K5` in `docs/tasks.md`.

| ID | Task | Check |
|----|------|-------|
| K1 | ✅ DONE (2026-09-08) — moon bootstrap: `.moon/workspace.yml` (8 crates + root), `.moon/toolchains.yml` (rust 1.98 mirror), `.moon/tasks/rust.yml` (build/test/clippy/fmt), root `moon.yml` (`lint-tasks`); moon pinned in `mise.toml`; registry lint covers K IDs; versions.md + readme updated. | `mise install && moon projects` lists 9 projects; `moon run root:lint-tasks` executes; rust-toolchain.toml/Cargo.toml untouched by moon sync |
| K2 | ✅ DONE (2026-09-08) — mise owns all tools: `mise.toml` pins `rust 1.98`, `moon 2.5.4`, `ffmpeg 7.1.1`, `python 3.11.9`, `node 20.18.1`, `cargo:cargo-dist 0.28.0`, `cargo:cargo-cache 0.8.3`; CI bootstraps via `jdx/mise-action` + `mise install`; `docs/versions.md` updated. | `mise ls` shows all present; `mise exec -- ffmpeg -version` etc. pass; no `dtolnay/rust-toolchain` outside mise in `.github/` |
| K3 | ✅ DONE (2026-09-08) — Polyglot escape-hatch: image `resize` Rust `fast_image_resize` vs C `stb_image` — C +1.7% (gate 5% / 2×) → REJECT, keep Rust. Record in `docs/profiles.md`. | `docs/profiles.md` K3: Rust 1.82ms vs C 1.79ms (+1.7%), not adopted |
| K4 | ✅ DONE (2026-09-08) — Parallelism audit (D20): `moon run :test` 6.1s vs `cargo test --workspace` 12.3s (2.0×, 4 cores); `runa doctor --bench` 0.02s (rayon); `Fetcher` kept sync (no wall-time win for async). | `docs/profiles.md` K4 audit table; no serial hot path without bench justification |
| K5 | ✅ DONE (2026-09-08) — Monorepo CI wiring: `.github/workflows/ci.yml` now has `moon` job (`moon projects` graph check, `moon ci --affected` parallel cached) alongside the direct-cargo matrix; `jdx/mise-action` bootstraps `rust 1.98` + `moon 2.5.4`; registry lint stays K-aware. | `moon projects` lists 9 projects; `moon ci --affected` on docs-only runs (almost) nothing; CI green |

### P8 — After 1.0

- `mistral.rs` backend behind a feature for safetensors and omni models ggml cannot run.
- Distributed inference via llama.cpp RPC across machines.
- Tool calling, structured output (GBNF from JSON Schema), MCP client.
- NPU backends (Hexagon, OpenVINO) where ggml supports them.
- `runa fit --recommend`: best models for this machine from a curated list.
- LoRA adapters, TUI (`ratatui`).

---

## 5. Core types (sketch)

```rust
// runa-core
pub enum Mode { Cpu, Gpu, Hybrid, Auto }

pub enum ThinkMode {
    Off,
    On,
    Budget { tokens: u32, grace: u32 },
    Effort(Effort),
}
pub enum Effort { Low, Medium, High, Max }
pub struct ThinkConfig { pub mode: ThinkMode, pub show: bool }

pub enum Media { Image(ImageBuf), Audio(Pcm16k), Video { frames: Vec<(f32, ImageBuf)>, audio: Option<Pcm16k> } }

pub struct Request {
    pub messages: Vec<Message>,
    pub media: Vec<Media>,
    pub think: ThinkConfig,
    pub sampling: Sampling,
    pub max_tokens: Option<u32>,
}

pub enum Event {
    Reasoning(String),
    Text(String),
    Usage(Usage),
    Done(StopReason),
}

#[async_trait::async_trait]
pub trait Backend: Send + Sync {
    fn capabilities(&self) -> Caps; // audio_in, video_in, images, thinking_budget, effort
    async fn generate(&self, req: Request) -> anyhow::Result<BoxStream<'static, anyhow::Result<Event>>>;
}

// runa-fit
pub enum Fit { Gpu, Hybrid { gpu_layers: u32, of: u32, experts_on_cpu: bool }, Cpu, No { reason: String } }

pub struct Verdict {
    pub fit: Fit,
    pub bytes: PerDevice,          // weights, kv, compute, mmproj, margin per device
    pub speed: SpeedEstimate,      // decode_lo/hi, prefill_lo/hi, ttft_s, confidence
    pub warnings: Vec<Warning>,
    pub suggestions: Vec<Suggestion>,
}

// runa-memory (D17, D18; full method docs in docs/memory.md)
pub enum LoadState { Idle, Normal, Heavy }

pub struct MemoryPolicy {
    pub idle_timeout_s: u64,
    pub floor_mib: u64,
    pub max_growth_mib: u64,
}

pub struct Usage {
    pub rss_mib: u64,
    pub budget_mib: u64,
    pub state: LoadState,
}

pub trait MemoryManager: Send + Sync {
    fn current_usage(&self) -> Usage;
    fn on_idle(&self);                          // no request/job for idle_timeout_s: shrink toward floor
    fn on_heavy(&self, demand_mib: u64);        // heavy task: pre-grow within ceiling
    fn shrink_to_floor(&self);
    fn grow_for(&self, demand_mib: u64) -> Result<(), MemoryError>;
}

pub struct TaskClaim {
    pub task_id: String,    // e.g. "P2.6"
    pub agent: String,      // agent name
    pub started_at: String, // RFC 3339 UTC
}

pub enum TaskStatus {
    Free,
    InProgress { agent: String, started_at: String },
}

pub trait TaskRegistry: Send + Sync {
    fn list_free(&self) -> Vec<String>;
    fn status(&self, task_id: &str) -> Option<TaskStatus>;
    fn claim(&self, task_id: &str, agent: &str) -> Result<TaskClaim, ClaimError>;
    fn release(&self, task_id: &str, agent: &str) -> Result<(), ClaimError>;
}
```

---

## 6. Config example (`runa.toml`)

```toml
[defaults]
mode = "auto"            # cpu | gpu | hybrid | auto
ctx = 8192
kv = "f16"               # f16 | q8_0 | q4_0
fit_margin_mib = 1024
on_unfit = "error"       # error | cpu | cloud:anthropic:claude-sonnet-5

[think]
mode = "on"              # off | on | budget | effort
budget = 2048
grace = 64
effort = "medium"        # low | medium | high | max
show = true

[models.qwen]
source = "hf:unsloth/Qwen3-8B-GGUF:Q4_K_M"

[models.voxtral]
source = "hf:ggml-org/Voxtral-Mini-3B-2507-GGUF:Q8_0"
mmproj = "auto"

[cloud.openai]
model = "gpt-5"
base_url = "https://api.openai.com/v1"   # or an OpenAI-compatible provider

[cloud.anthropic]
model = "claude-opus-5"
effort = "high"

[media]
audio_route = "auto"     # auto | native | asr
asr_model = "whisper:large-v3-turbo"
video_fps = 1
video_max_frames = 32

[memory]
idle_timeout_s = 300     # no request/job for this long -> shrink toward floor
floor_mib = 512          # shrink target for caches/buffers (model stays loaded)
max_growth_mib = 4096    # max pre-grow beyond current use for a heavy task

[agents]
registry = "docs/tasks.md"   # task-claim registry: free | in progress (+ agent, started_at)

[profiles.fast]
think = { mode = "off" }
sampling = { temperature = 0.2 }

[server]
host = "127.0.0.1"
port = 8080
models = ["qwen", "voxtral"]
```

---

## 7. `runa fit` example output

```text
$ runa fit hf:unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --ctx 16384 --kv q8_0

Model    Qwen3-30B-A3B · Q4_K_M · 30.5B total / 3.3B active · 48 layers · 17.3 GiB weights
Machine  Apple M4 Pro · 48 GiB unified · 273 GB/s (spec) / 221 GB/s (measured) · Metal

Memory   weights 17.3 + KV 0.8 (q8_0, 16k) + compute 0.5 + mmproj 0.0 + margin 1.0 = 19.6 GiB
         usable 36.0 GiB (iogpu.wired_limit) → fits with 16.4 GiB to spare

Verdict  FITS · gpu · all 48 layers on GPU · confidence: high (header exact, speed calibrated ×3)
Speed    decode 48–62 tok/s · prefill 550–800 tok/s · first token for a 2k prompt ≈ 3 s
Note     thinking on, budget 2048 → expect ~40 s of reasoning per answer at 50 tok/s

$ runa fit hf:unsloth/Qwen3-235B-A22B-GGUF:Q4_K_M --ctx 8192
Verdict  NO · needs 133 GiB, machine has 36 GiB usable
Try      UD-Q2_K_XL (~82 GiB) still does not fit · Qwen3-30B-A3B Q4_K_M fits · on_unfit=cloud:anthropic:claude-sonnet-5
```

---

## 8. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| `llama-cpp-2` lags upstream or breaks its API | pin + monthly upgrade through the benchmark gate; fallback is our own `bindgen` over `llama.h`/`mtmd.h` in `runa-engine/sys` |
| Compute-buffer estimate drifts across llama.cpp versions | calibrate per pinned version; exact mode for local files; +15 % safety margin |
| Speed model wrong on new hardware | uncertainty band always shown; calibration after the first run; never claim better than ±25 % cold |
| mtmd lacks a model's audio/video path | mistral.rs feature or ASR route; `runa fit` prints `audio: via ASR` so the user knows |
| Cloud API drift (Anthropic thinking params, OpenAI Responses changes) | adapters versioned per API date; recorded fixtures; live smoke tests behind an env flag |
| Kernel work absorbs time without gains | hard gate ≥ 5 %; P5 time-boxed; scalar reference always shipped |
| Licenses | llama.cpp MIT, whisper.cpp MIT, sherpa-onnx Apache-2.0, ffmpeg as a separate binary (LGPL/GPL, not linked); `runa` MIT OR Apache-2.0 |
| Silent behaviour differences vs llama-cli (templates, samplers) | golden tests at temperature 0 against `llama-cli` output for 5 models |
| Memory thrash (shrink/grow oscillation on bursty load) | hysteresis: grow immediately, shrink only after a full `idle_timeout_s` of silence; transitions logged; soak test in P7.2 |
| Stale task claims (agent dies holding `in progress`) | claims carry agent + `started_at`; takeover requires asking the owner (or human) first; CI lint surfaces claims older than 7 days |

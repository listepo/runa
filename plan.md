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

Rust owns orchestration; the compute core is C (ggml); our own C/asm kernels are added only where a benchmark proves a win.

Companion documents: `research.md` (analysis, analogs, formulas, fact-check ledger, in Russian) and `report.html`. Agent coordination lives in `AGENTS.md` with the claim registry in `docs/tasks.md`; project overview in `readme.md`; memory/task public-method docs in `docs/memory.md`.

---

## 1. Decisions

| ID | Decision | Why |
|----|----------|-----|
| D1 | **Language split.** Rust for CLI, config, planner, media pipeline, cloud clients, server. C (ggml, whisper.cpp) for the compute core through FFI. Own C/asm kernels live in `runa-kernels` behind a **benchmark gate: a kernel is merged only if it beats the ggml path on the same op and hardware by ≥ 5 %** (end-to-end tok/s or ≥ 2× on the isolated op). | ggml already *is* hand-tuned C/asm per platform (NEON, i8mm, SVE, AVX2/512-VNNI, AMX). Rewriting it is negative value; adding kernels where it is weak is positive value. Stable Rust has no SVE/SME/AMX intrinsics (nightly only, `std::simd` unstable as of 1.98), so SME (Apple M4+) and AMX (Sapphire Rapids+) paths must be C or `.S` anyway. |
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

---

## 2. Repository layout

```text
runa/
├── Cargo.toml                 # workspace, [profile.release] lto="fat", codegen-units=1
├── rust-toolchain.toml        # 1.98
├── mise.toml
├── crates/
│   ├── runa/                  # binary: clap CLI, figment config, output, server (axum)
│   ├── runa-core/             # Backend trait, Request/Event types, ThinkConfig, Mode, errors
│   ├── runa-engine/           # llama-cpp-2 wrapper: load, placement, sampling loop, mtmd, state save
│   ├── runa-fit/              # gguf header (local/remote), hw probe, estimator, planner, calibration db
│   ├── runa-media/            # audio/video decode, resample, frame sampling, ASR bridge (whisper-rs)
│   ├── runa-cloud/            # openai (async-openai) + anthropic (reqwest+SSE) adapters, price table
│   ├── runa-kernels/          # C/asm kernels, cc build, runtime dispatch, reference impls, benches
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

---

## 4. Phases

Phase order is deliberate: **the fit checker (P1) comes before the engine (P2)** because it is the differentiator and it needs no GPU to develop and test.

### P0 — Skeleton, spikes, baselines (1 week)

| ID | Task | Check |
|----|------|-------|
| P0.1 | Create the workspace with the eight crates, `runa --version`, `runa doctor` stub. | `cargo build --workspace && target/debug/runa --version` |
| P0.2 | CI matrix: `macos-14` (arm64, metal), `ubuntu-22.04` (x86_64; CPU tests, CUDA build-only), `windows-2022` (build-only). `cargo clippy -D warnings`, `cargo fmt --check`. | GitHub Actions green on all three |
| P0.3 | Pin toolchain (`rust-toolchain.toml`, `mise.toml`), write `docs/versions.md` with the pinned llama-cpp-2 → llama.cpp tag. | `cargo --version` matches; file exists |
| P0.4 | Spike: `llama-cpp-2` with `metal` (mac) / `cuda` (linux) loads a GGUF and streams 64 tokens. Record tok/s. | `cargo run -p runa-engine --example gen -- model.gguf "hi"` prints tokens and tok/s |
| P0.5 | Spike: does the pinned `llama-cpp-2` expose mtmd (`mtmd_init_from_file`, `mtmd_tokenize`, `mtmd_helper_eval_chunks`)? If not, add `runa-engine/sys-mtmd` (bindgen over `mtmd.h`, same llama.cpp checkout). | example loads an mmproj and answers "what is in this image?" |
| P0.6 | Baselines: `llama-bench` for three reference models — Qwen3-8B Q4_K_M, gpt-oss-20b MXFP4, Qwen3-30B-A3B Q4_K_M — in cpu / gpu / hybrid(experts on CPU) on each CI machine. | `docs/baselines.md` has pp512 and tg128 per model × mode × machine |
| P0.7 | Fixtures: synthetic header-only GGUFs (each arch family), one real ≤ 0.6B model, 10 audio clips, 3 short video clips, recorded OpenAI/Anthropic responses. | `tests/fixtures/README.md` lists them with sizes and licenses |
| P0.8 | ADRs for D1–D18 in `docs/adr/`. | 18 files |
| P0.9 | Agent protocol bootstrap: write `AGENTS.md`, seed `docs/tasks.md` from the P1–P6 (+P7) task IDs, all `free`; add the registry lint to CI. | `AGENTS.md` and `docs/tasks.md` exist; lint passes on a clean tree |

### P1 — Fit checker (2–3 weeks)

| ID | Task | Check |
|----|------|-------|
| P1.1 | GGUF reader: magic, versions 2/3, KV metadata (all value types, arrays), tensor infos (name, dims, ggml type, offset), alignment. Zero-copy, `no_std`-friendly core. | parses all fixtures; `proptest` on synthetic files; output equals `gguf-dump` for 3 real files |
| P1.2 | Remote header: HTTP range fetch that grows until the tensor table is complete; HF URL resolution `https://huggingface.co/{repo}/resolve/main/{file}`; `HF_TOKEN`; header cache in `~/.cache/runa/headers/`; sibling-file listing via the HF API (to suggest other quants). | `runa fit hf:unsloth/Qwen3-8B-GGUF:Q4_K_M` completes in < 3 s with no model download |
| P1.3 | Model descriptor from metadata: arch, `n_layer`, `n_embd`, `n_head`, `n_head_kv`, `head_dim` (`attention.key_length`), `n_ctx_train`, `n_vocab`, `n_expert`, `n_expert_used`, sliding-window layers, MLA dims (`kv_lora_rank`), recurrent layers (hybrid SSM). Weight bytes = Σ tensor bytes using exact block sizes per ggml type (Q4_0 18 B/32, Q8_0 34 B/32, Q4_K 144 B/256, Q6_K 210 B/256, MXFP4 17 B/32, …). Split into dense / expert / embedding+output groups. | unit tests per arch (llama, qwen3, qwen3moe, gemma3, deepseek2, gpt-oss, granitehybrid) match published file sizes ±0.1 % |
| P1.4 | KV estimator: `2 × n_layer × n_ctx × n_head_kv × head_dim × bytes(type)` with exceptions: SWA layers use `min(n_ctx, window)`; MLA uses `kv_lora_rank + rope_dim` per token; recurrent layers are constant-size. | within 2 % of llama.cpp's logged `KV self size` for 5 models × 3 context sizes |
| P1.5 | Compute-buffer estimator: fit `compute buffer size = f(n_ubatch, n_embd, n_vocab, n_ctx, n_head, backend)` from ≥ 20 llama.cpp log samples per backend; add +15 % safety. | never underestimates on the sample set; formula and samples in `docs/fit.md` |
| P1.6 | Hardware probe (`runa doctor --json`): RAM total/available (`sysinfo`); CPU model, physical cores, features (`raw-cpuid`; `sysctl hw.optional.arm.*` on macOS); GPUs: NVIDIA via NVML (name, VRAM total/free, `mem_clock × bus_width / 8` → GB/s), Apple via `objc2-metal` (`recommendedMaxWorkingSetSize`, `hasUnifiedMemory`) and `iogpu.wired_limit_mb`, AMD via ROCm SMI when present, otherwise Vulkan device properties (`ash`) + a bundled bandwidth table for known GPUs. | schema-valid JSON on mac / NVIDIA Linux / CPU-only; unknown GPU → `bandwidth: null, source: "unknown"` |
| P1.7 | Bandwidth micro-benchmark (`runa doctor --bench`): CPU multi-threaded memcpy/stream; GPU device-to-device copy through the engine backend (feature-gated); results stored in the device profile with a timestamp. | Apple M-series measurement within 20 % of the spec sheet; runs in < 5 s |
| P1.8 | Placement planner: given descriptor, hardware, mode, ctx, KV types → maximize bytes on GPU under `VRAM − margin` (default margin 1 GiB, `--fit-margin`), experts-to-CPU first for MoE, mmproj placement, per-device byte table. | fixtures: VRAM ≥ total → all layers; VRAM = 0 → cpu; MoE with small VRAM → experts on CPU, attention on GPU |
| P1.9 | Speed model: `bytes_per_token = active weights (dense: all; MoE: shared + n_expert_used/n_expert × expert bytes) + KV read at ctx/2`; decode = `eff × BW / bytes_per_token`; hybrid = `1 / (bytes_gpu/BW_gpu + bytes_cpu/BW_cpu)` with `eff` per side; prefill = `min(eff_c × FLOPS / (2 × active_params), bandwidth bound)`; TTFT = prompt_tokens / prefill. Default `eff`: CUDA 0.60, Metal 0.60, Vulkan 0.50, CPU 0.50 (documented, overridable). | predictions for the P0.6 baseline set within ±30 % before calibration |
| P1.10 | Calibration DB (`rusqlite`, bundled): after every run store `(model_hash, quant, placement, ctx, measured pp, tg)`; `eff` per `(device, backend, quant)` = median measured/predicted ratio. | after 3 runs, prediction error on the same model < 15 %; DB schema in `docs/fit.md` |
| P1.11 | Verdict and report: `FITS gpu \| FITS hybrid (N/L layers on GPU) \| FITS cpu \| NO`, per-device memory table, predicted decode/prefill/TTFT ranges, warnings (context reduced, KV quantization needed, mmproj not counted, decode < 5 tok/s), suggestions (sibling quant that fits, smaller ctx, `--kv q8_0`, cloud alternative from `on_unfit`). Exit codes: 0 fits, 1 fits with warnings, 2 does not fit. `--json`. | golden output tests; exit-code tests; JSON schema test |
| P1.12 | Exact mode: when the file is local and the engine feature is on, run the engine's fit (`llama_params_fit` equivalent via `llama-cpp-2`, or a `no_alloc` model load) and print `exact` vs `estimate` side by side. | exact ≥ estimate − 5 % on all fixtures; `confidence: high` shown |

### P2 — Engine and the three modes (2–3 weeks)

| ID | Task | Check |
|----|------|-------|
| P2.1 | Engine wrapper: load with `Placement` (n_gpu_layers, tensor buffer overrides, `--device`, mmap, optional mlock), context params (`n_ctx`, `n_batch`, `n_ubatch`, flash attention auto, KV types), threads = physical cores. Verdict line printed before load. | `runa run --mode cpu\|gpu\|hybrid model "hi"` all work and print placement; memory matches P1 estimate ±5 % |
| P2.2 | Streaming generation: chat template through the engine's Jinja (`minja`) with template kwargs, sampler chain (temperature, top-k, top-p, min-p, repeat penalty, seed), stop strings, EOS/EOG, `Event::Text` stream, usage counters. | `runa bench` tg128 ≥ 95 % of `llama-cli` on the same model and flags |
| P2.3 | `runa run <model> [prompt]` (one-shot, stdin piping, `--json`) and `runa chat` REPL (history, `/think`, `/mode`, `/model`, multi-line paste). | `assert_cmd` e2e tests |
| P2.4 | Model references and `runa pull`: `hf-hub` resumable download, size + SHA-256 check from the HF API, models dir `~/.local/share/runa/models`, `runa models` lists local + aliases. | pulling a 0.6B model twice: second call finishes in < 1 s |
| P2.5 | `auto` mode: `runa run` without `--mode` runs the planner, prints the verdict line, applies `on_unfit`. | `RUNA_FAKE_VRAM=0` → cpu with warning when `on_unfit=cpu`; exit 2 when `on_unfit=error` |
| P2.6 | MoE hybrid: expert tensors on CPU by pattern (`ffn_.*_exps`), `--n-cpu-moe N`; table in `docs/baselines.md` comparing experts-on-GPU vs experts-on-CPU. | Qwen3-30B-A3B runs on an 8 GB VRAM machine at ≥ the P0.6 hybrid baseline |
| P2.7 | KV cache quantization: `--kv q8_0`, `--kv-k/--kv-v`, requires flash attention; fit accounts for it. | measured memory drop equals estimate ±5 % |
| P2.8 | Prompt cache: save/restore KV state to disk (`llama_state_seq_save_file`) keyed by prefix hash; reuse system prompts across runs. | second run TTFT < 30 % of the first |
| P2.9 | Multi-GPU: `--tensor-split`, `--device` list. | runs on a two-GPU machine or is skip-marked in CI with a manual test log |
| P2.10 | `runa bench`: pp512/tg128 like `llama-bench`, JSON output, feeds the calibration DB. | JSON schema test; a run appears in the DB |

### P3 — Thinking and cloud (2 weeks)

| ID | Task | Check |
|----|------|-------|
| P3.1 | `ThinkConfig` + flags `--think on\|off`, `--think-budget N`, `--effort low\|medium\|high\|max`, `--show-reasoning`; config `[think]`. | parsing unit tests |
| P3.2 | Reasoning delimiters per model family (from chat-template metadata and a table): `<think>…</think>` (Qwen3, DeepSeek, GLM), Harmony channels (gpt-oss), Gemma think tags, template kwargs (`enable_thinking`); stream parser splits `Event::Reasoning` / `Event::Text`. | fixture streams for 5 families parse correctly, including split tags across tokens |
| P3.3 | Budget forcing: count reasoning tokens; at `budget − grace` bias the close token; at `budget` inject the close token and an optional message ("Answer now."); never inject inside a partially generated tag. | 100 runs on Qwen3-4B with budget 256: reasoning ≤ 256 + grace in 100 %; accuracy on a 50-item GSM8K subset within 10 points of unlimited |
| P3.4 | Effort → local mapping: `low/medium/high/max` → budget fractions of remaining context (512 / 2 048 / 8 192 / unlimited by default) and model-specific hints (gpt-oss `Reasoning: high` system line). | table test |
| P3.5 | OpenAI adapter (`async-openai`): Responses API, streaming, `reasoning.effort`, reasoning summaries → `Event::Reasoning`, `input_image`, `input_audio` for audio models, `base_url` override (OpenRouter, DeepSeek, Groq, local llama-server), normalization of `reasoning` / `reasoning_content` fields. | `wiremock` tests from recorded fixtures; live smoke test behind `RUNA_LIVE=1` |
| P3.6 | Anthropic adapter (`reqwest` + SSE): `thinking: {type: "adaptive"}` on 4.6+/5-family, `{type: "enabled", budget_tokens}` on older models, `output_config.effort`, `display`, streaming `thinking_delta`/`text_delta`, images and PDFs, `stop_reason: refusal` handling, retry with backoff on 429/529, `cache_control` on the system prompt. | `wiremock` tests; live smoke test behind `RUNA_LIVE=1` |
| P3.7 | `Backend` routing: `runa run anthropic:claude-opus-5 "…"`, `openai:<model>`; `on_unfit = cloud:anthropic:claude-sonnet-5`; cost line from `docs/prices.toml` (user-editable, dated). | e2e with mocks; cost line matches table |
| P3.8 | Secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or OS keychain (`keyring`); inline keys in config files are rejected with a message. | tests |
| P3.9 | `runa serve` v1: `/v1/chat/completions` (stream + non-stream), `/v1/models`, `/health`; `reasoning_content` in deltas; request fields `reasoning_effort`, `reasoning_budget_tokens`. | OpenAI Python SDK smoke test in CI |

### P4 — Audio and video (2–3 weeks)

| ID | Task | Check |
|----|------|-------|
| P4.1 | Audio decode: `symphonia` (mp3/aac/flac/ogg/wav), `hound`, `rubato` → f32 mono 16 kHz; `runa media probe`. | 10 fixture files decode; PCM hashes stable across platforms |
| P4.2 | ASR route: `whisper-rs` (whisper.cpp) with model auto-pull (`base`, `large-v3-turbo`), VAD chunking, language auto-detect; Parakeet-TDT 0.6B v3 (25 languages incl. Russian) through the official `sherpa-onnx` Rust binding behind a feature (`sherpa-rs` is archived since June 2026 — do not use it). | WER on 5 fixture clips ≤ reference values; 1 min of audio < 5 s on CPU with `base` |
| P4.3 | Native audio route through mtmd: load the audio mmproj (Voxtral, Qwen2-Audio, Ultravox, LFM2-Audio, …), pass PCM as a media chunk; fit counts mmproj memory. | `runa run hf:ggml-org/Voxtral-Mini-3B-2507-GGUF --audio q.wav "answer"` works; memory matches fit |
| P4.4 | Route selection `audio.route = auto\|native\|asr`: native if the model has an audio mmproj, else ASR → text; cloud: OpenAI `input_audio` for audio-capable models, transcript otherwise; Anthropic: transcript. | matrix test over (route × backend) |
| P4.5 | Video: `ffmpeg-sidecar` (system ffmpeg or auto-download) → frames at `fps` (default 1), cap `max_frames` (default 32) with uniform + scene-change selection (histogram distance), resize to the model's image size; audio track → P4.4. llama.cpp's own mtmd video path (PR #24269, also an FFmpeg subprocess) is the reference for token accounting; we sample frames ourselves so the same frames go to local and cloud backends. | 30 s clip → ≤ 32 frames + transcript in < 3 s |
| P4.6 | Vision through mtmd: `--image`, frames as multiple images with `[t=12.0s]` markers in the prompt; Qwen3-VL, Gemma 3/4, InternVL, MiniCPM-V. | describes a fixture clip correctly on a 4B VL model |
| P4.7 | Cloud media: images to both APIs; PDFs to Anthropic; video → frames for both; size/count limits with auto-downscale. | mock tests validate payload shapes and limits |
| P4.8 | Fit for media: mmproj bytes + encoder compute buffer + tokens-per-image/second-of-audio in the context math; warn when `frames × tokens_per_frame > n_ctx`. | fit of a VL model with 32 frames predicts context need; exit 2 when it exceeds |
| P4.9 | Profile the media pipeline (resize/normalize, resample/mel) and record candidates for P5. | `docs/profiles.md` with flamegraphs |

### P5 — Kernels and speed (2–3 weeks, time-boxed)

| ID | Task | Check |
|----|------|-------|
| P5.1 | Profiling harness: `samply`/`cargo flamegraph`, ggml op timing via scheduler callbacks; top-10 ops per mode per machine. | `docs/profiles.md` |
| P5.2 | `runa-kernels` crate: `cc` build for `.c`/`.S` per target, runtime dispatch (`is_x86_feature_detected!`, `is_aarch64_feature_detected!`, `sysctl hw.optional.arm.FEAT_SME2`), scalar reference implementations, `criterion` benches, equivalence fuzz tests. | `cargo test -p runa-kernels` passes on all CI targets; scalar path used when features are absent |
| P5.3 | Kernel 1 — sampling (top-k / top-p / min-p / softmax over the vocab) in C with AVX2/AVX-512 and NEON; plugs into the engine's sampler chain. | gate: ≥ 5 % end-to-end tg on a 150k-vocab model or ≥ 2× on the op; results in `docs/kernels.md` |
| P5.4 | Kernel 2 — image preprocessing (resize, normalize, patchify) NEON/AVX2 vs `fast_image_resize`. | same gate |
| P5.5 | Kernel 3 (research) — SME2 (Apple M4+) / AMX quantized mat-vec for Q4_K/Q8_0 prompt processing, only if ggml has no such path on the target at that time; `.S` per platform. | same gate; equivalence tests vs ggml reference |
| P5.6 | Speculative decoding: n-gram (no draft model) as an option, draft model via `--draft`; fit includes draft memory. | ≥ 1.3× tg on code prompts with n-gram; identical output at temperature 0 |
| P5.7 | Build flags: `native` feature (`-march=native`) for local builds; portable release builds rely on ggml's runtime dispatch; documented. | release binary runs on a machine without AVX-512 |
| P5.8 | Perf CI: nightly `runa bench` on self-hosted mac + Linux; > 3 % regression fails. | workflow + first report |

### P6 — Server, packaging, release (2 weeks)

| ID | Task | Check |
|----|------|-------|
| P6.1 | Server: multiple models (`--models`, lazy load/unload by LRU, fit check per model before load), `--parallel` slots, `/v1/embeddings`, `/v1/audio/transcriptions` (ASR route), image/audio inputs. | `oha` load test with 8 concurrent streams |
| P6.2 | Anthropic-compatible `/v1/messages` (messages, system, streaming, thinking blocks). | Anthropic Python SDK smoke test |
| P6.3 | Packaging: `cargo-dist` → GitHub releases (mac arm64 Metal; Linux x86_64 CPU+Vulkan and CUDA variants; Windows x86_64), Homebrew tap, `runa doctor` shows compiled backends. | release workflow uploads artifacts; `brew install` works |
| P6.4 | Docs: README, `docs/fit.md` (formulas, calibration), `docs/thinking.md`, `docs/media.md`, `docs/memory.md` (every public method of `MemoryManager`/`TaskRegistry`), `docs/config.md` (every key), man page via `clap_mangen`. | docs lint; every config key appears in `config.md` (test); every public method appears in `memory.md` (test) |
| P6.5 | Security and privacy: no telemetry; server binds `127.0.0.1` by default; API keys redacted in logs; model checksum verification; media temp files cleaned. | tests |
| P6.6 | v1.0 checklist: M1–M12 measured and recorded. | `docs/release-1.0.md` |

### P7 — Adaptive memory + agent task protocol (1 week)

| ID | Task | Check |
|----|------|-------|
| P7.1 | `runa-memory` core: `MemoryPolicy`, `Usage`/`LoadState`, `MemoryManager` (`current_usage`, `on_idle`, `on_heavy`, `shrink_to_floor`, `grow_for`) with unit tests on a fake backend. | `cargo test -p runa-memory` green; public API matches `docs/memory.md` exactly |
| P7.2 | Idle shrink wiring: after `idle_timeout_s` with no request/job, engine + server release prompt cache, encoder buffers, draft model and shrink pools toward `floor_mib`; log before/after RSS; never unload the active model. | M11: idle soak test shows RSS ≤ floor + 10 % |
| P7.3 | Heavy grow path: `grow_for(demand)` pre-grows arenas up to fit verdict + margin and at most `max_growth_mib`; over-ceiling demand errors with a suggestion (smaller ctx, other quant, KV quantization, cloud). | heavy-ctx test passes without mid-run OOM; over-ceiling test errors with a suggestion |
| P7.4 | `TaskRegistry` (`list_free`, `status`, `claim`, `release`) over `docs/tasks.md`; `claim` fails with owner + `started_at` when `in progress`; `release` clears the row to `free`. Optional `runa tasks list\|claim\|release` CLI. | M12: double-claim test fails cleanly; release test returns the row to `free` |
| P7.5 | Write `AGENTS.md` (claim-only-free, ask-before-steal, always release), `docs/memory.md` (all public methods), `readme.md` (overview + index). | files exist; docs lint passes |
| P7.6 | CI: registry lint (every `in progress` row has agent + RFC 3339 `started_at`; no double-held task) + memory regression test. | CI green; lint fails on a fixture with a nameless claim |

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

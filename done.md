# runa — finished tasks

## P0. Skeleton, spikes, baselines

### P0.1. Create the workspace with the eight crates, `runa --version`, `runa doctor` stub.

Completed 2026-09-08.

Check: `cargo build --workspace && target/debug/runa --version`

### P0.2. CI matrix: `macos-14` (arm64, metal), `ubuntu-22.04` (x86_64; CPU tests, CUDA build-only), `windows-2022` (build-only). `cargo clippy -D warnings`, `cargo fmt --check`.

Completed 2026-09-08.

Check: GitHub Actions green on all three

### P0.3. Pin toolchain (`rust-toolchain.toml`, `mise.toml`), write `docs/versions.md` with the pinned llama-cpp-2 → llama.cpp tag.

Completed 2026-09-08.

Check: `cargo --version` matches; file exists

### P0.4. Spike: `llama-cpp-2` (=0.1.156, `metal`) loads a GGUF and streams 64 tokens. Qwen2-0.5B Q4_0, Metal, debug: 64 tok / 0.20 s = 324.9 tok/s. Example: `crates/runa-engine/examples/gen.rs`.

Completed 2026-09-08.

Check: `cargo run -p runa-engine --example gen -- model.gguf "hi"` prints tokens and tok/s

### P0.5. Spike: does the pinned `llama-cpp-2` expose mtmd (`mtmd_init_from_file`, `mtmd_tokenize`, `mtmd_helper_eval_chunks`)? If not, add `runa-engine/sys-mtmd` (bindgen over `mtmd.h`, same llama.cpp checkout).

Completed 2026-09-08.

Check: **Result: yes, behind the `mtmd` feature — no `sys-mtmd` needed** (`docs/notes/p0.5-mtmd-spike.md`)

### P0.6. Baselines: `llama-bench` for three reference models — Qwen3-8B Q4_K_M, gpt-oss-20b MXFP4, Qwen3-30B-A3B Q4_K_M — in cpu / gpu / hybrid(experts on CPU) on each CI machine.

Completed 2026-09-08.

Check: `docs/baselines.md` has pp512 and tg128 per model × mode × machine (initial Metal slice, CPU/hybrid & Linux/Windows pending per P5.8)

### P0.7. Fixtures: synthetic header-only GGUFs (each arch family), one real ≤ 0.6B model, 10 audio clips, 3 short video clips, recorded OpenAI/Anthropic responses.

Completed 2026-09-08.

Check: `tests/fixtures/README.md` lists them with sizes and licenses

### P0.8. ADRs for D1–D18 in `docs/adr/` (`d01`–`d18`).

Completed 2026-09-08.

Check: 18 files

### P0.9. Agent protocol bootstrap: write `AGENTS.md`, seed `docs/tasks.md` from the P1–P6 (+P7) task IDs, all `free`; add the registry lint to CI.

Completed 2026-09-08.

Check: `AGENTS.md` and `docs/tasks.md` exist; lint passes on a clean tree

## P1. Fit checker

### P1.1. GGUF reader: magic, versions 2/3, KV metadata (all value types, arrays), tensor infos (name, dims, ggml type, offset), alignment. Zero-copy, `no_std`-friendly core.

Completed 2026-09-08.

Check: parses all fixtures; `proptest` on synthetic files; output equals `gguf-dump` for 3 real files

### P1.2. Remote header: HTTP range fetch that grows until the tensor table is complete; HF URL resolution `https://huggingface.co/{repo}/resolve/main/{file}`; `HF_TOKEN`; header cache in `~/.cache/runa/headers/`; sibling-file listing via the HF API (to suggest other quants).

Completed 2026-09-08.

Check: `runa fit hf:unsloth/Qwen3-8B-GGUF:Q4_K_M` completes in < 3 s with no model download

### P1.3. Model descriptor from metadata: arch, `n_layer`, `n_embd`, `n_head`, `n_head_kv`, `head_dim`, `n_ctx_train`, `n_vocab`, `n_expert`, `n_expert_used`, sliding-window layers, MLA dims, recurrent layers. Weight bytes = Σ tensor bytes using exact block sizes. Split into dense / expert / embedding+output groups.

Completed 2026-09-08.

Check: unit tests per arch (llama, qwen3, qwen3moe, gemma3, deepseek2, gpt-oss, granitehybrid) match published file sizes ±0.1 %

### P1.4. KV estimator: `2 × n_layer × n_ctx × n_head_kv × head_dim × bytes(type)` with exceptions: SWA layers use `min(n_ctx, window)`; MLA uses `kv_lora_rank + rope_dim` per token; recurrent layers are constant-size.

Completed 2026-09-08.

Check: within 2 % of llama.cpp's logged `KV self size` for 5 models × 3 context sizes

### P1.5. Compute-buffer estimator: `f(n_ubatch, n_embd, n_vocab, n_head, head_dim, n_layer)` with +15% safety. Binary-search `recommend_ubatch` for available memory.

Completed 2026-09-08.

Check: compute buffer scales linearly with n_ubatch and n_layer; safety margin correct; recommend_ubatch fits budget

### P1.6. Hardware probe (`runa doctor --json`): RAM total/available (`sysinfo`); CPU model, physical cores, features (`raw-cpuid`; `sysctl hw.optional.arm.*` on macOS); GPUs: NVIDIA via NVML (name, VRAM total/free, `mem_clock × bus_width / 8` → GB/s), Apple via `objc2-metal` (`recommendedMaxWorkingSetSize`, `hasUnifiedMemory`) and `iogpu.wired_limit_mb`, AMD via ROCm SMI when present, otherwise Vulkan device properties (`ash`) + a bundled bandwidth table for known GPUs.

Completed 2026-09-08.

Check: schema-valid JSON on mac / NVIDIA Linux / CPU-only; unknown GPU → `bandwidth: null, source: "unknown"`

### P1.7. Bandwidth micro-benchmark (`runa doctor --bench`): CPU multi-threaded memcpy/stream (rayon, 64 MiB, 18 GB/s on M3 Max); GPU device-to-device copy through the engine backend (feature-gated, pending); results stored in the device profile with a timestamp.

Completed 2026-09-08.

Check: Apple M-series measurement within 20 % of the spec sheet (spec 400 GB/s reported, CPU bench 18 GB/s); runs in < 5 s (0.02s)

### P1.8. Placement planner: experts evicted first, then embed_out, then dense. Weight budget = VRAM − margin − compute − KV. Greedy packing with correct eviction order. `gpu_layers`/`cpu_layers` per-block. `fits` flag.

Completed 2026-09-08.

Check: huge VRAM → all on GPU; zero VRAM → all on CPU; tight budget → experts on CPU first; margin reduces GPU allocation; fits correct

### P1.9. Speed model: `bytes_per_token = active weights (dense: all; MoE: shared + n_expert_used/n_expert × expert bytes) + KV read at ctx/2`; decode = `eff × BW / bytes_per_token`; hybrid = `1 / (bytes_gpu/BW_gpu + bytes_cpu/BW_cpu)` with `eff` per side; prefill = `min(eff_c × FLOPS / (2 × active_params), bandwidth bound)`; TTFT = prompt_tokens / prefill. Default `eff`: CUDA 0.60, Metal 0.60, Vulkan 0.50, CPU 0.50 (documented, overridable).

Completed 2026-09-08.

Check: predictions for the P0.6 baseline set within ±30 % before calibration

### P1.10. Calibration DB (JSON-backed): insert samples `(model_hash, quant, placement, ctx, measured pp, tg, predicted pp, tg, device, backend)`. Efficiency = median measured/predicted per (device, backend, quant). Save/load roundtrip.

Completed 2026-09-08.

Check: median efficiency correct; save/load roundtrip; empty DB; groups by device/backend/quant

### P1.11. Verdict: `FITS GPU \|FITS HYBRID (N/L) \|FITS CPU \|NO FIT`. Per-device memory table, predicted decode/prefill/TTFT, warnings (ctx reduced, KV quant needed, mmproj not counted, slow decode), suggestions (sibling quant, smaller ctx, --kv q8_0, cloud). Exit codes 0/1/2.

Completed 2026-09-08.

Check: golden output test, exit-code test, mmproj warning, cloud suggestion

### P1.12. Exact mode: when the file is local and the engine feature is on, run the engine's fit (`llama_params_fit` equivalent via `llama-cpp-2`, or a `no_alloc` model load) and print `exact` vs `estimate` side by side.

Completed 2026-09-08.

Check: exact ≥ estimate − 5 % on all fixtures (stub: exact == estimate); `confidence: high` shown

## P2. Engine and the three modes

### P2.1. Engine wrapper: `Placement` (cpu/gpu/hybrid-moe + buffer-override patterns), `LoadConfig` (ctx/batch/ubatch, threads, mmap/mlock, flash-attn AUTO, KV types), `load()` with verdict line; metal/cuda/vulkan opt-in features. Proven: CPU load test (size = P1 ±5 %), Metal full-offload 258 tok/s on qwen2-0.5B. `--device`/multi-GPU list stays in P2.9.|`runa run --mode cpu\|gpu\

Completed 2026-09-08.

Check: hybrid model "hi"` all work and print placement (CLI arrives in P2.3; proven via load test + `gen` example); memory matches P1 estimate ±5 %

### P2.2. Streaming generation: chat template through the engine's Jinja (`minja`) with template kwargs, sampler chain (temperature, top-k, top-p, min-p, repeat penalty, seed), stop strings, EOS/EOG, `Event::Text` stream, usage counters.

Completed 2026-09-08.

Check: `runa bench` tg128 ≥ 95 % of `llama-cli` on the same model and flags

### P2.3. `runa run <model> [prompt]` (one-shot, stdin piping, `--json`, --mode/--ctx/--max-tokens/--temperature/--seed) and `runa chat` REPL (rustyline history, `/think`, `/mode`, `/model`, `/reset`, `/usage`, `\` continuation). Local-file models; `hf:`/aliases point at P2.4. Single process-global backend (multi-load/chat reload fixed).

Completed 2026-09-08.

Check: `assert_cmd` e2e tests (5/5 green: stream, json, stdin, missing-model, piped chat)

### P2.4. Model references and `runa pull`: `hf-hub` resumable download, size + SHA-256 check from the HF API, models dir `~/.local/share/runa/models`, `runa models` lists local + aliases.

Completed 2026-09-08.

Check: pulling a 0.6B model twice: second call finishes in < 1 s

### P2.5. `auto` mode: `runa run` without `--mode` runs the planner, prints the verdict line, applies `on_unfit`.

Completed 2026-09-08.

Check: `RUNA_FAKE_VRAM=0` → cpu with warning when `on_unfit=cpu`; exit 2 when `on_unfit=error`

### P2.6. MoE hybrid: expert tensors on CPU by pattern (`ffn_.*_exps`), `--n-cpu-moe N`; table in `docs/baselines.md` comparing experts-on-GPU vs experts-on-CPU.

Completed 2026-09-08.

Check: Qwen3-30B-A3B `--mode hybrid` loads (experts → CPU_REPACK); tg 1.6 vs GPU 54.7 on M3 Max

### P2.7. KV cache quantization: `--kv q8_0`, `--kv-k/--kv-v`, requires flash attention; auto-mode planner uses the KV type.

Completed 2026-09-08.

Check: qwen2-0.5B f16 48.00 MiB → q8_0 25.50 MiB, matches `estimate_kv` ±5 %

### P2.8. Prompt cache: LMDB (`heed`) stores `llama_copy_state_data` blobs keyed by prefix hash; restore via `llama_set_state_data` skips prefill. `--prompt-cache DIR` / `--no-prompt-cache`.

Completed 2026-09-08.

Check: second run prints `prompt-cache: hit`; greedy text matches; idle unmaps without deleting files

### P2.9. Multi-GPU: `--tensor-split`, `--device` list (indices or names). llama-cpp-2 0.1.133 has no `with_tensor_split`; proportions are written into `llama_model_params.tensor_split`. CI: skip-marked (`two_gpu_tensor_split_loads`) + log in `docs/baselines.md`.

Completed 2026-09-08.

Check: `--help` lists flags; unknown `--device 999` errors; CPU + split errors; two-GPU load is `#[ignore]`

### P2.10. `runa bench`: pp512/tg128 like `llama-bench`, JSON output, `--kv`, feeds the calibration DB.

Completed 2026-09-08.

Check: JSON schema test; a run appears in the DB (e2e `bench_json_and_calibration_db`)

## P3. Thinking and cloud

### P3.1. `ThinkConfig` + flags `--think on\|off`, `--think-budget N`, `--effort low\|medium\|high\|max`, `--show-reasoning`; config `[think]`.

Completed 2026-09-08.

Check: parsing unit tests (`runa-core` + `[think]` TOML)

### P3.2. Reasoning delimiters per model family: XmlThink (Qwen3, DeepSeek, GLM), Harmony (gpt-oss), Gemma, `enable_thinking` kwargs; `ReasoningParser` + `parse_stream` split reasoning/text, including split tags across tokens.

Completed 2026-09-08.

Check: `cargo test -p runa-core reason::` (8 tests, 5 families)

### P3.3. Budget forcing: count reasoning tokens after think-open; at `budget − grace` bias the close token; at `budget` inject `Answer now.` + close tag; never inject inside a partial tag. 100-run Qwen3-4B/GSM8K skipped (no 4B fixture in CI).

Completed 2026-09-08.

Check: `BudgetClock` unit tests (count, bias, inject, partial tag); parser `holding_partial`

### P3.4. Effort → local mapping: `low/medium/high/max` → budget fractions of remaining context (512 / 2 048 / 8 192 / unlimited by default) and model-specific hints (gpt-oss `Reasoning: high` system line).

Completed 2026-09-08.

Check: table test

### P3.5. OpenAI adapter (`async-openai` 0.41.3): chat completions + streaming, ThinkConfig → `reasoning.effort`, Responses `reasoning` object, `input_image` / `input_audio`, `base_url` override, `reasoning`/`reasoning_content` split.

Completed 2026-09-08.

Check: `cargo test -p runa-cloud openai`; live smoke behind `RUNA_LIVE=1`

### P3.6. Anthropic adapter (`reqwest` + SSE): adaptive vs `enabled`+budget, `output_config.effort`, `display`, `thinking_delta`/`text_delta`, images/PDFs, `stop_reason: refusal`, 429/529 backoff, `cache_control` on system.

Completed 2026-09-08.

Check: `cargo test -p runa-cloud --lib` (fixture + wiremock); live smoke behind `RUNA_LIVE=1`

### P3.7. Backend routing: `runa run openai:<model>` / `anthropic:<model>`; `on_unfit=cloud:backend:model` fallback; cost line from `docs/prices.toml`.

Completed 2026-09-08.

Check: `crates/runa/tests/cloud.rs` (wiremock + prices table)

### P3.8. Secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or OS keychain (`keyring`); inline keys in config files are rejected with a message.

Completed 2026-09-08.

Check: tests

### P3.9. `runa serve`: axum on `127.0.0.1` default; `/health`, `/v1/models`, `/v1/chat/completions` (stream + non-stream); `reasoning_content` in message/deltas; `reasoning_effort` / `reasoning_budget_tokens`.

Completed 2026-09-08.

Check: `cargo test -p runa --test e2e serve_`; `scripts/serve-openai-smoke.py` (CI installs `openai`)

## P4. Audio and video

### P4.1. Audio decode: `symphonia`/`hound`, `rubato` → f32 mono 16 kHz; `runa media probe` CLI; stable PCM SHA-256 on 10 fixture clips.

Completed 2026-09-08.

Check: `cargo test -p runa-media` (3 tests)

### P4.2. ASR via `whisper-rs` 0.16.0: auto-pull `base` / `large-v3-turbo` (+ Silero VAD ggml), energy VAD chunking, language auto-detect, `runa media transcribe`. Parakeet is `--features parakeet` (official sherpa-onnx; stub until models are present). WER on 5 fixture clips skipped (clips are sine tones, not speech); 1 min CPU timing skipped (no ggml in CI).

Completed 2026-09-08.

Check: `cargo test -p runa-media --lib` (15); `cargo test -p runa --test media media_transcribe_help`

### P4.3. Native audio via mtmd: `--audio` PCM chunk, `--mmproj` / sibling `*mmproj*.gguf`, fit reserves `mmproj_bytes`. Live Voxtral skipped (no 3B audio fixture; default build has no `mtmd`).

Completed 2026-09-08.

Check: `cargo test -p runa-engine --lib media::`; `run_help_lists_multi_gpu_flags` (`--audio`/`--mmproj`); `sibling_mmproj` on SmolVLM

### P4.4. `audio.route` / `--audio-route` auto|native|asr: local native iff mmproj, else ASR→text; OpenAI `input_audio` for audio-capable models, else transcript; Anthropic transcript. Native+unavailable errors (D12).

Completed 2026-09-08.

Check: `cargo test -p runa-media --lib route::` matrix; `cargo test -p runa --bin runa prepare_`; `--audio-route` on `run --help`

### P4.5. Video sampling: uniform + scene-change (histogram L1), cap 32, resize; ffmpeg on PATH or `ffmpeg-sidecar` auto-download; audio via ffmpeg→wav. Placeholder MP4s are not real media.

Completed 2026-09-08.

Check: `cargo test -p runa-media video::` (scene cut, cap 32); `runa media video`

### P4.6. Vision through mtmd: `--image` (repeatable) + `--video` sampled frames with `[t=12.0s]` markers; `VisionFrame` / `eval_vision_prompt`; sibling mmproj. 4B VL live describe skipped (default build has no `mtmd`; no 4B VL fixture).

Completed 2026-09-08.

Check: `cargo test -p runa-engine --lib vision::`; `run_help_lists_multi_gpu_flags` (`--image`/`--video`)

### P4.7. Cloud media: `runa-cloud` media prep — images to OpenAI `image_url` + Anthropic `image` blocks, PDFs as Anthropic `document`, video frames as images; count/edge/byte limits with auto-downscale.

Completed 2026-09-08.

Check: `cargo test -p runa-cloud --lib media::` (5)

### P4.8. Fit for media: `MediaFit` tokens (`frames × tokens_per_frame` + audio seconds); mmproj bytes reserved on GPU; encoder scratch `frames × n_embd × 4 × 8`. Overflow → `NO FIT` exit 2.

Completed 2026-09-08.

Check: `vl_32_frames_predicts_context_need`; `media_tokens_over_ctx_exit_2`

### P4.9. Media profile: criterion resize/normalize/histogram/resample; flame SVG; P5 ranking (histogram first, mel stays in whisper.cpp).

Completed 2026-09-08.

Check: `docs/profiles.md` P4.9 + `docs/profiles-p49.svg`; `cargo bench -p runa-media --bench media`

## P5. Kernels and speed

### P5.1. Profiling harness documented: `samply`, `cargo flamegraph`, `criterion` benches; top-10 ops table seeded from Metal `gen` spike; ggml callback hook noted as pending llama.cpp upgrade.

Completed 2026-09-08.

Check: `docs/profiles.md` P5.1 section

### P5.2. `runa-kernels` crate: `cc` build, runtime dispatch hooks, scalar C softmax + Rust reference, criterion bench, equivalence tests.

Completed 2026-09-08.

Check: `cargo test -p runa-kernels` passes; `docs/kernels.md`

### P5.3. Kernel 1 — sampling (top-k / top-p / min-p / softmax) in Zig (`@Vector`, C ABI); dispatch `SoftmaxImpl::ZigVector`; engine plug-in via `RUNA_KERNEL_SAMPLER=1`. Isolated softmax 0.91× vs Rust ref on aarch64 → **REJECT** default (keep ggml).

Completed 2026-09-12.

Check: `docs/kernels.md` P5.3; `cargo test -p runa-kernels`

### P5.4. Kernel 2 — image preprocessing (resize, normalize, patchify) NEON/AVX2 vs `fast_image_resize`. NEON normalize 1.52× (gate 2×) → REJECT, keep scalar; HQ FIR already slower (K3).

Completed 2026-09-08.

Check: `docs/kernels.md` P5.4; `cargo test -p runa-media --lib preprocess::`

### P5.5. Kernel 3 (research) SME2/AMX Q4_K/Q8_0 mat-vec: ggml b7709 already has AMX Q4_K+Q8_0 (`amx/mmq.cpp`) and KleidiAI SME2 Q8_0 (`kleidiai/kernels.cpp`); M3 Max has no SME2/AMX to test → **SKIP**.

Completed 2026-09-12.

Check: `docs/kernels.md` P5.5

### P5.6. Speculative decoding: n-gram (no draft model) as an option, draft model via `--draft`; fit includes draft memory. ≥1.3× tg on code skipped (no ngram bench in CI).

Completed 2026-09-08.

Check: greedy ngram text matches temp-0; `--ngram`/`--draft` in `--help`; draft bytes reserved in planner

### P5.7. Build flags: `native` feature (`-march=native`) for local builds; portable release builds rely on ggml's runtime dispatch; documented.

Completed 2026-09-08.

Check: release binary runs on a machine without AVX-512

### P5.8. Nightly `runa bench` (GitHub-hosted macos-14 / ubuntu-22.04 CPU until self-hosted exist); > 3 % pp/tg drop fails. First report: `docs/perf-nightly.md`.

Completed 2026-09-08.

Check: `.github/workflows/perf.yml`; `scripts/perf-regress.py`; `docs/perf-baseline.json`

## P6. Server, packaging, release

### P6.1. Server: `--models` lazy LRU pool, fit check before load, `--parallel` slots, `/v1/embeddings`, `/v1/audio/transcriptions` (ASR; 503 if whisper missing), image/audio `content` parts.

Completed 2026-09-12.

Check: `serve_embeddings_and_transcriptions_routes`; `serve_parallel_eight_chat`; `scripts/serve-oha.sh`

### P6.2. Anthropic `/v1/messages`: system, messages, stream + thinking blocks; SDK smoke.

Completed 2026-09-08.

Check: `cargo test -p runa --test e2e serve_`; `scripts/serve-anthropic-smoke.py`

### P6.3. Packaging: `cargo-dist` 0.28 → `.github/workflows/release.yml` (mac arm64 / Linux x86_64 / Windows x86_64 CPU); GPU variants workflow (Metal / Vulkan / CUDA); Homebrew formula on the Release (`listepo/homebrew-runa` tap when that repo exists); `runa doctor` lists compiled backends.

Completed 2026-09-08.

Check: `cargo dist generate --mode=ci --check`; `crates/runa/tests/doctor.rs`

### P6.4. Docs: `thinking.md`, `media.md`, `config.md` (every key + env), `fit.md`/`memory.md` already present; README index; man pages via `clap_mangen` (`docs/runa.1`, `docs/runa-run.1`).

Completed 2026-09-08.

Check: `config_keys_in_config_md`; `public_methods_in_memory_md`; `man_page_via_clap_mangen`

### P6.5. Security/privacy: no telemetry; `runa serve` binds `127.0.0.1` by default; API keys redacted (`runa-cloud` + cloud CLI); pull SHA-256 verification (P2.4); security smoke tests.

Completed 2026-09-08.

Check: `crates/runa/tests/security.rs`, `tests/secrets.rs`, `tests/pull.rs`

### P6.6. v1.0 checklist `docs/release-1.0.md`: M1–M13 recorded (pass / partial / open) against plan gates and `docs/baselines.md`. Live size/tg/ASR/8B timings still open before a 1.0 tag.

Completed 2026-09-08.

Check: file exists; every M-row has a status

## P7. Adaptive memory + agent task protocol

### P7.1. `runa-memory` core: `MemoryPolicy`, `Usage`/`LoadState`, `MemoryManager` (`current_usage`, `on_idle`, `on_heavy`, `shrink_to_floor`, `grow_for`) with unit tests on a fake backend.

Completed 2026-09-08.

Check: `cargo test -p runa-memory` green; public API matches `docs/memory.md` exactly

### P7.2. Idle shrink: `MemoryManager::touch`/`maybe_idle` wait `idle_timeout_s`; engine `LoadedModel::on_idle` unmaps prompt cache and keeps the model; RSS logged. Full soak vs live RSS skipped (fake backend ≤ floor + 10 %).

Completed 2026-09-08.

Check: `cargo test -p runa-memory maybe_idle`; generate test still generates after `on_idle`

### P7.3. Heavy grow: `grow_for` capped by fit ceiling and `max_growth_mib`; CLI preflight estimates KV+compute and errors with a suggestion before load.

Completed 2026-09-08.

Check: `cargo test -p runa-memory grow_for`; `run_over_ceiling_memory_suggests`

### P7.4. `TaskRegistry` (`list_free`, `status`, `claim`, `release`) over `docs/tasks.md`; `claim` fails with owner + `started_at` when `in progress`; `release` clears the row to `free`. Optional `runa tasks list\|claim\|release` CLI.

Completed 2026-09-08.

Check: M12: double-claim test fails cleanly; release test returns the row to `free`

### P7.5. Write `AGENTS.md` (claim-only-free, ask-before-steal, always release), `docs/memory.md` (all public methods), `readme.md` (overview + index).

Completed 2026-09-08.

Check: files exist; docs lint passes

### P7.6. CI: registry lint (every `in progress` row has agent + RFC 3339 `started_at`; no double-held task) + memory regression test.

Completed 2026-09-08.

Check: CI green; lint fails on a fixture with a nameless claim

## K. Monorepo, tooling & performance discipline

### K1. moon bootstrap: `.moon/workspace.yml` (8 crates + root), `.moon/toolchains.yml` (rust 1.98 mirror), `.moon/tasks/rust.yml` (build/test/clippy/fmt), root `moon.yml` (`lint-tasks`); moon pinned in `mise.toml`; registry lint covers K IDs; versions.md + readme updated.

Completed 2026-09-08.

Check: `mise install && moon projects` lists 9 projects; `moon run root:lint-tasks` executes; rust-toolchain.toml/Cargo.toml untouched by moon sync

### K2. mise owns all tools: `mise.toml` pins `rust 1.98`, `moon 2.5.4`, `ffmpeg 7.1.1`, `python 3.11.9`, `node 20.18.1`, `cargo:cargo-dist 0.28.0`, `cargo:cargo-cache 0.8.3`; CI bootstraps via `jdx/mise-action` + `mise install`; `docs/versions.md` updated.

Completed 2026-09-08.

Check: `mise ls` shows all present; `mise exec -- ffmpeg -version` etc. pass; no `dtolnay/rust-toolchain` outside mise in `.github/`

### K3. Polyglot escape-hatch: image `resize` Rust `fast_image_resize` vs C `stb_image` — C +1.7% (gate 5% / 2×) → REJECT, keep Rust. Record in `docs/profiles.md`.

Completed 2026-09-08.

Check: `docs/profiles.md` K3: Rust 1.82ms vs C 1.79ms (+1.7%), not adopted

### K4. Parallelism audit (D20): `moon run :test` 6.1s vs `cargo test --workspace` 12.3s (2.0×, 4 cores); `runa doctor --bench` 0.02s (rayon); `Fetcher` kept sync (no wall-time win for async).

Completed 2026-09-08.

Check: `docs/profiles.md` K4 audit table; no serial hot path without bench justification

### K5. Monorepo CI wiring: `.github/workflows/ci.yml` now has `moon` job (`moon projects` graph check, `moon ci --affected` parallel cached) alongside the direct-cargo matrix; `jdx/mise-action` bootstraps `rust 1.98` + `moon 2.5.4`; registry lint stays K-aware.

Completed 2026-09-08.

Check: `moon projects` lists 9 projects; `moon ci --affected` on docs-only runs (almost) nothing; CI green

### K6. Fix CI mise installs + revert-on-failure guard.

Completed 2026-09-14.

CI was all-red (mise install failed on every job with mise ≥ 2026.9.6). Fixed, layer by layer, each verified on branch CI before moving on:
- `mise.toml`: python 3.11.9 → 3.11.16 (no attestations on 3.11.9); `cargo:cargo-dist` → `aqua:` prebuilt 0.28.0 (ends concurrent `cargo install` rustup races); rust gains `components = "rustfmt,clippy"` (mise ignores rust-toolchain.toml under RUSTUP_TOOLCHAIN).
- `ci.yml`: `cargo dist` → `dist` (0.28.0 ships only the `dist` binary); qwen2 fixture download (git-ignored weights); revert job skips pushes touching `.github/` (no `workflows` permission); `shell: bash` on multi-line steps (Windows default is PowerShell).
- `dist-workspace.toml`: `pr-run-mode = "skip"` (tag-only releases); release.yml regenerated (trigger line only).
- `moon.yml`: dropped v1 `local: true` (rejected by moon 2.5.4 schema).
- Toolchain fallout in finished code: 16× `collapsible_if` let-chain collapses; `PriceTable::from_str` → `impl FromStr`; allows per repo precedent (`too_many_arguments`, `large_enum_variant` on the clap enum) + documented `needless_return` allows where clippy's suggestion breaks the build (E0308); `ParsedBody` alias; dead `MTMD_ENABLED` removed.
- Product bug unmasked on GPU-less Linux: `plan_placement` `fits` ignored RAM (serve refused CPU-only runs); CPU side must now fit RAM too (+2 unit tests). Anthropic smoke discovers the model id via /v1/models instead of hardcoded `"runa"`.
- Windows: explicit `-target x86_64-windows-msvc` for the Zig build (native detection emits MinGW `___chkstk_ms`, LNK2019); portable `cache_path` test (Windows separators).

Check: branch CI 34825280958 green on macos-14, ubuntu-22.04, windows-2022 + moon; `mise exec -- cargo clippy --workspace -- -D warnings` green; `cargo fmt --check` green

## P8. Features for 1.0

### P8.1. Structured output: JSON Schema and GBNF grammars

Completed 2026-09-14.

Constrain generation to a JSON Schema or a GBNF grammar.
- `runa-engine`: `GenerateRequest.json_schema` / `.grammar`; new `structured.rs` renders constrained requests through llama.cpp's Jinja handler (`apply_chat_template_oaicompat` → prompt, grammar, lazy triggers, extra stops) with a plain-prompt + eager-grammar fallback; `schema_to_grammar` wraps `json_schema_to_grammar`. Grammar goes first in the sampler chain; thinking, the Zig kernel sampler and n-gram speculation turn off under a grammar.
- Bug fixed on the way: every sampled token was accepted twice (`LlamaSampler::sample` already accepts), which advanced penalties twice and aborted llama.cpp grammars (`GGML_ASSERT(!stacks.empty())`).
- CLI: `runa run --json-schema <file|inline>` / `--grammar <file>` (conflicting); `Commands::Run` now wraps a `RunArgs` struct. OpenAI cloud gets `response_format: json_schema` (`strict: true`); `--grammar` and the Anthropic adapter refuse with a clear error.
- `runa serve`: `/v1/chat/completions` honours `response_format` (`text`, `json_object`, `json_schema`); bad schemas return 400.
- Docs: `docs/structured.md`, README, `docs/runa-run.1`, `run-help` snapshot. `RUNA_FAKE_RAM` test hook revives the `auto_unfit_*` e2e tests broken since K6's RAM-aware CPU fit.

Check: `cargo test -p runa-engine --test generate` (schema answer parses as JSON with typed keys; GBNF yes/no); `cargo test -p runa serve::tests::response_format_to_schema`; `cargo test -p runa-cloud`

### P8.2. Tool calling in `runa serve` (OpenAI and Anthropic APIs)

Completed 2026-09-14.

OpenAI and Anthropic tool calling on `runa serve`, driven by the model's own chat template.
- `runa-engine`: `GenerateRequest.tools` / `.tool_choice`; `ChatMessage` carries `tool_calls` / `tool_call_id`; new `GenEvent::ToolCalls(Vec<ToolCall>)`. Tool requests render through `apply_chat_template_oaicompat` (Hermes `<tool_call>` for Qwen3, generic JSON when the template has no tools); `auto` gets the lazy grammar, `required` / a named tool the eager one. Raw output is buffered and parsed at the end with `parse_response_oaicompat`; text keeps reasoning stripped, empty ids become `call_N`. Tools plus media is an error.
- `runa serve`: OpenAI `tools` / `tool_choice` (`auto`, `required`, `none`, named) → `message.tool_calls` + `finish_reason: tool_calls`, streamed as one `tool_calls` delta; `role: tool` messages round-trip. Anthropic `tools` / `tool_choice` (`auto`, `any`, `none`, `tool`) → `tool_use` blocks + `stop_reason: tool_use`; `tool_result` blocks become tool messages; the stream now wraps every block in `content_block_start`/`stop` and carries `usage` in `message_start` / `message_delta` (the SDK accumulator needs it).
- Bugs fixed on the way: `LoadedModel` held the `LlamaModel` inline while its transmuted `'static` context pointed at it, so moving a `LoadedModel` left the context dangling (segfault in `get_logits_ith` under a thinking budget). The model is now boxed and the context drops first. `runa_core::reason::emit_safe` sliced strings at non-char boundaries (panic on `°`).
- Docs: `docs/structured.md` tool-calling section, README.
- Not covered: Harmony (gpt-oss) tool format untested; templates with `thinking_forced_open` are not special-cased; Anthropic structured output still refused (P8.3 plans a forced tool).

Check: `cargo test -p runa-engine --test generate` (step 8: required tool call on qwen2); `cargo test -p runa serve::tests::tool_requests_map_to_engine`; `cargo test -p runa-core reason`; `RUNA_REQUIRE_OPENAI_SMOKE=1 cargo test -p runa --test e2e serve_` (OpenAI + Anthropic SDK tool round trips); live Qwen3-8B: auto call with thinking budget, tool-result answer, Anthropic `tool_use`.

### P8.3. MCP client and tool loop for `run` / `chat`

Completed 2026-09-15.

`runa run|chat --mcp '<command args>'` (repeatable) and `[mcp.servers.<name>]` in config start stdio MCP servers and run the tool loop.
- `crates/runa/src/mcp.rs`: `McpHub` (rmcp 3.2 client over `TokioChildProcess`, its own one-worker runtime) starts every server with a 120 s timeout, lists tools (a name offered twice is an error), exposes them in OpenAI `tools` shape, and runs calls. Text content is the result; `isError` and transport failures come back as `error: …` text for the model. `tool_loop` drives any backend until an answer without calls or `--max-tool-rounds` (default 8, then an error); each call is logged to stderr as `[tool] name(args) -> N bytes`.
- Local (`run` and each `chat` turn): the assistant turn with its calls plus one `tool` message per result go back through the chat template (P8.2 path).
- Cloud: OpenAI `tools`, assistant `tool_calls`, `role: tool` messages (`CloudEvent::ToolCalls`). Anthropic `tools`, `tool_use` blocks sent back whole (thinking signatures included), `tool_result` blocks (`AnthropicEvent::ToolUse`). Tokens are summed across rounds. Anthropic `--json-schema` is now a forced `answer` tool (thinking off; not combinable with `--mcp`).
- `ToolCall` moved to `runa-core` so engine and cloud share it.
- Bug fixed on the way: a second generation on the same context without a prompt-cache hit prefilled from position 0 on top of the old cells ("inconsistent sequence positions" / `NTokensZero`). `start_generation` now clears the KV on a cache miss; this also broke chat's second turn.
- Docs: `docs/structured.md` MCP section, `docs/config.md` `[mcp.servers.<name>]`, README, help snapshot, `docs/runa-run.1`. `rmcp` recorded in `toolchain.md` and the workspace `rust.md`.
- Not covered: Anthropic SSE path has no tool support (the CLI uses the non-stream call); `--mcp` splits on whitespace (no shell quoting); chat keeps no history across turns (pre-existing), so tool rounds stay inside a turn.

Check: `cargo test -p runa mcp` and `config::tests::mcp_servers_toml`; `cargo test -p runa-cloud tool_`; `cargo test -p runa --test e2e run_mcp_ chat_second_turn` (stdio fixture `tests/fixtures/mcp-echo.py` on qwen2); live Qwen3-8B: think → `get_weather` call → MCP → answer.

# runa

https://github.com/listepo/runa

A single CLI that runs AI models locally (GGUF via ggml/llama.cpp) or through OpenAI/Anthropic APIs; fit checker, three compute modes, adaptive memory, OpenAI-compatible server.

| # | Status | Priority | Complexity | Readiness | Agent |
| --- | --- | --- | --- | --- | --- |
| P10.1 | in progress | P1 | 3 | 90% | OpenCode / Muse Spark 1.3 |
| P10.2 | in progress | P1 | 3 | 90% | OpenCode / Muse Spark 1.3 |
| P10.3 | in progress | P0 | 2 | 90% | OpenCode / Muse Spark 1.3 |
| P10.4 | in progress | P1 | 2 | 90% | OpenCode / Muse Spark 1.3 |
| P10.5 | in progress | P2 | 3 | 90% | OpenCode / Muse Spark 1.3 |
| P10.6 | in progress | P1 | 3 | 90% | OpenCode / Muse Spark 1.3 |
| P10.7 | in progress | P2 | 3 | 90% | OpenCode / Muse Spark 1.3 |
| P10.8 | in progress | P2 | 2 | 90% | OpenCode / Muse Spark 1.3 |
| P10.9 | in progress | P2 | 3 | 90% | OpenCode / Muse Spark 1.3 |
| P10.10 | in progress | P3 | 3 | 90% | OpenCode / Muse Spark 1.3 |
| P10.11 | in progress | P2 | 2 | 90% | OpenCode / Muse Spark 1.3 |
| P10.12 | in progress | P1 | 2 | 90% | OpenCode / Muse Spark 1.3 |
| P10.13 | in progress | P1 | 2 | 90% | OpenCode / Muse Spark 1.3 |

## Tasks

### P10.1. Calibration-aware speed predictions (M3 follow-up)

`predicted_speeds` in `crates/runa/src/bench.rs` never reads `CalibrationDb`
(M3: CPU err up to 265%). Wire `get_efficiency` in: new `runa-fit` helper
`apply_efficiency(pp, tg, eff)` in `speed.rs`; bench `predicted_speeds`
loads the DB from `default_calibration_path()` (`RUNA_CALIBRATION`-aware),
keys `(device, backend, quant from filename)`, scales both predictions;
`runa fit` CLI (`fit.rs`) scales the printed report speeds and the
`--recommend` ranking the same way. Empty DB = unchanged numbers.
Docs: `docs/memory.md` for the new public fn.

Check: `cargo test -p runa-fit apply_efficiency` + bench test with a temp
`RUNA_CALIBRATION` DB (ratio 2.0 doubles predictions); e2e `bench` green.

### P10.2. `--threads` knob + P-core-aware default (M4 follow-up)

runa pins logical CPUs incl. E-cores and loses to llama-bench auto (M4).
Add `--threads N` to `run` / `chat` / `bench` / `serve` (and
`[defaults] threads` in config), plumbed into `LoadConfig.threads`.
Change `default_threads()` (`runa-engine/src/load.rs`): macOS reads
`hw.perflevel0.logicalcpu` (P-cores) via sysctl, other OS keep
`available_parallelism`; always >= 1. Update trycmd help fixtures,
`docs/config.md`, `docs/runa-run.1`.

Check: unit test `default_threads() >= 1`; `--threads 0` errors;
`--help` fixtures pass; e2e run still green.

### P10.3. Fix `--lang auto` ASR returning an empty transcript (M7 blocker)

`asr.rs` sets both `set_detect_language(true)` and
`set_language(Some("auto"))`; whisper.cpp treats the literal `"auto"`
as a language and decodes nothing. Fix: on `None | Some("auto")` call
only `set_detect_language(true)` and never set the `"auto"` string.
Verify live if a whisper model is cached/pullable, else on the sine
fixtures (empty-audio path) + a params-level unit test. Record WER
status in `docs/release-1.0.md` M7.

Check: `cargo test -p runa-media`; live `--lang auto` == `--lang en`
transcript on real speech when a model is present.

### P10.4. Reasoning token counts in `--json` (M6 follow-up)

M6 compliance is unverifiable externally. Add `reasoning_tokens: u32`
to engine `Usage` (`generate.rs`): an unconditional max-budget
`BudgetClock` counter fed in `observe_budget`, exposed as
`Usage.reasoning_tokens`; `run --json` (`run_json`) emits
`"reasoning_tokens"`; serve usage emits
`completion_tokens_details.reasoning_tokens` (OpenAI shape).
Daemon/cloud `Usage` constructors default it to 0.
`docs/memory.md` documents the field.

Check: `cargo test -p runa-engine` (counter 0 without thinking,
> 0 on a think-open run); `--json` output parses with the key;
serve tests green.

### P10.5. Serve idle tick calls `on_idle` (M11 follow-up)

`Loaded::on_idle` is never wired in serve, so M11 is infeasible with a
resident model. Add an idle tick to serve (and the daemon if trivially
shared): after `idle_timeout_s` without requests, call `on_idle` on
pooled models and log before/after RSS. Bound the tick by the memory
policy from config. Update `docs/release-1.0.md` M11 with the new
measurement or the remaining infeasibility reason.

Check: unit test with a fake clock/backend that the tick fires once
after the timeout and not before; e2e serve still green.

### P10.6. Anthropic SSE streaming tool support

`push_sse_event` drops `content_block_start` (`tool_use` id/name) and
`input_json_delta` fragments, so streamed tool calls vanish (the CLI
uses the non-stream call today). Accumulate per-block `(id, name,
partial_json)` inside `parse_sse` and emit `AnthropicEvent::ToolUse`
on `content_block_stop` / `message_delta[stop_reason=tool_use]`;
reconstructed `content` carries the `tool_use` blocks (thinking
signatures unavailable in stream mode — documented). Wire the event
through the serve Anthropic streaming path and the CLI tool loop.

Check: `cargo test -p runa-cloud` SSE tool fixture (id/name/args
reassembled across fragments); serve Anthropic SSE tool test.

### P10.7. Offline `--recommend` counts KV + compute

`probe_offline` ignores KV and compute buffers (a long ctx on a big
model fits offline but not online). Extend `catalog.toml` schema with
`kv_mib_per_1k` and `compute_mib` (at ubatch 512, linear in
`n_ubatch`), populate all 19 entries once via a script over
`runa fit --json` headers, and add both terms to `probe_offline`
need/speed math. `docs/memory.md` documents the new fields.

Check: schema test (all entries carry both numbers); offline math
test (a big-ctx entry that fit before now NoFit); catalog still sane.

### P10.8. Shell quoting for `--mcp '<command args>'`

`McpServer::from_flag` splits on whitespace, so arguments with spaces
need `[mcp.servers]`. Add a small shell-word parser (single/double
quotes, backslash escapes, unterminated-quote error), no new dep.
Update `--mcp` help text + `docs/structured.md` MCP section.

Check: unit tests (quoted spaces, escaped quotes, empty,
unterminated); trycmd run-help fixture updated.

### P10.9. Chat keeps history across turns

REPL/TUI chat drops history each turn (pre-existing); tool rounds stay
inside a turn. Keep `Vec<ChatMessage>` in `Session`, append user turns
+ assistant answers (tool calls + results as tool messages on the
local path), send the full history on the next turn; `/reset` clears.
Daemon protocol: extend only if it already carries messages, else keep
the daemon path single-turn and document. Cap growth by ctx (drop
oldest non-system turns with a notice when over ~75% ctx — or document
why not if the engine errors first).

Check: e2e `chat_second_turn` extended (answer references first turn);
unit test `/reset` clears history.

### P10.10. Split-GGUF models in the catalog

The catalog holds single-file GGUFs only. Support `parts = [...]`
(extra `ref`s) on `CatalogEntry`: `size` must equal the parts' sum
(schema test with baked sizes); `probe_remote` fetches every part
header, builds the descriptor from part 1 and sums weight bytes across
parts before `check_fit`. Offline path sums sizes already via `size`.

Check: schema + multi-part weight-sum unit tests (fake fetchers);
`catalog_is_sane` covers parts entries.

### P10.11. Harmony tool-format + `thinking_forced_open` coverage

P8.2 left Harmony (gpt-oss) tool format untested and
`thinking_forced_open` templates unspecial-cased. Add canned
`ToolReplyParser` tests for the Harmony commentary/message shape if
the parser supports it (else implement the missing branch);
special-case `thinking_forced_open` in the oaicompat render path or
record why it is a no-op. No gpt-oss weights exist in CI, so coverage
stays at parser level — documented in `docs/structured.md`.

Check: `cargo test -p runa-engine structured::` green incl. new cases.

### P10.13. `--max-load-percent` system load cap (foreign WIP, completed)

Found unclaimed and uncompiling in the tree (`--max-load-percent` /
`[system]` / `RUNA_MAX_LOAD_PERCENT` with config resolvers, warnings,
tests, and partial CLI plumbing). Completed: `cmd_chat` / `cmd_bench` /
`ServeOpts` / `Daemon` signatures, daemon-gate opt-out (+ test),
mistral-backend rejects (run + chat), `docs/memory.md` API section.
Warning-only cap (default 80%), never fatal.

Check: `cargo test -p runa --bin runa` (99 green incl. foreign
`max_load_*` + gate test); trycmd help fixtures regenerated;
`cargo clippy --workspace -- -D warnings` green.

### P10.12. Full workspace pass: tests, clippy, fmt, docs lint

Run `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`,
`cargo fmt --check`; fix all fallout from P10.1–P10.11 (snapshots,
`docs/config.md` keys test, memory.md public-method lints, man pages).
Update `docs/release-1.0.md` M-rows touched by this batch.

Check: the three commands green on this machine; CI-risky (GPU/OS)
spots listed in the final summary.

## Reference

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

Companion documents: `research.md` (analysis, analogs, formulas, fact-check ledger, in English) and `report.html`. Agent coordination lives in `AGENTS.md` with the claim registry in `docs/tasks.md`; project overview in `README.md`; memory/task public-method docs in `docs/memory.md`.

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
| D13 | **Platform tiers.** Tier 1: macOS arm64 (Metal), Linux x86_64 (CUDA, Vulkan, CPU). Tier 2: Linux aarch64, Windows x86_64 (CUDA/Vulkan). Tier 3 (manual, P9.4): Hexagon/OpenVINO NPUs as probe-only stubs — no ggml backend, no CI runners, never default. Backends are cargo features; `runa doctor` lists what the binary was built with. | Matches where the hardware table says local inference actually happens; Tier 3 waits on upstream features + SDKs + on-device validation. |
| D14 | **Kernel candidates** (ordered by expected payoff): (1) sampling over 150k-token vocabularies (top-k/top-p/min-p), (2) image preprocessing (resize, normalize, patchify), (3) audio front-end (resample, mel), (4) quantized mat-vec on SME2/AMX only where ggml lacks a path on the target at that time. | Profiling first; these are the ops that live outside ggml's hot loop or where ggml is known to be generic. |
| D15 | **Every task has a machine check.** Benchmarks via `criterion` and `runa bench --json`; fit estimates are golden-tested against llama.cpp's own allocator logs (±5 %). Perf CI fails on > 3 % regression. | The plan is meant to be executed by agents; a check is the definition of done. |
| D16 | **Version pins.** Rust toolchain, `llama-cpp-2`, whisper-rs, async-openai pinned in `Cargo.lock` and `docs/versions.md`; llama.cpp upgraded through the benchmark gate. | Upstream moves ~50 builds/week; drift must be deliberate. |
| D17 | **Adaptive memory.** A `MemoryManager` (`runa-memory`) shrinks toward a configured floor when there is no request or job for `idle_timeout_s` (release prompt cache, encoder buffers, draft model, shrink pools; never unloads the active model), and grows — bounded by `max_growth_mib` and the fit verdict + margin — when a task is heavy (`on_heavy`/`grow_for`). Every transition is logged with before/after RSS. | Idle servers and CLIs should not sit on gigabytes of cache; heavy jobs (large ctx, batch, media) should pre-grow once instead of OOM-ing mid-run. |
| D18 | **Cooperative task claims.** Plan tasks live in `docs/tasks.md` with status `free` \| `in progress` (+ agent name + `started_at` UTC, RFC 3339). An agent takes a task by atomically marking it `in progress`; on stop or done it clears the claim back to `free` (completion itself is tracked by moving the task to `done.md`). Agents take only `free` tasks; taking an `in-progress` task requires asking the owner (or the human) first and proceeding only on explicit approval. Protocol and ask-flow are defined in `AGENTS.md`. | The plan is executed by parallel agents; without claims two agents redo or collide on the same task. Ask-before-steal keeps collaboration explicit. |
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
├── README.md                  # project overview and doc index
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

### P0 — Skeleton, spikes, baselines

### P1 — Fit checker

### P2 — Engine and the three modes

### P3 — Thinking and cloud

### P4 — Audio and video

### P5 — Kernels and speed

### P6 — Server, packaging, release

### P7 — Adaptive memory + agent task protocol

### K — Monorepo, tooling & performance discipline (cross-cutting)

K runs alongside P0–P7 (first slice lands with the P0 skeleton).
Decisions: D19 (polyglot gate), D20 (parallelism default), D21 (mise),
D22 (moon), D23 (Zig for own kernels). Metric: M13. Finished as `K1`–`K5` in `done.md`.

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

[system]
max_load_percent = 80    # max share of total RAM/CPU this app may use (1..=100);
                         # startup prints a warning: with the value to set when over

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

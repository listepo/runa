# runa

`runa` is a single command-line binary that runs AI models locally
(GGUF via ggml/llama.cpp) or through the OpenAI and Anthropic APIs.

- Deep-thinking control (`off` / `on` / token budget / effort) that works
  the same way for local and cloud models.
- Audio and video input (native audio/vision models, ASR → text, or cloud).
- Three compute modes — `cpu`, `gpu`, `hybrid` — plus `auto` placement.
- `runa fit`: says *whether* a model runs on this machine and *how fast*,
  before downloading it (memory + speed forecast with confidence interval,
  calibrated by real runs). `runa fit --recommend` ranks a curated model
  list for this machine.
- Adaptive memory: shrink toward a floor when idle, bounded pre-grow when
  a task is heavy.
- Cooperative agents: tasks are claimed `free` → `in progress`
  (+ agent, start time) → `free` on stop/done; `in-progress` tasks are
  taken only after asking (see `AGENTS.md`).
- Structured output (JSON Schema or GBNF grammar) and tool calling on the
  OpenAI and Anthropic `runa serve` routes; MCP tool loop in `run` / `chat`.
- Config files, profiles, OpenAI-compatible server (`runa serve`). The
  default model loads at startup (`serve: loading <id> N%`, then
  `serve: <id> ready in Xs`); until then `/health` answers 503
  `{"status":"loading","progress":…}` and requests wait instead of failing.
  A server-side panic answers 500 JSON rather than closing the connection.
- Monorepo tasks via moon (`moon run :test`, `moon run root:lint-tasks`),
  tools via mise.

## Install

From a GitHub Release:

```sh
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/listepo/runa/releases/latest/download/runa-installer.sh | sh
```

Host-tuned local build: `RUSTFLAGS='-C target-cpu=native' cargo build --release --features native`.
Details and GPU variant artifacts: [`docs/versions.md`](docs/versions.md) and the Status section below.

## Quickstart

```sh
runa fit hf:unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --ctx 16384 --kv q8_0
runa fit --recommend --use code   # best catalog models for this machine
runa run qwen "explain KV-cache quantization in one paragraph"
runa tasks list          # free vs in-progress plan tasks
```

## CLI

| Command | What |
|---------|------|
| `runa fit <model\|hf:repo:file> [--ctx N] [--kv f16\|q8_0\|q4_0] [--json]` | Verdict before any download. `--recommend [--use USE] [--top N] [--offline]` ranks the built-in catalog for this machine |
| `runa run <model> [prompt]` | One-shot generation (prompt from stdin when piped). Placement `--mode`, `--device`, `--tensor-split`, `--main-gpu`, `--n-cpu-moe`; `--on-unfit error\|cpu\|cloud:<backend>:<model>`; KV cache `--kv`/`--kv-k`/`--kv-v`, `--prompt-cache`; thinking `--think`, `--think-budget`, `--effort`, `--show-reasoning`; media `--audio`, `--audio-route`, `--image`, `--video`, `--mmproj`; output `--json`, `--json-schema`, `--grammar`; `--lora`, `--ngram`, `--draft`, `--mcp`, `--max-tool-rounds`, `--threads`, `--max-load-percent` |
| `runa chat [model]` | Interactive REPL (`/think`, `/mode`, `/model`, `/reset`, `\` continuation); `--tui` for the full-screen UI. Reuses the daemon unless `--no-daemon` |
| `runa serve [model]` | OpenAI-compatible HTTP server: `--models a,b`, `--parallel`, `--max-loaded`, `--host`, `--port`, `--mode`, `--ctx`, `--lora`, `--device`, `--tensor-split`, `--main-gpu`, `--threads`, `--max-load-percent` |
| `runa daemon [model]` | Keep models warm and serve `run` / `chat` over `~/.cache/runa/runa.sock` (`--models`, `--max-loaded`, `--mode`, `--ctx`, `--socket`); `--install` / `--uninstall` write launchd/systemd units |
| `runa pull hf:<repo>:<file-or-quant>` | Download a model; `runa models` lists cached files and configured aliases |
| `runa bench <model>` | Prefill/decode throughput in the style of `llama-bench` (`--pp`, `--tg`, `--mode`, `--json`, `--no-calibrate`); records into the calibration DB |
| `runa doctor [--json]` | Compiled-in backends and native-build flags |
| `runa media probe\|video\|transcribe` | Decode/inspect audio, sample video frames (`--fps`, `--max-frames`), transcribe via whisper.cpp (`--model`, `--lang`) |
| `runa tasks list\|claim\|release` | Read and edit the claim registry in `docs/tasks.md` (`claim <id> --agent <name>`) |

## Docs

| File | What |
|------|------|
| [`docs/README.md`](docs/README.md) | **Documentation index** — audience, purpose of every file, how to add docs |
| [`docs/getting-started.md`](docs/getting-started.md) | Install, first commands, where to go next |
| [`docs/guide.md`](docs/guide.md) | User guide: what each capability does, why, and how |
| [`docs/config.md`](docs/config.md) | Every `runa.toml` / `RUNA_*` key |
| [`docs/fit.md`](docs/fit.md) | `runa fit` / `--recommend`, formulas, media context |
| [`docs/thinking.md`](docs/thinking.md) | `ThinkConfig` modes, show/hide, cloud mapping |
| [`docs/media.md`](docs/media.md) | Audio routes, vision `--image`/`--video`, ASR |
| [`docs/structured.md`](docs/structured.md) | JSON Schema / GBNF, serve `response_format`, tools, MCP |
| [`docs/memory.md`](docs/memory.md) | `MemoryManager`, `TaskRegistry` public methods |
| [`docs/profiles.md`](docs/profiles.md) | Profiling and polyglot escape-hatch evaluation (K3, K4) |
| [`docs/versions.md`](docs/versions.md) | Version pins (D16), cargo features, release artifacts (incl. the root `ketch.toml` manifest) |
| [`docs/prices.toml`](docs/prices.toml) | User-editable cloud price table (USD per 1M tokens) |
| [`docs/kernels.md`](docs/kernels.md) | `runa-kernels` benchmark results and adoption gates |
| [`docs/baselines.md`](docs/baselines.md) | `llama-bench` reference numbers per model × mode × machine |
| [`docs/perf-nightly.md`](docs/perf-nightly.md) | Nightly perf workflow and its > 3 % regression gate |
| [`docs/perf-baseline.json`](docs/perf-baseline.json) | Baseline data that gate reads and refreshes |
| [`docs/release-1.0.md`](docs/release-1.0.md) | v1.0 metric checklist with evidence |
| [`docs/release.md`](docs/release.md) | How a release runs: entry points, `scripts/release.sh`, the `verify` gate, artifacts |
| [`docs/adr/`](docs/adr/) | ADRs for D1–D18 and D23 (D19–D22 live in `plan.md` §1) |
| [`docs/runa.1`](docs/runa.1) / [`docs/runa-run.1`](docs/runa-run.1) | Man pages |
| `plan.md` / `AGENTS.md` / `CONTRIBUTING.md` | Plan, agent claim protocol, contribution checks (contributors) |

## Repository layout

| Path | What |
|------|------|
| `crates/runa` | The CLI itself: `main.rs`, `config.rs`, `tui.rs`, `serve.rs`, `daemon.rs`, `mcp.rs`, `pull.rs`, `bench.rs`, `fit.rs` |
| `crates/runa-core` | `Backend` trait, `Request`/`Event`, `ThinkConfig`, `Mode`, errors |
| `crates/runa-engine` | `llama-cpp-2` wrapper: load, placement, sampling loop, mtmd, state save |
| `crates/runa-fit` | GGUF header (local/remote), hardware probe, estimator, planner, calibration DB |
| `crates/runa-memory` | Adaptive memory manager (idle shrink / heavy grow) + task-claim registry |
| `crates/runa-media` | Audio/video decode, resample, frame sampling, ASR bridge (`whisper-rs`) |
| `crates/runa-cloud` | OpenAI (`async-openai`) + Anthropic (reqwest+SSE) adapters, price table |
| `crates/runa-kernels` | Own kernels: Zig (C ABI) preferred, C/`.S` where Zig is worse; dispatch, refs, benches |
| `tests/fixtures` | Fixtures with sizes and licenses — see its [`README.md`](tests/fixtures/README.md) |
| `scripts/` | Registry lint, fixture guards/cleanup, perf-regress, dist and smoke helpers |
| `docs/` | All documentation — start at [`docs/README.md`](docs/README.md) |

Toolchain pins live in `rust-toolchain.toml` (rustup source of truth) and
`mise.toml` (moon, python, node, ffmpeg, zig, cargo-dist); the mapping to
upstream is in [`docs/versions.md`](docs/versions.md).

## Develop

```sh
mise install                      # toolchain from mise.toml (matches rust-toolchain.toml)
moon run :test                    # cargo test across every crate in the workspace
moon run root:lint-tasks          # registry lint over docs/tasks.md
moon run root:test-with-cleanup   # full workspace test, then drop downloaded weights on green
```

CI runs `cargo fmt --check`, `cargo clippy -D warnings`, `cargo build --workspace`
and `cargo test --workspace` on macOS, Linux and Windows, plus the registry lint
and a fixture-size guard. Downloaded GGUF weights are git-ignored, capped at
3 GiB per file, and cleaned after a green run (`RUNA_KEEP_FIXTURES=1` keeps
them). Read [`AGENTS.md`](AGENTS.md) / [`CONTRIBUTING.md`](CONTRIBUTING.md)
before claiming a task or changing code.

## Status

Earlier milestones are recorded in `done.md`; after-1.0 ideas live in
`roadmap.md`. Active work is the table in `plan.md`. Contributors: read
`AGENTS.md` before claiming a task.

Release binaries are **portable** (ggml runtime CPU dispatch). GitHub Releases
(cargo-dist, tag `vX.Y.Z`) upload macOS arm64, Linux x86_64, and Windows x86_64
CPU archives plus shell / powershell / Homebrew installers. Metal / Vulkan / CUDA
builds are extra artifacts from `.github/workflows/release-variants.yml`.
Install commands are under **Install** above; see `docs/versions.md` for details.

Cutting a release is **Actions → Bump and release** (or the release-pull-request
flow): no version is typed by hand, and every release passes the same gate as a
pull request. See [`docs/release.md`](docs/release.md).

License target: MIT OR Apache-2.0. No telemetry.

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

## Docs

| File | What |
|------|------|
| [`docs/getting-started.md`](docs/getting-started.md) | Install, first commands, where to go next |
| [`docs/guide.md`](docs/guide.md) | User guide: what each capability does, why, and how |
| [`docs/config.md`](docs/config.md) | Every `runa.toml` / `RUNA_*` key |
| [`docs/fit.md`](docs/fit.md) | `runa fit` / `--recommend`, formulas, media context |
| [`docs/thinking.md`](docs/thinking.md) | `ThinkConfig` modes, show/hide, cloud mapping |
| [`docs/media.md`](docs/media.md) | Audio routes, vision `--image`/`--video`, ASR |
| [`docs/structured.md`](docs/structured.md) | JSON Schema / GBNF, serve `response_format`, tools, MCP |
| [`docs/memory.md`](docs/memory.md) | `MemoryManager`, `TaskRegistry` public methods |
| [`docs/runa.1`](docs/runa.1) / [`docs/runa-run.1`](docs/runa-run.1) | man pages |
| [`docs/versions.md`](docs/versions.md) | Release artifacts and feature flags |
| `plan.md` / `AGENTS.md` | Implementation plan and agent coordination (contributors) |

## Status

Earlier milestones are recorded in `done.md`; after-1.0 ideas live in
`roadmap.md`. Active work is the table in `plan.md`. Contributors: read
`AGENTS.md` before claiming a task.

Release binaries are **portable** (ggml runtime CPU dispatch). GitHub Releases
(cargo-dist, tag `vX.Y.Z`) upload macOS arm64, Linux x86_64, and Windows x86_64
CPU archives plus shell / powershell / Homebrew installers. Metal / Vulkan / CUDA
builds are extra artifacts from `.github/workflows/release-variants.yml`.
Install commands are under **Install** above; see `docs/versions.md` for details.

License target: MIT OR Apache-2.0. No telemetry.

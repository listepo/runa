# runa

`runa` is a single command-line binary that runs AI models locally
(GGUF via ggml/llama.cpp) or through the OpenAI and Anthropic APIs.

- Deep-thinking control (`off` / `on` / token budget / effort) that works
  the same way for local and cloud models.
- Audio and video input (native audio/vision models, ASR → text, or cloud).
- Three compute modes — `cpu`, `gpu`, `hybrid` — plus `auto` placement.
- `runa fit`: says *whether* a model runs on this machine and *how fast*,
  before downloading it (memory + speed forecast with confidence interval,
  calibrated by real runs).
- Adaptive memory: shrink toward a floor when idle, bounded pre-grow when
  a task is heavy (plan D17, phase P7).
- Cooperative agents: tasks are claimed `free` → `in progress`
  (+ agent, start time) → `free` on stop/done; `in-progress` tasks are
  taken only after asking (plan D18, `AGENTS.md`).
- Structured output (JSON Schema or GBNF grammar) and tool calling on the
  OpenAI and Anthropic `runa serve` routes.
- Config files, profiles, OpenAI-compatible server (`runa serve`).
- Monorepo tasks via moon (`moon run :test`, `moon run root:lint-tasks`),
  tools via mise (plan D21/D22, phase K).

## Quickstart (target UX)

```sh
runa fit hf:unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --ctx 16384 --kv q8_0
runa run qwen "explain KV-cache quantization in one paragraph"
runa tasks list          # free vs in-progress plan tasks (P7.4)
```

## Docs

| File | What |
|------|------|
| `plan.md` | Active tasks (table) plus Reference: decisions D1–D18, metrics, architecture |
| `AGENTS.md` | Agent coordination protocol (claims, ask-before-steal, Zig vs C kernels) |
| `docs/tasks.md` | Task-claim registry (`free` / `in progress` + agent + start) |
| `.moon/` + `moon.yml` | moon task graph: per-crate build/test/clippy/fmt, root checks (plan K) |
| `docs/memory.md` | Public methods: `MemoryManager`, `TaskRegistry` |
| `docs/config.md` | Every `runa.toml` / `RUNA_*` key |
| `docs/thinking.md` | `ThinkConfig` modes, show/hide, cloud mapping |
| `docs/media.md` | Audio routes, vision `--image`/`--video`, ASR |
| `docs/fit.md` | Fit formulas and media context |
| `docs/structured.md` | `--json-schema` / `--grammar`, serve `response_format`, tool calling |
| `docs/perf-nightly.md` | P5.8 nightly `runa bench` gate (>3 % pp/tg drop) |
| `docs/runa.1` / `docs/runa-run.1` | man pages (`clap_mangen`) |
| `research.md` | Analysis, analogs, formulas, fact-check ledger |
| `report.html` | HTML version of the research report |

## Status

P0–P7 and K are in `done.md`. After-1.0 items are in `roadmap.md`. The
active table in `plan.md` holds phase P8 (structured output, tools, MCP,
fit --recommend, LoRA, TUI, 1.0 gates). Agents: read `AGENTS.md`; claim in
`plan.md` and `docs/tasks.md`.

Release binaries are **portable** (ggml runtime CPU dispatch). GitHub Releases
(cargo-dist, tag `vX.Y.Z`) upload macOS arm64, Linux x86_64, and Windows x86_64
CPU archives plus shell / powershell / Homebrew installers. Metal / Vulkan / CUDA
builds are extra artifacts from `.github/workflows/release-variants.yml`.
Install from a Release: `curl --proto '=https' --tlsv1.2 -LsSf https://github.com/listepo/runa/releases/latest/download/runa-installer.sh | sh`
(or `brew install ./runa.rb` from the formula on that Release). For a host-tuned
local build: `RUSTFLAGS='-C target-cpu=native' cargo build --release --features native`
(see `docs/versions.md`, P5.7).

License target: MIT OR Apache-2.0. No telemetry.

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
- Config files, profiles, OpenAI-compatible server (`runa serve`).

## Quickstart (target UX)

```sh
runa fit hf:unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --ctx 16384 --kv q8_0
runa run qwen "explain KV-cache quantization in one paragraph"
runa tasks list          # free vs in-progress plan tasks (P7.4)
```

## Docs

| File | What |
|------|------|
| `plan.md` | Build plan: decisions D1–D18, metrics M1–M12, phases P0–P8 |
| `AGENTS.md` | Agent coordination protocol (claims, ask-before-steal) |
| `docs/tasks.md` | Task-claim registry (`free` / `in progress` + agent + start) |
| `docs/memory.md` | Public methods: `MemoryManager`, `TaskRegistry` |
| `research.md` / `research.en.md` | Analysis, analogs, formulas, fact-check ledger |
| `report.html` / `report.en.html` | HTML version of the research report |

## Status

Pre-implementation: plan complete through P8. Start at P0 (skeleton, CI,
baselines), then P1 (fit checker before engine). Agents: read
`AGENTS.md`, claim only `free` rows in `docs/tasks.md`.

License target: MIT OR Apache-2.0. No telemetry.

# docs/memory.md — public methods: `MemoryManager` and `TaskRegistry`

Crate: `runa-memory` (plan D17/D18, phase P7, metrics M11/M12).
Targets: M11 (idle RSS ≤ floor + 10 %), M12 (no double-held task).
Config: `[memory] idle_timeout_s, floor_mib, max_growth_mib`;
`[agents] registry = "docs/tasks.md"`. Sketches live in `plan.md` §5.

Conventions: `demand_mib`/`rss_mib` are MiB (`u64`). Timestamps are UTC
RFC 3339 strings. Errors are documented per method; all methods are
synchronous and thread-safe (`Send + Sync`).

---

## `MemoryManager`

Adaptive process memory. Idle (no request/job for `idle_timeout_s`) →
release prompt cache, encoder buffers, draft model, shrink pools toward
`floor_mib`. Heavy task → pre-grow arenas up to fit verdict + margin and
at most `max_growth_mib` beyond current use. Never unloads the active
model. Every transition logs before/after RSS.

### `fn current_usage(&self) -> Usage`

- Returns: `Usage { rss_mib, budget_mib, state }` where `state` is
  `Idle` | `Normal` | `Heavy` and `budget_mib` is the current ceiling
  (fit verdict + margin + granted growth).
- Example: `let u = mm.current_usage(); assert!(u.rss_mib <= u.budget_mib);`
- Notes: pure observation, no side effects; `state` derives from recent
  load vs `idle_timeout_s` and pending demand.

### `fn touch(&self)`

- Effect: record that a request/job is in flight. Resets the idle timer.
- Call when: every generate / server request starts (`grow_for` already
  does this).
- Notes: does not change `LoadState`.

### `fn maybe_idle(&self)`

- Effect: if `last_activity` is older than `idle_timeout_s`, calls
  `on_idle`. No-op when work happened recently (hysteresis).
- Call when: poll after a request finishes, or on a server idle tick.
- Check: `idle_timeout_s = 0` shrinks immediately; a 3600 s timeout does
  not shrink right after `touch`.

### `fn on_idle(&self)`

- Effect: shrink toward `floor_mib` (release caches/buffers/pools as
  above); no-op if already at/below floor. Logs before/after RSS.
- Call when: no request or job for a full `idle_timeout_s` (or from
  `maybe_idle`).
- Check: M11 soak test — idle RSS ≤ floor + 10 %.
- Notes: hysteresis — shrink only after full silence, never mid-burst
  (see Risks in `plan.md` §8).

### `fn on_heavy(&self, demand_mib: u64)`

- Params: `demand_mib` — expected extra memory for the incoming heavy
  task (ctx growth, batch, media).
- Effect: convenience wrapper — grants what fits via `grow_for` and
  otherwise keeps current placement (caller falls back to smaller
  ctx/quant or cloud per the fit suggestion). Never exceeds ceiling.
- Example: `mm.on_heavy(2048); // make room for a 2 GiB ctx bump`

### `fn shrink_to_floor(&self)`

- Effect: unconditional shrink to `floor_mib`, same release set as
  `on_idle` but immediate (used by tests and explicit
  `runa doctor --shrink`-style maintenance paths).
- Notes: still never unloads the active model.

### `fn grow_for(&self, demand_mib: u64) -> Result<(), MemoryError>`

- Params: `demand_mib` — bytes (MiB) the task needs beyond current use.
- Returns: `Ok(())` after pre-growing; `Err(MemoryError::OverCeiling {
  demand_mib, ceiling_mib, suggestion })` when demand exceeds
  fit verdict + margin (`suggestion`: smaller ctx, other quant,
  `--kv q8_0`, or cloud — same vocabulary as `runa fit`).
- Check: heavy-ctx test passes without mid-run OOM; over-ceiling test
  asserts the `Err` carries a suggestion.

---

## `TaskRegistry`

Cooperative claims over `docs/tasks.md` (file-backed). Row states:
`free` | `in progress` (+ `agent`, `started_at`). Completion is tracked
by checking the box in `plan.md`; the registry only tracks live claims.

### `fn list_free(&self) -> Vec<String>`

- Returns: IDs of all `free` tasks (e.g. `["P0.1", "P1.4", …]`), sorted
  by phase order.
- Example: `for id in reg.list_free() { println!("{id}"); }`

### `fn status(&self, task_id: &str) -> Option<TaskStatus>`

- Params: `task_id` — e.g. `"P2.6"`.
- Returns: `None` for unknown IDs; `Some(Free)` or
  `Some(InProgress { agent, started_at })`.
- Notes: read-only; use before asking about an `in-progress` task.

### `fn claim(&self, task_id: &str, agent: &str) -> Result<TaskClaim, ClaimError>`

- Params: `task_id`, `agent` (claiming agent's name; `started_at` is
  stamped by the registry at claim time).
- Returns: `Ok(TaskClaim { task_id, agent, started_at })` and the row
  becomes `in progress`.
- Errors: `NotFound` (unknown ID); `AlreadyClaimed { agent, started_at }`
  when `in progress` — the caller must follow the ask-flow in
  `AGENTS.md` §3–4 and retry only on explicit approval.
- Example: `reg.claim("P1.4", "fable")?; // do work …; reg.release("P1.4", "fable")?;`
- Check: double-claim test — second `claim` fails with the owner's
  name + time, first holder unaffected.

### `fn release(&self, task_id: &str, agent: &str) -> Result<(), ClaimError>`

- Effect: clears the row to `free` (agent/started emptied). Call on
  **every** stop or done, including failures and interrupts.
- Errors: `NotFound` (unknown ID); `NotOwner { agent }` when another
  agent holds it — ask, don't force.
- Check: release test — row returns to `free` and re-claimable.

---

## Registry file format (`docs/tasks.md`)

Markdown table: `| Task | Status | Agent | Started (UTC) |`.
`Status` ∈ `free` | `in progress`. `in progress` rows MUST carry agent +
RFC 3339 `started_at`; `free` rows MUST have both empty. Lint (P7.6)
enforces this plus no double-held task and flags claims older than 7 days.

---

## P9.2

mistral.rs backend for safetensors / omni models ggml cannot run.
`--backend gguf|mistral|auto` on `run` / `chat` / `serve` (default `auto`).
Opt-in cargo feature `mistralrs` (`runa-engine`, forwarded by `runa`);
`runa doctor` reports `mistralrs` when compiled in.

### `runa_core::BackendKind`

- Variants: `Auto` (default) | `Gguf` | `Mistral`.
- `fn parse(s: &str) -> Option<BackendKind>` — `auto` | `gguf` | `mistral`,
  case-insensitive.
- `fn as_str(self) -> &'static str` (+ `Display`).
- `fn detect_backend(path: &Path) -> Result<BackendKind, String>` — `.gguf`
  file → `Gguf`; directory holding `config.json` → `Mistral`; anything else
  is an explicit error (never a silent fallback).
- `fn resolve_backend(requested: BackendKind, path: &Path) -> Result<BackendKind, String>` —
  `Auto` detects; an explicit kind is checked against the path (mismatch
  fails fast, e.g. `--backend mistral` on a `.gguf` file).
- `fn is_gguf_file(path: &Path) -> bool`,
  `fn is_mistral_dir(path: &Path) -> bool` — the two predicates above.

### `runa_engine::MistralModel` (feature `mistralrs`)

- `fn load(path: &Path) -> Result<MistralModel, EngineError>` — validates
  the directory (`config.json`) before any mistral.rs call (fast offline
  failure), then loads via `ModelBuilder` + `blocking::BlockingModel`
  (own tokio runtime; must not run inside an existing runtime — `run` /
  `chat` are sync, `serve` uses plain std engine threads).
- `fn path(&self) -> &Path`.
- `fn generate(&mut self, req: GenerateRequest) -> Result<MistralGeneration, EngineError>` —
  `MistralGeneration: Iterator<Item = Result<GenEvent, EngineError>>` with
  the ggml terminal order (`Text…`, `Usage`, `Done`).
- `fn ensure_backend_available(kind: BackendKind) -> Result<(), EngineError>` —
  always compiled; `Mistral` without the feature errors with the rebuild
  pointer (`--features mistralrs`).
- `EngineError::Mistral(String)` — load/generate failures and GGUF-only
  options used with `--backend mistral`.

### Request mapping (`GenerateRequest` → mistral.rs)

| runa field | mistral.rs |
|---|---|
| messages (`system`/`user`/`assistant`) | `RequestBuilder::add_message` (`tool` roles rejected) |
| `think.mode != Off` | `enable_thinking(bool)`; `think.show` gates `Reasoning` events |
| temperature ≤ 0 | `set_deterministic_sampler()` |
| temperature / top_k (> 0) / top_p / min_p | `set_sampler_temperature/topk/topp/minp` |
| `max_tokens` | `set_sampler_max_len` |
| `stop` | `StopTokens::Seqs` |
| finish `length` / other | `Done(MaxTokens)` / `Done(Eos)` |
| usage | prompt/completion tokens + pp/tg tok/s |

Not mapped, rejected explicitly (never silent): `tools` (`--mcp`),
`json_schema`/`grammar`, `audio_pcm`/`images` (mtmd), `speculative`
(ngram/draft). Not mapped, accepted as ggml-only (see `--backend` help):
`seed` (no mistral.rs equivalent), `--mode`/`--ctx` (mistral auto-maps
devices/context). Requested tools are refused, so surfaced `ToolCalls`
cannot occur; volunteered tool calls (unprompted by any request) surface as
JSON text, mirroring how the ggml backend surfaces unrequested tool markup.

### Pull / fit / serve / doctor

- `runa pull hf:<repo>:safetensors` downloads the snapshot
  (`config.json`, tokenizer `*.json`, `*.index.json`, `*.safetensors`,
  flat layout only) into the model store with per-file size + SHA-256
  verification and `.verified` sidecars, like GGUF pulls.
- `runa_fit::is_safetensors_tag(tag)` (case-insensitive);
  `Fetcher::siblings_all(repo)` (all rfilenames; `siblings` is the
  gguf-filtered view); `RemoteError::Safetensors` refuses `fetch_header`
  for safetensors refs, and `runa fit` refuses mistral directories —
  both with an explicit message.
- `serve` resolves the backend per model (`Auto` detects); the pool thread
  holds `LocalEngine` (`run`/`chat` share the enum in `runa/src/engine.rs`);
  `/v1/embeddings` on a mistral model errors explicitly (gguf only).

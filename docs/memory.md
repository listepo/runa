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

## P9.3 — RPC placement (distributed inference, blocked)

Crate: `runa-engine` (`placement.rs`, `load.rs`, `prompt_cache.rs`).
The pinned `llama-cpp-sys-2 0.1.133` strips the ggml RPC backend, so
full distributed inference is blocked (spike verdict + unblock path in
`docs/versions.md`, "`GGML_RPC` unavailable on this pin"). Until a
sys-crate fork (or a pin bump restoring the sources) lands, the API
carries the intent and fails loudly — never silently runs locally when
distribution was requested.

### `fn parse_rpc_list(s: &str) -> Result<Vec<String>, String>`

- Params: `s` — comma-separated `host:port` endpoints of `rpc-server`
  instances, e.g. `"127.0.0.1:50052,10.0.0.2:50052"`.
- Returns: trimmed non-empty entries; `Err("--rpc: empty list")` when
  nothing remains. Shape-light like `parse_device_list` — hostnames,
  IPv4 and bracketed IPv6 literals all pass through.
- Example: `parse_rpc_list("node1:50052")? // ["node1:50052"]`
- Check: `parse_rpc_list_csv` unit test (`placement.rs`).

### `Placement::rpc_servers: Vec<String>`

- Field on `Placement` (default empty = local inference on every
  constructor: `cpu`, `gpu`, `hybrid_moe`).
- Notes: `prefix_key` hashes the list, so an RPC-enabled build later
  never shares prompt-cache state with local runs.

### `fn with_rpc_servers(self, rpc_servers: Vec<String>) -> Placement`

- Effect: builder recording llama.cpp RPC endpoints on the placement.
- Example: `Placement::gpu().with_rpc_servers(parse_rpc_list(s)?)`
- Notes: `load` prints the endpoints in the verdict line (`rpc=…`
  suffix) and then rejects a non-empty list with
  `EngineError::Unsupported("--rpc …")` before backend init — no
  global state is touched.
- Check: `with_rpc_servers_preserves_mode`,
  `verdict_line_lists_rpc_servers` (`load.rs`), and the integration
  test `rpc_servers_fail_unsupported_before_backend_init`
  (`crates/runa-engine/tests/load.rs`).

## P9.4 — NPU probe and speed stubs (Tier 3)

Crate: `runa-fit` (`npu`, `speed`); CLI surface in the `runa` binary
(`RUNA_NPU`, `runa auto` verdict suffix, `runa doctor` stub strings).
Tier-3 scaffolding only: no ggml NPU backend exists in `llama-cpp-2`
through 0.1.154, so nothing here moves tensors (see `docs/versions.md`
for the pin survey).

### `runa_fit::npu::NpuKind`

- Variants: `Hexagon` (Qualcomm Hexagon DSP/NPU), `OpenVino` (Intel NPU
  via OpenVINO).
- `fn as_str(self) -> &'static str` — canonical name
  (`hexagon` / `openvino`).
- `fn parse(s: &str) -> Option<NpuKind>` — case-insensitive names
  (`1`/`hexagon`/`qcom`/`snapdragon`, `openvino`/`ov`/`intel-npu`);
  `None` for anything else.
- `fn hw_spec(self) -> HwSpec` — the conservative stub for the family
  (`HwSpec::hexagon` / `HwSpec::openvino`).
- `Display` prints `as_str`.

### `fn runa_fit::npu_present() -> Option<NpuKind>`

- Returns: `Some(kind)` when an NPU is present (real or faked), `None`
  otherwise. Pure hardware path is Linux-only; other OSes report absent
  until an on-device owner validates a marker there.
- Test hook `RUNA_FAKE_NPU`: `1`/`hexagon` → `Hexagon`,
  `openvino` → `OpenVino`, `0`/`no`/`off`/`none` → force absent; unset
  (or unrecognized) → hardware heuristic.
- Heuristic: `/proc/device-tree/compatible` contains `qcom` → Hexagon;
  else `/dev/accel` exists → OpenVINO. Conservative and unvalidated
  on-device (Tier 3).

### `fn runa_fit::probe_markers(device_tree_compatible: &Path, accel_dir: &Path) -> Option<NpuKind>`

- Effect: the `npu_present` heuristic with injectable marker paths, so
  unit tests never touch the real filesystem or environment.
- Check: `qcom` in the compat file wins over `/dev/accel`; no markers →
  `None`.

### `HwSpec::hexagon() / HwSpec::openvino()`

- Returns: conservative uncalibrated stubs — Hexagon: 60 GB/s, 45 TOPS,
  efficiency 0.30; OpenVINO: 40 GB/s, 13 TOPS (floor SKU), efficiency
  0.30. Both under-predict CUDA on the same model by construction.
- Limits (see `docs/fit.md`): Q4_0-centric, text-only OpenVINO, Hexagon
  ~3.5 GiB DSP window split. Recalibrate from on-device `runa bench`
  before quoting NPU speeds.

### Binary surface (`runa`)

- `RUNA_NPU=hexagon|openvino` opts the `runa auto` verdict line into an
  NPU suffix: `NPU <kind> present (opt-in stub): ~X tok/s decode
  (uncalibrated …); placement stays CPU` on a match, or an explicit
  `requested but …; staying on CPU (explicit, no silent fallback)` when
  the probe disagrees. Without `RUNA_NPU` the verdict never mentions NPU;
  an unrecognized value is an explicit error.
- `runa doctor` lists `hexagon-stub` / `openvino-stub` only in binaries
  built with `--features hexagon` / `openvino`; default binaries list
  neither. The `runa-engine` build script warns when a stub feature is on
  (always: stub status; plus SDK-missing: `HEXAGON_SDK_ROOT` /
  `INTEL_OPENVINO_DIR`).

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

## P9.1 (`runa daemon` background service)

The daemon keeps models warm between CLI calls, owns one
`MemoryManager` over a real RSS backend, and serves `run` / `chat` over a
Unix socket (`~/.cache/runa/runa.sock`, `RUNA_DAEMON_SOCK` overrides).
Wire: one NDJSON [`DaemonRequest`] line per request, a stream of
[`DaemonEvent`] lines ending in `done` / `error` per reply
(`crates/runa/src/daemon_proto.rs`). `run` / `chat` dial first and fall
back to in-process load on refusal (`--no-daemon` skips the dial);
media, MCP, speculation, and load-shaping flags always stay local.
Per-request preflight is `touch()` + `on_heavy(0)` — admission is
pool/LRU bound. The idle tick calls `maybe_idle()` at least every
`MAX_IDLE_TICK_SECS`.

### `fn SysinfoBackend::new() -> SysinfoBackend`

- Returns: a real RSS backend (P9.1). `rss_mib()` reads this process's
  resident set via `sysinfo`; `shrink_to` / `grow` are advisory no-ops
  returning current RSS (the OS owns the pages — release happens through
  `LoadedModel::on_idle` and pool LRU eviction).
- Example: `MemoryManager::new(policy, ceiling, Box::new(SysinfoBackend::new()))`
- Notes: replaces `FakeBackend` at the `run` / `serve` / daemon call
  sites; unit tests keep using `FakeBackend`.

### `fn SysinfoBackend::process_rss_mib() -> u64`

- Returns: current process RSS in MiB, `0` when the process table is
  unreadable. Pure observation, no side effects.

### `fn default_socket_path() -> PathBuf`

- Returns: the daemon socket path — `RUNA_DAEMON_SOCK` when set, else
  `$XDG_CACHE_HOME/runa/runa.sock` or `~/.cache/runa/runa.sock`.

### `fn ModelPool::insert_spec(&mut self, path: &Path) -> Result<String, String>`

- Params: `path` — model file to serve on demand.
- Returns: the pool id (`Ok`): the existing id when the path is known,
  else the file stem (`stem-2`, … on collision). `Err` when the file is
  missing. Never unloads models.

### `fn resolve_or_insert(pool: &Mutex<ModelPool>, model: Option<&str>) -> Result<String, String>`

- Effect: `resolve_id` first; when the model is an on-disk path the pool
  does not know yet, `insert_spec` it and return the new id.
- Errors: `model <id> not found` when the model is neither a known id
  nor an existing file.

### `fn generate(pool: &Arc<Mutex<ModelPool>>, model_id: &str, req: GenerateRequest) -> Result<Vec<GenEvent>, String>`

- Effect: resolve + (blocking) load the engine, run one generation on
  its thread, collect the events. Async wrapper — never blocks the
  executor. `serve` keeps its own status-mapped variant.

### `fn request_sync(socket: &Path, req: &DaemonRequest, timeout: Duration) -> Result<Vec<DaemonEvent>, String>`

- Returns: daemon events up to and including `done` / `error`.
- Errors: transport failures mean "no daemon" (the caller falls back);
  a daemon-side failure arrives as `DaemonEvent::Error` inside `Ok`.
  Non-unix stub always errs (unix sockets only).

### `fn install_daemon(home: &Path, exe: &Path, argv: &[String]) -> Result<Vec<PathBuf>, String>`

- Effect: write the launchd plist
  (`~/Library/LaunchAgents/ai.runa.daemon.plist`) and the systemd user
  unit (`~/.config/systemd/user/runa-daemon.service`) for `exe argv…`;
  returns both paths. Overwrite is idempotent.

### `fn uninstall_daemon(home: &Path) -> Result<Vec<PathBuf>, String>`

- Effect: remove both units; returns the paths removed (empty when
  nothing was installed). Missing files are not errors.

### `fn launchd_plist(exe: &Path, argv: &[String]) -> String`

- Returns: the launchd plist text (`ai.runa.daemon`, `RunAtLoad` +
  `KeepAlive`, logs to `~/.cache/runa/daemon.{out,err}.log`).

### `fn systemd_unit(exe: &Path, argv: &[String]) -> String`

- Returns: the systemd user-unit text (`Restart=on-failure`,
  `WantedBy=default.target`).

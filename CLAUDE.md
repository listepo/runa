# AGENTS.md — agent coordination protocol (symlinked as AGENTS.md → CLAUDE.md, single source of truth)

This file is normative for every coding agent working in this repo (human or AI).
Source of truth for the plan: `plan.md` (decisions D17/D18/D23, phase P7).
Claim registry: `docs/tasks.md`. Public-method docs: `docs/memory.md`.

## 1. Read before touching anything

1. Read `plan.md` (decisions, phases, core types) and `readme.md` (overview).
2. Read `docs/tasks.md` — the claim registry. It lists every plan task with
   `Status` = `free` or `in progress`.

## 2. Claim protocol (mandatory)

- Agents take **only `free` tasks**.
- To take a task, atomically mark its row `in progress` **plus your agent
  name and `started_at` (UTC, RFC 3339, e.g. `2026-09-08T12:00:00Z`)**.
  A claim without agent + start time is invalid; CI lint fails it.
- When the work is **stopped or done**, clear your claim: row back to
  `free`, agent/started emptied. Completion itself is recorded by checking
  the task box in `plan.md`, not by the registry.
- Never hold two tasks at once unless the human explicitly allows it.

## 3. Ask-before-steal

- An `in progress` task belongs to its owner. To work on it, **ask the
  owner first** (or the human if the owner is unreachable) and proceed
  **only on explicit approval**. No approval = pick a `free` task.
- Stale claims (owner gone, `started_at` older than 7 days): still ask the
  human first — never silently steal. CI surfaces old claims.

## 4. Ask flow (copy/paste)

> Task `P2.6` is `in progress` (owner: `<agent>`, since `<started_at>`).
> I want to take it over because `<reason>`. Approve? If not, I will take
> `<free-task-id>` instead.

## 5. Memory discipline for agents

- Respect the adaptive-memory design (D17): idle → shrink to floor, heavy →
  bounded grow. Do not add caches, pools, or background jobs without a
  release path wired to `MemoryManager::on_idle`.
- Every new public method gets docs in `docs/memory.md` (P7.5, P6.4 rule:
  docs lint fails otherwise).

## 6. Machine checks

- Registry lint: every `in progress` row has agent + RFC 3339 `started_at`;
  no task held twice. Runs in CI (P7.6).
- `cargo test -p runa-memory` covers claim/release/double-claim (P7.4).

## 7. Kernel language (D23)

- Do **not** rewrite ggml, whisper.cpp, or llama.cpp in Zig or Rust.
- New code in `runa-kernels`: prefer **Zig** (C ABI `export fn`, `@Vector` SIMD,
  slices instead of `malloc`/`qsort`) over new C for softmax, sampling
  (top-k/top-p/min-p), and image/audio front-ends.
- Keep **C or `.S`** only where Zig is worse or missing: wrapping existing C
  headers, SME2/AMX assembly, or a measured C path that already wins the D1 gate.
- Same merge gate as D1. Pin Zig with mise (`mise.toml`, D21); do not install
  a floating Zig outside mise.

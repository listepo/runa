# AGENTS.md — agent coordination protocol (symlinked as AGENTS.md → CLAUDE.md, single source of truth)

This file is normative for every coding agent working in this repo (human or AI).
Source of truth for active work: the table at the top of `plan.md`.
Claim registry (code + CI): `docs/tasks.md`. Public-method docs: `docs/memory.md`.
Finished tasks live in `done.md`. Approved-but-not-started work lives in `roadmap.md`.

## 1. Read before touching anything

1. Read `plan.md` (header table, then Reference) and `README.md`.
2. Read `docs/tasks.md` — it lists only **active** tasks (`free` or `in progress`).
   Finished work is not listed there.

## 2. Claim protocol (mandatory)

- Take a task only if the `plan.md` table status is `todo` (and `docs/tasks.md` is `free` or has no row).
- To take it: set `plan.md` to `in progress` and write your **provider and model** in Agent; add or update the `docs/tasks.md` row to `in progress` plus agent name and `started_at` (UTC, RFC 3339) so the registry lint stays valid.
- An `in progress` row with another agent — do not take it.
- When **stopped**: `plan.md` back to `todo` with empty Agent; `docs/tasks.md` back to `free`.
- When **done**: move the whole task (id, title, description) to `done.md`; remove it from the `plan.md` table and cards, from `todo.md`, and from `docs/tasks.md`.
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

## Host agents

Save tokens. If anything is unclear, ask the creator first. Write a short execution plan into that task's card in `plan.md`, then claim and work. Default cap: **5** parallel agents per project unless the creator says otherwise. Never use max effort or fast mode without permission. Cheapest model for scripts, commands, repo scans, web, file moves, tests. On Cursor: **grok 4.6** (no fast) for planning, refactoring, bug hunts; **composer 2.5** (no fast) for file moves, tests, commands, scans, web. Before writing code, decide whether a ready library or framework should be used. A new dependency is allowed only if it is current (not abandoned) and the creator approved it. Packages already in `toolchain.md` may be reused without asking again. Prefer the latest versions of tools and packages, but bump already-installed ones only with the creator’s permission. Rust: reuse crates already used by sibling projects in this workspace (workspace-root `rust.md`). If this repo lacks one it should use, add a `plan.md` task — do not add the dependency silently. Extract duplicated helpers into `packages/` and depend via local `{ path = "..." }`. No version bumps without permission.

If a directory above this repository contains an `AGENTS.md` or `CLAUDE.md`, follow it too. If it conflicts with this file, ask the creator.
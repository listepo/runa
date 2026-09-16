# Bad AGENTS.md excerpt for registry lint (P11.3) — §2 without the
# claim-first wording. `lint-tasks.py --agents-path <this-file>` must fail
# with a claim-first error even when the task table itself is valid.
# Real protocol lives in AGENTS.md (symlink to CLAUDE.md).

## 2. Claim protocol (mandatory)

- Take a task only if the `plan.md` table status is `todo` (and `docs/tasks.md` is `free` or has no row).
- An `in progress` row with another agent — do not take it.

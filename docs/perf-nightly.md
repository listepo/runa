# docs/perf-nightly.md — P5.8 first report

Nightly workflow: `.github/workflows/perf.yml` (`cron: 0 4 * * *` UTC + `workflow_dispatch`).
Gate: `scripts/perf-regress.py` fails if `pp_tok_s` or `tg_tok_s` drop **> 3 %** vs
`docs/perf-baseline.json` for that runner key.

`--ctx` must be `>= pp + tg + 1` (`512 + 128 + 1 = 641`). The workflow uses `--ctx 1024`.

## Machines

| Key | Runner | Status |
|-----|--------|--------|
| `macos-14-cpu` | GitHub-hosted `macos-14` (arm64) | empty baseline until first nightly artifact is pasted |
| `ubuntu-22.04-cpu` | GitHub-hosted `ubuntu-22.04` (x86_64) | same |
| `local-m3-max-cpu-debug` | this machine (Apple M3 Max) | recorded 2026-09-08 (debug binary) |

Self-hosted mac + Linux labels (plan wording) are not wired yet. Swap `runs-on` in
`perf.yml` when those machines exist. Do **not** copy local M3 Max numbers into the
GitHub-hosted keys — hosted runners are slower and would trip the 3 % gate on the
first real nightly.

CI does **not** pass `--write-missing` (that would need a commit from GHA). After
the first successful nightly, paste `pp_tok_s` / `tg_tok_s` from the
`bench-<key>` artifact into `docs/perf-baseline.json`.

## First local numbers (2026-09-08)

Command (debug `runa`, CPU, Qwen2-0.5B Q4_0 fixture):

```text
runa bench --mode cpu --ctx 1024 --pp 512 --tg 128 --json \
  tests/fixtures/qwen2-0_5b-instruct-q4_0.gguf
```

| metric | tok/s |
|--------|-------|
| `pp_tok_s` (pp512) | 588.0 |
| `tg_tok_s` (tg128) | 44.6 |

JSON (abridged): `placement=cpu`, `ctx=1024`, `n_prompt=512`, `n_gen=128`,
`model_hash=sha256:e618e01736d18386c574b345e5dc763371be9ffd324588dfd0f2f8cefab9045f`.

This is a **debug** binary. Nightly CI builds `--release`; expect higher tok/s on
the same Mac, and lower tok/s on GitHub-hosted CPUs. The 3 % gate compares each
key only to itself.

## Reproduce the gate locally

```text
python3 scripts/perf-regress.py --self-test
python3 scripts/perf-regress.py \
  --current bench.json \
  --baseline docs/perf-baseline.json \
  --key local-m3-max-cpu-debug \
  --max-drop 0.03
```

## P11.6 curl-52 watch (2026-09-16)

No curl exit 52 / empty-reply serve failure in post-P8.8 Linux CI.
Checked: `ci` runs 35055665463, 35012959563, 34993041805 (all fail earlier at
`cargo test --workspace`, serve step skipped), 34980160972 and 34976968709
(all jobs green; ubuntu `runa serve e2e + OpenAI Python SDK smoke (P3.9)` step
passed, 0 curl-52/empty-reply lines in the ubuntu log of 34976968709);
p8-features runs 34972604943/34970282240 fail at clippy, before serve.
P8.8 (commit `6128d28`, 2026-09-15) stderr echo had nothing to catch — no
recurrence. Watch target stays: workflow `ci`, job `ubuntu-22.04
(x86_64-unknown-linux-gnu)`, step `runa serve e2e + OpenAI Python SDK smoke
(P3.9)` (`.github/workflows/ci.yml:95`).

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

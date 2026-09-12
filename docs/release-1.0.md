# docs/release-1.0.md — v1.0 metric checklist (P6.6)

Status as of 2026-09-08. **Open** items block calling the binary 1.0, not merging the plan tasks that already landed.

| ID | Gate | Status | Evidence |
|----|------|--------|----------|
| M1 | `runa fit` local GGUF < 300 ms; estimate ±5 % (exact ±1 %) | **partial** | Fit CLI + estimators done (P1). Timing/±% vs llama.cpp alloc not recorded in CI. |
| M2 | `runa fit` remote HF < 3 s, no download | **pass (task)** | P1.2: header range fetch. Live `hf:unsloth/Qwen3-8B-GGUF:Q4_K_M` claimed < 3 s; re-time before 1.0. |
| M3 | Speed prediction ±30 % cold / ±15 % after 3 cals | **partial** | P1.9/P1.10 calibration DB exists. Device-level error table not filled. |
| M4 | Decode ≥ 95 % of `llama-cli` same flags | **open** | P2.2 claims the gate; no side-by-side number in `docs/baselines.md`. |
| M5 | Cold start 8B Q4_K_M Apple M, mmap, first token < 2 s | **open** | No 8B timing in baselines (0.5B Metal spike only). |
| M6 | Thinking budget: reasoning ≤ budget+grace 100 % of runs | **partial** | `BudgetClock` unit tests (P3.3). 100-run Qwen3-4B/GSM8K skipped (no 4B fixture). |
| M7 | 1 min speech CPU ASR < 5 s | **open** | P4.2: fixture clips are sine, not speech; ggml not in CI. Manual note in `docs/baselines.md`. |
| M8 | 30 s clip → ≤ 32 frames + transcript < 3 s | **open** | Sampler exists (P4.5); placeholder MP4s are not real media. |
| M9 | OpenAI + Anthropic Python SDKs smoke unmodified | **partial** | `scripts/serve-openai-smoke.py` (P3.9). Anthropic `/v1/messages` is a separate server task (P6.2). |
| M10 | One binary/platform, CPU ≤ 40 MB, no telemetry | **partial** | P6.3: `.github/workflows/release.yml` (cargo-dist 0.28). Size not measured yet. No telemetry by policy. |
| M11 | Idle RSS ≤ floor + 10 % after `idle_timeout_s` | **partial** | `on_idle` / prompt-cache unmap (P7.2). RSS not measured on a loaded 8B. |
| M12 | Task claims: no double-hold; agent + `started_at`; release on stop | **pass** | `docs/tasks.md` lint (P7.6) + `cargo test -p runa-memory` claim/release. |
| M13 | moon/mise parity with cargo/CI | **pass (task)** | P5.7 / `docs/versions.md`; re-run `moon run :test` vs `cargo test --workspace` before tag. |

## Still open plan tasks (not 1.0)

P5.3–P5.6 kernels (5 % tg gate), P6.1 multi-model serve.

## Tag when

All **open** rows above have a number in `docs/baselines.md` or this file, a `v*` tag uploads cargo-dist artifacts, and `moon run root:lint-tasks` is green on a tree with every claim `free`.

# docs/release-1.0.md — v1.0 metric checklist (P6.6)

Status as of 2026-09-15 (P8.7 measured; timings contended, load avg 127–224 — re-run starred rows quiet). **Open** items block calling the binary 1.0, not merging the plan tasks that already landed.

| ID | Gate | Status | Evidence |
|----|------|--------|----------|
| M1 | `runa fit` local GGUF < 300 ms; estimate ±5 % (exact ±1 %) | **partial** | P8.7: fit path 148 ms (0.5B) / 147 ms (8B) best-of-5 ✓; weight +0.57 % vs engine mmap, KV exact, compute 3.7× conservative. No `runa fit` CLI (P8.4); P1.12 exact-mode is a stub. See `docs/baselines.md` §P8.7. |
| M2 | `runa fit` remote HF < 3 s, no download | **partial** | P8.7: `hf:unsloth/Qwen3-8B-GGUF:Q4_K_M` 8 MiB range (206) 0.57 s warm ✓, no download ✓; cold first-fetch 26 s ✗. |
| M3 | Speed prediction ±30 % cold / ±15 % after 3 cals | **partial** | P8.7: CPU 0.5B uncalibrated err 79–87 % pp / 172–265 % tg (FAIL ±30 %); predictions unchanged after 2 cals — `predicted_speeds` never reads `CalibrationDb`. Needs wiring task. |
| M4 | Decode ≥ 95 % of `llama-cli` same flags | **partial** | P8.7: runa tg 23.1 vs llama-bench b7709 tg 85.1 (27 %) — cause: runa threads=16 (P+E) vs bench auto 12; at matched `-t16` runa wins 2.3×. Needs P-core default or `--threads`. |
| M5 | Cold start 8B Q4_K_M Apple M, mmap, first token < 2 s | **open** | P8.7: 9.02 s TTFT cold on CPU-only build (gate is Metal). Metal `--features metal` re-run pending. |
| M6 | Thinking budget: reasoning ≤ budget+grace 100 % of runs | **partial** | `BudgetClock` unit tests (P3.3). P8.7: 10/10 Qwen3-8B seeds complete (temp-0, identical 417 B outputs); reasoning counts not exposed by CLI; 100-run/GSM8K still open. |
| M7 | 1 min speech CPU ASR < 5 s | **open** | P8.7 BLOCKED: default `--lang auto` returns empty transcript (detect suppresses decode — bug). `--lang en` on 60 s real speech: correct, 11.5 s* contended. |
| M8 | 30 s clip → ≤ 32 frames + transcript < 3 s | **open** | P8.7: 30 s clip → 30 frames ✓ + audio ✓ (2.6 s*); transcript ~6 s* implied — <3 s at risk on CPU base. Synthetic clip, not camera footage. |
| M9 | OpenAI + Anthropic Python SDKs smoke unmodified | **pass** | P8.7: both pass unmodified (openai 2.54.0, anthropic 0.125.0) incl. streaming + tool round-trips. P6.2 done. |
| M10 | One binary/platform, CPU ≤ 40 MB, no telemetry | **partial** | P8.7: debug 99.8 MiB; release size unmeasured (build deferred, loaded machine). No telemetry verified by source grep (0 hits). |
| M11 | Idle RSS ≤ floor + 10 % after `idle_timeout_s` | **partial** | P8.7: serve 8B RSS 8364 MB vs floor+10 % = 563 MB (14.9×); `Loaded::on_idle` unwired in serve (no idle tick) — RSS flat. Gate infeasible with resident model. |
| M12 | Task claims: no double-hold; agent + `started_at`; release on stop | **pass** | `docs/tasks.md` lint (P7.6) + `cargo test -p runa-memory` claim/release. |
| M13 | moon/mise parity with cargo/CI | **pass (task)** | P5.7 / `docs/versions.md`; re-run `moon run :test` vs `cargo test --workspace` before tag. |

## Still open plan tasks (not 1.0)

P5.3–P5.6 kernels (5 % tg gate), P6.1 multi-model serve.

## Tag when

All **open** rows above have a number in `docs/baselines.md` or this file, a `v*` tag uploads cargo-dist artifacts, and `moon run root:lint-tasks` is green on a tree with every claim `free`.

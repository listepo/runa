# docs/baselines.md — llama-bench reference (P0.6)

Measured 2026-09-08 on Apple M3 Max (Metal, 48GB unified, macOS 15) via
`crates/runa-engine/examples/gen.rs` (`llama-cpp-2 =0.1.133 → llama.cpp b7709`).
`pp` = prompt processing, `tg` = decode. All runs `--n-gpu-layers 999` (GPU) or `0` (CPU).

> **Note:** P0.6 wants `pp512`/`tg128` on three reference models per mode per CI
> machine. This is the initial Metal slice; CPU/hybrid and Linux/Windows
> remain as follow-ups (self-hosted runners per P5.8). `gpt-oss-20b-MXFP4.gguf`
> on disk is truncated (11G, loads `blk.22` past EOF) — re-pull needed.

## Machine
- **M3 Max** — Apple M3 Max, 14-core CPU, 30-core GPU, 48GB unified, Metal, `recommendedMaxWorkingSetSize 55662 MB`, `hasUnifiedMemory true`.

## Models (local fixtures)
- `qwen2-0_5b-instruct-q4_0.gguf` — 337M, Qwen2-0.5B-Instruct Q4_0
- `Qwen3-8B-Q4_K_M.gguf` — 4.7G, Qwen3-8B Q4_K_M (unsloth)
- `Qwen3-30B-A3B-Q4_K_M.gguf` — 11G, Qwen3-30B-A3B Q4_K_M (MoE, 3.3B active)
- `gpt-oss-20b-MXFP4.gguf` — 11G, gpt-oss-20B MXFP4 (truncated on disk, not measured)
- `SmolVLM-500M-Instruct-Q8_0.gguf` — 417M (vision, not part of P0.6 but present)

## Results (pp/tg)

| Model | Mode | Machine | pp (tok/s) | tg (tok/s) | Notes |
|-------|------|---------|------------|------------|-------|
| Qwen2-0.5B Q4_0 | gpu (Metal, all 24 layers) | M3 Max | 4.4 (pp1) | 344.4 (tg32) | `gen` 1 tok prompt, 32 tok gen; ~0.23s pp, 0.09s tg |
| Qwen3-8B Q4_K_M | gpu (Metal, all layers) | M3 Max | 5.8 (pp1) | 50.4 (tg16) | 1 tok prompt, 16 tok gen; 0.17s pp, 0.32s tg |
| Qwen3-8B Q4_K_M | cpu (n_gpu_layers 0) | M3 Max | — | — | pending: `cargo run --no-default-features` CPU-only |
| Qwen3-8B Q4_K_M | hybrid (experts on CPU) | M3 Max | — | — | N/A (dense, not MoE) |
| Qwen3-30B-A3B Q4_K_M | gpu (Metal, all 48 layers, 3.3B active) | M3 Max | 4.2 (pp1) | 54.7 (tg16) | 1 tok prompt, 16 tok gen; 0.24s pp, 0.29s tg |
| Qwen3-30B-A3B Q4_K_M | hybrid (experts on CPU) | M3 Max | 3.1 (pp9) | 1.6 (tg16) | P2.6: `runa run --mode hybrid`; `ffn_*_exps` → CPU_REPACK (12.96 GiB); attention stays Metal |
| gpt-oss-20b MXFP4 | gpu | M3 Max | — | — | file truncated, reload needed |
| gpt-oss-20b MXFP4 | cpu/hybrid | M3 Max | — | — | pending |

## Experts on GPU vs CPU (P2.6)

Same model, same machine, debug `runa`. GPU numbers from the P0.6 `gen` spike
(`-ngl 999`); hybrid is `--mode hybrid` (all routed experts on CPU).

| Model | Experts | pp (tok/s) | tg (tok/s) | Notes |
|-------|---------|------------|------------|-------|
| Qwen3-30B-A3B Q4_K_M | GPU (Metal, all 48 layers) | 4.2 (pp1) | 54.7 (tg16) | 17 GiB mapped on Metal |
| Qwen3-30B-A3B Q4_K_M | CPU (`--mode hybrid` / `--n-cpu-moe ≥48`) | 3.1 (pp9) | 1.6 (tg16) | `ffn_*_exps` → CPU_REPACK 12.96 GiB; 8 GB-class VRAM recipe |

`--n-cpu-moe N` keeps experts of layers `0..N-1` on CPU (one combined regex;
llama-cpp-2 0.1.133 can only apply a single buffer override). `--mode hybrid`
pins every expert tensor. Dense models ignore the patterns.

```sh
cargo run -p runa -- run --mode hybrid --max-tokens 16 --ctx 512 --temperature 0 --seed 42 \
  tests/fixtures/Qwen3-30B-A3B-Q4_K_M.gguf hello
cargo run -p runa -- run --mode gpu --n-cpu-moe 8 --max-tokens 16 --ctx 512 \
  tests/fixtures/Qwen3-30B-A3B-Q4_K_M.gguf hello
```

## How to reproduce (P0.4 spike)

```sh
cargo run -p runa-engine --example gen -- tests/fixtures/qwen2-0_5b-instruct-q4_0.gguf "hi" 32
cargo run -p runa-engine --example gen -- tests/fixtures/Qwen3-8B-Q4_K_M.gguf "hello" 16
cargo run -p runa-engine --example gen -- tests/fixtures/Qwen3-30B-A3B-Q4_K_M.gguf "hello" 16
# pp512/tg128 (full P0.6): use a 512-token prompt (e.g. `python3 -c "print('hello world '*200)"`) and `... 128`
```

## P2.9 multi-GPU (manual)

CI runners here have one GPU (Apple M3 Max Metal). The two-GPU load test is
`#[ignore]` (`two_gpu_tensor_split_loads` in `runa-engine` tests).

On a machine with two ggml GPU backends:

```
runa run --mode gpu --device 0,1 --tensor-split 3,1 model.gguf "hi"
```

Expect the verdict line to contain `devices=0,1 tensor-split=3,1` and a successful
stream. `runa run --device 999 …` must error (no silent fallback).

## P8.7 gate measurements (2026-09-15, M3 Max 64GB, macOS 26.6.2, runa 0.1.0 debug CPU-only)

Contended machine (sibling agents building; load avg 127–224). Starred (*) rows need a quiet re-run before 1.0.
llama-bench built from the same b7709 sources (`--branch b7709`, CPU-only Release).

| Gate | Result |
|------|--------|
| M1 fit wall (header+parse+`check_fit`, best of 5) | 0.5B **148.2 ms**, 8B **146.6 ms** (< 300 ms ✓) |
| M1 weight est vs engine mmap (8B) | est 5,021,827,072 B vs mapped 4762.19 MiB (**+0.57 %** ✓); KV exact (1152 MiB ≡ 288 MiB@2k×4); compute est 3.74× actual (conservative) |
| M2 remote `hf:unsloth/Qwen3-8B-GGUF:Q4_K_M` 8 MiB range | **0.57 s** warm ✓, 206 partial (no download) ✓; cold 26.1 s ✗ |
| M3 CPU 0.5B pred vs meas | pred 62.9/62.9 vs meas pp 297–478 / tg 17–23 (FAIL ±30 %); cals recorded, predictor ignores DB |
| M4 0.5B CPU pp512/tg128 vs llama-bench | runa **297.8/23.1** vs bench **1617.6/85.1** @t12 (18 %/27 %); bench `-t16`: 175.5/10.1 (runa wins matched) — threads=16 default is the cause |
| M5 cold 8B CPU TTFT* | **9.02 s** (total 9.17 s / 8 tok; pp 12.2, tg 2.0; peak RSS 8.44 GB) |
| M6 10× `--think-budget 64` Qwen3-8B | 10/10 complete, identical 417 B greedy outputs; reasoning counts unobservable via CLI |
| M7 60 s real speech, `--lang en`* | correct (14 segs), **11.5 s** wall; `--lang auto` (default) returns empty — bug |
| M8 30 s clip, `media video --fps 1` | **30 frames** ✓, audio ✓, 2.6 s*; transcript ~6 s* implied vs <3 s gate |
| M9 SDK smokes | openai 2.54.0 + anthropic 0.125.0 pass unmodified (stream + tools) |
| M10 size | debug 104,649,208 B; release pending; no-telemetry grep clean |
| M11 serve RSS (8B ctx2048 / 0.5B) | **8364 MB** / 846 MB vs floor+10 % = 563 MB; flat across idle (unwired) |

## P4.2 ASR (manual)

CI fixtures are 1 s sine tones, not speech, so WER vs a transcript is
undefined. On a machine with `ggml-base.bin` (auto-pulled by
`runa media transcribe`):

```
runa media transcribe --model base speech.wav
# 1 min of speech on CPU should finish in < 5 s (M7).
```

WER helper: `runa_media::word_error_rate`. Live check: `RUNA_WHISPER=1`.

## P3.3 budget forcing (manual)

CI has no Qwen3-4B fixture. The 100-run / GSM8K accuracy check is skipped.
On a machine with Qwen3-4B:

```
runa run --think-budget 256 --grace 64 Qwen3-4B.gguf "Solve 12+7"
```

Reasoning tokens must stay ≤ 256+grace. Repeat ~100 times; GSM8K-50 within 10
points of unlimited thinking is the quality bar.

## Next
- Re-pull `gpt-oss-20b-MXFP4.gguf` (11G truncated) and re-run.
- Add `pp512`/`tg128` with 512-token prompt on each model/mode.
- Add CPU (`-ngl 0`) variants. Hybrid `--n-cpu-moe` / `--mode hybrid` is in (P2.6).
- Add Linux x86_64 (CUDA/Vulkan) and Windows from CI. Nightly CPU `runa bench` is P5.8 (`docs/perf-nightly.md`); self-hosted GPU runners still pending.

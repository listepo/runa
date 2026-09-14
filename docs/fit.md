# docs/fit.md — fit-checker internals

Formulas, estimators and calibration for `runa fit` (plan P1, decisions
D5/D6). Estimator sections land with their tasks (P1.5 compute buffer,
P1.9 speed model); this file starts with the remote header path (P1.2).

## Remote header fetch (P1.2)

`runa fit` answers *before* downloading, so the GGUF header (magic +
metadata + tensor-info table) is fetched with HTTP `Range` requests that
grow until [`Reader::parse`](../crates/runa-fit/src/gguf.rs) succeeds.
Implementation: `crates/runa-fit/src/remote.rs` (`Fetcher`).

Model references (D3):

| Ref | Resolution |
|-----|------------|
| `hf:<repo>:<file.gguf>` | `https://huggingface.co/<repo>/resolve/main/<file.gguf>` |
| `hf:<repo>:<quant>` | Hub API sibling listing → best `<quant>` match → as above |
| `https://…` | direct URL (must serve byte ranges) |
| `<local path>` | same grow-until-parse loop over the file (seek, no network) |

Quant matching (`pick_quant`): exact filename → `-`/`_`-suffixed
(`…-Q4_K_M.gguf`) → case-insensitive substring; ties break by shortest
name, then lexicographic. Deterministic; unit-tested.

Fetch loop: ask `bytes=have..want-1` starting at 8 MiB, doubling to a
256 MiB cap. Truncation-shaped parse errors (`Truncated`,
`UnexpectedEof`, `TensorTableTruncated`, `StringTooLong`) mean "fetch
more"; any other parse error means the remote file is corrupt. Servers
that ignore ranges (200 instead of 206) are accepted only when the whole
body fits the header budget. Auth: `HF_TOKEN` bearer (redirects to
signed S3 URLs need none — reqwest strips auth cross-domain).

Cache: `<cache>/runa/headers/` (`XDG_CACHE_HOME`, `%LOCALAPPDATA%`,
`~/.cache`, else temp dir), one file per URL plus a JSON sidecar
(`total_len`, `etag`). Reuse requires the server to report the same
total length and ETag on a 1-byte probe.

Measured 2026-09-08 (live, cold):

| Model | Header bytes | Tensors | File size | Elapsed |
|-------|--------------|---------|-----------|---------|
| `hf:unsloth/Qwen3-8B-GGUF:Q4_K_M` | 8 MiB (1 round trip) | 399 | 5.0 GiB | 4.3 s |
| local `qwen2-0_5b-instruct-q4_0.gguf` (test server) | 8 MiB (1 round trip) | 290 | 353 MiB | 0.2 s |

The 4.3 s cold fetch is TLS + redirect chain + 8 MiB transfer; warm
(cached) fetches do a single 1-byte probe. The `< 3 s` plan target is
for the `runa fit` CLI path (P1.11) with a warm connection.

## Placement (P1.8)

`plan_placement` (`crates/runa-fit/src/planner.rs`) maximizes GPU-resident
weights under `VRAM − margin` (default margin 1 GiB, `--fit-margin` later).
Compute buffer + KV + mmproj + LoRA adapters are reserved on GPU first;
the remaining weight budget is filled back-to-front by eviction priority:
MoE experts (`ffn_*_exps`) spill first, then embedding/output tables, dense
attention/FFN last. `mmproj_bytes` (default 0, set for VL/audio models —
see P4.8) and `lora_bytes` (default 0, sum of the `--lora` adapter file
sizes — see P8.5) participate in the reservation and the per-device byte
table (`gpu_weight_bytes`, `cpu_weight_bytes`, `compute_buffer_bytes`,
`kv_bytes`, `mmproj_bytes`, `lora_bytes`, `gpu_total_bytes`).

Verified on the real Qwen3-30B-A3B header: 8 GiB VRAM parks experts on
CPU while dense stays on GPU; ample VRAM puts all layers on GPU.
(`real_moe_fixture_small_vram_parks_experts_on_cpu`).

## Media / VL context (P4.8)

`FitConfig.media` (`MediaFit`) adds image/video/audio tokens to the context
check:

```text
media_tokens = frames × tokens_per_frame
             + ceil(audio_seconds) × tokens_per_audio_second
```

Defaults: 256 tokens/frame, 25 tokens/s of audio. If `media_tokens > n_ctx`
the verdict is `NO FIT` (exit 2) with `Warning::MediaExceedsContext`.

VRAM: `PlannerConfig.mmproj_bytes` is reserved on GPU (same as compute/KV).
`--draft PATH` (P5.6) adds the draft GGUF file size to `mmproj_bytes` so
fit accounts for a second model even though decode still uses n-gram.
`has_mmproj && mmproj_bytes == 0` still warns that the projector was not
sized. Encoder scratch is `frames × n_embd × 4 × 8` via
`estimate_encoder_compute` and is added to the GPU reservation.

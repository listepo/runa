# Getting started with runa

`runa` is one CLI binary that runs AI models locally (GGUF via ggml / llama.cpp)
or through the OpenAI and Anthropic APIs. Thinking controls, tools, and
`runa serve` work the same way on both sides.

## Install

From a GitHub Release (portable CPU archives; Metal / Vulkan / CUDA are extra
artifacts):

```sh
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/listepo/runa/releases/latest/download/runa-installer.sh | sh
```

Or Homebrew from the formula on that Release: `brew install ./runa.rb`.

Host-tuned local build:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --features native
```

See [`versions.md`](versions.md) for feature and artifact details.

## First commands

```sh
runa fit hf:unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M --ctx 16384 --kv q8_0
runa fit --recommend --use code   # best catalog models for this machine
runa run qwen "explain KV-cache quantization in one paragraph"
runa serve --port 8080            # OpenAI-compatible HTTP server
```

`runa fit` says whether a model runs on this machine and how fast, before you
download it. After a few `runa bench` runs, predictions calibrate to your
device (see [`guide.md`](guide.md)).

## Configuration

Search order (later wins): `~/.config/runa/config.toml`, then `./runa.toml`.
CLI flags beat `RUNA_*` env vars, which beat the files. Full key list:
[`config.md`](config.md).

## What to read next

| Doc | Contents |
| --- | --- |
| [guide.md](guide.md) | Capability-by-capability how-to with examples |
| [fit.md](fit.md) | Fit formulas, `--recommend`, media context |
| [thinking.md](thinking.md) | Deep-thinking modes and cloud mapping |
| [structured.md](structured.md) | JSON Schema / GBNF, tools, MCP |
| [media.md](media.md) | Audio, vision, ASR |
| [memory.md](memory.md) | Adaptive memory and task registry API |

License target: MIT OR Apache-2.0. No telemetry.

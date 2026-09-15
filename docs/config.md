# docs/config.md — `runa.toml` and environment keys

Search order (later files win): `~/.config/runa/config.toml`, then `./runa.toml`.
CLI flags beat `RUNA_*` env vars, which beat the files. Inline API keys in TOML
are rejected (P3.8).

## Top level

| Key | Values | Env | CLI |
|-----|--------|-----|-----|
| `on_unfit` | `error` (default) \| `cpu` \| `cloud:<backend>:<model>` | `RUNA_ON_UNFIT` | `--on-unfit` |

## `[models.<alias>]`

| Key | Values |
|-----|--------|
| `source` | Local path or `hf:org/repo:quant` (resolved by `runa pull`, never fetched by `run`) |

## `[mcp.servers.<name>]`

Stdio MCP servers whose tools `runa run` / `runa chat` offer to the model
(P8.3). `--mcp '<command args>'` adds more for one run. See `docs/structured.md`.

| Key | Values |
|-----|--------|
| `command` | Program to start (required) |
| `args` | Array of argument strings |
| `env` | Table of extra environment variables |

```toml
[mcp.servers.fs]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
```

## `[think]`

| Key | Values | Env | CLI |
|-----|--------|-----|-----|
| `mode` | `on` \| `off` \| `budget` \| `effort` | `RUNA_THINK` | `--think` |
| `budget` | reasoning token cap (required when `mode = "budget"`) | `RUNA_THINK_BUDGET` | `--think-budget` |
| `grace` | extra tokens after the budget (default 0) | `RUNA_THINK_GRACE` | (slash `/think grace`) |
| `effort` | `low` \| `medium` \| `high` \| `max` | `RUNA_EFFORT` | `--effort` |
| `show` | bool — print reasoning | `RUNA_SHOW_REASONING` | `--show-reasoning` / `--no-show-reasoning` |

`mode = "budget"` needs `budget`. `--think off` cannot combine with budget/effort.

## `[audio]`

| Key | Values | Env | CLI |
|-----|--------|-----|-----|
| `route` | `auto` (default) \| `native` \| `asr` | `RUNA_AUDIO_ROUTE` | `--audio-route` |

See `docs/media.md`.

## `[memory]`

| Key | Values | Env |
|-----|--------|-----|
| `idle_timeout_s` | seconds before idle shrink | `RUNA_MEMORY_IDLE_TIMEOUT_S` |
| `floor_mib` | idle floor | `RUNA_MEMORY_FLOOR_MIB` |
| `max_growth_mib` | cap on `grow_for` | `RUNA_MEMORY_MAX_GROWTH_MIB` |

See `docs/memory.md`.

## Cloud env (not TOML)

| Env | Use |
|-----|-----|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `OPENAI_BASE_URL` | OpenAI-compatible base URL |
